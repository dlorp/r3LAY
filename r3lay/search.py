"""Three-stage retrieval pipeline: KNN -> RRF fusion -> MMR rerank.

This module implements the shared search pipeline used by both r3LAY and the
knowledge vault. The MMR + RRF code lives here — other consumers import it.

Search stages:
  1. Overfetch (5:1 ratio -> top 50 candidates)
     a. sqlite-vec KNN: top 25 by cosine distance
     b. FTS5 BM25: top 25 by rank * quality_weight
     c. Graph expansion: 2-hop neighbors of top-5 vec results (weight 0.5x in RRF)
  2. RRF fusion + cosine dedup at 0.95 threshold
  3. MMR rerank (NumPy vectorized, NOT LangChain)
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .ingest import embed_text

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result with metadata."""

    chunk_id: str
    content: str
    score: float
    file_path: str = ""
    project_id: str = ""
    file_type: str = ""
    provenance: str = ""
    quality_weight: float = 1.0
    metadata: dict = field(default_factory=dict)


# =============================================================================
# Stage 1: Overfetch
# =============================================================================


def knn_search(
    conn: sqlite3.Connection,
    query_embedding: np.ndarray,
    k: int = 25,
    project_id: str | None = None,
) -> list[tuple[str, float]]:
    """Vector KNN search using sqlite-vec.

    Args:
        conn: SQLite connection with sqlite-vec loaded.
        query_embedding: Query embedding as float32 numpy array.
        k: Number of results to return.
        project_id: Optional filter by project.

    Returns:
        List of (chunk_id, distance) tuples, sorted by ascending distance.
    """
    query_bytes = query_embedding.astype(np.float32).tobytes()

    if project_id:
        rows = conn.execute(
            """SELECT v.chunk_id, v.distance
               FROM vec_chunks v
               JOIN chunks c ON c.chunk_id = v.chunk_id
               WHERE v.embedding MATCH ? AND k = ?
                 AND c.project_id = ?
               ORDER BY v.distance""",
            (query_bytes, k, project_id),
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT chunk_id, distance
               FROM vec_chunks
               WHERE embedding MATCH ? AND k = ?
               ORDER BY distance""",
            (query_bytes, k),
        ).fetchall()

    return [(row[0], row[1]) for row in rows]


def fts_search(
    conn: sqlite3.Connection,
    query: str,
    k: int = 25,
    project_id: str | None = None,
) -> list[tuple[str, float]]:
    """Full-text BM25 search using FTS5.

    Args:
        conn: SQLite connection.
        query: Search query string.
        k: Number of results to return.
        project_id: Optional filter by project.

    Returns:
        List of (chunk_id, score) tuples, sorted by descending score.
    """
    if project_id:
        rows = conn.execute(
            """SELECT c.chunk_id, fts.rank * COALESCE(f.quality_weight, 1.0) AS score
               FROM fts_chunks fts
               JOIN chunks c ON c.rowid = fts.rowid
               LEFT JOIN files f ON f.id = c.file_id
               WHERE fts_chunks MATCH ?
                 AND c.project_id = ?
               ORDER BY score
               LIMIT ?""",
            (query, project_id, k),
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT c.chunk_id, fts.rank * COALESCE(f.quality_weight, 1.0) AS score
               FROM fts_chunks fts
               JOIN chunks c ON c.rowid = fts.rowid
               LEFT JOIN files f ON f.id = c.file_id
               WHERE fts_chunks MATCH ?
               ORDER BY score
               LIMIT ?""",
            (query, k),
        ).fetchall()

    return [(row[0], abs(row[1])) for row in rows]


def graph_expand(
    conn: sqlite3.Connection,
    seed_chunk_ids: list[str],
    max_hops: int = 2,
) -> list[str]:
    """Expand seed chunks via 2-hop graph neighbors.

    Args:
        conn: SQLite connection.
        seed_chunk_ids: Initial chunk IDs from top-5 vector results.
        max_hops: Number of hops to expand.

    Returns:
        List of neighbor chunk IDs (deduplicated).
    """
    visited: set[str] = set(seed_chunk_ids)
    frontier = set(seed_chunk_ids)

    for _ in range(max_hops):
        if not frontier:
            break

        placeholders = ",".join("?" * len(frontier))
        # Get file_ids for current frontier chunks
        file_rows = conn.execute(
            f"SELECT DISTINCT file_id FROM chunks WHERE chunk_id IN ({placeholders})",
            list(frontier),
        ).fetchall()
        file_ids = [r[0] for r in file_rows if r[0]]

        if not file_ids:
            break

        # Follow edges from those files
        fp = ",".join("?" * len(file_ids))
        edge_rows = conn.execute(
            f"""SELECT target_id FROM edges WHERE source_id IN ({fp})
                UNION
                SELECT source_id FROM edges WHERE target_id IN ({fp})""",
            file_ids + file_ids,
        ).fetchall()

        new_ids = set()
        for row in edge_rows:
            target = row[0]
            if target not in visited:
                # Check if target is a chunk or resolve file->chunks
                chunk_rows = conn.execute(
                    "SELECT chunk_id FROM chunks WHERE file_id = ? OR chunk_id = ?",
                    (target, target),
                ).fetchall()
                for cr in chunk_rows:
                    if cr[0] not in visited:
                        new_ids.add(cr[0])

        visited.update(new_ids)
        frontier = new_ids

    return list(visited - set(seed_chunk_ids))


# =============================================================================
# Stage 2: RRF Fusion
# =============================================================================


def rrf_fuse(
    ranked_lists: list[list[tuple[str, float]]],
    weights: list[float] | None = None,
    k: int = 60,
) -> dict[str, float]:
    """Reciprocal Rank Fusion across multiple retrieval signals.

    score(d) = sum(weight_i / (k + rank_i)) for each retrieval signal

    Args:
        ranked_lists: List of ranked result lists, each being (chunk_id, score).
        weights: Optional per-list weights. Defaults to 1.0 for all.
        k: RRF constant (default 60 per the paper).

    Returns:
        Dict mapping chunk_id -> fused RRF score, sorted by descending score.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    scores: dict[str, float] = {}

    for ranked_list, weight in zip(ranked_lists, weights):
        for rank, (chunk_id, _) in enumerate(ranked_list):
            if chunk_id not in scores:
                scores[chunk_id] = 0.0
            scores[chunk_id] += weight / (k + rank + 1)

    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))


def cosine_dedup(
    conn: sqlite3.Connection,
    chunk_ids: list[str],
    threshold: float = 0.95,
) -> list[str]:
    """Remove near-duplicate chunks by cosine similarity.

    Args:
        conn: SQLite connection with sqlite-vec loaded.
        chunk_ids: Candidate chunk IDs to deduplicate.
        threshold: Cosine similarity threshold for dedup.

    Returns:
        Deduplicated list of chunk IDs.
    """
    if len(chunk_ids) <= 1:
        return chunk_ids

    # Fetch embeddings for candidates
    embeddings = {}
    for cid in chunk_ids:
        row = conn.execute("SELECT embedding FROM vec_chunks WHERE chunk_id = ?", (cid,)).fetchone()
        if row:
            embeddings[cid] = np.frombuffer(row[0], dtype=np.float32)

    if not embeddings:
        return chunk_ids

    # Greedy dedup: keep first, skip any too similar to kept set
    kept = []
    kept_embs = []

    for cid in chunk_ids:
        if cid not in embeddings:
            kept.append(cid)
            continue

        emb = embeddings[cid]
        norm = np.linalg.norm(emb)
        if norm == 0:
            kept.append(cid)
            continue

        is_dup = False
        for kept_emb in kept_embs:
            sim = np.dot(emb, kept_emb) / (norm * np.linalg.norm(kept_emb))
            if sim >= threshold:
                is_dup = True
                break

        if not is_dup:
            kept.append(cid)
            kept_embs.append(emb)

    return kept


# =============================================================================
# Stage 3: MMR Rerank
# =============================================================================


def mmr_select(
    query_emb: np.ndarray,
    candidate_embs: np.ndarray,
    lambda_mult: float = 0.7,
    k: int = 10,
) -> list[int]:
    """Maximal Marginal Relevance selection (NumPy vectorized).

    Balances relevance to query with diversity among selected results.

    Args:
        query_emb: Query embedding, shape (dim,).
        candidate_embs: Candidate embeddings, shape (n, dim).
        lambda_mult: Balance between relevance (1.0) and diversity (0.0).
        k: Number of results to select.

    Returns:
        List of selected indices into candidate_embs.
    """
    n = candidate_embs.shape[0]
    if n == 0:
        return []
    k = min(k, n)

    # Normalize for cosine similarity
    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)
    norms = np.linalg.norm(candidate_embs, axis=1, keepdims=True) + 1e-10
    cand_normed = candidate_embs / norms

    sim_to_query = (cand_normed @ query_norm).flatten()
    sim_between = cand_normed @ cand_normed.T

    selected = [int(np.argmax(sim_to_query))]

    for _ in range(1, k):
        redundant = np.max(sim_between[:, selected], axis=1)
        scores = lambda_mult * sim_to_query - (1 - lambda_mult) * redundant
        scores[selected] = -np.inf
        selected.append(int(np.argmax(scores)))

    return selected


# =============================================================================
# Full Pipeline
# =============================================================================


async def search(
    conn: sqlite3.Connection,
    query: str,
    k: int = 10,
    project_id: str | None = None,
    lambda_mult: float = 0.7,
) -> list[SearchResult]:
    """Execute the full three-stage retrieval pipeline.

    Stage 1: Overfetch (KNN + FTS5 + graph expansion)
    Stage 2: RRF fusion + cosine dedup
    Stage 3: MMR rerank

    Args:
        conn: SQLite connection with extensions loaded.
        query: Search query string.
        k: Number of final results to return.
        project_id: Optional filter to a specific project.
        lambda_mult: MMR diversity parameter (0.7 = relevance-heavy).

    Returns:
        List of SearchResult objects, ranked by MMR score.
    """
    # Stage 1a: KNN vector search
    query_emb = np.array(await embed_text(query, prefix="query: "), dtype=np.float32)
    knn_results = knn_search(conn, query_emb, k=25, project_id=project_id)

    # Stage 1b: FTS5 BM25 search
    fts_results = fts_search(conn, query, k=25, project_id=project_id)

    # Stage 1c: Graph expansion from top-5 vector results
    top5_ids = [cid for cid, _ in knn_results[:5]]
    graph_neighbors = graph_expand(conn, top5_ids)
    graph_results = [(cid, 0.0) for cid in graph_neighbors]

    # Stage 2: RRF fusion (graph signal at 0.5x weight)
    fused_scores = rrf_fuse(
        [knn_results, fts_results, graph_results],
        weights=[1.0, 1.0, 0.5],
    )

    # Cosine dedup at 0.95 threshold
    candidate_ids = list(fused_scores.keys())
    deduped_ids = cosine_dedup(conn, candidate_ids)

    if not deduped_ids:
        return []

    # Stage 3: MMR rerank
    # Fetch embeddings for deduped candidates
    cand_embs = []
    valid_ids = []
    for cid in deduped_ids:
        row = conn.execute("SELECT embedding FROM vec_chunks WHERE chunk_id = ?", (cid,)).fetchone()
        if row:
            cand_embs.append(np.frombuffer(row[0], dtype=np.float32))
            valid_ids.append(cid)

    if not cand_embs:
        return []

    cand_matrix = np.stack(cand_embs)
    selected_indices = mmr_select(query_emb, cand_matrix, lambda_mult=lambda_mult, k=k)

    # Build results with metadata
    results = []
    for idx in selected_indices:
        cid = valid_ids[idx]
        row = conn.execute(
            """SELECT c.content, c.project_id, f.path, f.file_type, f.provenance, f.quality_weight
               FROM chunks c
               LEFT JOIN files f ON f.id = c.file_id
               WHERE c.chunk_id = ?""",
            (cid,),
        ).fetchone()

        if row:
            results.append(
                SearchResult(
                    chunk_id=cid,
                    content=row[0],
                    score=fused_scores.get(cid, 0.0),
                    file_path=row[2] or "",
                    project_id=row[1] or "",
                    file_type=row[3] or "",
                    provenance=row[4] or "",
                    quality_weight=row[5] or 1.0,
                )
            )

    return results


def main() -> None:
    """CLI entry point for search testing."""
    import argparse
    import asyncio

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Search r3LAY index")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--k", type=int, default=10, help="Number of results")
    parser.add_argument("--project", type=str, default=None, help="Filter by project ID")
    parser.add_argument("--db", type=str, default=None, help="Database path override")
    args = parser.parse_args()

    from .db import get_db

    db_path = Path(args.db) if args.db else None
    conn = get_db(db_path)

    results = asyncio.run(search(conn, args.query, k=args.k, project_id=args.project))

    for i, r in enumerate(results, 1):
        print(f"\n--- Result {i} (score: {r.score:.4f}) ---")
        print(f"File: {r.file_path}")
        print(f"Type: {r.file_type} | Provenance: {r.provenance} | Weight: {r.quality_weight}")
        print(r.content[:300])

    conn.close()

    if not results:
        print("\nNo results found.")


if __name__ == "__main__":
    main()
