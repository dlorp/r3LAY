"""FAISS-backed vector storage for hybrid search.

Replaces raw NumPy .npy storage with FAISS indexes for faster similarity search.
Supports exact (IndexFlatIP) for small collections and approximate (IndexIVFFlat)
for larger ones.

Falls back to a NumPy-based implementation if faiss-cpu is not installed.

Usage:
    store = create_vector_store(dimension=384, persist_path=Path("./index"))
    store.add(vectors, ids=["chunk1", "chunk2"])
    results = store.search(query_vector, k=10)
    store.save()
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import math
import threading
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

FAISS_AVAILABLE = importlib.util.find_spec("faiss") is not None

# Threshold for switching from flat to IVF index
IVF_THRESHOLD = 100_000


class VectorStoreBase(ABC):
    """Abstract interface for vector stores.

    Both FAISSVectorStore and NumpyFallbackStore implement this interface
    so HybridIndex can use either without changes.
    """

    @abstractmethod
    def add(self, vectors: np.ndarray, ids: list[str] | None = None) -> None:
        """Add vectors to the index.

        Args:
            vectors: Array of shape (n, dimension).
            ids: Optional chunk IDs to associate with each vector.
        """

    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int = 10) -> list[tuple[int, float]]:
        """Search for k nearest neighbors.

        Args:
            query_vector: Query vector of shape (dimension,).
            k: Number of results to return.

        Returns:
            List of (index, score) tuples. Score is cosine similarity (0-1 range).
        """

    @abstractmethod
    def remove(self, indices: list[int]) -> None:
        """Remove vectors by index position.

        Args:
            indices: List of vector indices to remove.
        """

    @abstractmethod
    def save(self) -> None:
        """Persist index to disk."""

    @abstractmethod
    def load(self) -> bool:
        """Load index from disk.

        Returns:
            True if loaded successfully, False if no index exists.
        """

    @property
    @abstractmethod
    def count(self) -> int:
        """Number of vectors in the index."""

    @abstractmethod
    def rebuild_index(self) -> None:
        """Rebuild index structure (e.g., switch from Flat to IVF if needed)."""

    @classmethod
    @abstractmethod
    def from_numpy(cls, vectors: np.ndarray, persist_path: Path | None = None) -> VectorStoreBase:
        """Create a store from existing NumPy vectors (migration helper).

        Args:
            vectors: Array of shape (n, dimension).
            persist_path: Optional path for persistence.

        Returns:
            Initialized vector store with the provided vectors.
        """

    def get_id(self, index: int) -> str | None:
        """Get the chunk ID for a given vector index.

        Args:
            index: Vector position in the store.

        Returns:
            The chunk ID string, or None if no mapping exists.
        """
        return None


class FAISSVectorStore(VectorStoreBase):
    """FAISS-backed vector storage for hybrid search.

    Replaces raw NumPy .npy storage with FAISS indexes for faster
    similarity search. Supports exact (IndexFlatIP) for small collections
    and approximate (IndexIVFFlat) for larger ones.

    All vectors are L2-normalized before insertion so that inner product
    equals cosine similarity.

    Thread safety:
        Read operations (search) are thread-safe in FAISS.
        Write operations (add/remove/rebuild/save) are protected by a lock.
    """

    def __init__(self, dimension: int, persist_path: Path | None = None) -> None:
        """Initialize store. Loads existing index from persist_path if available.

        Args:
            dimension: Embedding vector dimensionality.
            persist_path: Directory for persisting index and ID mappings.
        """
        import faiss

        self._faiss = faiss
        self._dimension = dimension
        self._persist_path = persist_path
        self._write_lock = threading.Lock()

        # ID mapping: faiss internal index -> chunk_id
        self._id_map: dict[int, str] = {}
        self._next_id: int = 0

        # Create initial flat index
        self._index: faiss.Index = faiss.IndexFlatIP(dimension)
        self._is_ivf = False

        # Try loading from disk
        if persist_path is not None:
            self.load()

    @property
    def _index_file(self) -> Path | None:
        """Path to the FAISS index file."""
        if self._persist_path is None:
            return None
        return self._persist_path / "faiss.index"

    @property
    def _id_map_file(self) -> Path | None:
        """Path to the ID mapping JSON file."""
        if self._persist_path is None:
            return None
        return self._persist_path / "faiss_id_map.json"

    @property
    def _hash_file(self) -> Path | None:
        """Path to the FAISS index integrity hash file."""
        if self._persist_path is None:
            return None
        return self._persist_path / "faiss.index.sha256"

    @staticmethod
    def _compute_file_hash(path: Path) -> str:
        """Compute SHA-256 hash of a file for integrity verification."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def add(self, vectors: np.ndarray, ids: list[str] | None = None) -> None:
        """Add vectors to the index.

        Vectors are L2-normalized in-place before insertion so inner product
        equals cosine similarity.

        Args:
            vectors: Array of shape (n, dimension). Modified in-place (normalized).
            ids: Optional chunk IDs. If None, indices are auto-assigned.

        Raises:
            ValueError: If vectors have wrong dimensionality.
        """
        if vectors.ndim != 2 or vectors.shape[1] != self._dimension:
            raise ValueError(
                f"Expected vectors of shape (n, {self._dimension}), got {vectors.shape}"
            )

        # Always copy to avoid modifying caller's array during normalization
        vectors = np.array(vectors, dtype=np.float32, copy=True)
        vectors = np.ascontiguousarray(vectors)
        self._faiss.normalize_L2(vectors)

        with self._write_lock:
            start_idx = self._next_id
            n = vectors.shape[0]

            if self._is_ivf and not self._index.is_trained:
                logger.warning("IVF index not trained, falling back to flat index")
                self._index = self._faiss.IndexFlatIP(self._dimension)
                self._is_ivf = False

            self._index.add(vectors)

            # Update ID mapping
            for i in range(n):
                faiss_idx = start_idx + i
                if ids is not None:
                    self._id_map[faiss_idx] = ids[i]
                else:
                    self._id_map[faiss_idx] = str(faiss_idx)

            self._next_id = start_idx + n

        logger.debug("Added %d vectors to FAISS index (total: %d)", n, self.count)

    def search(self, query_vector: np.ndarray, k: int = 10) -> list[tuple[int, float]]:
        """Search for k nearest neighbors.

        The query vector is L2-normalized so inner product equals cosine similarity.
        Thread-safe: snapshots the index reference to avoid TOCTOU races with
        concurrent write operations (add/remove/rebuild).

        Args:
            query_vector: Query vector of shape (dimension,).
            k: Number of results to return.

        Returns:
            List of (index, score) tuples sorted by descending score.
            Score is cosine similarity clamped to [0, 1].
        """
        # Snapshot index reference for thread safety
        index = self._index
        count = index.ntotal
        if count == 0:
            return []

        query = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)
        self._faiss.normalize_L2(query)

        # Clamp k to available vectors
        effective_k = min(k, count)

        scores, indices = index.search(query, effective_k)

        results: list[tuple[int, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                # FAISS returns -1 for missing results
                continue
            # Clamp score: IP on normalized vectors is in [-1, 1], map to [0, 1]
            clamped = max(0.0, float(score))
            results.append((int(idx), clamped))

        return results

    def remove(self, indices: list[int]) -> None:
        """Remove vectors by index position.

        FAISS flat indexes don't support direct removal, so this rebuilds
        the index without the specified vectors.

        Args:
            indices: List of FAISS internal indices to remove.
        """
        if not indices:
            return

        with self._write_lock:
            remove_set = set(indices)
            n = self._index.ntotal

            if n == 0:
                return

            # Reconstruct all vectors
            all_vectors = (
                self._faiss.rev_swig_ptr(self._index.get_xb(), n * self._dimension)
                .reshape(n, self._dimension)
                .copy()
            )

            # Filter out removed indices
            keep_mask = np.ones(n, dtype=bool)
            for idx in remove_set:
                if 0 <= idx < n:
                    keep_mask[idx] = False

            kept_vectors = all_vectors[keep_mask]

            # Rebuild ID mapping
            old_to_new: dict[int, int] = {}
            new_id_map: dict[int, str] = {}
            new_idx = 0
            for old_idx in range(n):
                if keep_mask[old_idx]:
                    old_to_new[old_idx] = new_idx
                    if old_idx in self._id_map:
                        new_id_map[new_idx] = self._id_map[old_idx]
                    new_idx += 1

            # Reset index
            self._index = self._faiss.IndexFlatIP(self._dimension)
            self._is_ivf = False

            if len(kept_vectors) > 0:
                self._index.add(kept_vectors)

            self._id_map = new_id_map
            self._next_id = new_idx

        logger.debug(
            "Removed %d vectors, %d remaining",
            len(remove_set),
            self.count,
        )

    def save(self) -> None:
        """Persist index, ID mapping, and integrity hash to disk.

        Raises:
            RuntimeError: If no persist_path was configured.
        """
        if self._persist_path is None or self._index_file is None or self._id_map_file is None:
            raise RuntimeError("Cannot save: no persist_path configured")

        with self._write_lock:
            self._persist_path.mkdir(parents=True, exist_ok=True)
            self._faiss.write_index(self._index, str(self._index_file))

            # Write integrity hash for the FAISS index
            if self._hash_file is not None:
                file_hash = self._compute_file_hash(self._index_file)
                self._hash_file.write_text(file_hash, encoding="utf-8")

            # Store ID map with string keys for JSON compatibility
            serializable = {str(k): v for k, v in self._id_map.items()}
            serializable["__next_id"] = str(self._next_id)
            self._id_map_file.write_text(json.dumps(serializable), encoding="utf-8")

        logger.info(
            "Saved FAISS index (%d vectors) to %s",
            self.count,
            self._persist_path,
        )

    def load(self) -> bool:
        """Load index and ID mapping from disk.

        Returns:
            True if loaded successfully, False if no index file exists.
        """
        if self._index_file is None or self._id_map_file is None:
            return False

        if not self._index_file.exists():
            return False

        try:
            # Verify FAISS index integrity before loading
            if self._hash_file is not None and self._hash_file.exists():
                expected = self._hash_file.read_text(encoding="utf-8").strip()
                actual = self._compute_file_hash(self._index_file)
                if expected != actual:
                    logger.error(
                        "FAISS index integrity check failed: %s (expected %s, got %s)",
                        self._index_file,
                        expected[:16],
                        actual[:16],
                    )
                    return False

            self._index = self._faiss.read_index(str(self._index_file))
            self._is_ivf = hasattr(self._index, "nprobe")

            # Load ID mapping
            if self._id_map_file.exists():
                raw = json.loads(self._id_map_file.read_text(encoding="utf-8"))
                self._next_id = int(raw.pop("__next_id", "0"))
                self._id_map = {int(k): v for k, v in raw.items()}
            else:
                # No ID map file — generate sequential IDs
                self._id_map = {i: str(i) for i in range(self._index.ntotal)}
                self._next_id = self._index.ntotal

            logger.info(
                "Loaded FAISS index (%d vectors, ivf=%s) from %s",
                self.count,
                self._is_ivf,
                self._persist_path,
            )
            return True

        except Exception:
            logger.exception("Failed to load FAISS index from %s", self._persist_path)
            # Reset to empty
            self._index = self._faiss.IndexFlatIP(self._dimension)
            self._is_ivf = False
            self._id_map = {}
            self._next_id = 0
            return False

    @property
    def count(self) -> int:
        """Number of vectors in the index."""
        return self._index.ntotal

    def rebuild_index(self) -> None:
        """Rebuild index, switching from Flat to IVF if above threshold.

        For collections >= IVF_THRESHOLD vectors, creates an IVFFlat index
        with nlist = sqrt(n) and nprobe = min(nlist, 10). The IVF index
        is trained on the existing vectors before adding them.
        """
        with self._write_lock:
            n = self._index.ntotal
            if n == 0:
                return

            # Reconstruct all vectors from current index
            all_vectors = (
                self._faiss.rev_swig_ptr(self._index.get_xb(), n * self._dimension)
                .reshape(n, self._dimension)
                .copy()
            )

            if n >= IVF_THRESHOLD:
                nlist = int(math.sqrt(n))
                nlist = max(1, nlist)

                quantizer = self._faiss.IndexFlatIP(self._dimension)
                new_index = self._faiss.IndexIVFFlat(
                    quantizer, self._dimension, nlist, self._faiss.METRIC_INNER_PRODUCT
                )
                new_index.train(all_vectors)
                new_index.nprobe = min(nlist, 10)
                new_index.add(all_vectors)

                self._index = new_index
                self._is_ivf = True
                logger.info(
                    "Rebuilt as IVFFlat index (nlist=%d, nprobe=%d, vectors=%d)",
                    nlist,
                    new_index.nprobe,
                    n,
                )
            else:
                new_index = self._faiss.IndexFlatIP(self._dimension)
                new_index.add(all_vectors)
                self._index = new_index
                self._is_ivf = False
                logger.info("Rebuilt as Flat index (vectors=%d)", n)

    @classmethod
    def from_numpy(
        cls,
        vectors: np.ndarray,
        persist_path: Path | None = None,
    ) -> FAISSVectorStore:
        """Create a FAISSVectorStore from existing NumPy vectors.

        This is a migration helper for converting .npy-based indexes to FAISS.
        Vectors are L2-normalized during insertion.

        Args:
            vectors: Array of shape (n, dimension).
            persist_path: Optional directory for persistence.

        Returns:
            Initialized FAISSVectorStore containing the provided vectors.
        """
        if vectors.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {vectors.shape}")

        dimension = vectors.shape[1]
        store = cls(dimension=dimension, persist_path=persist_path)
        store.add(vectors)
        return store

    def get_id(self, index: int) -> str | None:
        """Get the chunk ID for a given vector index.

        Args:
            index: FAISS internal index position.

        Returns:
            The chunk ID string, or None if not found.
        """
        return self._id_map.get(index)


class NumpyFallbackStore(VectorStoreBase):
    """NumPy-based vector store fallback when FAISS is not installed.

    Uses brute-force cosine similarity via normalized dot product.
    Same interface as FAISSVectorStore so HybridIndex can use either.
    """

    def __init__(self, dimension: int, persist_path: Path | None = None) -> None:
        """Initialize store.

        Args:
            dimension: Embedding vector dimensionality.
            persist_path: Directory for persisting vectors and ID mappings.
        """
        self._dimension = dimension
        self._persist_path = persist_path
        self._write_lock = threading.Lock()

        self._vectors: np.ndarray | None = None  # Shape: (n, dimension), normalized
        self._id_map: dict[int, str] = {}
        self._next_id: int = 0

        if persist_path is not None:
            self.load()

    @property
    def _vectors_file(self) -> Path | None:
        """Path to the NumPy vectors file."""
        if self._persist_path is None:
            return None
        return self._persist_path / "vectors.npy"

    @property
    def _id_map_file(self) -> Path | None:
        """Path to the ID mapping JSON file."""
        if self._persist_path is None:
            return None
        return self._persist_path / "numpy_id_map.json"

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """L2-normalize vectors for cosine similarity via dot product.

        Args:
            vectors: Array of shape (n, dimension).

        Returns:
            Normalized copy of the input array.
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.maximum(norms, 1e-9)
        return vectors / norms

    def add(self, vectors: np.ndarray, ids: list[str] | None = None) -> None:
        """Add vectors to the store.

        Args:
            vectors: Array of shape (n, dimension).
            ids: Optional chunk IDs.

        Raises:
            ValueError: If vectors have wrong dimensionality.
        """
        if vectors.ndim != 2 or vectors.shape[1] != self._dimension:
            raise ValueError(
                f"Expected vectors of shape (n, {self._dimension}), got {vectors.shape}"
            )

        normalized = self._normalize(vectors.astype(np.float32))

        with self._write_lock:
            start_idx = self._next_id
            n = vectors.shape[0]

            if self._vectors is None:
                self._vectors = normalized
            else:
                self._vectors = np.vstack([self._vectors, normalized])

            for i in range(n):
                idx = start_idx + i
                if ids is not None:
                    self._id_map[idx] = ids[i]
                else:
                    self._id_map[idx] = str(idx)

            self._next_id = start_idx + n

        logger.debug("Added %d vectors to NumPy store (total: %d)", n, self.count)

    def search(self, query_vector: np.ndarray, k: int = 10) -> list[tuple[int, float]]:
        """Search for k nearest neighbors using brute-force cosine similarity.

        Args:
            query_vector: Query vector of shape (dimension,).
            k: Number of results to return.

        Returns:
            List of (index, score) tuples sorted by descending score.
        """
        if self._vectors is None or len(self._vectors) == 0:
            return []

        query = query_vector.astype(np.float32).reshape(1, -1)
        query_norm = self._normalize(query).squeeze()

        similarities = self._vectors @ query_norm

        effective_k = min(k, len(similarities))
        top_indices = np.argsort(similarities)[::-1][:effective_k]

        results: list[tuple[int, float]] = []
        for idx in top_indices:
            score = max(0.0, float(similarities[idx]))
            results.append((int(idx), score))

        return results

    def remove(self, indices: list[int]) -> None:
        """Remove vectors by index position.

        Args:
            indices: List of vector indices to remove.
        """
        if not indices or self._vectors is None:
            return

        with self._write_lock:
            n = len(self._vectors)
            remove_set = set(indices)

            keep_mask = np.ones(n, dtype=bool)
            for idx in remove_set:
                if 0 <= idx < n:
                    keep_mask[idx] = False

            self._vectors = self._vectors[keep_mask]

            # Rebuild ID mapping with new indices
            new_id_map: dict[int, str] = {}
            new_idx = 0
            for old_idx in range(n):
                if keep_mask[old_idx]:
                    if old_idx in self._id_map:
                        new_id_map[new_idx] = self._id_map[old_idx]
                    new_idx += 1

            self._id_map = new_id_map
            self._next_id = new_idx

            if len(self._vectors) == 0:
                self._vectors = None

        logger.debug("Removed %d vectors, %d remaining", len(remove_set), self.count)

    def save(self) -> None:
        """Persist vectors and ID mapping to disk.

        Raises:
            RuntimeError: If no persist_path was configured.
        """
        if self._persist_path is None or self._vectors_file is None or self._id_map_file is None:
            raise RuntimeError("Cannot save: no persist_path configured")

        with self._write_lock:
            self._persist_path.mkdir(parents=True, exist_ok=True)

            if self._vectors is not None and len(self._vectors) > 0:
                np.save(self._vectors_file, self._vectors)
            elif self._vectors_file.exists():
                self._vectors_file.unlink()

            serializable = {str(k): v for k, v in self._id_map.items()}
            serializable["__next_id"] = str(self._next_id)
            self._id_map_file.write_text(json.dumps(serializable), encoding="utf-8")

        logger.info(
            "Saved NumPy vectors (%d) to %s",
            self.count,
            self._persist_path,
        )

    def load(self) -> bool:
        """Load vectors and ID mapping from disk.

        Returns:
            True if loaded successfully, False if no vectors file exists.
        """
        if self._vectors_file is None or self._id_map_file is None:
            return False

        if not self._vectors_file.exists():
            return False

        try:
            self._vectors = np.load(self._vectors_file, allow_pickle=False)

            if self._id_map_file.exists():
                raw = json.loads(self._id_map_file.read_text(encoding="utf-8"))
                self._next_id = int(raw.pop("__next_id", "0"))
                self._id_map = {int(k): v for k, v in raw.items()}
            else:
                n = len(self._vectors)
                self._id_map = {i: str(i) for i in range(n)}
                self._next_id = n

            logger.info(
                "Loaded %d NumPy vectors from %s",
                self.count,
                self._persist_path,
            )
            return True

        except Exception:
            logger.exception("Failed to load NumPy vectors from %s", self._persist_path)
            self._vectors = None
            self._id_map = {}
            self._next_id = 0
            return False

    @property
    def count(self) -> int:
        """Number of vectors in the store."""
        if self._vectors is None:
            return 0
        return len(self._vectors)

    def rebuild_index(self) -> None:
        """No-op for NumPy store (no index structure to rebuild)."""
        logger.debug("NumPy fallback store has no index to rebuild")

    @classmethod
    def from_numpy(
        cls,
        vectors: np.ndarray,
        persist_path: Path | None = None,
    ) -> NumpyFallbackStore:
        """Create a NumpyFallbackStore from existing NumPy vectors.

        Args:
            vectors: Array of shape (n, dimension).
            persist_path: Optional directory for persistence.

        Returns:
            Initialized NumpyFallbackStore containing the provided vectors.
        """
        if vectors.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {vectors.shape}")

        dimension = vectors.shape[1]
        store = cls(dimension=dimension, persist_path=persist_path)
        store.add(vectors)
        return store

    def get_id(self, index: int) -> str | None:
        """Get the chunk ID for a given vector index.

        Args:
            index: Vector position in the store.

        Returns:
            The chunk ID string, or None if not found.
        """
        return self._id_map.get(index)


def create_vector_store(
    dimension: int,
    persist_path: Path | None = None,
    force_numpy: bool = False,
) -> VectorStoreBase:
    """Factory function to create the best available vector store.

    Prefers FAISS when available, falls back to NumPy brute-force search.

    Args:
        dimension: Embedding vector dimensionality.
        persist_path: Optional directory for persistence.
        force_numpy: If True, use NumPy fallback even if FAISS is available.

    Returns:
        A VectorStoreBase implementation (FAISSVectorStore or NumpyFallbackStore).
    """
    if FAISS_AVAILABLE and not force_numpy:
        logger.info("Using FAISS vector store (dimension=%d)", dimension)
        return FAISSVectorStore(dimension=dimension, persist_path=persist_path)

    if not force_numpy:
        logger.warning(
            "faiss-cpu not installed, using NumPy fallback. Install with: pip install faiss-cpu"
        )
    return NumpyFallbackStore(dimension=dimension, persist_path=persist_path)
