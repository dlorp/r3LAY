#!/usr/bin/env python3
"""Cross-encoder reranker worker subprocess for isolated reranking.

This module runs as a standalone subprocess to completely isolate cross-encoder
model interactions from Textual's TUI. This avoids fd conflicts with
sentence-transformers/transformers tokenizer parallelism.

Communication happens via JSON lines over stdin/stdout:
- stdin: JSON commands (one per line)
- stdout: JSON responses (one per line)
- stderr: redirected to /dev/null

Commands:
    {"cmd": "load", "model": "cross-encoder/ms-marco-MiniLM-L-6-v2"}
    {"cmd": "rerank", "query": "...", "passages": ["...", "..."], "top_k": 10, "threshold": 0.35}
    {"cmd": "unload"}

Responses:
    {"type": "loaded", "success": true, "model": "..."}
    {"type": "loaded", "success": false, "error": "message"}
    {"type": "reranked", "results": [{"index": 0, "score": 0.87, "passage": "..."}, ...]}
    {"type": "error", "message": "..."}

Usage:
    python -m r3lay.core.reranker_worker
"""

from __future__ import annotations

import gc
import json
import os
import sys
from typing import Any

# Minimum number of words in a query to perform actual reranking.
# Queries shorter than this get uniform scores (reranking would be meaningless).
MIN_QUERY_WORDS = 5


def setup_isolation() -> None:
    """Set up environment isolation before any imports.

    MUST be called before importing transformers, sentence-transformers, etc.
    """
    # Disable terminal features
    os.environ["TERM"] = "dumb"
    os.environ["NO_COLOR"] = "1"
    os.environ["FORCE_COLOR"] = "0"
    os.environ["TQDM_DISABLE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    # Redirect stderr to /dev/null (keep stdout for JSON responses)
    sys.stderr = open(os.devnull, "w")  # noqa: SIM115


def send_response(response: dict[str, Any]) -> None:
    """Send a JSON response to stdout."""
    print(json.dumps(response), flush=True)


def main() -> None:
    """Main worker loop."""
    # Set up isolation FIRST
    setup_isolation()

    model = None
    model_name: str = ""

    # Read commands from stdin
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            cmd = json.loads(line)
        except json.JSONDecodeError as e:
            send_response({"type": "error", "message": f"Invalid JSON: {e}"})
            continue

        cmd_type = cmd.get("cmd")

        if cmd_type == "load":
            requested_model = cmd.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")

            try:
                from sentence_transformers import CrossEncoder

                model = CrossEncoder(requested_model)
                model_name = requested_model

                send_response(
                    {
                        "type": "loaded",
                        "success": True,
                        "model": model_name,
                    }
                )

            except ImportError:
                send_response(
                    {
                        "type": "loaded",
                        "success": False,
                        "error": (
                            "sentence-transformers is required for cross-encoder reranking. "
                            "Install with: pip install sentence-transformers"
                        ),
                    }
                )

            except Exception as e:
                send_response(
                    {
                        "type": "loaded",
                        "success": False,
                        "error": f"Failed to load cross-encoder model: {e}",
                    }
                )

        elif cmd_type == "rerank":
            if model is None:
                send_response({"type": "error", "message": "Model not loaded"})
                continue

            query = cmd.get("query", "")
            passages = cmd.get("passages", [])
            top_k = cmd.get("top_k", 10)
            threshold = cmd.get("threshold", 0.35)

            if not passages:
                send_response({"type": "reranked", "results": []})
                continue

            try:
                # Smart skip: if query is too short, reranking is meaningless
                query_words = len(query.split())
                if query_words < MIN_QUERY_WORDS:
                    # Return all passages with uniform score, preserving original order
                    uniform_score = 1.0
                    results = [
                        {"index": i, "score": uniform_score, "passage": p}
                        for i, p in enumerate(passages[:top_k])
                    ]
                    send_response({"type": "reranked", "results": results})
                    continue

                # Score all query-passage pairs
                pairs = [(query, passage) for passage in passages]
                scores = model.predict(pairs)

                # Convert numpy types to native Python floats
                scored = []
                for i, (score, passage) in enumerate(zip(scores, passages)):
                    score_val = float(score)
                    if score_val > threshold:
                        scored.append(
                            {
                                "index": i,
                                "score": score_val,
                                "passage": passage,
                            }
                        )

                # Sort by score descending, limit to top_k
                scored.sort(key=lambda x: x["score"], reverse=True)
                results = scored[:top_k]

                send_response({"type": "reranked", "results": results})

            except Exception as e:
                send_response({"type": "error", "message": f"Reranking error: {e}"})

        elif cmd_type == "unload":
            # Cleanup and exit
            if model is not None:
                try:
                    del model
                    gc.collect()

                    # Try to clear torch MPS cache if available
                    try:
                        import torch

                        if torch.backends.mps.is_available():
                            torch.mps.empty_cache()
                    except (ImportError, AttributeError):
                        pass

                except Exception:
                    pass

            break

        else:
            send_response({"type": "error", "message": f"Unknown command: {cmd_type}"})


if __name__ == "__main__":
    main()
