#!/usr/bin/env python3
"""MLX text embedding worker subprocess for isolated embedding generation.

This module runs as a standalone subprocess to completely isolate embedding
model interactions from Textual's TUI. This avoids fd conflicts with
sentence-transformers/transformers tokenizer parallelism.

Communication happens via JSON lines over stdin/stdout:
- stdin: JSON commands (one per line)
- stdout: JSON responses (one per line)
- stderr: redirected to /dev/null

Commands:
    {"cmd": "load", "model": "sentence-transformers/all-MiniLM-L6-v2"}
    {"cmd": "embed", "texts": ["Hello", "World"]}
    {"cmd": "unload"}

Responses:
    {"type": "loaded", "success": true, "dimension": 384}
    {"type": "loaded", "success": false, "error": "message"}
    {"type": "embeddings", "vectors": "<base64>", "shape": [2, 384], "dtype": "float32"}
    {"type": "error", "message": "..."}

Usage:
    python -m r3lay.core.embeddings.mlx_text_worker
"""

from __future__ import annotations

import base64
import gc
import json
import os
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


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
    sys.stderr = open(os.devnull, "w")


def send_response(response: dict[str, Any]) -> None:
    """Send a JSON response to stdout."""
    print(json.dumps(response), flush=True)


def encode_array(arr: "np.ndarray") -> tuple[str, list[int], str]:
    """Encode numpy array as base64 for JSON transport.

    Returns:
        Tuple of (base64_data, shape, dtype_str)
    """
    data = base64.b64encode(arr.tobytes()).decode("ascii")
    return data, list(arr.shape), str(arr.dtype)


def main() -> None:
    """Main worker loop."""
    # Set up isolation FIRST
    setup_isolation()

    model = None
    dimension: int = 0

    # Import numpy after isolation (it's safe, but be consistent)
    import numpy as np

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
            requested_model = cmd.get("model", "sentence-transformers/all-MiniLM-L6-v2")

            try:
                # Try mlx-embeddings first (native MLX)
                try:
                    from mlx_embeddings import load as mlx_load

                    model = mlx_load(requested_model)

                    # Get dimension by embedding a test string
                    test_embedding = model.encode(["test"])
                    if hasattr(test_embedding, "shape"):
                        dimension = test_embedding.shape[-1]
                    else:
                        dimension = len(test_embedding[0])

                    send_response(
                        {
                            "type": "loaded",
                            "success": True,
                            "dimension": dimension,
                            "backend": "mlx_embeddings",
                        }
                    )
                    continue

                except ImportError:
                    pass

                # Fall back to sentence-transformers
                try:
                    from sentence_transformers import SentenceTransformer

                    # Use MPS on Apple Silicon if available
                    device = "mps" if _mps_available() else "cpu"
                    model = SentenceTransformer(requested_model, device=device)

                    # Get embedding dimension
                    dimension = model.get_sentence_embedding_dimension()

                    send_response(
                        {
                            "type": "loaded",
                            "success": True,
                            "dimension": dimension,
                            "backend": "sentence_transformers",
                            "device": device,
                        }
                    )
                    continue

                except ImportError:
                    pass

                # No embedding library available
                send_response(
                    {
                        "type": "loaded",
                        "success": False,
                        "error": (
                            "No embedding library available. "
                            "Install with: pip install sentence-transformers "
                            "or pip install mlx-embeddings"
                        ),
                    }
                )

            except Exception as e:
                send_response(
                    {
                        "type": "loaded",
                        "success": False,
                        "error": f"Failed to load model: {e}",
                    }
                )

        elif cmd_type == "embed":
            if model is None:
                send_response({"type": "error", "message": "Model not loaded"})
                continue

            texts = cmd.get("texts", [])
            if not texts:
                send_response(
                    {
                        "type": "embeddings",
                        "vectors": "",
                        "shape": [0, dimension],
                        "dtype": "float32",
                    }
                )
                continue

            try:
                # Generate embeddings
                embeddings = model.encode(texts, convert_to_numpy=True)

                # Ensure it's a numpy array with correct dtype
                if not isinstance(embeddings, np.ndarray):
                    embeddings = np.array(embeddings)
                embeddings = embeddings.astype(np.float32)

                # Encode and send
                data, shape, dtype = encode_array(embeddings)
                send_response(
                    {
                        "type": "embeddings",
                        "vectors": data,
                        "shape": shape,
                        "dtype": dtype,
                    }
                )

            except Exception as e:
                send_response({"type": "error", "message": f"Embedding error: {e}"})

        elif cmd_type == "unload":
            # Cleanup and exit
            if model is not None:
                try:
                    del model
                    gc.collect()

                    # Try to clear MLX cache if available
                    try:
                        import mlx.core as mx

                        if hasattr(mx.metal, "clear_cache"):
                            mx.metal.clear_cache()
                    except ImportError:
                        pass

                    # Try to clear torch cache if available
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


def _mps_available() -> bool:
    """Check if MPS (Metal Performance Shaders) is available."""
    try:
        import torch

        return torch.backends.mps.is_available()
    except (ImportError, AttributeError):
        return False


if __name__ == "__main__":
    main()
