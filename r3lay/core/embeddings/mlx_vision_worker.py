#!/usr/bin/env python3
"""MLX vision embedding worker subprocess for isolated embedding generation.

This module runs as a standalone subprocess to completely isolate vision
embedding model interactions from Textual's TUI. This avoids fd conflicts with
transformers/tokenizer parallelism and mlx-vlm's terminal output.

Communication happens via JSON lines over stdin/stdout:
- stdin: JSON commands (one per line)
- stdout: JSON responses (one per line)
- stderr: redirected to /dev/null

Commands:
    {"cmd": "load", "model": "mlx-community/colqwen2.5-v0.2-bf16"}
    {"cmd": "embed", "images": ["/path/to/img1.png", "/path/to/img2.jpg"]}
    {"cmd": "unload"}

Responses:
    {"type": "loaded", "success": true, "dimension": 768, "backend": "colqwen2",
     "late_interaction": true, "num_vectors": 256}
    {"type": "loaded", "success": false, "error": "message"}
    {"type": "embeddings", "vectors": "<base64>", "shape": [2, 768], "dtype": "float32"}
    {"type": "embeddings", "vectors": "<base64>", "shape": [2, 256, 768], "dtype": "float32",
     "late_interaction": true}
    {"type": "error", "message": "..."}

Supported backends (tried in order):
    1. mlx-vlm with ColQwen2.5 support (late interaction)
    2. transformers CLIP/SigLIP (single vector)
    3. sentence-transformers CLIP (single vector)

Usage:
    python -m r3lay.core.embeddings.mlx_vision_worker
"""

from __future__ import annotations

import base64
import gc
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from PIL import Image


def setup_isolation() -> None:
    """Set up environment isolation before any imports.

    MUST be called before importing transformers, mlx-vlm, etc.
    """
    # Disable terminal features
    os.environ["TERM"] = "dumb"
    os.environ["NO_COLOR"] = "1"
    os.environ["FORCE_COLOR"] = "0"
    os.environ["TQDM_DISABLE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["MLX_SHOW_PROGRESS"] = "0"

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
    # Ensure contiguous memory layout for correct serialization
    arr = np.ascontiguousarray(arr)
    data = base64.b64encode(arr.tobytes()).decode("ascii")
    return data, list(arr.shape), str(arr.dtype)


def load_and_preprocess_image(
    image_path: str,
    max_size: int = 512,
) -> "Image.Image":
    """Load and preprocess an image for embedding.

    Args:
        image_path: Path to the image file.
        max_size: Maximum dimension for resizing.

    Returns:
        PIL Image in RGB format.
    """
    from PIL import Image

    img = Image.open(image_path)

    # Convert to RGB (handles RGBA, grayscale, etc.)
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Resize if too large (preserve aspect ratio)
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    return img


class VisionEmbedder:
    """Abstract interface for vision embedding backends."""

    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self.dimension: int = 0
        self.backend_name: str = "unknown"
        self.late_interaction: bool = False
        self.num_vectors: int = 1

    def load(self, model_name: str) -> bool:
        """Load the model. Returns True on success."""
        raise NotImplementedError

    def embed(self, images: list["Image.Image"]) -> "np.ndarray":
        """Embed images. Returns (N, D) or (N, P, D) array."""
        raise NotImplementedError

    def unload(self) -> None:
        """Unload model and free memory."""
        pass


class ColQwen2Embedder(VisionEmbedder):
    """ColQwen2.5 embedder using mlx-vlm for late interaction retrieval.

    ColQwen2.5 is specifically designed for visual document retrieval with
    late interaction (multi-vector representations).
    """

    def __init__(self) -> None:
        super().__init__()
        self.backend_name = "colqwen2"
        self.late_interaction = True

    def load(self, model_name: str) -> bool:
        """Load ColQwen2.5 model."""
        try:
            # Try colpali-engine first (designed for ColQwen2)
            try:
                from colpali_engine.models import ColQwen2, ColQwen2Processor

                self.processor = ColQwen2Processor.from_pretrained(model_name)
                self.model = ColQwen2.from_pretrained(model_name)

                # Get dimension from model config
                self.dimension = self.model.config.hidden_size
                self.num_vectors = 256  # Typical for ColQwen2

                return True

            except ImportError:
                pass

            # Fallback to transformers Qwen2-VL (not ideal but works)
            try:
                from transformers import AutoModel, AutoProcessor
                import torch

                self.processor = AutoProcessor.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                )

                # Get dimension
                if hasattr(self.model.config, "hidden_size"):
                    self.dimension = self.model.config.hidden_size
                else:
                    self.dimension = 768  # Default

                return True

            except Exception:
                pass

            return False

        except Exception:
            return False

    def embed(self, images: list["Image.Image"]) -> "np.ndarray":
        """Embed images using ColQwen2.5."""
        import numpy as np

        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded")

        try:
            # colpali-engine path (imports already done in load)
            # Process images
            batch_images = self.processor.process_images(images)
            batch_images = {k: v.to(self.model.device) for k, v in batch_images.items()}

            # Get embeddings
            with self.model.no_grad():
                embeddings = self.model(**batch_images)

            # Shape: (N, P, D) for late interaction
            result = embeddings.cpu().numpy().astype(np.float32)

            # Update num_vectors based on actual output
            if len(result.shape) == 3:
                self.num_vectors = result.shape[1]

            # L2 normalize each vector
            norms = np.linalg.norm(result, axis=-1, keepdims=True)
            result = result / (norms + 1e-8)

            return result

        except ImportError:
            # Fallback: transformers path (single vector)
            import torch

            self.late_interaction = False
            self.num_vectors = 1

            # Process each image
            all_embeddings = []
            for img in images:
                inputs = self.processor(images=img, return_tensors="pt")
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model.get_image_features(**inputs)

                embedding = outputs.cpu().numpy().astype(np.float32)
                all_embeddings.append(embedding)

            result = np.concatenate(all_embeddings, axis=0)

            # L2 normalize
            norms = np.linalg.norm(result, axis=-1, keepdims=True)
            result = result / (norms + 1e-8)

            return result

    def unload(self) -> None:
        """Unload model and free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None

        gc.collect()

        # Try to clear GPU memory
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except (ImportError, AttributeError):
            pass


class CLIPEmbedder(VisionEmbedder):
    """CLIP/SigLIP embedder using transformers or sentence-transformers.

    Provides single-vector embeddings suitable for image similarity search.
    """

    def __init__(self) -> None:
        super().__init__()
        self.backend_name = "clip"
        self.late_interaction = False
        self.num_vectors = 1
        self._use_sentence_transformers = False

    def load(self, model_name: str) -> bool:
        """Load CLIP model."""
        # Try sentence-transformers first (simpler API)
        try:
            from sentence_transformers import SentenceTransformer

            # Use MPS on Apple Silicon
            device = "mps" if _mps_available() else "cpu"
            self.model = SentenceTransformer(model_name, device=device)

            # Get dimension by encoding a test image
            from PIL import Image

            test_img = Image.new("RGB", (64, 64), color="white")
            test_emb = self.model.encode([test_img])
            self.dimension = test_emb.shape[-1]

            self._use_sentence_transformers = True
            self.backend_name = "sentence_transformers_clip"
            return True

        except (ImportError, Exception):
            pass

        # Try transformers CLIP
        try:
            from transformers import CLIPModel, CLIPProcessor

            device = "mps" if _mps_available() else "cpu"
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model = CLIPModel.from_pretrained(model_name).to(device)
            self.model.eval()

            # Get dimension
            self.dimension = self.model.config.projection_dim

            self.backend_name = "transformers_clip"
            return True

        except (ImportError, Exception):
            pass

        return False

    def embed(self, images: list["Image.Image"]) -> "np.ndarray":
        """Embed images using CLIP."""
        import numpy as np

        if self.model is None:
            raise RuntimeError("Model not loaded")

        if self._use_sentence_transformers:
            # sentence-transformers path
            embeddings = self.model.encode(images, convert_to_numpy=True)
            result = embeddings.astype(np.float32)
        else:
            # transformers path
            import torch

            device = next(self.model.parameters()).device
            inputs = self.processor(images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)

            result = outputs.cpu().numpy().astype(np.float32)

        # L2 normalize
        norms = np.linalg.norm(result, axis=-1, keepdims=True)
        result = result / (norms + 1e-8)

        return result

    def unload(self) -> None:
        """Unload model and free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None

        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except (ImportError, AttributeError):
            pass


class MLXCLIPEmbedder(VisionEmbedder):
    """CLIP embedder using MLX for Apple Silicon.

    Uses mlx-vlm or mlx-clip for native Apple Silicon acceleration.
    """

    def __init__(self) -> None:
        super().__init__()
        self.backend_name = "mlx_clip"
        self.late_interaction = False
        self.num_vectors = 1

    def load(self, model_name: str) -> bool:
        """Load CLIP model using MLX."""
        try:
            # Try mlx-vlm first
            from mlx_vlm import load as load_vlm

            self.model, self.processor = load_vlm(model_name)

            # Try to determine dimension
            self.dimension = 768  # Default for CLIP
            if hasattr(self.model, "config"):
                if hasattr(self.model.config, "projection_dim"):
                    self.dimension = self.model.config.projection_dim
                elif hasattr(self.model.config, "hidden_size"):
                    self.dimension = self.model.config.hidden_size

            self.backend_name = "mlx_vlm"
            return True

        except Exception:
            # Catch ALL exceptions (not just ImportError) so fallback can happen
            # mlx_vlm may be installed but fail to load non-MLX models (e.g., CLIP without safetensors)
            pass

        # Try mlx-clip if available
        try:
            import mlx_clip

            self.model = mlx_clip.load(model_name)
            self.dimension = self.model.vision_model.config.hidden_size
            self.backend_name = "mlx_clip"
            return True

        except Exception:
            # Catch ALL exceptions for proper fallback
            pass

        return False

    def embed(self, images: list["Image.Image"]) -> "np.ndarray":
        """Embed images using MLX."""
        import mlx.core as mx
        import numpy as np

        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Process images through vision encoder
        if self.backend_name == "mlx_vlm":
            # mlx-vlm path - get vision features
            all_embeddings = []
            for img in images:
                # Process single image
                if self.processor is not None:
                    inputs = self.processor(images=img)
                    features = self.model.get_image_features(**inputs)
                else:
                    features = self.model.encode_image(img)

                all_embeddings.append(features)

            embeddings = mx.concatenate(all_embeddings, axis=0)
        else:
            # mlx-clip path
            embeddings = self.model.encode_images(images)

        # Convert to numpy
        result = np.array(embeddings).astype(np.float32)

        # L2 normalize
        norms = np.linalg.norm(result, axis=-1, keepdims=True)
        result = result / (norms + 1e-8)

        return result

    def unload(self) -> None:
        """Unload model and free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None

        gc.collect()

        # Clear MLX cache
        try:
            import mlx.core as mx
            if hasattr(mx.metal, "clear_cache"):
                mx.metal.clear_cache()
                mx.eval(mx.zeros(1))  # Force sync
                mx.metal.clear_cache()
        except (ImportError, AttributeError):
            pass


def _mps_available() -> bool:
    """Check if MPS (Metal Performance Shaders) is available."""
    try:
        import torch
        return torch.backends.mps.is_available()
    except (ImportError, AttributeError):
        return False


def _is_colqwen_model(model_name: str) -> bool:
    """Check if model name suggests ColQwen2."""
    name_lower = model_name.lower()
    return "colqwen" in name_lower or "col-qwen" in name_lower


def _is_clip_model(model_name: str) -> bool:
    """Check if model name suggests CLIP/SigLIP."""
    name_lower = model_name.lower()
    return any(x in name_lower for x in ["clip", "siglip", "eva", "openai/clip"])


def main() -> None:
    """Main worker loop."""
    # Set up isolation FIRST
    setup_isolation()

    embedder: VisionEmbedder | None = None
    model_name: str | None = None

    # Import numpy and PIL after isolation
    import numpy as np
    from PIL import Image

    # Make these available to helper functions
    globals()["np"] = np
    globals()["Image"] = Image

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
            requested_model = cmd.get("model", "openai/clip-vit-base-patch32")

            try:
                # Determine which embedder to try based on model name
                if _is_colqwen_model(requested_model):
                    # ColQwen2 model
                    embedder = ColQwen2Embedder()
                    if not embedder.load(requested_model):
                        # Fall back to CLIP
                        embedder = CLIPEmbedder()
                        if not embedder.load(requested_model):
                            embedder = None
                elif _is_clip_model(requested_model):
                    # CLIP-style model - try MLX first on Apple Silicon
                    import platform
                    if platform.system() == "Darwin" and platform.machine() == "arm64":
                        embedder = MLXCLIPEmbedder()
                        if not embedder.load(requested_model):
                            embedder = CLIPEmbedder()
                            if not embedder.load(requested_model):
                                embedder = None
                    else:
                        embedder = CLIPEmbedder()
                        if not embedder.load(requested_model):
                            embedder = None
                else:
                    # Unknown model - try in order
                    for EmbedderClass in [MLXCLIPEmbedder, ColQwen2Embedder, CLIPEmbedder]:
                        embedder = EmbedderClass()
                        if embedder.load(requested_model):
                            break
                        embedder = None

                if embedder is not None:
                    model_name = requested_model
                    send_response({
                        "type": "loaded",
                        "success": True,
                        "dimension": embedder.dimension,
                        "backend": embedder.backend_name,
                        "late_interaction": embedder.late_interaction,
                        "num_vectors": embedder.num_vectors,
                    })
                else:
                    send_response({
                        "type": "loaded",
                        "success": False,
                        "error": (
                            "No vision embedding library available. "
                            "Install with: pip install transformers pillow "
                            "or pip install sentence-transformers pillow"
                        ),
                    })

            except Exception as e:
                send_response({
                    "type": "loaded",
                    "success": False,
                    "error": f"Failed to load model: {e}",
                })

        elif cmd_type == "embed":
            if embedder is None:
                send_response({"type": "error", "message": "Model not loaded"})
                continue

            image_paths = cmd.get("images", [])
            max_size = cmd.get("max_size", 512)

            if not image_paths:
                # Return empty array with correct shape
                if embedder.late_interaction:
                    shape = [0, embedder.num_vectors, embedder.dimension]
                else:
                    shape = [0, embedder.dimension]
                send_response({
                    "type": "embeddings",
                    "vectors": "",
                    "shape": shape,
                    "dtype": "float32",
                    "late_interaction": embedder.late_interaction,
                })
                continue

            try:
                # Load and preprocess images
                images = []
                for path in image_paths:
                    if not Path(path).exists():
                        send_response({
                            "type": "error",
                            "message": f"Image not found: {path}"
                        })
                        images = []  # Clear and skip
                        break
                    images.append(load_and_preprocess_image(path, max_size))

                if not images:
                    continue

                # Generate embeddings
                embeddings = embedder.embed(images)

                # Encode and send
                data, shape, dtype = encode_array(embeddings)
                send_response({
                    "type": "embeddings",
                    "vectors": data,
                    "shape": shape,
                    "dtype": dtype,
                    "late_interaction": embedder.late_interaction,
                })

            except Exception as e:
                send_response({"type": "error", "message": f"Embedding error: {e}"})

        elif cmd_type == "unload":
            # Cleanup and exit
            if embedder is not None:
                try:
                    embedder.unload()
                except Exception:
                    pass

            break

        else:
            send_response({"type": "error", "message": f"Unknown command: {cmd_type}"})


if __name__ == "__main__":
    main()
