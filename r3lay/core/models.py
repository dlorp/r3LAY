"""Model discovery and management for r3LAY.

Scans three sources for available models:
1. HuggingFace cache - safetensors and GGUF models
2. GGUF drop folder - standalone .gguf files
3. Ollama API - running Ollama models

Phase 2 implementation based on plans/2026-01-02_phase2-model-discovery.md
Phase C: Added capability detection (text, vision, embedding)
"""

from __future__ import annotations

import json
import platform
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field, computed_field

# =============================================================================
# Model Capability Types
# =============================================================================


class ModelCapability(str, Enum):
    """Model capability types."""

    TEXT = "text"  # Standard text generation (chat, completion)
    VISION = "vision"  # Vision-language model (image understanding)
    TEXT_EMBEDDING = "text_embedding"  # Text embedding model
    VISION_EMBEDDING = "vision_embedding"  # Vision embedding model (CLIP-style)


# =============================================================================
# Phase 2.1 - Core Enums
# =============================================================================


class ModelSource(str, Enum):
    """Where a model was discovered from."""

    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    GGUF_FILE = "gguf_file"


class ModelFormat(str, Enum):
    """Model file format."""

    GGUF = "gguf"
    SAFETENSORS = "safetensors"
    PYTORCH = "pytorch"
    OLLAMA = "ollama"


class Backend(str, Enum):
    """Inference backend to use for a model."""

    MLX = "mlx"
    LLAMA_CPP = "llama_cpp"
    VLLM = "vllm"
    OLLAMA = "ollama"
    OPENCLAW = "openclaw"


# =============================================================================
# Phase 2.1 - ModelInfo Pydantic Model
# =============================================================================


class ModelInfo(BaseModel):
    """Information about an available model.

    Attributes:
        name: Model identifier (e.g., "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF")
        source: Where the model was discovered (HuggingFace, Ollama, GGUF file)
        path: Full filesystem path to model files (None for Ollama)
        format: Model file format (GGUF, safetensors, etc.)
        size_bytes: Total size in bytes (None if unknown)
        backend: Recommended inference backend
        capabilities: Set of model capabilities (text, vision, embedding)
        last_accessed: When the model was last accessed (if available)
        metadata: Additional source-specific information
    """

    name: str
    source: ModelSource
    path: Path | None = None
    format: ModelFormat | None = None
    size_bytes: int | None = None
    backend: Backend
    capabilities: set[ModelCapability] = Field(default_factory=lambda: {ModelCapability.TEXT})
    last_accessed: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    @computed_field
    @property
    def display_name(self) -> str:
        """Human-readable display name with source tag.

        Returns:
            Format: "[HF] model-name" or "[OL] model-name" or "[GF] model-name"
        """
        tags = {
            ModelSource.HUGGINGFACE: "HF",
            ModelSource.OLLAMA: "OL",
            ModelSource.GGUF_FILE: "GF",
        }
        tag = tags.get(self.source, "??")
        return f"[{tag}] {self.name}"

    @computed_field
    @property
    def size_human(self) -> str:
        """Human-readable size string.

        Returns:
            Format: "5.0 GB" or "?" if unknown
        """
        if self.size_bytes is None:
            return "?"

        size = float(self.size_bytes)
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} PB"

    @property
    def is_text_model(self) -> bool:
        """Check if this is a text generation model."""
        return ModelCapability.TEXT in self.capabilities

    @property
    def is_vision_model(self) -> bool:
        """Check if this is a vision-language model."""
        return ModelCapability.VISION in self.capabilities

    @property
    def is_text_embedder(self) -> bool:
        """Check if this is a text embedding model."""
        return ModelCapability.TEXT_EMBEDDING in self.capabilities

    @property
    def is_vision_embedder(self) -> bool:
        """Check if this is a vision embedding model."""
        return ModelCapability.VISION_EMBEDDING in self.capabilities

    @property
    def capabilities_display(self) -> str:
        """Human-readable capabilities string."""
        caps = []
        if ModelCapability.TEXT in self.capabilities:
            caps.append("text")
        if ModelCapability.VISION in self.capabilities:
            caps.append("vision")
        if ModelCapability.TEXT_EMBEDDING in self.capabilities:
            caps.append("embed")
        if ModelCapability.VISION_EMBEDDING in self.capabilities:
            caps.append("v-embed")
        return ", ".join(caps) if caps else "unknown"


# =============================================================================
# Phase 2.2 - Format Detection and Backend Selection
# =============================================================================


# GGUF magic bytes: "GGUF" = 0x47475546 (big endian) or 0x46554747 (little endian)
GGUF_MAGIC_BE = b"GGUF"
GGUF_MAGIC_LE = b"FUGG"


def detect_format(path: Path) -> ModelFormat | None:
    """Detect model format from a directory or file.

    Checks for:
    - .gguf files (verifies magic bytes 0x47475546)
    - .safetensors files
    - pytorch_model.bin or .bin files

    Args:
        path: Path to a model directory or file

    Returns:
        Detected ModelFormat or None if unknown
    """
    if not path.exists():
        return None

    # If path is a file, check it directly
    if path.is_file():
        return _detect_file_format(path)

    # If path is a directory, scan for model files
    # Check snapshots subdirectory first (HuggingFace cache structure)
    snapshots = path / "snapshots"
    if snapshots.exists():
        for snapshot in snapshots.iterdir():
            if snapshot.is_dir():
                fmt = _scan_directory_for_format(snapshot)
                if fmt:
                    return fmt

    # Check the directory itself
    return _scan_directory_for_format(path)


def _detect_file_format(file_path: Path) -> ModelFormat | None:
    """Detect format of a single file."""
    suffix = file_path.suffix.lower()

    if suffix == ".gguf":
        # Verify GGUF magic bytes
        try:
            with open(file_path, "rb") as f:
                magic = f.read(4)
                if magic == GGUF_MAGIC_BE or magic == GGUF_MAGIC_LE:
                    return ModelFormat.GGUF
        except (OSError, IOError):
            pass
        # Fall back to extension-based detection if magic check fails
        return ModelFormat.GGUF

    if suffix == ".safetensors":
        return ModelFormat.SAFETENSORS

    if suffix == ".bin":
        # Check if it's a PyTorch model
        if "pytorch" in file_path.name.lower():
            return ModelFormat.PYTORCH
        return ModelFormat.PYTORCH

    return None


def _scan_directory_for_format(directory: Path) -> ModelFormat | None:
    """Scan a directory for model files and detect format."""
    try:
        for file_path in directory.iterdir():
            if not file_path.is_file():
                continue
            fmt = _detect_file_format(file_path)
            if fmt:
                return fmt
    except (OSError, PermissionError):
        pass
    return None


def select_backend(
    fmt: ModelFormat | None,
    source: ModelSource,
    capabilities: set[ModelCapability] | None = None,
) -> Backend:
    """Select the best inference backend for a model.

    Selection priority:
    - Ollama source -> OLLAMA backend
    - Embedding models -> MLX (they use sentence-transformers/mlx-embeddings anyway)
    - GGUF format -> LLAMA_CPP (universal)
    - Safetensors on Apple Silicon -> MLX
    - Safetensors with CUDA -> VLLM
    - Default -> LLAMA_CPP

    Args:
        fmt: Model file format
        source: Model source
        capabilities: Model capabilities (used for special handling of embeddings)

    Returns:
        Recommended Backend for this model
    """
    # Ollama models use Ollama backend
    if source == ModelSource.OLLAMA:
        return Backend.OLLAMA

    # Embedding models always use MLX (they have their own runtime)
    if capabilities:
        if ModelCapability.TEXT_EMBEDDING in capabilities:
            return Backend.MLX
        if ModelCapability.VISION_EMBEDDING in capabilities:
            return Backend.MLX

    # GGUF always uses llama.cpp (most compatible)
    if fmt == ModelFormat.GGUF:
        return Backend.LLAMA_CPP

    # Safetensors/PyTorch: prefer MLX on Apple Silicon
    if fmt in (ModelFormat.SAFETENSORS, ModelFormat.PYTORCH):
        if _is_apple_silicon():
            return Backend.MLX
        if _has_cuda():
            return Backend.VLLM
        return Backend.LLAMA_CPP

    # Default fallback
    return Backend.LLAMA_CPP


def _is_apple_silicon() -> bool:
    """Check if running on Apple Silicon."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _has_cuda() -> bool:
    """Check if CUDA is available (lazy import to avoid torch dependency)."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


# =============================================================================
# Phase C - Capability Detection
# =============================================================================


def detect_capabilities(model_path: Path) -> set[ModelCapability]:
    """Infer model capabilities from config.json architecture.

    Analyzes the model's config.json to detect:
    - Vision-language models (VL, VLM, LLaVA, etc.)
    - Embedding models (BERT, RoBERTa, sentence-transformers, etc.)
    - Standard text LLMs (default)

    Args:
        model_path: Path to model directory (must contain config.json)

    Returns:
        Set of detected ModelCapability values. Defaults to {TEXT} if detection fails.
    """
    capabilities: set[ModelCapability] = set()

    # Find config.json - could be in model_path directly or in a subdirectory
    config_path = model_path / "config.json"
    if not config_path.exists():
        # Try looking in snapshots for HF cache structure
        snapshots = model_path / "snapshots"
        if snapshots.exists():
            for snapshot in snapshots.iterdir():
                if snapshot.is_dir():
                    candidate = snapshot / "config.json"
                    if candidate.exists():
                        config_path = candidate
                        break

    if not config_path.exists():
        return {ModelCapability.TEXT}  # Default assumption

    try:
        config = json.loads(config_path.read_text())

        # Extract architecture info
        architectures = config.get("architectures", [])
        architectures_str = str(architectures).lower()
        model_type = config.get("model_type", "").lower()
        model_name = config.get("_name_or_path", "").lower()

        # Combine all text for pattern matching
        all_text = f"{architectures_str} {model_type} {model_name}"

        # Vision-language models (VLM, VL, LLaVA, etc.)
        vision_patterns = [
            "qwen2_vl",
            "qwen2-vl",  # Hyphen variant
            "qwen2vl",
            "llava",
            "vision",
            "vlm",
            "vl_",
            "_vl",
            "-vl",  # Hyphen variant (e.g., Model-VL-7B)
            "paligemma",
            "idefics",
            "fuyu",
            "cogvlm",
            "internvl",
            "blip",
            "flamingo",
        ]
        if any(p in all_text for p in vision_patterns):
            capabilities.add(ModelCapability.VISION)
            capabilities.add(ModelCapability.TEXT)
            return capabilities

        # Text embedding models
        embedding_patterns = [
            "bert",
            "roberta",
            "xlm",
            "minilm",
            "sentence",
            "e5",
            "bge",
            "gte",
            "nomic",
            "instructor",
            "embeddings",
            "encoder",
        ]
        if any(p in all_text for p in embedding_patterns):
            capabilities.add(ModelCapability.TEXT_EMBEDDING)
            # Check for vision embedding (CLIP-style)
            clip_patterns = ["clip", "siglip", "vision_encoder"]
            if any(p in all_text for p in clip_patterns):
                capabilities.add(ModelCapability.VISION_EMBEDDING)
            return capabilities

        # CLIP-style vision encoders
        clip_patterns = ["clip", "siglip", "eva", "dinov2"]
        if any(p in all_text for p in clip_patterns):
            capabilities.add(ModelCapability.VISION_EMBEDDING)
            if "text" in all_text:
                capabilities.add(ModelCapability.TEXT_EMBEDDING)
            return capabilities

        # Standard LLMs
        llm_patterns = [
            "llama",
            "qwen",
            "qwen2",
            "mistral",
            "phi",
            "gemma",
            "gpt",
            "falcon",
            "starcoder",
            "codellama",
            "deepseek",
            "yi",
            "mixtral",
            "command",
            "orca",
            "dolphin",
            "vicuna",
            "openchat",
        ]
        if any(p in all_text for p in llm_patterns):
            capabilities.add(ModelCapability.TEXT)
            return capabilities

        # Default to text if nothing matched
        capabilities.add(ModelCapability.TEXT)

    except (json.JSONDecodeError, OSError, KeyError):
        capabilities.add(ModelCapability.TEXT)

    return capabilities if capabilities else {ModelCapability.TEXT}


def detect_capabilities_from_name(name: str) -> set[ModelCapability]:
    """Infer capabilities from model name when config.json is unavailable.

    Used for Ollama models and GGUF files without config.json.

    Args:
        name: Model name/identifier

    Returns:
        Set of detected ModelCapability values.
    """
    name_lower = name.lower()
    capabilities: set[ModelCapability] = set()

    # Vision patterns in name
    vision_patterns = [
        "-vl-",
        "_vl_",
        "-vl",
        "_vl",
        "vision",
        "vlm",
        "llava",
        "cogvlm",
        "qwen-vl",
        "qwen2-vl",
        "joycaption",  # JoyCaption is a vision model
        "pixtral",
        "paligemma",
    ]
    if any(p in name_lower for p in vision_patterns):
        capabilities.add(ModelCapability.VISION)
        capabilities.add(ModelCapability.TEXT)
        return capabilities

    # Embedding patterns in name
    embedding_patterns = [
        "embed",
        "e5-",
        "bge-",
        "gte-",
        "minilm",
        "bert",
        "sentence",
        "nomic-embed",
    ]
    if any(p in name_lower for p in embedding_patterns):
        capabilities.add(ModelCapability.TEXT_EMBEDDING)
        return capabilities

    # Default to text
    capabilities.add(ModelCapability.TEXT)
    return capabilities


# =============================================================================
# Phase 2.3 - HuggingFace Cache Scanner
# =============================================================================


def scan_huggingface_cache(
    cache_path: Path | None = None,
) -> list[ModelInfo]:
    """Scan HuggingFace cache directory for available models.

    Walks the models--* directories directly (no subprocess).
    Parses repo_id from directory names like:
        models--Qwen--Qwen2.5-Coder-14B-Instruct-GGUF -> Qwen/Qwen2.5-Coder-14B-Instruct-GGUF

    Args:
        cache_path: Path to HuggingFace cache hub directory.
                   Default: ~/.cache/huggingface/hub

    Returns:
        List of ModelInfo for discovered models. Empty list if path doesn't exist.
    """
    if cache_path is None:
        cache_path = Path.home() / ".cache" / "huggingface" / "hub"

    if not cache_path.exists():
        return []

    models: list[ModelInfo] = []

    try:
        for item in cache_path.iterdir():
            if not item.is_dir():
                continue

            # Match models--org--name pattern
            if not item.name.startswith("models--"):
                continue

            # Parse repo_id: models--Qwen--Model -> Qwen/Model
            parts = item.name.split("--")
            if len(parts) < 3:
                continue

            repo_id = "/".join(parts[1:])

            # Find snapshot directory for actual model files
            snapshots_dir = item / "snapshots"
            if not snapshots_dir.exists():
                continue

            # Get the latest snapshot (usually only one, or pick first)
            snapshot_path: Path | None = None
            for snapshot in snapshots_dir.iterdir():
                if snapshot.is_dir():
                    snapshot_path = snapshot
                    break

            if not snapshot_path:
                continue

            # Detect format and calculate size
            fmt = detect_format(item)
            size_bytes = _calculate_directory_size(snapshot_path)

            # For GGUF models, find the actual .gguf file path
            model_path: Path = snapshot_path
            if fmt == ModelFormat.GGUF:
                gguf_files = list(snapshot_path.glob("*.gguf"))
                if gguf_files:
                    # Use the first .gguf file found (usually only one)
                    model_path = gguf_files[0]
                    size_bytes = model_path.stat().st_size

            # Get last accessed time from refs/main if available
            last_accessed: datetime | None = None
            refs_main = item / "refs" / "main"
            if refs_main.exists():
                try:
                    stat = refs_main.stat()
                    last_accessed = datetime.fromtimestamp(stat.st_atime)
                except OSError:
                    pass

            # Detect capabilities from config.json (do this first for backend selection)
            capabilities = detect_capabilities(snapshot_path)

            # Select backend based on format and capabilities
            backend = select_backend(fmt, ModelSource.HUGGINGFACE, capabilities)

            models.append(
                ModelInfo(
                    name=repo_id,
                    source=ModelSource.HUGGINGFACE,
                    path=model_path,
                    format=fmt,
                    size_bytes=size_bytes,
                    backend=backend,
                    capabilities=capabilities,
                    last_accessed=last_accessed,
                    metadata={
                        "cache_dir": str(item),
                        "snapshot": snapshot_path.name if snapshot_path else None,
                    },
                )
            )

    except (OSError, PermissionError):
        # Graceful handling of filesystem errors
        pass

    return models


def _calculate_directory_size(directory: Path) -> int:
    """Calculate total size of all files in a directory.

    Follows symlinks to get actual file sizes (important for HuggingFace cache
    where model files are symlinked to blobs/).
    """
    total = 0
    try:
        for item in directory.rglob("*"):
            if item.is_file() or item.is_symlink():
                try:
                    # Use resolve() to follow symlinks, then stat the resolved path
                    resolved = item.resolve()
                    if resolved.exists() and resolved.is_file():
                        total += resolved.stat().st_size
                except (OSError, RuntimeError):
                    # RuntimeError can occur with circular symlinks
                    pass
    except (OSError, PermissionError):
        pass
    return total


# =============================================================================
# Phase 2.4 - MLX Direct Folder Scanner
# =============================================================================


def scan_mlx_folder(
    folder_path: Path | None = None,
) -> list[ModelInfo]:
    """Scan MLX folder for direct model downloads (not HF cache format).

    Looks for directories containing safetensors files and config.json.
    This handles models downloaded via mlx-lm or manually cloned repos.

    Args:
        folder_path: Path to MLX models folder.
                    Default: ~/.r3lay/models/mlx

    Returns:
        List of ModelInfo for discovered MLX models.
    """
    if folder_path is None:
        folder_path = Path.home() / ".r3lay" / "models" / "mlx"

    if not folder_path.exists():
        return []

    models: list[ModelInfo] = []

    try:
        for item in folder_path.iterdir():
            if not item.is_dir():
                continue

            # Skip hidden directories
            if item.name.startswith("."):
                continue

            # Check for safetensors files (MLX model indicator)
            safetensors_files = list(item.glob("*.safetensors"))
            if not safetensors_files:
                continue

            # Check for config.json (required for mlx-lm)
            config_file = item / "config.json"
            if not config_file.exists():
                continue

            # Model name is the directory name
            model_name = f"mlx-community/{item.name}"

            # Calculate total size
            size_bytes = _calculate_directory_size(item)

            # Get last accessed time
            last_accessed: datetime | None = None
            try:
                stat = item.stat()
                last_accessed = datetime.fromtimestamp(stat.st_atime)
            except OSError:
                pass

            # Detect capabilities from config.json
            capabilities = detect_capabilities(item)

            models.append(
                ModelInfo(
                    name=model_name,
                    source=ModelSource.HUGGINGFACE,  # Treat as HF source for sorting
                    path=item,  # MLX uses directory path
                    format=ModelFormat.SAFETENSORS,
                    size_bytes=size_bytes,
                    backend=Backend.MLX,  # Force MLX backend
                    capabilities=capabilities,
                    last_accessed=last_accessed,
                    metadata={
                        "mlx_direct": True,
                        "safetensors_count": len(safetensors_files),
                    },
                )
            )

    except (OSError, PermissionError):
        pass

    return models


# =============================================================================
# Phase 2.5 - GGUF Folder Scanner
# =============================================================================


def scan_gguf_folder(
    folder_path: Path | None = None,
) -> list[ModelInfo]:
    """Scan GGUF drop folder for standalone .gguf files.

    Auto-creates the folder if it doesn't exist.

    Args:
        folder_path: Path to GGUF folder. Default: ~/.r3lay/models/

    Returns:
        List of ModelInfo for discovered .gguf files. Empty list if none found.
    """
    if folder_path is None:
        folder_path = Path.home() / ".r3lay" / "models"

    # Auto-create folder if missing
    try:
        folder_path.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError):
        return []

    if not folder_path.exists():
        return []

    models: list[ModelInfo] = []

    try:
        for item in folder_path.iterdir():
            if not item.is_file():
                continue

            if item.suffix.lower() != ".gguf":
                continue

            # Extract model name from filename (remove .gguf extension)
            model_name = item.stem

            # Get file size
            try:
                size_bytes = item.stat().st_size
            except OSError:
                size_bytes = None

            # Get last accessed time
            last_accessed: datetime | None = None
            try:
                stat = item.stat()
                last_accessed = datetime.fromtimestamp(stat.st_atime)
            except OSError:
                pass

            # Detect capabilities from name (no config.json for standalone GGUF)
            capabilities = detect_capabilities_from_name(model_name)

            models.append(
                ModelInfo(
                    name=model_name,
                    source=ModelSource.GGUF_FILE,
                    path=item,
                    format=ModelFormat.GGUF,
                    size_bytes=size_bytes,
                    backend=Backend.LLAMA_CPP,
                    capabilities=capabilities,
                    last_accessed=last_accessed,
                    metadata={"filename": item.name},
                )
            )

    except (OSError, PermissionError):
        pass

    return models


# =============================================================================
# Phase 5.7 - LLM Models Folder Scanner (with mmproj detection)
# =============================================================================


def scan_llm_models_folder(
    folder_path: Path | None = None,
) -> list[ModelInfo]:
    """Scan llm-models folder for GGUF models with optional mmproj files.

    Looks for subdirectories containing .gguf files and detects
    mmproj files for vision model support (LLaVA, JoyCaption, etc.).

    Args:
        folder_path: Path to llm-models folder.
                    Default: ~/.r3lay/models

    Returns:
        List of ModelInfo for discovered models.
    """
    if folder_path is None:
        folder_path = Path("~/.r3lay/models")

    if not folder_path.exists():
        return []

    models: list[ModelInfo] = []

    try:
        for item in folder_path.iterdir():
            if not item.is_dir():
                continue

            # Skip hidden directories and known non-model dirs
            if item.name.startswith(".") or item.name in ("hub", "ollama-models"):
                continue

            # Find GGUF files in this directory
            gguf_files = list(item.glob("*.gguf"))
            if not gguf_files:
                continue

            # Separate main model from mmproj file
            main_model: Path | None = None
            mmproj_file: Path | None = None

            for gguf in gguf_files:
                name_lower = gguf.name.lower()
                if "mmproj" in name_lower or "clip" in name_lower:
                    mmproj_file = gguf
                elif main_model is None or gguf.stat().st_size > main_model.stat().st_size:
                    # Pick the largest non-mmproj file as main model
                    main_model = gguf

            if not main_model:
                continue

            # Model name is the directory name
            model_name = item.name

            # Get file size
            try:
                size_bytes = main_model.stat().st_size
            except OSError:
                size_bytes = None

            # Get last accessed time
            last_accessed: datetime | None = None
            try:
                stat = main_model.stat()
                last_accessed = datetime.fromtimestamp(stat.st_atime)
            except OSError:
                pass

            # Detect capabilities from name
            capabilities = detect_capabilities_from_name(model_name)

            # If mmproj present, ensure VISION capability
            if mmproj_file:
                capabilities.add(ModelCapability.VISION)
                capabilities.add(ModelCapability.TEXT)

            # Build metadata
            metadata: dict[str, Any] = {
                "filename": main_model.name,
                "llm_models_dir": True,
            }
            if mmproj_file:
                metadata["mmproj_path"] = str(mmproj_file)

            models.append(
                ModelInfo(
                    name=model_name,
                    source=ModelSource.GGUF_FILE,
                    path=main_model,
                    format=ModelFormat.GGUF,
                    size_bytes=size_bytes,
                    backend=Backend.LLAMA_CPP,
                    capabilities=capabilities,
                    last_accessed=last_accessed,
                    metadata=metadata,
                )
            )

    except (OSError, PermissionError):
        pass

    return models


# =============================================================================
# Phase 2.5 - Ollama Scanner
# =============================================================================


async def scan_ollama(
    endpoint: str = "http://localhost:11434",
) -> list[ModelInfo]:
    """Scan Ollama API for available models.

    Makes GET request to /api/tags endpoint.
    Times out after 5 seconds.
    Returns empty list if Ollama is not running.

    Args:
        endpoint: Ollama API endpoint. Default: http://localhost:11434

    Returns:
        List of ModelInfo for available Ollama models. Empty list if unavailable.
    """
    models: list[ModelInfo] = []

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{endpoint}/api/tags")
            response.raise_for_status()
            data = response.json()

            for model in data.get("models", []):
                name = model.get("name", "unknown")
                size = model.get("size")

                # Parse modified_at timestamp if available
                last_accessed: datetime | None = None
                modified_at = model.get("modified_at")
                if modified_at:
                    try:
                        # Ollama uses ISO format with timezone
                        last_accessed = datetime.fromisoformat(
                            modified_at.replace("Z", "+00:00")
                        )
                    except (ValueError, TypeError):
                        pass

                # Detect capabilities from name (Ollama doesn't expose config.json)
                capabilities = detect_capabilities_from_name(name)

                models.append(
                    ModelInfo(
                        name=name,
                        source=ModelSource.OLLAMA,
                        path=None,
                        format=ModelFormat.OLLAMA,
                        size_bytes=size,
                        backend=Backend.OLLAMA,
                        capabilities=capabilities,
                        last_accessed=last_accessed,
                        metadata={
                            "digest": model.get("digest"),
                            "details": model.get("details", {}),
                            "modified_at": modified_at,
                        },
                    )
                )

    except (httpx.HTTPError, httpx.TimeoutException, Exception):
        # Silent skip if Ollama not running or any error
        pass

    return models


# =============================================================================
# Phase 2.6 - ModelScanner Class
# =============================================================================


class ModelScanner:
    """Unified model discovery across all sources.

    Scans HuggingFace cache, GGUF drop folder, and Ollama API.
    Caches results for efficient lookups.

    Example:
        scanner = ModelScanner()
        models = await scanner.scan_all()
        for model in models:
            print(f"{model.display_name} - {model.size_human}")
    """

    def __init__(
        self,
        hf_cache_path: Path | None = None,
        mlx_folder: Path | None = None,
        gguf_folder: Path | None = None,
        llm_models_folder: Path | None = None,
        ollama_endpoint: str = "http://localhost:11434",
    ):
        """Initialize the model scanner.

        Args:
            hf_cache_path: HuggingFace cache path.
                          Default: ~/.cache/huggingface/hub
            mlx_folder: MLX models folder (direct downloads).
                       Default: ~/.r3lay/models/mlx
            gguf_folder: GGUF drop folder path.
                        Default: ~/.r3lay/models/
            llm_models_folder: LLM models folder (GGUF with mmproj).
                              Default: ~/.r3lay/models
            ollama_endpoint: Ollama API endpoint.
                            Default: http://localhost:11434
        """
        self.hf_cache_path = hf_cache_path
        self.mlx_folder = mlx_folder
        self.gguf_folder = gguf_folder
        self.llm_models_folder = llm_models_folder
        self.ollama_endpoint = ollama_endpoint
        self._models: list[ModelInfo] = []

    async def scan_all(self) -> list[ModelInfo]:
        """Scan all sources for available models.

        Scans HuggingFace cache, MLX folder, GGUF folder, llm-models folder,
        and Ollama API.
        Results are sorted by source: HuggingFace/MLX first, then Ollama, then GGUF files.
        Within each source, models are sorted by name.

        Returns:
            List of ModelInfo sorted by source and name.
        """
        self._models = []

        # Scan HuggingFace cache (sync)
        hf_models = scan_huggingface_cache(self.hf_cache_path)
        self._models.extend(hf_models)

        # Scan MLX direct folder (sync)
        mlx_models = scan_mlx_folder(self.mlx_folder)
        self._models.extend(mlx_models)

        # Scan llm-models folder for GGUF with mmproj (sync)
        llm_models = scan_llm_models_folder(self.llm_models_folder)
        self._models.extend(llm_models)

        # Scan Ollama (async)
        ollama_models = await scan_ollama(self.ollama_endpoint)
        self._models.extend(ollama_models)

        # Scan GGUF drop folder (sync)
        gguf_models = scan_gguf_folder(self.gguf_folder)
        self._models.extend(gguf_models)

        # Sort: HuggingFace (0) -> Ollama (1) -> GGUF_FILE (2), then by name
        source_order = {
            ModelSource.HUGGINGFACE: 0,
            ModelSource.OLLAMA: 1,
            ModelSource.GGUF_FILE: 2,
        }
        self._models.sort(
            key=lambda m: (source_order.get(m.source, 99), m.name.lower())
        )

        return self._models

    def get_by_name(self, name: str) -> ModelInfo | None:
        """Find a model by exact name.

        Args:
            name: Model name to search for

        Returns:
            ModelInfo if found, None otherwise
        """
        for model in self._models:
            if model.name == name:
                return model
        return None

    def get_by_source(self, source: ModelSource) -> list[ModelInfo]:
        """Get all models from a specific source.

        Args:
            source: ModelSource to filter by

        Returns:
            List of ModelInfo from that source
        """
        return [m for m in self._models if m.source == source]

    def get_cached_models(self) -> list[ModelInfo]:
        """Get the cached model list from the last scan.

        Returns:
            Cached list of ModelInfo (may be empty if scan_all not called)
        """
        return self._models

    def get_by_capability(self, capability: ModelCapability) -> list[ModelInfo]:
        """Get all models with a specific capability.

        Args:
            capability: ModelCapability to filter by

        Returns:
            List of ModelInfo with that capability
        """
        return [m for m in self._models if capability in m.capabilities]

    def get_text_models(self) -> list[ModelInfo]:
        """Get all text generation models."""
        return self.get_by_capability(ModelCapability.TEXT)

    def get_vision_models(self) -> list[ModelInfo]:
        """Get all vision-language models."""
        return self.get_by_capability(ModelCapability.VISION)

    def get_text_embedders(self) -> list[ModelInfo]:
        """Get all text embedding models."""
        return self.get_by_capability(ModelCapability.TEXT_EMBEDDING)

    def get_vision_embedders(self) -> list[ModelInfo]:
        """Get all vision embedding models."""
        return self.get_by_capability(ModelCapability.VISION_EMBEDDING)

    def clear_cache(self) -> None:
        """Clear the cached model list."""
        self._models = []


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # Enums
    "ModelSource",
    "ModelFormat",
    "Backend",
    "ModelCapability",
    # Data model
    "ModelInfo",
    # Functions
    "detect_format",
    "select_backend",
    "detect_capabilities",
    "detect_capabilities_from_name",
    "scan_huggingface_cache",
    "scan_mlx_folder",
    "scan_gguf_folder",
    "scan_llm_models_folder",
    "scan_ollama",
    # Class
    "ModelScanner",
]
