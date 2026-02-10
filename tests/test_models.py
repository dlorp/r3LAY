"""Comprehensive tests for r3lay.core.models module.

Tests cover:
- Enums: ModelCapability, ModelSource, ModelFormat, Backend
- ModelInfo: creation, computed fields, properties
- Format detection: detect_format, GGUF magic bytes
- Backend selection: select_backend with platform detection
- Capability detection: from config.json and from name
- Scanners: HuggingFace, MLX, GGUF, llm-models folders
- ModelScanner class: unified scanning, filtering, caching
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from r3lay.core.models import (
    GGUF_MAGIC_BE,
    GGUF_MAGIC_LE,
    Backend,
    ModelCapability,
    ModelFormat,
    ModelInfo,
    ModelScanner,
    ModelSource,
    _calculate_directory_size,
    _detect_file_format,
    _is_apple_silicon,
    _scan_directory_for_format,
    detect_capabilities,
    detect_capabilities_from_name,
    detect_format,
    scan_gguf_folder,
    scan_huggingface_cache,
    scan_llm_models_folder,
    scan_mlx_folder,
    scan_ollama,
    select_backend,
    validate_model_name,
)

# =============================================================================
# Enum Tests
# =============================================================================


class TestModelCapability:
    """Tests for ModelCapability enum."""

    def test_all_capabilities(self):
        """Test all capability values exist."""
        assert ModelCapability.TEXT == "text"
        assert ModelCapability.VISION == "vision"
        assert ModelCapability.TEXT_EMBEDDING == "text_embedding"
        assert ModelCapability.VISION_EMBEDDING == "vision_embedding"

    def test_capability_count(self):
        """Test expected number of capabilities."""
        assert len(ModelCapability) == 4


class TestModelSource:
    """Tests for ModelSource enum."""

    def test_all_sources(self):
        """Test all source values exist."""
        assert ModelSource.HUGGINGFACE == "huggingface"
        assert ModelSource.OLLAMA == "ollama"
        assert ModelSource.GGUF_FILE == "gguf_file"

    def test_source_count(self):
        """Test expected number of sources."""
        assert len(ModelSource) == 3


class TestModelFormat:
    """Tests for ModelFormat enum."""

    def test_all_formats(self):
        """Test all format values exist."""
        assert ModelFormat.GGUF == "gguf"
        assert ModelFormat.SAFETENSORS == "safetensors"
        assert ModelFormat.PYTORCH == "pytorch"
        assert ModelFormat.OLLAMA == "ollama"

    def test_format_count(self):
        """Test expected number of formats."""
        assert len(ModelFormat) == 4


class TestBackend:
    """Tests for Backend enum."""

    def test_all_backends(self):
        """Test all backend values exist."""
        assert Backend.MLX == "mlx"
        assert Backend.LLAMA_CPP == "llama_cpp"
        assert Backend.VLLM == "vllm"
        assert Backend.OLLAMA == "ollama"

    def test_backend_count(self):
        """Test expected number of backends."""
        assert len(Backend) == 5  # MLX, LLAMA_CPP, OLLAMA, VLLM, OPENCLAW


# =============================================================================
# ModelInfo Tests
# =============================================================================


class TestModelInfo:
    """Tests for ModelInfo Pydantic model."""

    def test_minimal_creation(self):
        """Test creating ModelInfo with minimal required fields."""
        model = ModelInfo(
            name="test-model",
            source=ModelSource.OLLAMA,
            backend=Backend.OLLAMA,
        )
        assert model.name == "test-model"
        assert model.source == ModelSource.OLLAMA
        assert model.backend == Backend.OLLAMA
        assert model.path is None
        assert model.format is None
        assert model.size_bytes is None
        assert ModelCapability.TEXT in model.capabilities

    def test_full_creation(self):
        """Test creating ModelInfo with all fields."""
        now = datetime.now()
        model = ModelInfo(
            name="Qwen/Qwen2.5-7B",
            source=ModelSource.HUGGINGFACE,
            path=Path("/models/qwen"),
            format=ModelFormat.SAFETENSORS,
            size_bytes=15_000_000_000,
            backend=Backend.MLX,
            capabilities={ModelCapability.TEXT, ModelCapability.VISION},
            last_accessed=now,
            metadata={"snapshot": "abc123"},
        )
        assert model.name == "Qwen/Qwen2.5-7B"
        assert model.path == Path("/models/qwen")
        assert model.format == ModelFormat.SAFETENSORS
        assert model.size_bytes == 15_000_000_000
        assert model.last_accessed == now
        assert model.metadata == {"snapshot": "abc123"}

    def test_display_name_huggingface(self):
        """Test display_name for HuggingFace source."""
        model = ModelInfo(
            name="meta-llama/Llama-3.2-8B",
            source=ModelSource.HUGGINGFACE,
            backend=Backend.MLX,
        )
        assert model.display_name == "[HF] meta-llama/Llama-3.2-8B"

    def test_display_name_ollama(self):
        """Test display_name for Ollama source."""
        model = ModelInfo(
            name="llama3.2:8b",
            source=ModelSource.OLLAMA,
            backend=Backend.OLLAMA,
        )
        assert model.display_name == "[OL] llama3.2:8b"

    def test_display_name_gguf(self):
        """Test display_name for GGUF file source."""
        model = ModelInfo(
            name="mistral-7b-instruct",
            source=ModelSource.GGUF_FILE,
            backend=Backend.LLAMA_CPP,
        )
        assert model.display_name == "[GF] mistral-7b-instruct"

    def test_size_human_gigabytes(self):
        """Test size_human for GB-sized models."""
        model = ModelInfo(
            name="test",
            source=ModelSource.OLLAMA,
            backend=Backend.OLLAMA,
            size_bytes=5_368_709_120,  # 5 GB
        )
        assert model.size_human == "5.0 GB"

    def test_size_human_megabytes(self):
        """Test size_human for MB-sized models."""
        model = ModelInfo(
            name="test",
            source=ModelSource.OLLAMA,
            backend=Backend.OLLAMA,
            size_bytes=524_288_000,  # 500 MB
        )
        assert model.size_human == "500.0 MB"

    def test_size_human_unknown(self):
        """Test size_human when size is unknown."""
        model = ModelInfo(
            name="test",
            source=ModelSource.OLLAMA,
            backend=Backend.OLLAMA,
            size_bytes=None,
        )
        assert model.size_human == "?"

    def test_is_text_model(self):
        """Test is_text_model property."""
        model = ModelInfo(
            name="test",
            source=ModelSource.OLLAMA,
            backend=Backend.OLLAMA,
            capabilities={ModelCapability.TEXT},
        )
        assert model.is_text_model is True
        assert model.is_vision_model is False

    def test_is_vision_model(self):
        """Test is_vision_model property."""
        model = ModelInfo(
            name="test",
            source=ModelSource.OLLAMA,
            backend=Backend.OLLAMA,
            capabilities={ModelCapability.TEXT, ModelCapability.VISION},
        )
        assert model.is_vision_model is True
        assert model.is_text_model is True

    def test_is_text_embedder(self):
        """Test is_text_embedder property."""
        model = ModelInfo(
            name="test",
            source=ModelSource.HUGGINGFACE,
            backend=Backend.MLX,
            capabilities={ModelCapability.TEXT_EMBEDDING},
        )
        assert model.is_text_embedder is True
        assert model.is_text_model is False

    def test_is_vision_embedder(self):
        """Test is_vision_embedder property."""
        model = ModelInfo(
            name="test",
            source=ModelSource.HUGGINGFACE,
            backend=Backend.MLX,
            capabilities={ModelCapability.VISION_EMBEDDING},
        )
        assert model.is_vision_embedder is True

    def test_capabilities_display(self):
        """Test capabilities_display property."""
        model = ModelInfo(
            name="test",
            source=ModelSource.HUGGINGFACE,
            backend=Backend.MLX,
            capabilities={ModelCapability.TEXT, ModelCapability.VISION},
        )
        assert "text" in model.capabilities_display
        assert "vision" in model.capabilities_display

    def test_capabilities_display_embedding(self):
        """Test capabilities_display for embedding models."""
        model = ModelInfo(
            name="test",
            source=ModelSource.HUGGINGFACE,
            backend=Backend.MLX,
            capabilities={ModelCapability.TEXT_EMBEDDING, ModelCapability.VISION_EMBEDDING},
        )
        assert "embed" in model.capabilities_display
        assert "v-embed" in model.capabilities_display


# =============================================================================
# Format Detection Tests
# =============================================================================


class TestDetectFormat:
    """Tests for format detection functions."""

    def test_detect_gguf_file_with_magic(self):
        """Test detecting GGUF file with correct magic bytes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_file = Path(tmpdir) / "model.gguf"
            # Write GGUF magic bytes (big endian)
            with open(gguf_file, "wb") as f:
                f.write(GGUF_MAGIC_BE)
                f.write(b"\x00" * 100)

            assert _detect_file_format(gguf_file) == ModelFormat.GGUF

    def test_detect_gguf_file_little_endian(self):
        """Test detecting GGUF file with little-endian magic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_file = Path(tmpdir) / "model.gguf"
            with open(gguf_file, "wb") as f:
                f.write(GGUF_MAGIC_LE)
                f.write(b"\x00" * 100)

            assert _detect_file_format(gguf_file) == ModelFormat.GGUF

    def test_detect_gguf_by_extension(self):
        """Test detecting GGUF file by extension (fallback)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_file = Path(tmpdir) / "model.gguf"
            # Write non-GGUF content but with .gguf extension
            gguf_file.write_bytes(b"not real gguf content")

            # Should still detect as GGUF by extension
            assert _detect_file_format(gguf_file) == ModelFormat.GGUF

    def test_detect_safetensors(self):
        """Test detecting safetensors file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            st_file = Path(tmpdir) / "model.safetensors"
            st_file.write_bytes(b"safetensors content")

            assert _detect_file_format(st_file) == ModelFormat.SAFETENSORS

    def test_detect_pytorch_bin(self):
        """Test detecting PyTorch model file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pt_file = Path(tmpdir) / "pytorch_model.bin"
            pt_file.write_bytes(b"pytorch content")

            assert _detect_file_format(pt_file) == ModelFormat.PYTORCH

    def test_detect_unknown_format(self):
        """Test that unknown formats return None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            unknown_file = Path(tmpdir) / "model.xyz"
            unknown_file.write_bytes(b"unknown content")

            assert _detect_file_format(unknown_file) is None

    def test_detect_format_directory(self):
        """Test detect_format on a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            model_dir.mkdir()
            (model_dir / "model.safetensors").write_bytes(b"content")

            assert detect_format(model_dir) == ModelFormat.SAFETENSORS

    def test_detect_format_nonexistent(self):
        """Test detect_format on nonexistent path."""
        assert detect_format(Path("/nonexistent/path")) is None

    def test_scan_directory_for_format(self):
        """Test scanning directory for model format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "config.json").write_text("{}")
            (Path(tmpdir) / "model.safetensors").write_bytes(b"content")

            assert _scan_directory_for_format(Path(tmpdir)) == ModelFormat.SAFETENSORS


# =============================================================================
# Backend Selection Tests
# =============================================================================


class TestSelectBackend:
    """Tests for backend selection logic."""

    def test_ollama_source_uses_ollama(self):
        """Test that Ollama source always uses Ollama backend."""
        backend = select_backend(ModelFormat.OLLAMA, ModelSource.OLLAMA)
        assert backend == Backend.OLLAMA

    def test_gguf_uses_llama_cpp(self):
        """Test that GGUF format uses llama.cpp backend."""
        backend = select_backend(ModelFormat.GGUF, ModelSource.HUGGINGFACE)
        assert backend == Backend.LLAMA_CPP

    def test_gguf_file_uses_llama_cpp(self):
        """Test that GGUF file source uses llama.cpp."""
        backend = select_backend(ModelFormat.GGUF, ModelSource.GGUF_FILE)
        assert backend == Backend.LLAMA_CPP

    def test_embedding_uses_mlx(self):
        """Test that embedding models use MLX backend."""
        backend = select_backend(
            ModelFormat.SAFETENSORS,
            ModelSource.HUGGINGFACE,
            capabilities={ModelCapability.TEXT_EMBEDDING},
        )
        assert backend == Backend.MLX

    def test_vision_embedding_uses_mlx(self):
        """Test that vision embedding models use MLX backend."""
        backend = select_backend(
            ModelFormat.SAFETENSORS,
            ModelSource.HUGGINGFACE,
            capabilities={ModelCapability.VISION_EMBEDDING},
        )
        assert backend == Backend.MLX

    @patch("r3lay.core.models._is_apple_silicon", return_value=True)
    def test_safetensors_apple_silicon_uses_mlx(self, mock_silicon):
        """Test safetensors on Apple Silicon uses MLX."""
        backend = select_backend(ModelFormat.SAFETENSORS, ModelSource.HUGGINGFACE)
        assert backend == Backend.MLX

    @patch("r3lay.core.models._is_apple_silicon", return_value=False)
    @patch("r3lay.core.models._has_cuda", return_value=True)
    def test_safetensors_cuda_uses_vllm(self, mock_cuda, mock_silicon):
        """Test safetensors with CUDA uses VLLM."""
        backend = select_backend(ModelFormat.SAFETENSORS, ModelSource.HUGGINGFACE)
        assert backend == Backend.VLLM

    @patch("r3lay.core.models._is_apple_silicon", return_value=False)
    @patch("r3lay.core.models._has_cuda", return_value=False)
    def test_safetensors_cpu_uses_llama_cpp(self, mock_cuda, mock_silicon):
        """Test safetensors on CPU uses llama.cpp."""
        backend = select_backend(ModelFormat.SAFETENSORS, ModelSource.HUGGINGFACE)
        assert backend == Backend.LLAMA_CPP

    def test_default_fallback(self):
        """Test that unknown format falls back to llama.cpp."""
        backend = select_backend(None, ModelSource.HUGGINGFACE)
        assert backend == Backend.LLAMA_CPP


# =============================================================================
# Capability Detection Tests
# =============================================================================


class TestDetectCapabilities:
    """Tests for capability detection from config.json."""

    def test_detect_vision_model_qwen2_vl(self):
        """Test detecting Qwen2-VL as vision model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "architectures": ["Qwen2VLForConditionalGeneration"],
                "model_type": "qwen2_vl",
            }
            (Path(tmpdir) / "config.json").write_text(json.dumps(config))

            caps = detect_capabilities(Path(tmpdir))

            assert ModelCapability.VISION in caps
            assert ModelCapability.TEXT in caps

    def test_detect_vision_model_llava(self):
        """Test detecting LLaVA as vision model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "architectures": ["LlavaForConditionalGeneration"],
                "model_type": "llava",
            }
            (Path(tmpdir) / "config.json").write_text(json.dumps(config))

            caps = detect_capabilities(Path(tmpdir))

            assert ModelCapability.VISION in caps

    def test_detect_text_embedding_bert(self):
        """Test detecting BERT as text embedding model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "architectures": ["BertModel"],
                "model_type": "bert",
            }
            (Path(tmpdir) / "config.json").write_text(json.dumps(config))

            caps = detect_capabilities(Path(tmpdir))

            assert ModelCapability.TEXT_EMBEDDING in caps

    def test_detect_text_embedding_e5(self):
        """Test detecting E5 as text embedding model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "architectures": ["XLMRobertaModel"],
                "_name_or_path": "intfloat/e5-large",
            }
            (Path(tmpdir) / "config.json").write_text(json.dumps(config))

            caps = detect_capabilities(Path(tmpdir))

            assert ModelCapability.TEXT_EMBEDDING in caps

    def test_detect_clip_vision_embedding(self):
        """Test detecting CLIP as vision embedding model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "architectures": ["CLIPModel"],
                "model_type": "clip",
            }
            (Path(tmpdir) / "config.json").write_text(json.dumps(config))

            caps = detect_capabilities(Path(tmpdir))

            assert ModelCapability.VISION_EMBEDDING in caps

    def test_detect_llm_qwen(self):
        """Test detecting Qwen as standard LLM."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "architectures": ["Qwen2ForCausalLM"],
                "model_type": "qwen2",
            }
            (Path(tmpdir) / "config.json").write_text(json.dumps(config))

            caps = detect_capabilities(Path(tmpdir))

            assert ModelCapability.TEXT in caps
            assert ModelCapability.VISION not in caps

    def test_detect_llm_llama(self):
        """Test detecting Llama as standard LLM."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
            }
            (Path(tmpdir) / "config.json").write_text(json.dumps(config))

            caps = detect_capabilities(Path(tmpdir))

            assert ModelCapability.TEXT in caps

    def test_detect_no_config(self):
        """Test detection with no config.json defaults to TEXT."""
        with tempfile.TemporaryDirectory() as tmpdir:
            caps = detect_capabilities(Path(tmpdir))
            assert caps == {ModelCapability.TEXT}

    def test_detect_invalid_json(self):
        """Test detection with invalid JSON defaults to TEXT."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "config.json").write_text("not valid json")
            caps = detect_capabilities(Path(tmpdir))
            assert caps == {ModelCapability.TEXT}

    def test_detect_hf_cache_structure(self):
        """Test detection in HuggingFace cache structure (snapshots subdir)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_dir = Path(tmpdir) / "snapshots" / "abc123"
            snapshot_dir.mkdir(parents=True)

            config = {"architectures": ["LlavaModel"], "model_type": "llava"}
            (snapshot_dir / "config.json").write_text(json.dumps(config))

            caps = detect_capabilities(Path(tmpdir))

            assert ModelCapability.VISION in caps


class TestDetectCapabilitiesFromName:
    """Tests for capability detection from model name."""

    def test_detect_vision_from_vl_suffix(self):
        """Test detecting vision from -VL suffix."""
        caps = detect_capabilities_from_name("Qwen2-VL-7B")
        assert ModelCapability.VISION in caps
        assert ModelCapability.TEXT in caps

    def test_detect_vision_from_llava(self):
        """Test detecting vision from llava in name."""
        caps = detect_capabilities_from_name("llava-1.5-7b")
        assert ModelCapability.VISION in caps

    def test_detect_vision_from_joycaption(self):
        """Test detecting vision from JoyCaption."""
        caps = detect_capabilities_from_name("JoyCaption-alpha-two")
        assert ModelCapability.VISION in caps

    def test_detect_embedding_from_embed(self):
        """Test detecting embedding from embed in name."""
        caps = detect_capabilities_from_name("nomic-embed-text")
        assert ModelCapability.TEXT_EMBEDDING in caps

    def test_detect_embedding_from_bge(self):
        """Test detecting embedding from bge prefix."""
        caps = detect_capabilities_from_name("bge-large-en")
        assert ModelCapability.TEXT_EMBEDDING in caps

    def test_detect_text_default(self):
        """Test default text detection for standard names."""
        caps = detect_capabilities_from_name("llama3.2:8b")
        assert ModelCapability.TEXT in caps
        assert ModelCapability.VISION not in caps


# =============================================================================
# Security Validation Tests
# =============================================================================


class TestValidateModelName:
    """Tests for model name validation (security)."""

    def test_valid_model_names(self):
        """Test that valid model names pass validation."""
        valid_names = [
            "Qwen/Qwen2.5-7B",
            "meta-llama/Llama-3.2-8B",
            "mistral-7b-instruct",
            "model_name_v1.2",
            "organization/model-name-2024",
            "mlx-community/Qwen2.5-7B-4bit",
        ]
        for name in valid_names:
            assert validate_model_name(name) is True, f"Valid name rejected: {name}"

    def test_path_traversal_attack(self):
        """Test that path traversal attempts are blocked."""
        dangerous_names = [
            "../../../etc/passwd",
            "models--org--..--malicious",
            "safe/../../../dangerous",
            "model/../config",
        ]
        for name in dangerous_names:
            assert validate_model_name(name) is False, f"Path traversal not blocked: {name}"

    def test_home_expansion_attack(self):
        """Test that home directory expansion is blocked."""
        dangerous_names = [
            "~/malicious/model",
            "model-~/config",
            "org~/model",
        ]
        for name in dangerous_names:
            assert validate_model_name(name) is False, f"Home expansion not blocked: {name}"

    def test_variable_expansion_attack(self):
        """Test that variable expansion is blocked."""
        dangerous_names = [
            "$HOME/model",
            "model-$USER",
            "${PATH}/malicious",
            "$SHELL",
        ]
        for name in dangerous_names:
            assert validate_model_name(name) is False, f"Variable expansion not blocked: {name}"

    def test_command_substitution_attack(self):
        """Test that command substitution is blocked."""
        dangerous_names = [
            "`whoami`",
            "model-`id`",
            "`rm -rf /`",
        ]
        for name in dangerous_names:
            assert validate_model_name(name) is False, f"Command substitution not blocked: {name}"

    def test_shell_operator_attack(self):
        """Test that shell operators are blocked."""
        dangerous_names = [
            "model|malicious",
            "name&background",
            "cmd;rm -rf",
            "test||exploit",
        ]
        for name in dangerous_names:
            assert validate_model_name(name) is False, f"Shell operator not blocked: {name}"

    def test_line_break_injection(self):
        """Test that line breaks are blocked (newline injection)."""
        dangerous_names = [
            "model\nmalicious",
            "name\rexploit",
            "test\r\ninjection",
        ]
        for name in dangerous_names:
            assert validate_model_name(name) is False, f"Line break not blocked: {name}"

    def test_combined_attacks(self):
        """Test combinations of attack vectors."""
        dangerous_names = [
            "../$HOME/model",
            "`cat ../../../etc/passwd`",
            "model;rm -rf ~/",
            "$USER/../config",
        ]
        for name in dangerous_names:
            assert validate_model_name(name) is False, f"Combined attack not blocked: {name}"


# =============================================================================
# Scanner Tests
# =============================================================================


class TestScanGgufFolder:
    """Tests for GGUF folder scanner."""

    def test_scan_empty_folder(self):
        """Test scanning empty folder returns empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            models = scan_gguf_folder(Path(tmpdir))
            assert models == []

    def test_scan_rejects_path_traversal(self):
        """Test that GGUF scanner rejects path traversal filenames."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create malicious filename
            malicious = Path(tmpdir) / "..--etc--passwd.gguf"
            malicious.write_bytes(GGUF_MAGIC_BE + b"\x00" * 100)

            models = scan_gguf_folder(Path(tmpdir))

            # Should be rejected due to ".." in name
            assert len(models) == 0

    def test_scan_rejects_shell_operators(self):
        """Test that GGUF scanner rejects shell operators in filenames."""
        with tempfile.TemporaryDirectory() as tmpdir:
            malicious = Path(tmpdir) / "model|malicious.gguf"
            malicious.write_bytes(GGUF_MAGIC_BE + b"\x00" * 100)

            models = scan_gguf_folder(Path(tmpdir))

            assert len(models) == 0

    def test_scan_with_gguf_files(self):
        """Test scanning folder with GGUF files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create fake GGUF files
            gguf1 = Path(tmpdir) / "model1.gguf"
            gguf2 = Path(tmpdir) / "model2.gguf"
            gguf1.write_bytes(GGUF_MAGIC_BE + b"\x00" * 100)
            gguf2.write_bytes(GGUF_MAGIC_BE + b"\x00" * 200)

            models = scan_gguf_folder(Path(tmpdir))

            assert len(models) == 2
            names = {m.name for m in models}
            assert "model1" in names
            assert "model2" in names

    def test_scan_ignores_non_gguf(self):
        """Test that non-GGUF files are ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "model.gguf").write_bytes(GGUF_MAGIC_BE)
            (Path(tmpdir) / "readme.txt").write_text("readme")
            (Path(tmpdir) / "config.json").write_text("{}")

            models = scan_gguf_folder(Path(tmpdir))

            assert len(models) == 1
            assert models[0].name == "model"

    def test_scan_model_properties(self):
        """Test that scanned models have correct properties."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_file = Path(tmpdir) / "test-model.gguf"
            gguf_file.write_bytes(GGUF_MAGIC_BE + b"\x00" * 1000)

            models = scan_gguf_folder(Path(tmpdir))

            assert len(models) == 1
            model = models[0]
            assert model.name == "test-model"
            assert model.source == ModelSource.GGUF_FILE
            assert model.format == ModelFormat.GGUF
            assert model.backend == Backend.LLAMA_CPP
            assert model.path == gguf_file
            assert model.size_bytes is not None

    def test_scan_creates_folder_if_missing(self):
        """Test that scanning auto-creates the folder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_folder = Path(tmpdir) / "new" / "gguf" / "folder"
            models = scan_gguf_folder(new_folder)

            assert new_folder.exists()
            assert models == []

    def test_scan_nonexistent_parent(self):
        """Test scanning with nonexistent parent returns empty."""
        models = scan_gguf_folder(Path("/nonexistent/path/models"))
        assert models == []


class TestScanMlxFolder:
    """Tests for MLX folder scanner."""

    def test_scan_empty_folder(self):
        """Test scanning empty folder returns empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            models = scan_mlx_folder(Path(tmpdir))
            assert models == []

    def test_scan_rejects_path_traversal(self):
        """Test that MLX scanner rejects path traversal in directory names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            malicious_dir = Path(tmpdir) / "..--etc--passwd"
            malicious_dir.mkdir()
            (malicious_dir / "model.safetensors").write_bytes(b"content")
            (malicious_dir / "config.json").write_text("{}")

            models = scan_mlx_folder(Path(tmpdir))

            # Should be rejected
            assert len(models) == 0

    def test_scan_with_mlx_models(self):
        """Test scanning folder with MLX model structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "Qwen2.5-7B-4bit"
            model_dir.mkdir()
            (model_dir / "model.safetensors").write_bytes(b"content")
            (model_dir / "config.json").write_text('{"model_type": "qwen2"}')

            models = scan_mlx_folder(Path(tmpdir))

            assert len(models) == 1
            assert "Qwen2.5-7B-4bit" in models[0].name
            assert models[0].format == ModelFormat.SAFETENSORS
            assert models[0].backend == Backend.MLX

    def test_scan_ignores_incomplete_models(self):
        """Test that models without config.json are ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Model without config.json
            incomplete = Path(tmpdir) / "incomplete"
            incomplete.mkdir()
            (incomplete / "model.safetensors").write_bytes(b"content")

            # Complete model
            complete = Path(tmpdir) / "complete"
            complete.mkdir()
            (complete / "model.safetensors").write_bytes(b"content")
            (complete / "config.json").write_text("{}")

            models = scan_mlx_folder(Path(tmpdir))

            assert len(models) == 1
            assert "complete" in models[0].name

    def test_scan_nonexistent(self):
        """Test scanning nonexistent folder returns empty."""
        models = scan_mlx_folder(Path("/nonexistent/mlx/folder"))
        assert models == []


class TestScanLlmModelsFolder:
    """Tests for llm-models folder scanner with mmproj detection."""

    def test_scan_empty_folder(self):
        """Test scanning empty folder returns empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            models = scan_llm_models_folder(Path(tmpdir))
            assert models == []

    def test_scan_rejects_dangerous_directory_names(self):
        """Test that llm-models scanner rejects dangerous directory names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            malicious_dir = Path(tmpdir) / "model;rm -rf ~"
            malicious_dir.mkdir()
            (malicious_dir / "model.gguf").write_bytes(GGUF_MAGIC_BE + b"\x00" * 1000)

            models = scan_llm_models_folder(Path(tmpdir))

            # Should be rejected
            assert len(models) == 0

    def test_scan_with_gguf_model(self):
        """Test scanning folder with GGUF model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "llama-7b"
            model_dir.mkdir()
            (model_dir / "llama-7b-q4.gguf").write_bytes(GGUF_MAGIC_BE + b"\x00" * 1000)

            models = scan_llm_models_folder(Path(tmpdir))

            assert len(models) == 1
            assert models[0].name == "llama-7b"
            assert models[0].format == ModelFormat.GGUF

    def test_scan_with_mmproj(self):
        """Test scanning folder with mmproj file adds VISION capability."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "llava-model"
            model_dir.mkdir()
            # Main model (larger)
            (model_dir / "llava-7b-q4.gguf").write_bytes(GGUF_MAGIC_BE + b"\x00" * 10000)
            # mmproj file (smaller)
            (model_dir / "mmproj-llava.gguf").write_bytes(GGUF_MAGIC_BE + b"\x00" * 1000)

            models = scan_llm_models_folder(Path(tmpdir))

            assert len(models) == 1
            model = models[0]
            assert ModelCapability.VISION in model.capabilities
            assert "mmproj_path" in model.metadata

    def test_scan_ignores_hub_folder(self):
        """Test that 'hub' folder is ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Should be ignored
            hub_dir = Path(tmpdir) / "hub"
            hub_dir.mkdir()
            (hub_dir / "model.gguf").write_bytes(GGUF_MAGIC_BE)

            # Should be found
            model_dir = Path(tmpdir) / "real-model"
            model_dir.mkdir()
            (model_dir / "model.gguf").write_bytes(GGUF_MAGIC_BE)

            models = scan_llm_models_folder(Path(tmpdir))

            assert len(models) == 1
            assert models[0].name == "real-model"


class TestScanHuggingfaceCache:
    """Tests for HuggingFace cache scanner."""

    def test_scan_nonexistent(self):
        """Test scanning nonexistent path returns empty."""
        models = scan_huggingface_cache(Path("/nonexistent/hf/cache"))
        assert models == []

    def test_scan_rejects_path_traversal_in_repo_id(self):
        """Test that HF cache scanner rejects path traversal in repo IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create HF cache structure with malicious repo_id
            model_dir = Path(tmpdir) / "models--org--..--etc--passwd"
            snapshot_dir = model_dir / "snapshots" / "abc123"
            snapshot_dir.mkdir(parents=True)

            (snapshot_dir / "model.safetensors").write_bytes(b"content")
            (snapshot_dir / "config.json").write_text('{"model_type": "test"}')

            models = scan_huggingface_cache(Path(tmpdir))

            # Should be rejected due to ".." in repo_id
            assert len(models) == 0

    def test_scan_rejects_variable_expansion(self):
        """Test that HF cache scanner rejects variable expansion in names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "models--$USER--malicious"
            snapshot_dir = model_dir / "snapshots" / "abc123"
            snapshot_dir.mkdir(parents=True)

            (snapshot_dir / "model.safetensors").write_bytes(b"content")
            (snapshot_dir / "config.json").write_text("{}")

            models = scan_huggingface_cache(Path(tmpdir))

            assert len(models) == 0

    def test_scan_with_model(self):
        """Test scanning HuggingFace cache structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create HF cache structure: models--org--name/snapshots/hash/files
            model_dir = Path(tmpdir) / "models--Qwen--Qwen2.5-7B"
            snapshot_dir = model_dir / "snapshots" / "abc123"
            snapshot_dir.mkdir(parents=True)

            (snapshot_dir / "model.safetensors").write_bytes(b"content")
            (snapshot_dir / "config.json").write_text('{"model_type": "qwen2"}')

            # Create refs/main for last_accessed
            refs_dir = model_dir / "refs"
            refs_dir.mkdir()
            (refs_dir / "main").write_text("abc123")

            models = scan_huggingface_cache(Path(tmpdir))

            assert len(models) == 1
            assert models[0].name == "Qwen/Qwen2.5-7B"
            assert models[0].source == ModelSource.HUGGINGFACE

    def test_scan_ignores_non_model_dirs(self):
        """Test that non-model directories are ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Not a model directory
            (Path(tmpdir) / "cache-version").write_text("1")
            (Path(tmpdir) / "blobs").mkdir()

            models = scan_huggingface_cache(Path(tmpdir))
            assert models == []


class TestScanOllama:
    """Tests for Ollama API scanner."""

    @pytest.mark.asyncio
    async def test_scan_ollama_success(self):
        """Test successful Ollama scan."""
        mock_response = {
            "models": [
                {
                    "name": "llama3.2:8b",
                    "size": 5_000_000_000,
                    "modified_at": "2026-01-31T10:00:00Z",
                    "digest": "sha256:abc123",
                    "details": {"family": "llama"},
                }
            ]
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_instance.get.return_value = MagicMock(
                json=lambda: mock_response,
                raise_for_status=lambda: None,
            )
            mock_client.return_value = mock_instance

            models = await scan_ollama()

            assert len(models) == 1
            assert models[0].name == "llama3.2:8b"
            assert models[0].source == ModelSource.OLLAMA
            assert models[0].backend == Backend.OLLAMA

    @pytest.mark.asyncio
    async def test_scan_ollama_not_running(self):
        """Test Ollama scan when server not running."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_instance.get.side_effect = Exception("Connection refused")
            mock_client.return_value = mock_instance

            models = await scan_ollama()

            assert models == []


# =============================================================================
# ModelScanner Class Tests
# =============================================================================


class TestModelScanner:
    """Tests for ModelScanner class."""

    @pytest.mark.asyncio
    async def test_scan_all_empty(self):
        """Test scan_all with no models found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scanner = ModelScanner(
                hf_cache_path=Path(tmpdir) / "hf",
                mlx_folder=Path(tmpdir) / "mlx",
                gguf_folder=Path(tmpdir) / "gguf",
                llm_models_folder=Path(tmpdir) / "llm",
            )

            # Mock Ollama to return empty
            with patch("r3lay.core.models.scan_ollama", return_value=[]):
                models = await scanner.scan_all()

            assert models == []

    @pytest.mark.asyncio
    async def test_scan_all_with_models(self):
        """Test scan_all finds models from multiple sources."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create GGUF model
            gguf_dir = Path(tmpdir) / "gguf"
            gguf_dir.mkdir()
            (gguf_dir / "test.gguf").write_bytes(GGUF_MAGIC_BE + b"\x00" * 100)

            scanner = ModelScanner(
                hf_cache_path=Path(tmpdir) / "nonexistent",
                mlx_folder=Path(tmpdir) / "nonexistent",
                gguf_folder=gguf_dir,
                llm_models_folder=Path(tmpdir) / "nonexistent",
            )

            with patch("r3lay.core.models.scan_ollama", return_value=[]):
                models = await scanner.scan_all()

            assert len(models) == 1
            assert models[0].name == "test"

    def test_get_by_name(self):
        """Test get_by_name lookup."""
        scanner = ModelScanner()
        scanner._models = [
            ModelInfo(name="model-a", source=ModelSource.OLLAMA, backend=Backend.OLLAMA),
            ModelInfo(name="model-b", source=ModelSource.OLLAMA, backend=Backend.OLLAMA),
        ]

        result = scanner.get_by_name("model-a")
        assert result is not None
        assert result.name == "model-a"

        result = scanner.get_by_name("nonexistent")
        assert result is None

    def test_get_by_source(self):
        """Test get_by_source filtering."""
        scanner = ModelScanner()
        scanner._models = [
            ModelInfo(name="hf-model", source=ModelSource.HUGGINGFACE, backend=Backend.MLX),
            ModelInfo(name="ollama-model", source=ModelSource.OLLAMA, backend=Backend.OLLAMA),
            ModelInfo(name="gguf-model", source=ModelSource.GGUF_FILE, backend=Backend.LLAMA_CPP),
        ]

        hf_models = scanner.get_by_source(ModelSource.HUGGINGFACE)
        assert len(hf_models) == 1
        assert hf_models[0].name == "hf-model"

    def test_get_by_capability(self):
        """Test get_by_capability filtering."""
        scanner = ModelScanner()
        scanner._models = [
            ModelInfo(
                name="text-model",
                source=ModelSource.OLLAMA,
                backend=Backend.OLLAMA,
                capabilities={ModelCapability.TEXT},
            ),
            ModelInfo(
                name="vision-model",
                source=ModelSource.OLLAMA,
                backend=Backend.OLLAMA,
                capabilities={ModelCapability.TEXT, ModelCapability.VISION},
            ),
        ]

        vision_models = scanner.get_by_capability(ModelCapability.VISION)
        assert len(vision_models) == 1
        assert vision_models[0].name == "vision-model"

    def test_get_text_models(self):
        """Test get_text_models helper."""
        scanner = ModelScanner()
        scanner._models = [
            ModelInfo(
                name="text",
                source=ModelSource.OLLAMA,
                backend=Backend.OLLAMA,
                capabilities={ModelCapability.TEXT},
            ),
            ModelInfo(
                name="embed",
                source=ModelSource.HUGGINGFACE,
                backend=Backend.MLX,
                capabilities={ModelCapability.TEXT_EMBEDDING},
            ),
        ]

        text_models = scanner.get_text_models()
        assert len(text_models) == 1
        assert text_models[0].name == "text"

    def test_get_vision_models(self):
        """Test get_vision_models helper."""
        scanner = ModelScanner()
        scanner._models = [
            ModelInfo(
                name="vision",
                source=ModelSource.OLLAMA,
                backend=Backend.OLLAMA,
                capabilities={ModelCapability.VISION, ModelCapability.TEXT},
            ),
        ]

        vision_models = scanner.get_vision_models()
        assert len(vision_models) == 1

    def test_get_cached_models(self):
        """Test get_cached_models returns internal list."""
        scanner = ModelScanner()
        scanner._models = [
            ModelInfo(name="cached", source=ModelSource.OLLAMA, backend=Backend.OLLAMA)
        ]

        cached = scanner.get_cached_models()
        assert len(cached) == 1
        assert cached[0].name == "cached"

    def test_clear_cache(self):
        """Test clear_cache empties the model list."""
        scanner = ModelScanner()
        scanner._models = [
            ModelInfo(name="model", source=ModelSource.OLLAMA, backend=Backend.OLLAMA)
        ]

        scanner.clear_cache()
        assert scanner._models == []


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_calculate_directory_size(self):
        """Test directory size calculation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "file1.txt").write_bytes(b"a" * 100)
            (Path(tmpdir) / "file2.txt").write_bytes(b"b" * 200)

            size = _calculate_directory_size(Path(tmpdir))

            assert size == 300

    def test_calculate_directory_size_with_subdirs(self):
        """Test directory size calculation with subdirectories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "file.txt").write_bytes(b"a" * 100)
            subdir = Path(tmpdir) / "subdir"
            subdir.mkdir()
            (subdir / "nested.txt").write_bytes(b"b" * 50)

            size = _calculate_directory_size(Path(tmpdir))

            assert size == 150

    def test_calculate_directory_size_empty(self):
        """Test directory size calculation for empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            size = _calculate_directory_size(Path(tmpdir))
            assert size == 0

    def test_is_apple_silicon(self):
        """Test Apple Silicon detection returns bool."""
        result = _is_apple_silicon()
        assert isinstance(result, bool)


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_model_info_empty_capabilities(self):
        """Test ModelInfo with empty capabilities set."""
        model = ModelInfo(
            name="test",
            source=ModelSource.OLLAMA,
            backend=Backend.OLLAMA,
            capabilities=set(),
        )
        assert model.capabilities_display == "unknown"

    def test_detect_format_permission_error(self):
        """Test format detection handles permission errors gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "restricted"
            model_dir.mkdir(mode=0o000)

            try:
                fmt = detect_format(model_dir)
                # Should return None or handle gracefully
                assert fmt is None or isinstance(fmt, ModelFormat)
            finally:
                # Restore permissions for cleanup
                model_dir.chmod(0o755)

    def test_model_info_with_none_path(self):
        """Test ModelInfo serialization with None path."""
        model = ModelInfo(
            name="ollama-model",
            source=ModelSource.OLLAMA,
            backend=Backend.OLLAMA,
            path=None,
        )
        # Should not raise
        assert model.path is None
        assert model.display_name == "[OL] ollama-model"

    def test_large_size_human(self):
        """Test size_human with very large sizes (TB+)."""
        model = ModelInfo(
            name="huge",
            source=ModelSource.HUGGINGFACE,
            backend=Backend.MLX,
            size_bytes=5_000_000_000_000,  # 5 TB
        )
        assert "TB" in model.size_human
