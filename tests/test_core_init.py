"""Tests for r3lay.core module top-level functionality.

Tests cover:
- Utility functions: embeddings_available, vision_embeddings_available, pdf_extraction_available
- R3LayState: initialization, model loading, router management, session management
- Error handling and edge cases
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from r3lay.core import (
    R3LayState,
    embeddings_available,
    pdf_extraction_available,
    vision_embeddings_available,
)

if TYPE_CHECKING:
    from r3lay.core.models import ModelInfo


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestEmbeddingsAvailable:
    """Tests for embeddings_available function."""

    def test_returns_bool(self):
        """Test that function returns a boolean."""
        result = embeddings_available()
        assert isinstance(result, bool)

    def test_with_sentence_transformers(self):
        """Test detection when sentence_transformers is available."""
        with patch.dict(sys.modules, {"sentence_transformers": MagicMock()}):
            with patch("importlib.util.find_spec") as mock_spec:
                mock_spec.return_value = MagicMock()
                result = embeddings_available()
                assert result is True

    def test_without_dependencies(self):
        """Test detection when no embedding libs are available."""
        with patch("importlib.util.find_spec") as mock_spec:
            mock_spec.return_value = None
            result = embeddings_available()
            assert result is False


class TestVisionEmbeddingsAvailable:
    """Tests for vision_embeddings_available function."""

    def test_returns_bool(self):
        """Test that function returns a boolean."""
        result = vision_embeddings_available()
        assert isinstance(result, bool)

    def test_requires_both_deps(self):
        """Test that vision embeddings require both PIL and embedder."""
        with patch("importlib.util.find_spec") as mock_spec:
            # Mock only PIL available
            def spec_side_effect(name):
                if name == "PIL":
                    return MagicMock()
                return None

            mock_spec.side_effect = spec_side_effect
            result = vision_embeddings_available()
            assert result is False

    def test_with_all_deps(self):
        """Test detection when all vision deps are available."""
        with patch("importlib.util.find_spec") as mock_spec:
            mock_spec.return_value = MagicMock()
            result = vision_embeddings_available()
            assert result is True


class TestPdfExtractionAvailable:
    """Tests for pdf_extraction_available function."""

    def test_returns_bool(self):
        """Test that function returns a boolean."""
        result = pdf_extraction_available()
        assert isinstance(result, bool)

    def test_detects_fitz(self):
        """Test detection of pymupdf (fitz)."""
        with patch("importlib.util.find_spec") as mock_spec:
            mock_spec.return_value = MagicMock()
            result = pdf_extraction_available()
            assert result is True
            mock_spec.assert_called_with("fitz")


# =============================================================================
# R3LayState Tests
# =============================================================================


class TestR3LayStateInit:
    """Tests for R3LayState initialization."""

    def test_default_initialization(self, tmp_path: Path):
        """Test default state initialization."""
        state = R3LayState(project_path=tmp_path)
        assert state.project_path == tmp_path
        assert state.current_model is None
        assert state.current_backend is None
        assert state.session is not None
        assert state.scanner is not None

    def test_string_path_conversion(self, tmp_path: Path):
        """Test that string paths are converted to Path objects."""
        state = R3LayState(project_path=str(tmp_path))
        assert isinstance(state.project_path, Path)
        assert state.project_path == tmp_path

    def test_config_lazy_load(self, tmp_path: Path):
        """Test config property creates default config."""
        state = R3LayState(project_path=tmp_path)
        config = state.config
        assert config is not None
        # Second access should return same instance
        assert state.config is config


class TestR3LayStateRouter:
    """Tests for R3LayState router methods."""

    def test_init_router(self, tmp_path: Path):
        """Test router initialization."""
        state = R3LayState(project_path=tmp_path)
        router = state.init_router(text_model="test-model", vision_model="test-vision")
        assert router is not None
        assert state.router is router
        assert state.router_config is not None
        assert state.router_config.text_model == "test-model"
        assert state.router_config.vision_model == "test-vision"

    def test_init_router_text_only(self, tmp_path: Path):
        """Test router initialization with text model only."""
        state = R3LayState(project_path=tmp_path)
        router = state.init_router(text_model="text-only-model")
        assert router is not None
        assert state.router_config.vision_model is None


class TestR3LayStateModelLoading:
    """Tests for R3LayState model loading methods."""

    @pytest.fixture
    def mock_model_info(self) -> "ModelInfo":
        """Create mock model info."""
        from r3lay.core.models import Backend, ModelFormat, ModelInfo, ModelSource

        return ModelInfo(
            name="test-model",
            path=Path("/fake/path"),
            source=ModelSource.GGUF_FILE,
            format=ModelFormat.GGUF,
            backend=Backend.OLLAMA,
            size_bytes=1000,
            is_vision_model=False,
        )

    @pytest.fixture
    def mock_vision_model_info(self) -> "ModelInfo":
        """Create mock vision model info."""
        from r3lay.core.models import (
            Backend,
            ModelCapability,
            ModelFormat,
            ModelInfo,
            ModelSource,
        )

        return ModelInfo(
            name="test-vision-model",
            path=Path("/fake/path"),
            source=ModelSource.GGUF_FILE,
            format=ModelFormat.GGUF,
            backend=Backend.OLLAMA,
            size_bytes=1000,
            capabilities={ModelCapability.VISION, ModelCapability.TEXT},
        )

    @pytest.mark.asyncio
    async def test_load_model_text(self, tmp_path: Path, mock_model_info: "ModelInfo"):
        """Test loading a text model."""
        state = R3LayState(project_path=tmp_path)

        mock_backend = AsyncMock()
        mock_backend.load = AsyncMock()

        with patch("r3lay.core.backends.create_backend", return_value=mock_backend):
            await state.load_model(mock_model_info)

        assert state.current_model == "test-model"
        assert state.current_backend is mock_backend
        mock_backend.load.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_model_vision(self, tmp_path: Path, mock_vision_model_info: "ModelInfo"):
        """Test loading a vision model."""
        state = R3LayState(project_path=tmp_path)

        mock_backend = AsyncMock()
        mock_backend.load = AsyncMock()

        with patch("r3lay.core.backends.create_backend", return_value=mock_backend):
            await state.load_model(mock_vision_model_info)

        assert state.current_model == "test-vision-model"
        assert state.router is not None
        # Note: The router's current_model_type depends on the backend being set
        # as vision_backend, which requires the backend to be recognized as vision
        assert state.router.vision_backend is mock_backend

    @pytest.mark.asyncio
    async def test_load_model_unloads_existing(self, tmp_path: Path, mock_model_info: "ModelInfo"):
        """Test that loading a new model unloads the existing one."""
        state = R3LayState(project_path=tmp_path)

        old_backend = AsyncMock()
        old_backend.unload = AsyncMock()
        state.current_backend = old_backend
        state.current_model = "old-model"

        new_backend = AsyncMock()
        new_backend.load = AsyncMock()

        with patch("r3lay.core.backends.create_backend", return_value=new_backend):
            await state.load_model(mock_model_info)

        old_backend.unload.assert_called_once()

    @pytest.mark.asyncio
    async def test_unload_model(self, tmp_path: Path):
        """Test unloading a model."""
        state = R3LayState(project_path=tmp_path)

        mock_backend = AsyncMock()
        mock_backend.unload = AsyncMock()
        state.current_backend = mock_backend
        state.current_model = "test-model"

        await state.unload_model()

        mock_backend.unload.assert_called_once()
        assert state.current_backend is None
        assert state.current_model is None

    @pytest.mark.asyncio
    async def test_unload_model_when_none(self, tmp_path: Path):
        """Test unloading when no model is loaded."""
        state = R3LayState(project_path=tmp_path)
        # Should not raise
        await state.unload_model()
        assert state.current_backend is None


class TestR3LayStateSwitchModel:
    """Tests for R3LayState.switch_model method."""

    @pytest.fixture
    def state_with_roles(self, tmp_path: Path) -> R3LayState:
        """Create state with model roles configured."""
        from r3lay.config import ModelRoles

        state = R3LayState(project_path=tmp_path)
        state.model_roles = ModelRoles(
            text_model="text-model-name",
            vision_model="vision-model-name",
        )
        return state

    @pytest.mark.asyncio
    async def test_switch_model_no_roles(self, tmp_path: Path):
        """Test switch_model fails without model_roles."""
        state = R3LayState(project_path=tmp_path)
        result = await state.switch_model("text")
        assert result is False

    @pytest.mark.asyncio
    async def test_switch_model_not_in_available(self, state_with_roles: R3LayState):
        """Test switch_model fails when model not in available_models."""
        result = await state_with_roles.switch_model("text")
        assert result is False

    @pytest.mark.asyncio
    async def test_switch_model_success(self, state_with_roles: R3LayState):
        """Test successful model switch."""
        from r3lay.core.models import Backend, ModelFormat, ModelInfo, ModelSource

        text_model = ModelInfo(
            name="text-model-name",
            path=Path("/fake/path"),
            source=ModelSource.GGUF_FILE,
            format=ModelFormat.GGUF,
            backend=Backend.OLLAMA,
            size_bytes=1000,
            is_vision_model=False,
        )
        state_with_roles.available_models = [text_model]

        mock_backend = AsyncMock()
        mock_backend.load = AsyncMock()

        with patch("r3lay.core.backends.create_backend", return_value=mock_backend):
            result = await state_with_roles.switch_model("text")

        assert result is True
        assert state_with_roles.current_model == "text-model-name"

    @pytest.mark.asyncio
    async def test_switch_model_load_failure(self, state_with_roles: R3LayState):
        """Test switch_model handles load failure gracefully."""
        from r3lay.core.models import Backend, ModelFormat, ModelInfo, ModelSource

        text_model = ModelInfo(
            name="text-model-name",
            path=Path("/fake/path"),
            source=ModelSource.GGUF_FILE,
            format=ModelFormat.GGUF,
            backend=Backend.OLLAMA,
            size_bytes=1000,
            is_vision_model=False,
        )
        state_with_roles.available_models = [text_model]

        with patch("r3lay.core.backends.create_backend") as mock_create:
            mock_create.side_effect = RuntimeError("Load failed")
            result = await state_with_roles.switch_model("text")

        assert result is False


class TestR3LayStateIndex:
    """Tests for R3LayState index methods."""

    def test_init_index(self, tmp_path: Path):
        """Test index initialization."""
        state = R3LayState(project_path=tmp_path)
        index = state.init_index()
        assert index is not None
        assert state.index is index
        # Second call returns same instance
        assert state.init_index() is index

    def test_init_index_with_embedder(self, tmp_path: Path):
        """Test index initialization with embedder."""
        state = R3LayState(project_path=tmp_path)
        mock_embedder = MagicMock()
        state.text_embedder = mock_embedder

        index = state.init_index(with_embedder=True)
        assert index.text_embedder is mock_embedder

    def test_init_index_attach_embedder_later(self, tmp_path: Path):
        """Test attaching embedder to existing index."""
        state = R3LayState(project_path=tmp_path)
        # Create index without embedder
        index = state.init_index(with_embedder=False)
        assert index.text_embedder is None

        # Now add embedder
        mock_embedder = MagicMock()
        state.text_embedder = mock_embedder
        index2 = state.init_index(with_embedder=True)

        assert index2 is index  # Same instance
        assert index.text_embedder is mock_embedder


class TestR3LayStateEmbedder:
    """Tests for R3LayState embedder methods."""

    @pytest.mark.asyncio
    async def test_init_embedder_unavailable(self, tmp_path: Path):
        """Test init_embedder when dependencies unavailable."""
        state = R3LayState(project_path=tmp_path)

        with patch("r3lay.core.embeddings_available", return_value=False):
            result = await state.init_embedder()

        assert result is None
        assert state.text_embedder is None

    @pytest.mark.asyncio
    async def test_init_embedder_already_loaded(self, tmp_path: Path):
        """Test init_embedder returns existing embedder if loaded."""
        state = R3LayState(project_path=tmp_path)
        mock_embedder = MagicMock()
        mock_embedder.is_loaded = True
        state.text_embedder = mock_embedder

        result = await state.init_embedder()
        assert result is mock_embedder

    @pytest.mark.asyncio
    async def test_unload_embedder(self, tmp_path: Path):
        """Test unloading text embedder."""
        state = R3LayState(project_path=tmp_path)
        mock_embedder = AsyncMock()
        mock_embedder.unload = AsyncMock()
        state.text_embedder = mock_embedder

        await state.unload_embedder()

        mock_embedder.unload.assert_called_once()
        assert state.text_embedder is None

    @pytest.mark.asyncio
    async def test_unload_embedder_when_none(self, tmp_path: Path):
        """Test unloading when no embedder loaded."""
        state = R3LayState(project_path=tmp_path)
        await state.unload_embedder()  # Should not raise


class TestR3LayStateVisionEmbedder:
    """Tests for R3LayState vision embedder methods."""

    @pytest.mark.asyncio
    async def test_init_vision_embedder_unavailable(self, tmp_path: Path):
        """Test init_vision_embedder when dependencies unavailable."""
        state = R3LayState(project_path=tmp_path)

        with patch("r3lay.core.vision_embeddings_available", return_value=False):
            result = await state.init_vision_embedder()

        assert result is None

    @pytest.mark.asyncio
    async def test_init_vision_embedder_already_loaded(self, tmp_path: Path):
        """Test init_vision_embedder returns existing embedder if loaded."""
        state = R3LayState(project_path=tmp_path)
        mock_embedder = MagicMock()
        mock_embedder.is_loaded = True
        state.vision_embedder = mock_embedder

        result = await state.init_vision_embedder()
        assert result is mock_embedder

    @pytest.mark.asyncio
    async def test_unload_vision_embedder(self, tmp_path: Path):
        """Test unloading vision embedder."""
        state = R3LayState(project_path=tmp_path)
        mock_embedder = AsyncMock()
        mock_embedder.unload = AsyncMock()
        state.vision_embedder = mock_embedder

        await state.unload_vision_embedder()

        mock_embedder.unload.assert_called_once()
        assert state.vision_embedder is None


class TestR3LayStateSession:
    """Tests for R3LayState session methods."""

    def test_new_session(self, tmp_path: Path):
        """Test creating a new session."""
        state = R3LayState(project_path=tmp_path)
        old_session = state.session

        new_session = state.new_session()

        assert new_session is not old_session
        assert state.session is new_session

    def test_new_session_resets_router(self, tmp_path: Path):
        """Test that new_session resets the router."""
        state = R3LayState(project_path=tmp_path)
        state.init_router(text_model="test")

        mock_reset = MagicMock()
        state.router.reset = mock_reset

        state.new_session()

        mock_reset.assert_called_once()

    def test_get_sessions_dir(self, tmp_path: Path):
        """Test getting sessions directory."""
        state = R3LayState(project_path=tmp_path)
        sessions_dir = state.get_sessions_dir()

        assert sessions_dir == tmp_path / ".r3lay" / "sessions"
        assert sessions_dir.exists()


class TestR3LayStateSignalsAxioms:
    """Tests for R3LayState signals and axioms methods."""

    def test_init_signals(self, tmp_path: Path):
        """Test signals manager initialization."""
        state = R3LayState(project_path=tmp_path)
        signals = state.init_signals()

        assert signals is not None
        assert state.signals_manager is signals
        # Second call returns same instance
        assert state.init_signals() is signals

    def test_init_axioms(self, tmp_path: Path):
        """Test axiom manager initialization."""
        state = R3LayState(project_path=tmp_path)
        axioms = state.init_axioms()

        assert axioms is not None
        assert state.axiom_manager is axioms
        # Second call returns same instance
        assert state.init_axioms() is axioms


class TestR3LayStateResearch:
    """Tests for R3LayState research methods."""

    def test_init_research_no_backend(self, tmp_path: Path):
        """Test init_research raises without backend."""
        state = R3LayState(project_path=tmp_path)

        with pytest.raises(ValueError, match="No LLM backend loaded"):
            state.init_research()

    def test_init_research_with_backend(self, tmp_path: Path):
        """Test init_research with backend loaded."""
        state = R3LayState(project_path=tmp_path)
        state.current_backend = MagicMock()

        orchestrator = state.init_research()

        assert orchestrator is not None
        assert state.research_orchestrator is orchestrator
        assert state.signals_manager is not None
        assert state.axiom_manager is not None
        assert state.search_client is not None

    def test_init_research_custom_endpoint(self, tmp_path: Path):
        """Test init_research with custom searxng endpoint."""
        state = R3LayState(project_path=tmp_path)
        state.current_backend = MagicMock()

        orchestrator = state.init_research(searxng_endpoint="http://custom:9999")

        assert orchestrator is not None
        assert state.search_client.endpoint == "http://custom:9999"

    def test_init_research_returns_existing(self, tmp_path: Path):
        """Test init_research returns existing orchestrator."""
        state = R3LayState(project_path=tmp_path)
        state.current_backend = MagicMock()

        orch1 = state.init_research()
        orch2 = state.init_research()

        assert orch1 is orch2

    @pytest.mark.asyncio
    async def test_close_research(self, tmp_path: Path):
        """Test closing research orchestrator."""
        state = R3LayState(project_path=tmp_path)
        state.current_backend = MagicMock()
        state.init_research()

        await state.close_research()

        assert state.search_client is None
        assert state.research_orchestrator is None
