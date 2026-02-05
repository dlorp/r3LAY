"""Tests for r3lay.ui.widgets.index_panel module.

Tests cover:
- IndexPanel: initialization, CSS styling
- Stats refresh for various index states
- Button interactions (reindex, clear)
- Progress and button state updates
- Reindexing workflow with mocked dependencies
- Vision embedder configuration
- Welcome message refresh
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from textual.containers import Vertical
from textual.widgets import Button, Static

from r3lay.ui.widgets.index_panel import IndexPanel

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_state(tmp_path: Path) -> MagicMock:
    """Create a mock R3LayState with a temp project path."""
    state = MagicMock()
    state.project_path = tmp_path
    state.index = None
    state.vision_embedder = None
    return state


@pytest.fixture
def mock_state_with_index(mock_state: MagicMock) -> MagicMock:
    """Create a mock R3LayState with an initialized index."""
    mock_index = MagicMock()
    mock_index.get_stats.return_value = {
        "count": 100,
        "image_count": 10,
        "collection": "test_collection",
        "hybrid_enabled": True,
        "vectors_count": 100,
        "image_vectors_count": 10,
        "embedding_dim": 768,
        "image_embedding_dim": 512,
    }
    mock_index.clear = MagicMock()
    mock_state.index = mock_index
    return mock_state


@pytest.fixture
def mock_state_with_basic_index(mock_state: MagicMock) -> MagicMock:
    """Create a mock R3LayState with a basic index (no vectors)."""
    mock_index = MagicMock()
    mock_index.get_stats.return_value = {
        "count": 50,
        "image_count": 0,
        "collection": "basic_collection",
        "hybrid_enabled": False,
        "vectors_count": 0,
        "image_vectors_count": 0,
        "embedding_dim": 0,
        "image_embedding_dim": 0,
    }
    mock_index.clear = MagicMock()
    mock_state.index = mock_index
    return mock_state


# =============================================================================
# IndexPanel Initialization Tests
# =============================================================================


class TestIndexPanelInit:
    """Tests for IndexPanel initialization."""

    def test_creation(self, mock_state: MagicMock) -> None:
        """Test creating an IndexPanel."""
        panel = IndexPanel(mock_state)

        assert panel.state is mock_state
        assert panel._is_indexing is False
        assert panel._embedder_loaded is False

    def test_inherits_from_vertical(self, mock_state: MagicMock) -> None:
        """Test IndexPanel inherits from Vertical."""
        panel = IndexPanel(mock_state)
        assert isinstance(panel, Vertical)

    def test_initial_state_flags(self, mock_state: MagicMock) -> None:
        """Test initial state flags are set correctly."""
        panel = IndexPanel(mock_state)

        assert not panel._is_indexing
        assert not panel._embedder_loaded


# =============================================================================
# IndexPanel CSS Tests
# =============================================================================


class TestIndexPanelCSS:
    """Tests for IndexPanel CSS styling."""

    def test_has_default_css(self) -> None:
        """Test that IndexPanel has DEFAULT_CSS defined."""
        assert IndexPanel.DEFAULT_CSS is not None
        assert "IndexPanel" in IndexPanel.DEFAULT_CSS

    def test_css_includes_index_header(self) -> None:
        """Test CSS includes index-header ID."""
        assert "#index-header" in IndexPanel.DEFAULT_CSS

    def test_css_includes_index_stats(self) -> None:
        """Test CSS includes index-stats ID."""
        assert "#index-stats" in IndexPanel.DEFAULT_CSS

    def test_css_includes_index_progress(self) -> None:
        """Test CSS includes index-progress ID."""
        assert "#index-progress" in IndexPanel.DEFAULT_CSS

    def test_css_includes_button_row(self) -> None:
        """Test CSS includes button-row ID."""
        assert "#button-row" in IndexPanel.DEFAULT_CSS

    def test_css_includes_reindex_button(self) -> None:
        """Test CSS includes reindex-button ID."""
        assert "#reindex-button" in IndexPanel.DEFAULT_CSS

    def test_css_includes_clear_button(self) -> None:
        """Test CSS includes clear-button ID."""
        assert "#clear-button" in IndexPanel.DEFAULT_CSS


# =============================================================================
# RefreshStats Tests
# =============================================================================


class TestRefreshStats:
    """Tests for _refresh_stats method."""

    def test_refresh_stats_no_index_all_available(self, mock_state: MagicMock) -> None:
        """Test stats display when index is None but features available."""
        panel = IndexPanel(mock_state)

        mock_stats = MagicMock(spec=Static)
        panel.query_one = MagicMock(return_value=mock_stats)

        # Patch the imports at the point where _refresh_stats imports them
        with patch.dict(
            "sys.modules",
            {
                "r3lay.core": MagicMock(
                    embeddings_available=MagicMock(return_value=True),
                    vision_embeddings_available=MagicMock(return_value=True),
                    pdf_extraction_available=MagicMock(return_value=True),
                )
            },
        ):
            # Need to reimport to get the patched version

            # Call the method directly
            panel._refresh_stats()

        mock_stats.update.assert_called_once()
        call_text = mock_stats.update.call_args[0][0]
        assert "Index not initialized" in call_text

    def test_refresh_stats_with_index_hybrid_enabled(
        self, mock_state_with_index: MagicMock
    ) -> None:
        """Test stats display with hybrid search enabled."""
        panel = IndexPanel(mock_state_with_index)

        mock_stats = MagicMock(spec=Static)
        panel.query_one = MagicMock(return_value=mock_stats)

        panel._refresh_stats()

        call_text = mock_stats.update.call_args[0][0]
        assert "Chunks: 100 (text) + 10 (images)" in call_text
        assert "Collection: test_collection" in call_text
        assert "Hybrid: Enabled" in call_text
        assert "100 text (dim=768)" in call_text
        assert "10 image (dim=512)" in call_text

    def test_refresh_stats_text_only(self, mock_state_with_basic_index: MagicMock) -> None:
        """Test stats display with text only (no images)."""
        panel = IndexPanel(mock_state_with_basic_index)

        mock_stats = MagicMock(spec=Static)
        panel.query_one = MagicMock(return_value=mock_stats)

        panel._refresh_stats()

        call_text = mock_stats.update.call_args[0][0]
        assert "Chunks: 50" in call_text
        assert "(images)" not in call_text

    def test_refresh_stats_vectors_ready_no_embedder(
        self, mock_state_with_index: MagicMock
    ) -> None:
        """Test stats display when vectors exist but hybrid not enabled."""
        mock_state_with_index.index.get_stats.return_value = {
            "count": 100,
            "image_count": 0,
            "collection": "test_collection",
            "hybrid_enabled": False,
            "vectors_count": 100,
            "image_vectors_count": 0,
            "embedding_dim": 768,
            "image_embedding_dim": 0,
        }
        panel = IndexPanel(mock_state_with_index)

        mock_stats = MagicMock(spec=Static)
        panel.query_one = MagicMock(return_value=mock_stats)

        panel._refresh_stats()

        call_text = mock_stats.update.call_args[0][0]
        assert "Vectors ready (embedder not loaded)" in call_text

    def test_refresh_stats_handles_exception(self, mock_state_with_index: MagicMock) -> None:
        """Test stats display handles exceptions gracefully."""
        mock_state_with_index.index.get_stats.side_effect = RuntimeError("Test error")
        panel = IndexPanel(mock_state_with_index)

        mock_stats = MagicMock(spec=Static)
        panel.query_one = MagicMock(return_value=mock_stats)

        panel._refresh_stats()

        call_text = mock_stats.update.call_args[0][0]
        assert "Error: Test error" in call_text

    def test_refresh_stats_image_only_vectors(self, mock_state_with_index: MagicMock) -> None:
        """Test stats display with only image vectors."""
        mock_state_with_index.index.get_stats.return_value = {
            "count": 0,
            "image_count": 20,
            "collection": "image_collection",
            "hybrid_enabled": True,
            "vectors_count": 0,
            "image_vectors_count": 20,
            "embedding_dim": 0,
            "image_embedding_dim": 512,
        }
        panel = IndexPanel(mock_state_with_index)

        mock_stats = MagicMock(spec=Static)
        panel.query_one = MagicMock(return_value=mock_stats)

        panel._refresh_stats()

        call_text = mock_stats.update.call_args[0][0]
        assert "20 image (dim=512)" in call_text


# =============================================================================
# Button Press Tests
# =============================================================================


class TestOnButtonPressed:
    """Tests for on_button_pressed handler."""

    @pytest.mark.asyncio
    async def test_clear_button_clears_index(self, mock_state_with_index: MagicMock) -> None:
        """Test clear button clears the index."""
        panel = IndexPanel(mock_state_with_index)

        # Mock the methods needed
        panel._refresh_stats = MagicMock()

        # Mock app.notify using patch.object on the property
        mock_app = MagicMock()
        with patch.object(type(panel), "app", new_callable=lambda: property(lambda s: mock_app)):
            # Create mock button event
            event = MagicMock()
            event.button.id = "clear-button"

            await panel.on_button_pressed(event)

        mock_state_with_index.index.clear.assert_called_once()
        panel._refresh_stats.assert_called_once()
        mock_app.notify.assert_called_once_with("Index cleared")

    @pytest.mark.asyncio
    async def test_clear_button_no_index(self, mock_state: MagicMock) -> None:
        """Test clear button does nothing when no index."""
        panel = IndexPanel(mock_state)

        panel._refresh_stats = MagicMock()

        event = MagicMock()
        event.button.id = "clear-button"

        await panel.on_button_pressed(event)

        # Should not crash, refresh_stats should not be called
        panel._refresh_stats.assert_not_called()

    @pytest.mark.asyncio
    async def test_reindex_button_calls_reindex(self, mock_state: MagicMock) -> None:
        """Test reindex button calls _do_reindex_sync."""
        panel = IndexPanel(mock_state)
        panel._do_reindex_sync = AsyncMock()

        event = MagicMock()
        event.button.id = "reindex-button"

        await panel.on_button_pressed(event)

        panel._do_reindex_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_reindex_button_skipped_when_indexing(self, mock_state: MagicMock) -> None:
        """Test reindex button is skipped when already indexing."""
        panel = IndexPanel(mock_state)
        panel._is_indexing = True
        panel._do_reindex_sync = AsyncMock()

        event = MagicMock()
        event.button.id = "reindex-button"

        await panel.on_button_pressed(event)

        panel._do_reindex_sync.assert_not_called()

    @pytest.mark.asyncio
    async def test_unknown_button_id(self, mock_state: MagicMock) -> None:
        """Test unknown button ID is handled gracefully."""
        panel = IndexPanel(mock_state)
        panel._do_reindex_sync = AsyncMock()

        event = MagicMock()
        event.button.id = "unknown-button"

        # Should not raise
        await panel.on_button_pressed(event)

        panel._do_reindex_sync.assert_not_called()


# =============================================================================
# Update Progress Tests
# =============================================================================


class TestUpdateProgress:
    """Tests for _update_progress method."""

    def test_update_progress(self, mock_state: MagicMock) -> None:
        """Test progress text is updated."""
        panel = IndexPanel(mock_state)

        mock_progress = MagicMock(spec=Static)
        panel.query_one = MagicMock(return_value=mock_progress)

        panel._update_progress("Loading...")

        panel.query_one.assert_called_with("#index-progress", Static)
        mock_progress.update.assert_called_once_with("Loading...")

    def test_update_progress_empty(self, mock_state: MagicMock) -> None:
        """Test progress can be cleared."""
        panel = IndexPanel(mock_state)

        mock_progress = MagicMock(spec=Static)
        panel.query_one = MagicMock(return_value=mock_progress)

        panel._update_progress("")

        mock_progress.update.assert_called_once_with("")

    def test_update_progress_multiline(self, mock_state: MagicMock) -> None:
        """Test progress can have multiline text."""
        panel = IndexPanel(mock_state)

        mock_progress = MagicMock(spec=Static)
        panel.query_one = MagicMock(return_value=mock_progress)

        panel._update_progress("Line 1\nLine 2")

        mock_progress.update.assert_called_once_with("Line 1\nLine 2")


# =============================================================================
# Update Button Tests
# =============================================================================


class TestUpdateButton:
    """Tests for _update_button method."""

    def test_update_button_label_and_disabled(self, mock_state: MagicMock) -> None:
        """Test button label and disabled state are updated."""
        panel = IndexPanel(mock_state)

        mock_button = MagicMock(spec=Button)
        panel.query_one = MagicMock(return_value=mock_button)

        panel._update_button("Indexing...", True)

        panel.query_one.assert_called_with("#reindex-button", Button)
        assert mock_button.label == "Indexing..."
        assert mock_button.disabled is True

    def test_update_button_reset(self, mock_state: MagicMock) -> None:
        """Test button can be reset to default state."""
        panel = IndexPanel(mock_state)

        mock_button = MagicMock(spec=Button)
        panel.query_one = MagicMock(return_value=mock_button)

        panel._update_button("Reindex", False)

        assert mock_button.label == "Reindex"
        assert mock_button.disabled is False


# =============================================================================
# Vision Embedder Config Tests
# =============================================================================


class TestGetVisionEmbedderConfig:
    """Tests for _get_vision_embedder_config method."""

    def test_get_config_from_app(self, mock_state: MagicMock) -> None:
        """Test getting vision embedder config from app."""
        panel = IndexPanel(mock_state)

        mock_app = MagicMock()
        mock_app.config.model_roles.vision_embedder = "clip-model"

        with patch.object(type(panel), "app", new_callable=lambda: property(lambda s: mock_app)):
            result = panel._get_vision_embedder_config()

        assert result == "clip-model"

    def test_get_config_returns_none_on_exception(self, mock_state: MagicMock) -> None:
        """Test returns None when exception occurs accessing config."""
        panel = IndexPanel(mock_state)

        # Mock app that raises when config is accessed
        mock_app = MagicMock()
        type(mock_app).config = property(
            fget=lambda self: (_ for _ in ()).throw(RuntimeError("Test"))
        )

        with patch.object(type(panel), "app", new_callable=lambda: property(lambda s: mock_app)):
            result = panel._get_vision_embedder_config()

        # Should not raise, should return None
        assert result is None

    def test_get_config_no_model_roles(self, mock_state: MagicMock) -> None:
        """Test returns None when model_roles not available."""
        panel = IndexPanel(mock_state)

        mock_app = MagicMock(spec=["config"])
        mock_app.config = MagicMock(spec=[])  # No model_roles

        with patch.object(type(panel), "app", new_callable=lambda: property(lambda s: mock_app)):
            result = panel._get_vision_embedder_config()

        assert result is None


# =============================================================================
# Refresh Welcome Message Tests
# =============================================================================


class TestRefreshWelcomeMessage:
    """Tests for _refresh_welcome_message method."""

    def test_refresh_welcome_success(self, mock_state: MagicMock) -> None:
        """Test refreshing welcome message when ResponsePane exists."""
        panel = IndexPanel(mock_state)

        mock_response_pane = MagicMock()
        mock_response_pane.refresh_welcome = MagicMock()

        mock_screen = MagicMock()
        mock_screen.query_one.return_value = mock_response_pane

        with patch.object(
            type(panel), "screen", new_callable=lambda: property(lambda s: mock_screen)
        ):
            panel._refresh_welcome_message()

        mock_screen.query_one.assert_called_once_with("ResponsePane")
        mock_response_pane.refresh_welcome.assert_called_once()

    def test_refresh_welcome_no_response_pane(self, mock_state: MagicMock) -> None:
        """Test graceful handling when ResponsePane doesn't exist."""
        panel = IndexPanel(mock_state)

        mock_screen = MagicMock()
        mock_screen.query_one.side_effect = Exception("No match")

        with patch.object(
            type(panel), "screen", new_callable=lambda: property(lambda s: mock_screen)
        ):
            # Should not raise
            panel._refresh_welcome_message()

    def test_refresh_welcome_no_refresh_method(self, mock_state: MagicMock) -> None:
        """Test graceful handling when refresh_welcome method doesn't exist."""
        panel = IndexPanel(mock_state)

        mock_response_pane = MagicMock(spec=[])  # No refresh_welcome
        mock_screen = MagicMock()
        mock_screen.query_one.return_value = mock_response_pane

        with patch.object(
            type(panel), "screen", new_callable=lambda: property(lambda s: mock_screen)
        ):
            # Should not raise
            panel._refresh_welcome_message()


# =============================================================================
# Do Reindex Sync Tests
# =============================================================================


class TestDoReindexSync:
    """Tests for _do_reindex_sync method."""

    @pytest.mark.asyncio
    async def test_reindex_sets_and_clears_indexing_flag(self, mock_state: MagicMock) -> None:
        """Test reindexing sets and clears _is_indexing flag."""
        panel = IndexPanel(mock_state)

        # Mock all dependencies
        mock_index = MagicMock()
        mock_index.add_chunks.return_value = 0
        mock_state.init_index.return_value = mock_index

        panel._update_button = MagicMock()
        panel._update_progress = MagicMock()
        panel._refresh_stats = MagicMock()
        panel._refresh_welcome_message = MagicMock()
        panel.refresh = MagicMock()

        mock_app = MagicMock()
        mock_loader = MagicMock()
        mock_result = MagicMock()
        mock_result.chunks = []
        mock_result.image_paths = []
        mock_result.image_metadata = []
        mock_loader.load_directory_with_images.return_value = mock_result

        with (
            patch.object(type(panel), "app", new_callable=lambda: property(lambda s: mock_app)),
            patch("r3lay.core.embeddings_available", return_value=False),
            patch("r3lay.core.pdf_extraction_available", return_value=False),
            patch("r3lay.core.index.DocumentLoader", return_value=mock_loader),
        ):
            await panel._do_reindex_sync()

        # Flag should be cleared after completion
        assert panel._is_indexing is False

    @pytest.mark.asyncio
    async def test_reindex_initializes_index(self, mock_state: MagicMock) -> None:
        """Test reindexing initializes and clears the index."""
        panel = IndexPanel(mock_state)

        mock_index = MagicMock()
        mock_index.add_chunks.return_value = 5
        mock_state.init_index.return_value = mock_index

        panel._update_button = MagicMock()
        panel._update_progress = MagicMock()
        panel._refresh_stats = MagicMock()
        panel._refresh_welcome_message = MagicMock()
        panel.refresh = MagicMock()

        mock_app = MagicMock()
        mock_loader = MagicMock()
        mock_result = MagicMock()
        mock_result.chunks = ["chunk1", "chunk2", "chunk3", "chunk4", "chunk5"]
        mock_result.image_paths = []
        mock_result.image_metadata = []
        mock_loader.load_directory_with_images.return_value = mock_result

        with (
            patch.object(type(panel), "app", new_callable=lambda: property(lambda s: mock_app)),
            patch("r3lay.core.embeddings_available", return_value=False),
            patch("r3lay.core.pdf_extraction_available", return_value=False),
            patch("r3lay.core.index.DocumentLoader", return_value=mock_loader),
        ):
            await panel._do_reindex_sync()

        mock_state.init_index.assert_called_once()
        mock_index.clear.assert_called_once()
        mock_index.add_chunks.assert_called_once()

    @pytest.mark.asyncio
    async def test_reindex_with_embeddings(self, mock_state: MagicMock) -> None:
        """Test reindexing generates embeddings when available."""
        panel = IndexPanel(mock_state)

        mock_index = MagicMock()
        mock_index.add_chunks.return_value = 10
        mock_index.generate_embeddings = AsyncMock(return_value=10)
        mock_state.init_index.return_value = mock_index

        mock_embedder = MagicMock()
        mock_state.init_embedder = AsyncMock(return_value=mock_embedder)

        panel._update_button = MagicMock()
        panel._update_progress = MagicMock()
        panel._refresh_stats = MagicMock()
        panel._refresh_welcome_message = MagicMock()
        panel.refresh = MagicMock()

        mock_app = MagicMock()
        mock_loader = MagicMock()
        mock_result = MagicMock()
        mock_result.chunks = ["c"] * 10
        mock_result.image_paths = []
        mock_result.image_metadata = []
        mock_loader.load_directory_with_images.return_value = mock_result

        with (
            patch.object(type(panel), "app", new_callable=lambda: property(lambda s: mock_app)),
            patch("r3lay.core.embeddings_available", return_value=True),
            patch("r3lay.core.pdf_extraction_available", return_value=False),
            patch("r3lay.core.index.DocumentLoader", return_value=mock_loader),
        ):
            await panel._do_reindex_sync()

        mock_state.init_embedder.assert_called_once()
        mock_index.generate_embeddings.assert_called_once()
        assert panel._embedder_loaded is True

    @pytest.mark.asyncio
    async def test_reindex_with_images(self, mock_state: MagicMock) -> None:
        """Test reindexing handles images with vision embedder."""
        panel = IndexPanel(mock_state)

        mock_index = MagicMock()
        mock_index.add_chunks.return_value = 5
        mock_index.add_images = AsyncMock(return_value=3)
        mock_state.init_index.return_value = mock_index

        mock_vision_embedder = MagicMock()
        mock_vision_embedder.is_loaded = True
        mock_state.vision_embedder = mock_vision_embedder

        panel._update_button = MagicMock()
        panel._update_progress = MagicMock()
        panel._refresh_stats = MagicMock()
        panel._refresh_welcome_message = MagicMock()
        panel.refresh = MagicMock()

        mock_app = MagicMock()
        mock_loader = MagicMock()
        mock_result = MagicMock()
        mock_result.chunks = ["c"] * 5
        mock_result.image_paths = ["/path/to/img1.png", "/path/to/img2.png", "/path/to/img3.png"]
        mock_result.image_metadata = [{}, {}, {}]
        mock_loader.load_directory_with_images.return_value = mock_result

        with (
            patch.object(type(panel), "app", new_callable=lambda: property(lambda s: mock_app)),
            patch("r3lay.core.embeddings_available", return_value=False),
            patch("r3lay.core.pdf_extraction_available", return_value=False),
            patch("r3lay.core.index.DocumentLoader", return_value=mock_loader),
        ):
            await panel._do_reindex_sync()

        mock_index.add_images.assert_called_once()

    @pytest.mark.asyncio
    async def test_reindex_loads_vision_embedder(self, mock_state: MagicMock) -> None:
        """Test reindexing loads vision embedder when images found."""
        panel = IndexPanel(mock_state)

        mock_index = MagicMock()
        mock_index.add_chunks.return_value = 5
        mock_index.add_images = AsyncMock(return_value=2)
        mock_state.init_index.return_value = mock_index

        mock_vision_embedder = MagicMock()
        mock_vision_embedder.is_loaded = True
        mock_state.init_vision_embedder = AsyncMock(return_value=mock_vision_embedder)

        panel._update_button = MagicMock()
        panel._update_progress = MagicMock()
        panel._refresh_stats = MagicMock()
        panel._refresh_welcome_message = MagicMock()
        panel._get_vision_embedder_config = MagicMock(return_value="clip-model")
        panel.refresh = MagicMock()

        mock_app = MagicMock()
        mock_loader = MagicMock()
        mock_result = MagicMock()
        mock_result.chunks = ["c"] * 5
        mock_result.image_paths = ["/img1.png", "/img2.png"]
        mock_result.image_metadata = [{}, {}]
        mock_loader.load_directory_with_images.return_value = mock_result

        with (
            patch.object(type(panel), "app", new_callable=lambda: property(lambda s: mock_app)),
            patch("r3lay.core.embeddings_available", return_value=False),
            patch("r3lay.core.pdf_extraction_available", return_value=False),
            patch("r3lay.core.vision_embeddings_available", return_value=True),
            patch("r3lay.core.index.DocumentLoader", return_value=mock_loader),
        ):
            await panel._do_reindex_sync()

        mock_state.init_vision_embedder.assert_called_once_with(model_name="clip-model")

    @pytest.mark.asyncio
    async def test_reindex_handles_embedder_failure(self, mock_state: MagicMock) -> None:
        """Test reindexing continues on embedder failure."""
        panel = IndexPanel(mock_state)

        mock_index = MagicMock()
        mock_index.add_chunks.return_value = 5
        mock_state.init_index.return_value = mock_index
        mock_state.init_embedder = AsyncMock(side_effect=RuntimeError("Embedder failed"))

        panel._update_button = MagicMock()
        panel._update_progress = MagicMock()
        panel._refresh_stats = MagicMock()
        panel._refresh_welcome_message = MagicMock()
        panel.refresh = MagicMock()

        mock_app = MagicMock()
        mock_loader = MagicMock()
        mock_result = MagicMock()
        mock_result.chunks = ["c"] * 5
        mock_result.image_paths = []
        mock_result.image_metadata = []
        mock_loader.load_directory_with_images.return_value = mock_result

        with (
            patch.object(type(panel), "app", new_callable=lambda: property(lambda s: mock_app)),
            patch("r3lay.core.embeddings_available", return_value=True),
            patch("r3lay.core.pdf_extraction_available", return_value=False),
            patch("r3lay.core.index.DocumentLoader", return_value=mock_loader),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            # Should not raise
            await panel._do_reindex_sync()

        # Should still complete
        assert panel._is_indexing is False
        mock_app.notify.assert_called()

    @pytest.mark.asyncio
    async def test_reindex_handles_general_exception(self, mock_state: MagicMock) -> None:
        """Test reindexing handles general exceptions."""
        panel = IndexPanel(mock_state)

        mock_state.init_index.side_effect = RuntimeError("Init failed")

        panel._update_button = MagicMock()
        panel._update_progress = MagicMock()
        panel._refresh_stats = MagicMock()
        panel._refresh_welcome_message = MagicMock()
        panel.refresh = MagicMock()

        mock_app = MagicMock()

        with (
            patch.object(type(panel), "app", new_callable=lambda: property(lambda s: mock_app)),
            patch("r3lay.core.embeddings_available", return_value=False),
            patch("r3lay.core.pdf_extraction_available", return_value=False),
        ):
            # Should not raise
            await panel._do_reindex_sync()

        # Error should be shown in progress
        panel._update_progress.assert_called()
        # Check that error was shown
        calls = [str(c) for c in panel._update_progress.call_args_list]
        assert any("Error" in str(c) for c in calls)

        # Should still clean up
        assert panel._is_indexing is False

    @pytest.mark.asyncio
    async def test_reindex_notifies_on_completion(self, mock_state: MagicMock) -> None:
        """Test reindexing sends notification on completion."""
        panel = IndexPanel(mock_state)

        mock_index = MagicMock()
        mock_index.add_chunks.return_value = 25
        mock_state.init_index.return_value = mock_index

        panel._update_button = MagicMock()
        panel._update_progress = MagicMock()
        panel._refresh_stats = MagicMock()
        panel._refresh_welcome_message = MagicMock()
        panel.refresh = MagicMock()

        mock_app = MagicMock()
        mock_loader = MagicMock()
        mock_result = MagicMock()
        mock_result.chunks = ["c"] * 25
        mock_result.image_paths = []
        mock_result.image_metadata = []
        mock_loader.load_directory_with_images.return_value = mock_result

        with (
            patch.object(type(panel), "app", new_callable=lambda: property(lambda s: mock_app)),
            patch("r3lay.core.embeddings_available", return_value=False),
            patch("r3lay.core.pdf_extraction_available", return_value=False),
            patch("r3lay.core.index.DocumentLoader", return_value=mock_loader),
        ):
            await panel._do_reindex_sync()

        mock_app.notify.assert_called_once_with("Indexed 25 chunks")

    @pytest.mark.asyncio
    async def test_reindex_notifies_hybrid_enabled(self, mock_state: MagicMock) -> None:
        """Test notification mentions hybrid search when enabled."""
        panel = IndexPanel(mock_state)

        mock_index = MagicMock()
        mock_index.add_chunks.return_value = 10
        mock_index.generate_embeddings = AsyncMock(return_value=10)
        mock_state.init_index.return_value = mock_index

        mock_embedder = MagicMock()
        mock_state.init_embedder = AsyncMock(return_value=mock_embedder)

        panel._update_button = MagicMock()
        panel._update_progress = MagicMock()
        panel._refresh_stats = MagicMock()
        panel._refresh_welcome_message = MagicMock()
        panel.refresh = MagicMock()

        mock_app = MagicMock()
        mock_loader = MagicMock()
        mock_result = MagicMock()
        mock_result.chunks = ["c"] * 10
        mock_result.image_paths = []
        mock_result.image_metadata = []
        mock_loader.load_directory_with_images.return_value = mock_result

        with (
            patch.object(type(panel), "app", new_callable=lambda: property(lambda s: mock_app)),
            patch("r3lay.core.embeddings_available", return_value=True),
            patch("r3lay.core.pdf_extraction_available", return_value=False),
            patch("r3lay.core.index.DocumentLoader", return_value=mock_loader),
        ):
            await panel._do_reindex_sync()

        mock_app.notify.assert_called_once_with("Indexed 10 chunks (hybrid search enabled)")


# =============================================================================
# On Mount Tests
# =============================================================================


class TestOnMount:
    """Tests for on_mount handler."""

    def test_on_mount_calls_refresh_stats(self, mock_state: MagicMock) -> None:
        """Test that on_mount calls _refresh_stats."""
        panel = IndexPanel(mock_state)
        panel._refresh_stats = MagicMock()

        panel.on_mount()

        panel._refresh_stats.assert_called_once()


# =============================================================================
# Module Exports Tests
# =============================================================================


class TestModuleExports:
    """Tests for module exports."""

    def test_all_exports(self) -> None:
        """Test __all__ exports."""
        from r3lay.ui.widgets import index_panel

        assert "IndexPanel" in index_panel.__all__

    def test_index_panel_importable(self) -> None:
        """Test IndexPanel can be imported from widgets."""
        from r3lay.ui.widgets import IndexPanel

        assert IndexPanel is not None


__all__ = [
    "TestIndexPanelInit",
    "TestIndexPanelCSS",
    "TestRefreshStats",
    "TestOnButtonPressed",
    "TestUpdateProgress",
    "TestUpdateButton",
    "TestGetVisionEmbedderConfig",
    "TestRefreshWelcomeMessage",
    "TestDoReindexSync",
    "TestOnMount",
    "TestModuleExports",
]
