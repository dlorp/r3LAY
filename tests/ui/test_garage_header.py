"""Tests for r3lay.ui.widgets.garage_header module.

Tests cover:
- GarageHeader: initialization, compose, display updates
- Reactive attributes: project_name, mileage, model_name
- Update methods: update_model, update_mileage, update_project
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from textual.widgets import Static

from r3lay.ui.widgets.garage_header import GarageHeader

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_state(tmp_path: Path) -> MagicMock:
    """Create a mock R3LayState."""
    state = MagicMock()
    state.project_path = tmp_path
    state.current_model = "llama3.2"
    return state


# =============================================================================
# GarageHeader Initialization Tests
# =============================================================================


class TestGarageHeaderInit:
    """Tests for GarageHeader initialization."""

    def test_creation_without_state(self) -> None:
        """Test creating a GarageHeader without state."""
        header = GarageHeader()

        assert header._state is None
        assert header.project_name == "No Project"
        assert header.mileage == 0
        assert header.model_name == "no model"

    def test_creation_with_state(self, mock_state: MagicMock) -> None:
        """Test creating a GarageHeader with state."""
        header = GarageHeader(state=mock_state)

        assert header._state is mock_state

    def test_creation_with_name(self) -> None:
        """Test creating a GarageHeader with name."""
        header = GarageHeader(name="test-header")

        assert header.name == "test-header"

    def test_creation_with_id(self) -> None:
        """Test creating a GarageHeader with ID."""
        header = GarageHeader(id="my-header")

        assert header.id == "my-header"

    def test_creation_with_classes(self) -> None:
        """Test creating a GarageHeader with CSS classes."""
        header = GarageHeader(classes="custom-class")

        assert "custom-class" in header.classes

    def test_inherits_from_static(self) -> None:
        """Test GarageHeader inherits from Static."""
        header = GarageHeader()
        assert isinstance(header, Static)


class TestGarageHeaderCSS:
    """Tests for GarageHeader CSS styling."""

    def test_has_default_css(self) -> None:
        """Test that GarageHeader has DEFAULT_CSS defined."""
        assert GarageHeader.DEFAULT_CSS is not None
        assert "GarageHeader" in GarageHeader.DEFAULT_CSS

    def test_css_includes_header_line(self) -> None:
        """Test CSS includes header-line classes."""
        css = GarageHeader.DEFAULT_CSS
        assert ".header-line" in css

    def test_css_includes_line_colors(self) -> None:
        """Test CSS includes line color classes."""
        css = GarageHeader.DEFAULT_CSS
        assert ".header-line-1" in css
        assert ".header-line-2" in css


class TestGarageHeaderReactiveAttributes:
    """Tests for GarageHeader reactive attributes."""

    def test_project_name_reactive(self) -> None:
        """Test project_name is reactive."""
        header = GarageHeader()

        # Set via attribute
        header.project_name = "My Project"
        assert header.project_name == "My Project"

    def test_mileage_reactive(self) -> None:
        """Test mileage is reactive."""
        header = GarageHeader()

        header.mileage = 50000
        assert header.mileage == 50000

    def test_model_name_reactive(self) -> None:
        """Test model_name is reactive."""
        header = GarageHeader()

        header.model_name = "gpt-4"
        assert header.model_name == "gpt-4"

    def test_reactive_default_values(self) -> None:
        """Test reactive attributes have correct defaults."""
        header = GarageHeader()

        assert header.project_name == "No Project"
        assert header.mileage == 0
        assert header.model_name == "no model"


class TestGarageHeaderCompose:
    """Tests for GarageHeader compose method."""

    def test_compose_yields_two_lines(self) -> None:
        """Test that compose yields two Static widgets for lines."""
        header = GarageHeader()
        widgets = list(header.compose())

        assert len(widgets) == 2

    def test_compose_first_line(self) -> None:
        """Test first line widget."""
        header = GarageHeader()
        widgets = list(header.compose())

        line1 = widgets[0]
        assert isinstance(line1, Static)
        assert line1.id == "line1"
        assert "header-line" in line1.classes
        assert "header-line-1" in line1.classes

    def test_compose_second_line(self) -> None:
        """Test second line widget."""
        header = GarageHeader()
        widgets = list(header.compose())

        line2 = widgets[1]
        assert isinstance(line2, Static)
        assert line2.id == "line2"
        assert "header-line" in line2.classes
        assert "header-line-2" in line2.classes


class TestGarageHeaderUpdateMethods:
    """Tests for GarageHeader update methods."""

    def test_update_model(self) -> None:
        """Test update_model method."""
        header = GarageHeader()

        header.update_model("claude-3")

        assert header.model_name == "claude-3"

    def test_update_mileage(self) -> None:
        """Test update_mileage method."""
        header = GarageHeader()

        header.update_mileage(123456)

        assert header.mileage == 123456

    def test_update_project(self) -> None:
        """Test update_project method with name only."""
        header = GarageHeader()

        header.update_project("New Project")

        assert header.project_name == "New Project"

    def test_update_project_with_mileage(self) -> None:
        """Test update_project method with name and mileage."""
        header = GarageHeader()

        header.update_project("New Project", mileage=75000)

        assert header.project_name == "New Project"
        assert header.mileage == 75000

    def test_update_project_none_mileage(self) -> None:
        """Test update_project with None mileage doesn't change mileage."""
        header = GarageHeader()
        header.mileage = 50000

        header.update_project("New Project", mileage=None)

        assert header.project_name == "New Project"
        assert header.mileage == 50000


class TestGarageHeaderWatchers:
    """Tests for GarageHeader reactive watchers."""

    def test_watch_project_name_calls_update_display(self) -> None:
        """Test project_name watcher calls _update_display."""
        header = GarageHeader()
        header._update_display = MagicMock()

        header.watch_project_name("New Name")

        header._update_display.assert_called_once()

    def test_watch_mileage_calls_update_display(self) -> None:
        """Test mileage watcher calls _update_display."""
        header = GarageHeader()
        header._update_display = MagicMock()

        header.watch_mileage(100000)

        header._update_display.assert_called_once()

    def test_watch_model_name_calls_update_display(self) -> None:
        """Test model_name watcher calls _update_display."""
        header = GarageHeader()
        header._update_display = MagicMock()

        header.watch_model_name("new-model")

        header._update_display.assert_called_once()


class TestGarageHeaderUpdateDisplay:
    """Tests for GarageHeader _update_display method."""

    def test_update_display_handles_query_exception(self) -> None:
        """Test _update_display handles query exceptions gracefully."""
        header = GarageHeader()

        # Mock query_one to raise
        def raise_error(*args, **kwargs):
            raise RuntimeError("Not mounted")

        header.query_one = MagicMock(side_effect=raise_error)

        # Should not raise even without proper size
        header._update_display()

    def test_model_name_truncation_logic(self) -> None:
        """Test that model name truncation to 20 chars is in the code."""
        import inspect

        source = inspect.getsource(GarageHeader._update_display)
        # Check the truncation logic exists
        assert "[:20]" in source

    def test_mileage_comma_formatting_logic(self) -> None:
        """Test that mileage is formatted with commas."""
        import inspect

        source = inspect.getsource(GarageHeader._update_display)
        # Check the formatting logic exists
        assert ":," in source  # Format specifier for thousand separators

    def test_project_name_truncation_logic(self) -> None:
        """Test that project name truncation to 30 chars is in the code."""
        import inspect

        source = inspect.getsource(GarageHeader._update_display)
        # Check the truncation logic exists
        assert "[:30]" in source


class TestGarageHeaderLoadProjectState:
    """Tests for GarageHeader _load_project_state method."""

    def test_load_project_state_no_state(self) -> None:
        """Test _load_project_state with no state does nothing."""
        header = GarageHeader(state=None)

        # Should not raise
        header._load_project_state()

        # Values should remain at defaults
        assert header.project_name == "No Project"
        assert header.mileage == 0

    def test_load_project_state_with_state_sets_model(self, mock_state: MagicMock) -> None:
        """Test _load_project_state sets model from state."""
        mock_state.current_model = "mistral-7b"
        header = GarageHeader(state=mock_state)

        # The method tries to load project but handles exceptions
        # So we just verify it doesn't crash and sets model
        header._load_project_state()

        # Model should be set from state
        assert header.model_name == "mistral-7b"

    def test_load_project_state_handles_missing_project(
        self, mock_state: MagicMock, tmp_path: Path
    ) -> None:
        """Test _load_project_state handles missing project gracefully."""
        mock_state.project_path = tmp_path / "nonexistent"
        mock_state.current_model = "test-model"
        header = GarageHeader(state=mock_state)

        # Should not raise even if project doesn't exist
        header._load_project_state()

        # Model should still be set
        assert header.model_name == "test-model"


# =============================================================================
# Module Exports
# =============================================================================


class TestModuleExports:
    """Tests for module exports."""

    def test_garage_header_importable(self) -> None:
        """Test GarageHeader can be imported."""
        from r3lay.ui.widgets.garage_header import GarageHeader

        assert GarageHeader is not None
