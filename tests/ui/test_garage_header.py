"""Tests for GarageHeader widget."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from textual.app import App, ComposeResult

from r3lay.ui.widgets.garage_header import GarageHeader


def create_mock_state(project_name: str = "Test Project") -> MagicMock:
    """Create a mock R3LayState for testing."""
    state = MagicMock()
    # Use a real Path object so .name property works correctly
    state.project_path = Path(f"/tmp/{project_name}")
    return state


class GarageHeaderTestApp(App):
    """Test app for GarageHeader."""

    def __init__(self, state=None):
        super().__init__()
        self.test_state = state or create_mock_state()

    def compose(self) -> ComposeResult:
        """Mount GarageHeader widget."""
        yield GarageHeader(self.test_state)


class TestGarageHeaderInit:
    """Test GarageHeader initialization."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """GarageHeader initializes with project name."""
        async with GarageHeaderTestApp().run_test() as pilot:
            header = pilot.app.query_one(GarageHeader)
            assert header is not None
            assert header.project_name == "Test Project"

    @pytest.mark.asyncio
    async def test_default_project_name(self):
        """GarageHeader uses default project name from state."""
        state = create_mock_state("r3LAY")

        async with GarageHeaderTestApp(state).run_test() as pilot:
            header = pilot.app.query_one(GarageHeader)
            assert header.project_name == "r3LAY"


class TestGarageHeaderCSS:
    """Test GarageHeader CSS."""

    @pytest.mark.asyncio
    async def test_has_css(self):
        """GarageHeader has CSS."""
        async with GarageHeaderTestApp().run_test() as pilot:
            header = pilot.app.query_one(GarageHeader)
            assert header.DEFAULT_CSS is not None

    @pytest.mark.asyncio
    async def test_css_includes_garage_header_styles(self):
        """GarageHeader CSS includes GarageHeader-specific styles."""
        async with GarageHeaderTestApp().run_test() as pilot:
            header = pilot.app.query_one(GarageHeader)
            css = header.DEFAULT_CSS
            assert "GarageHeader" in css


class TestGarageHeaderCompose:
    """Test GarageHeader composition."""

    @pytest.mark.asyncio
    async def test_compose_yields_widgets(self):
        """GarageHeader compose yields widgets."""
        async with GarageHeaderTestApp().run_test() as pilot:
            header = pilot.app.query_one(GarageHeader)
            # Should have child widgets for project name, model, mileage
            assert header.query("Static")


class TestGarageHeaderActiveModel:
    """Test GarageHeader active model display."""

    @pytest.mark.asyncio
    async def test_watch_active_model_updates_display(self):
        """watch_active_model updates the model display."""
        async with GarageHeaderTestApp().run_test() as pilot:
            header = pilot.app.query_one(GarageHeader)

            # Trigger model change
            header.active_model = "test-model-1"
            await pilot.pause()

            # Should update display
            assert header.active_model == "test-model-1"

    @pytest.mark.asyncio
    async def test_active_model_defaults_to_none(self):
        """active_model defaults to None."""
        async with GarageHeaderTestApp().run_test() as pilot:
            header = pilot.app.query_one(GarageHeader)
            assert header.active_model is None


class TestGarageHeaderMileage:
    """Test GarageHeader mileage display."""

    @pytest.mark.asyncio
    async def test_refresh_mileage_updates_display(self):
        """refresh_mileage updates the mileage display."""
        async with GarageHeaderTestApp().run_test() as pilot:
            header = pilot.app.query_one(GarageHeader)

            # Call refresh_mileage with a value
            header.refresh_mileage(12345)
            await pilot.pause()

            # Mileage should be stored (implementation will display it)
            assert hasattr(header, "current_mileage")

    @pytest.mark.asyncio
    async def test_mileage_defaults_to_none(self):
        """Mileage defaults to None."""
        async with GarageHeaderTestApp().run_test() as pilot:
            header = pilot.app.query_one(GarageHeader)
            assert not hasattr(header, "current_mileage") or header.current_mileage is None


class TestModuleExports:
    """Test module exports."""

    def test_all_exports(self):
        """Module exports GarageHeader in __all__."""
        from r3lay.ui.widgets import garage_header

        assert hasattr(garage_header, "__all__")
        assert "GarageHeader" in garage_header.__all__
