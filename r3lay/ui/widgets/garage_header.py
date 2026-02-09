"""GarageHeader widget for r3LAY."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

if TYPE_CHECKING:
    from ...core import R3LayState


class GarageHeader(Widget):
    """Custom header widget showing project, model, and mileage.

    Displays:
        - Project name
        - Currently active model
        - Current mileage (if available)

    Attributes:
        state: Application state
        active_model: Currently selected model name
        current_mileage: Current vehicle mileage (optional)
    """

    DEFAULT_CSS = """
    GarageHeader {
        height: 3;
        background: $surface;
        border-bottom: solid $primary;
        padding: 0 2;
    }

    GarageHeader Static {
        height: 1;
        content-align: center middle;
    }

    GarageHeader .project-name {
        text-style: bold;
        color: $primary;
    }

    GarageHeader .model-info {
        color: $text;
    }

    GarageHeader .mileage-info {
        color: $accent;
    }
    """

    active_model: reactive[Optional[str]] = reactive(None)

    def __init__(
        self,
        state: "R3LayState",
        *,
        name: Optional[str] = None,
        id: Optional[str] = None,
        classes: Optional[str] = None,
        disabled: bool = False,
    ) -> None:
        """Initialize GarageHeader.

        Args:
            state: Application state
            name: Widget name
            id: Widget ID
            classes: Widget CSS classes
            disabled: Whether widget is disabled
        """
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        self.state = state
        self.current_mileage: Optional[int] = None

    @property
    def project_name(self) -> str:
        """Get project name from state."""
        return self.state.project_path.name if self.state.project_path else "r3LAY"

    def compose(self) -> ComposeResult:
        """Compose the header layout.

        Yields:
            Static widgets for project name, model, and mileage
        """
        yield Static(self._get_header_text(), id="header-text", classes="project-name")
        yield Static(self._get_model_text(), id="model-text", classes="model-info")
        yield Static(self._get_mileage_text(), id="mileage-text", classes="mileage-info")

    def watch_active_model(self, new_model: Optional[str]) -> None:
        """React to active model changes.

        Args:
            new_model: New active model name
        """
        # Only update if widget is mounted and composed
        if self.is_mounted:
            try:
                model_static = self.query_one("#model-text", Static)
                model_static.update(self._get_model_text())
            except Exception:
                # Widget not ready yet, will be set during compose
                pass

    def refresh_mileage(self, mileage: int) -> None:
        """Update mileage display.

        Args:
            mileage: New mileage value
        """
        self.current_mileage = mileage
        # Only update if widget is mounted
        if self.is_mounted:
            try:
                mileage_static = self.query_one("#mileage-text", Static)
                mileage_static.update(self._get_mileage_text())
            except Exception:
                # Widget not ready yet
                pass

    def _get_header_text(self) -> str:
        """Get project name header text."""
        return f"[[ {self.project_name} ]]"

    def _get_model_text(self) -> str:
        """Get active model display text."""
        if self.active_model:
            return f"Model: {self.active_model}"
        return "Model: None"

    def _get_mileage_text(self) -> str:
        """Get mileage display text."""
        if self.current_mileage is not None:
            return f"Mileage: {self.current_mileage:,} mi"
        return "Mileage: --"


__all__ = ["GarageHeader"]
