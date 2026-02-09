"""GarageHeader widget for r3LAY."""

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static


class GarageHeader(Widget):
    """Custom header widget showing project, model, and mileage.

    Displays:
        - Project name
        - Currently active model
        - Current mileage (if available)

    Attributes:
        project_name: Name of the project
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

    active_model: reactive[str | None] = reactive(None)

    def __init__(
        self,
        project_name: str = "r3LAY",
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        """Initialize GarageHeader.

        Args:
            project_name: Name of the project (default: "r3LAY")
            name: Widget name
            id: Widget ID
            classes: Widget CSS classes
            disabled: Whether widget is disabled
        """
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        self.project_name = project_name
        self.current_mileage: int | None = None

    def compose(self) -> ComposeResult:
        """Compose the header layout.

        Yields:
            Static widgets for project name, model, and mileage
        """
        yield Static(self._get_header_text(), id="header-text", classes="project-name")
        yield Static(self._get_model_text(), id="model-text", classes="model-info")
        yield Static(self._get_mileage_text(), id="mileage-text", classes="mileage-info")

    def watch_active_model(self, new_model: str | None) -> None:
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
