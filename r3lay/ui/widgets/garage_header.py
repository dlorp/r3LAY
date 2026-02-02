"""Custom garage-style header for r3LAY.

Shows project name, mileage, active model, and time in retrofuturistic style.

Layout:
╔═══════════════════════════════════════════════════════════════════════════╗
║ r3LAY ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ llama3.2 ░░░░░░░░░░░░░░░░ 14:32 ║
║ 97-IMPREZA ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 102,847 mi ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from textual.reactive import reactive
from textual.widgets import Static

if TYPE_CHECKING:
    from textual.app import ComposeResult

    from ...core import R3LayState


class GarageHeader(Static):
    """Two-line header showing project and system status.

    Attributes:
        project_name: Vehicle/project display name
        mileage: Current odometer reading
        model_name: Active LLM model name
    """

    DEFAULT_CSS = """
    GarageHeader {
        height: 4;
        width: 100%;
        background: #0d0d0d;
        border-bottom: solid #636764;
        padding: 0 1;
    }

    GarageHeader .header-line {
        width: 100%;
        height: 1;
    }

    GarageHeader .header-line-1 {
        color: #F4E409;
    }

    GarageHeader .header-line-2 {
        color: #50D8D7;
    }
    """

    project_name: reactive[str] = reactive("No Project")
    mileage: reactive[int] = reactive(0)
    model_name: reactive[str] = reactive("no model")

    def __init__(
        self,
        state: "R3LayState | None" = None,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        """Initialize the garage header.

        Args:
            state: R3LayState instance for project/model info
            name: Widget name
            id: Widget ID
            classes: CSS classes
        """
        super().__init__(name=name, id=id, classes=classes)
        self._state = state

    def compose(self) -> "ComposeResult":
        """Compose the header lines."""
        yield Static(id="line1", classes="header-line header-line-1")
        yield Static(id="line2", classes="header-line header-line-2")

    def on_mount(self) -> None:
        """Initialize header content on mount."""
        self._load_project_state()
        self._update_display()
        # Update time every minute
        self.set_interval(60, self._update_display)

    def _load_project_state(self) -> None:
        """Load project state from disk if available."""
        if self._state is None:
            return

        try:
            from ...core.project import ProjectManager

            pm = ProjectManager(self._state.project_path)
            project_state = pm.load()

            if project_state:
                self.project_name = project_state.profile.short_name
                self.mileage = project_state.current_mileage
        except Exception:
            # Silently fail - project may not exist
            pass

        # Get current model
        if self._state.current_model:
            self.model_name = self._state.current_model

    def _update_display(self) -> None:
        """Update the header display with current values."""
        width = self.size.width or 80
        now = datetime.now()
        time_str = now.strftime("%H:%M")

        # Line 1: r3LAY ░░░░░░ model ░░░░░░ time
        title = "r³LAY"
        model = self.model_name[:20]  # Truncate long model names

        # Calculate spacers
        line1_content = f"{title}  {model}  {time_str}"
        spacer_len = max(0, width - len(line1_content) - 4)
        spacer1_len = spacer_len // 2
        spacer2_len = spacer_len - spacer_len // 2

        spacer1 = "░" * spacer1_len
        spacer2 = "░" * spacer2_len

        line1 = f" {title} {spacer1} {model} {spacer2} {time_str} "

        # Line 2: project ░░░░░░░░░░░░░░░░░░░░░░░░░░░░ mileage mi
        project = self.project_name[:30]
        mileage_str = f"{self.mileage:,} mi"

        line2_content = f"{project}  {mileage_str}"
        spacer3_len = max(0, width - len(line2_content) - 4)
        spacer3 = "░" * spacer3_len

        line2 = f" {project} {spacer3} {mileage_str} "

        # Update static widgets
        try:
            self.query_one("#line1", Static).update(line1)
            self.query_one("#line2", Static).update(line2)
        except Exception:
            pass

    def watch_project_name(self, value: str) -> None:
        """React to project name changes."""
        self._update_display()

    def watch_mileage(self, value: int) -> None:
        """React to mileage changes."""
        self._update_display()

    def watch_model_name(self, value: str) -> None:
        """React to model name changes."""
        self._update_display()

    def update_model(self, model_name: str) -> None:
        """Update the displayed model name.

        Args:
            model_name: New model name to display
        """
        self.model_name = model_name

    def update_mileage(self, mileage: int) -> None:
        """Update the displayed mileage.

        Args:
            mileage: New mileage value
        """
        self.mileage = mileage

    def update_project(self, project_name: str, mileage: int | None = None) -> None:
        """Update project information.

        Args:
            project_name: New project name
            mileage: Optional new mileage value
        """
        self.project_name = project_name
        if mileage is not None:
            self.mileage = mileage
