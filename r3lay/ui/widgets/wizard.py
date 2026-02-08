"""Project Configuration Wizard - TUI wizard for initial project setup.

A multi-step wizard that guides users through project configuration:
1. Domain selection (Automotive, Electronics, Software, Home/DIY)
2. Domain-specific questions
3. Save to .r3lay/project.yaml

The wizard can be pushed on app startup for new projects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Center, Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Button, Input, Label, Select, Static, TextArea

if TYPE_CHECKING:
    from textual.app import App

logger = logging.getLogger(__name__)


# =============================================================================
# Domain Constants
# =============================================================================

DOMAINS = [
    ("automotive", "Automotive - Vehicles, maintenance, repairs"),
    ("electronics", "Electronics - Circuits, microcontrollers, IoT"),
    ("software", "Software - Code projects, documentation"),
    ("home", "Home/DIY - Home improvement, repairs, automation"),
]

# Import automotive makes from project_context
from ...core.project_context import (
    AUTOMOTIVE_MAKES,
    ELECTRONICS_BOARDS,
    SOFTWARE_LANGUAGES,
)

# Common automotive makes for dropdown (sorted, with "Other" option)
AUTOMOTIVE_MAKES_LIST = sorted(AUTOMOTIVE_MAKES.keys())

# Electronics platforms
ELECTRONICS_PLATFORMS = [
    "Arduino IDE",
    "PlatformIO",
    "ESP-IDF",
    "MicroPython",
    "CircuitPython",
    "Raspberry Pi OS",
    "Zephyr RTOS",
    "FreeRTOS",
    "Bare Metal",
    "Other",
]

# Software frameworks by language
SOFTWARE_FRAMEWORKS = {
    "python": ["Django", "Flask", "FastAPI", "Textual", "PyQt", "None/Other"],
    "javascript": ["React", "Vue", "Angular", "Node.js", "Express", "None/Other"],
    "typescript": ["React", "Vue", "Angular", "Node.js", "NestJS", "None/Other"],
    "rust": ["Actix", "Axum", "Rocket", "Tokio", "None/Other"],
    "go": ["Gin", "Echo", "Fiber", "Chi", "None/Other"],
    "java": ["Spring Boot", "Quarkus", "Micronaut", "None/Other"],
    "cpp": ["Qt", "Boost", "SFML", "None/Other"],
    "c": ["None/Other"],
}

# Home/DIY project types
HOME_PROJECT_TYPES = [
    "Plumbing",
    "Electrical",
    "HVAC",
    "Renovation",
    "Automation",
    "Landscaping",
    "Furniture",
    "Repair",
    "Other",
]

SKILL_LEVELS = [
    ("beginner", "Beginner - Learning the basics"),
    ("intermediate", "Intermediate - Some experience"),
    ("advanced", "Advanced - Experienced practitioner"),
]


# =============================================================================
# Project Configuration Manager
# =============================================================================

@dataclass
class ProjectConfig:
    """Project configuration data model.

    Stores all project metadata for personalized research.
    """

    domain: str = "general"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Common fields
    nickname: str | None = None
    goals: str | None = None
    current_issues: str | None = None

    # Automotive-specific
    make: str | None = None
    model: str | None = None
    year: int | None = None
    engine_code: str | None = None
    mileage: int | None = None

    # Electronics-specific
    device_type: str | None = None
    manufacturer: str | None = None
    device_model: str | None = None
    platform: str | None = None
    symptoms: str | None = None
    tools_available: str | None = None

    # Software-specific
    language: str | None = None
    framework: str | None = None
    version: str | None = None
    platform_target: str | None = None
    description: str | None = None

    # Home/DIY-specific
    project_type: str | None = None
    location: str | None = None
    skill_level: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        data: dict[str, Any] = {
            "domain": self.domain,
            "created_at": self.created_at,
        }

        # Add common fields if set
        if self.nickname:
            data["nickname"] = self.nickname
        if self.goals:
            data["goals"] = self.goals
        if self.current_issues:
            data["current_issues"] = self.current_issues

        # Add domain-specific fields
        if self.domain == "automotive":
            auto_data: dict[str, Any] = {}
            if self.make:
                auto_data["make"] = self.make
            if self.model:
                auto_data["model"] = self.model
            if self.year:
                auto_data["year"] = self.year
            if self.engine_code:
                auto_data["engine_code"] = self.engine_code
            if self.mileage:
                auto_data["mileage"] = self.mileage
            if auto_data:
                data["automotive"] = auto_data

        elif self.domain == "electronics":
            elec_data: dict[str, Any] = {}
            if self.device_type:
                elec_data["device_type"] = self.device_type
            if self.manufacturer:
                elec_data["manufacturer"] = self.manufacturer
            if self.device_model:
                elec_data["model"] = self.device_model
            if self.platform:
                elec_data["platform"] = self.platform
            if self.symptoms:
                elec_data["symptoms"] = self.symptoms
            if self.tools_available:
                elec_data["tools_available"] = self.tools_available
            if elec_data:
                data["electronics"] = elec_data

        elif self.domain == "software":
            soft_data: dict[str, Any] = {}
            if self.language:
                soft_data["language"] = self.language
            if self.framework:
                soft_data["framework"] = self.framework
            if self.version:
                soft_data["version"] = self.version
            if self.platform_target:
                soft_data["platform"] = self.platform_target
            if self.description:
                soft_data["description"] = self.description
            if soft_data:
                data["software"] = soft_data

        elif self.domain == "home":
            home_data: dict[str, Any] = {}
            if self.project_type:
                home_data["project_type"] = self.project_type
            if self.location:
                home_data["location"] = self.location
            if self.skill_level:
                home_data["skill_level"] = self.skill_level
            if home_data:
                data["home"] = home_data

        return data


class ProjectConfigManager:
    """Manages project configuration persistence.

    Saves and loads project.yaml from .r3lay/ directory.
    """

    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.config_dir = project_path / ".r3lay"
        self.config_file = self.config_dir / "project.yaml"

    def exists(self) -> bool:
        """Check if project configuration exists."""
        return self.config_file.exists()

    def save(self, config: ProjectConfig) -> None:
        """Save project configuration to YAML file.

        Uses atomic write (temp file + replace) for safety.
        """
        import tempfile
        from ruamel.yaml import YAML

        # Ensure directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)

        yaml = YAML()
        yaml.default_flow_style = False

        data = config.to_dict()

        # Atomic write
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=self.config_dir,
                suffix=".yaml",
                delete=False,
            ) as tmp:
                yaml.dump(data, tmp)
                tmp_path = Path(tmp.name)

            tmp_path.replace(self.config_file)
            logger.info(f"Saved project config to {self.config_file}")

        except Exception as e:
            logger.error(f"Failed to save project config: {e}")
            # Clean up temp file if it exists
            if 'tmp_path' in locals() and tmp_path.exists():
                tmp_path.unlink()
            raise

    def load(self) -> ProjectConfig | None:
        """Load project configuration from YAML file.

        Returns None if file doesn't exist or is invalid.
        """
        from ruamel.yaml import YAML

        if not self.config_file.exists():
            return None

        try:
            yaml = YAML()
            with self.config_file.open() as f:
                data = yaml.load(f)

            if not data:
                return None

            config = ProjectConfig(
                domain=data.get("domain", "general"),
                created_at=data.get("created_at", datetime.now().isoformat()),
                nickname=data.get("nickname"),
                goals=data.get("goals"),
                current_issues=data.get("current_issues"),
            )

            # Load domain-specific data
            if config.domain == "automotive" and "automotive" in data:
                auto = data["automotive"]
                config.make = auto.get("make")
                config.model = auto.get("model")
                config.year = auto.get("year")
                config.engine_code = auto.get("engine_code")
                config.mileage = auto.get("mileage")

            elif config.domain == "electronics" and "electronics" in data:
                elec = data["electronics"]
                config.device_type = elec.get("device_type")
                config.manufacturer = elec.get("manufacturer")
                config.device_model = elec.get("model")
                config.platform = elec.get("platform")
                config.symptoms = elec.get("symptoms")
                config.tools_available = elec.get("tools_available")

            elif config.domain == "software" and "software" in data:
                soft = data["software"]
                config.language = soft.get("language")
                config.framework = soft.get("framework")
                config.version = soft.get("version")
                config.platform_target = soft.get("platform")
                config.description = soft.get("description")

            elif config.domain == "home" and "home" in data:
                home = data["home"]
                config.project_type = home.get("project_type")
                config.location = home.get("location")
                config.skill_level = home.get("skill_level")

            return config

        except Exception as e:
            logger.error(f"Failed to load project config: {e}")
            return None


# =============================================================================
# Wizard Screen
# =============================================================================

class WizardComplete(Message):
    """Message posted when wizard completes successfully."""

    def __init__(self, config: ProjectConfig):
        self.config = config
        super().__init__()


class WizardSkipped(Message):
    """Message posted when wizard is skipped."""
    pass


class ProjectWizard(Screen):
    """Multi-step project configuration wizard.

    Steps:
    1. Domain selection
    2. Domain-specific questions
    3. Common questions (goals, issues)

    Keybindings:
    - Enter: Proceed to next step
    - Escape: Skip wizard (minimal config)
    """

    BINDINGS = [
        Binding("escape", "skip", "Skip"),
        Binding("enter", "next", "Next", show=False),
    ]

    DEFAULT_CSS = """
    ProjectWizard {
        align: center middle;
        background: $surface;
    }

    #wizard-container {
        width: 80;
        height: auto;
        max-height: 90%;
        padding: 1 2;
        border: solid $primary;
        background: $surface;
    }

    #wizard-title {
        text-align: center;
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }

    #wizard-subtitle {
        text-align: center;
        color: $text-muted;
        margin-bottom: 1;
    }

    .step-indicator {
        text-align: center;
        color: $text-muted;
        margin-bottom: 1;
    }

    .section-title {
        text-style: bold;
        color: $secondary;
        margin-top: 1;
        margin-bottom: 0;
    }

    .field-label {
        margin-top: 1;
        color: $text;
    }

    .field-hint {
        color: $text-muted;
        text-style: italic;
    }

    #domain-select {
        width: 100%;
        margin: 1 0;
    }

    .form-input {
        width: 100%;
        margin-bottom: 0;
    }

    .form-select {
        width: 100%;
        margin-bottom: 0;
    }

    .form-textarea {
        height: 4;
        width: 100%;
        margin-bottom: 0;
    }

    .year-input {
        width: 100%;
    }

    #error-label {
        color: $error;
        text-align: center;
        height: auto;
        min-height: 1;
        margin-top: 1;
    }

    #wizard-buttons {
        margin-top: 1;
        align: center middle;
        height: auto;
        min-height: 3;
    }

    #wizard-buttons Button {
        margin: 0 1;
    }

    #skip-button {
        background: $surface;
        color: $text-muted;
    }

    #next-button {
        background: $primary;
    }

    #back-button {
        background: $surface;
    }

    #finish-button {
        background: $success;
    }

    .hidden {
        display: none;
    }

    #step-content {
        height: 1fr;
        min-height: 10;
        max-height: 50;
        scrollbar-gutter: stable;
    }

    #wizard-footer {
        height: auto;
        dock: bottom;
        padding-top: 1;
    }
    """

    def __init__(
        self,
        project_path: Path,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ):
        super().__init__(name=name, id=id, classes=classes)
        self.project_path = project_path
        self.config_manager = ProjectConfigManager(project_path)
        self.config = ProjectConfig()
        self._step = 1  # 1=domain, 2=domain-specific, 3=common
        self._max_steps = 3

    def compose(self) -> ComposeResult:
        with Center():
            with Vertical(id="wizard-container"):
                yield Static("r3LAY Project Setup", id="wizard-title")
                yield Static(
                    "Configure your project for personalized research",
                    id="wizard-subtitle",
                )
                yield Static("Step 1 of 3", id="step-indicator", classes="step-indicator")

                with VerticalScroll(id="step-content"):
                    # Step 1: Domain selection
                    with Vertical(id="step-1"):
                        yield Label("What type of project is this?", classes="field-label")
                        yield Select(
                            [(label, value) for value, label in DOMAINS],
                            id="domain-select",
                            prompt="Select domain...",
                        )

                    # Step 2: Domain-specific (initially hidden)
                    with Vertical(id="step-2", classes="hidden"):
                        # Automotive fields
                        with Vertical(id="automotive-fields", classes="hidden"):
                            yield Static("Vehicle Information", classes="section-title")

                            yield Label("Make", classes="field-label")
                            yield Select(
                                [(m.title(), m) for m in AUTOMOTIVE_MAKES_LIST] + [("Other", "other")],
                                id="auto-make",
                                prompt="Select make...",
                                classes="form-select",
                            )

                            yield Label("Model", classes="field-label")
                            yield Input(
                                placeholder="e.g., Outback, Civic, F-150",
                                id="auto-model",
                                classes="form-input",
                            )

                            yield Label("Year", classes="field-label")
                            yield Input(
                                placeholder="e.g., 1997",
                                id="auto-year",
                                classes="year-input",
                            )

                            yield Label("Engine Code", classes="field-label")
                            yield Static(
                                "e.g., EJ25, K24, 5.0 Coyote",
                                classes="field-hint",
                            )
                            yield Input(
                                placeholder="Engine code (optional)",
                                id="auto-engine",
                                classes="form-input",
                            )

                            yield Label("Mileage", classes="field-label")
                            yield Input(
                                placeholder="Current mileage (optional)",
                                id="auto-mileage",
                                classes="form-input",
                            )

                            yield Label("Nickname", classes="field-label")
                            yield Static(
                                "A name for your vehicle (optional)",
                                classes="field-hint",
                            )
                            yield Input(
                                placeholder="e.g., Brighton, Blue Thunder",
                                id="auto-nickname",
                                classes="form-input",
                            )

                        # Electronics fields
                        with Vertical(id="electronics-fields", classes="hidden"):
                            yield Static("Device Information", classes="section-title")

                            yield Label("Device Type", classes="field-label")
                            yield Input(
                                placeholder="e.g., Weather Station, Robot Arm, LED Controller",
                                id="elec-device-type",
                                classes="form-input",
                            )

                            yield Label("Manufacturer / Model", classes="field-label")
                            yield Input(
                                placeholder="e.g., ESP32-WROOM-32, Arduino Nano",
                                id="elec-model",
                                classes="form-input",
                            )

                            yield Label("Platform", classes="field-label")
                            yield Select(
                                [(p, p.lower().replace(" ", "-")) for p in ELECTRONICS_PLATFORMS],
                                id="elec-platform",
                                prompt="Select platform...",
                                classes="form-select",
                            )

                            yield Label("Symptoms / Issues", classes="field-label")
                            yield TextArea(
                                id="elec-symptoms",
                                classes="form-textarea",
                            )

                            yield Label("Tools Available", classes="field-label")
                            yield Static(
                                "e.g., multimeter, oscilloscope, logic analyzer",
                                classes="field-hint",
                            )
                            yield Input(
                                placeholder="Available test equipment",
                                id="elec-tools",
                                classes="form-input",
                            )

                        # Software fields
                        with Vertical(id="software-fields", classes="hidden"):
                            yield Static("Project Information", classes="section-title")

                            yield Label("Primary Language", classes="field-label")
                            yield Select(
                                [(lang.title(), lang) for lang in sorted(SOFTWARE_LANGUAGES)],
                                id="soft-language",
                                prompt="Select language...",
                                classes="form-select",
                            )

                            yield Label("Framework", classes="field-label")
                            yield Select(
                                [],  # Populated dynamically based on language
                                id="soft-framework",
                                prompt="Select framework...",
                                classes="form-select",
                            )

                            yield Label("Version", classes="field-label")
                            yield Input(
                                placeholder="e.g., 3.11, 20.x, 1.0.0",
                                id="soft-version",
                                classes="form-input",
                            )

                            yield Label("Target Platform", classes="field-label")
                            yield Input(
                                placeholder="e.g., Linux, macOS, Web, Cross-platform",
                                id="soft-platform",
                                classes="form-input",
                            )

                            yield Label("Project Description", classes="field-label")
                            yield TextArea(
                                id="soft-description",
                                classes="form-textarea",
                            )

                        # Home/DIY fields
                        with Vertical(id="home-fields", classes="hidden"):
                            yield Static("Project Information", classes="section-title")

                            yield Label("Project Type", classes="field-label")
                            yield Select(
                                [(t, t.lower()) for t in HOME_PROJECT_TYPES],
                                id="home-type",
                                prompt="Select type...",
                                classes="form-select",
                            )

                            yield Label("Location", classes="field-label")
                            yield Input(
                                placeholder="e.g., Kitchen, Garage, Backyard",
                                id="home-location",
                                classes="form-input",
                            )

                            yield Label("Skill Level", classes="field-label")
                            yield Select(
                                [(label, value) for value, label in SKILL_LEVELS],
                                id="home-skill",
                                prompt="Select skill level...",
                                classes="form-select",
                            )

                    # Step 3: Common questions
                    with Vertical(id="step-3", classes="hidden"):
                        yield Static("Goals & Issues", classes="section-title")

                        yield Label("Current Issues", classes="field-label")
                        yield Static(
                            "What problems are you trying to solve?",
                            classes="field-hint",
                        )
                        yield TextArea(
                            id="common-issues",
                            classes="form-textarea",
                        )

                        yield Label("Goals", classes="field-label")
                        yield Static(
                            "What do you want to achieve with this project?",
                            classes="field-hint",
                        )
                        yield TextArea(
                            id="common-goals",
                            classes="form-textarea",
                        )

                with Vertical(id="wizard-footer"):
                    yield Label("", id="error-label")
                    with Horizontal(id="wizard-buttons"):
                        yield Button("Skip", id="skip-button", variant="default")
                        yield Button("Back", id="back-button", variant="default", classes="hidden")
                        yield Button("Next", id="next-button", variant="primary")
                        yield Button("Finish", id="finish-button", variant="success", classes="hidden")

    def on_mount(self) -> None:
        """Focus the domain select on mount."""
        self.query_one("#domain-select", Select).focus()

    def _update_step_display(self) -> None:
        """Update the UI to show the current step."""
        # Clear any error message when changing steps
        self._clear_error()

        # Update step indicator
        indicator = self.query_one("#step-indicator", Static)
        indicator.update(f"Step {self._step} of {self._max_steps}")

        # Show/hide step containers
        for i in range(1, self._max_steps + 1):
            step = self.query_one(f"#step-{i}", Vertical)
            if i == self._step:
                step.remove_class("hidden")
            else:
                step.add_class("hidden")

        # Update buttons
        back_btn = self.query_one("#back-button", Button)
        next_btn = self.query_one("#next-button", Button)
        finish_btn = self.query_one("#finish-button", Button)

        if self._step == 1:
            back_btn.add_class("hidden")
            next_btn.remove_class("hidden")
            finish_btn.add_class("hidden")
        elif self._step == self._max_steps:
            back_btn.remove_class("hidden")
            next_btn.add_class("hidden")
            finish_btn.remove_class("hidden")
        else:
            back_btn.remove_class("hidden")
            next_btn.remove_class("hidden")
            finish_btn.add_class("hidden")

    def _show_domain_fields(self, domain: str) -> None:
        """Show the appropriate domain-specific fields."""
        # Hide all domain fields first
        for domain_id in ["automotive", "electronics", "software", "home"]:
            fields = self.query_one(f"#{domain_id}-fields", Vertical)
            fields.add_class("hidden")

        # Show selected domain fields
        if domain in ["automotive", "electronics", "software", "home"]:
            fields = self.query_one(f"#{domain}-fields", Vertical)
            fields.remove_class("hidden")

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle select widget changes."""
        select_id = event.select.id
        value = event.value

        if select_id == "domain-select" and value != Select.BLANK:
            self.config.domain = str(value)
            self._show_domain_fields(str(value))

        elif select_id == "soft-language" and value != Select.BLANK:
            # Update framework options based on language
            self.config.language = str(value)
            framework_select = self.query_one("#soft-framework", Select)

            # Get frameworks for selected language
            lang_key = str(value).lower()
            if lang_key in SOFTWARE_FRAMEWORKS:
                frameworks = SOFTWARE_FRAMEWORKS[lang_key]
                framework_select.set_options(
                    [(f, f.lower().replace(" ", "-")) for f in frameworks]
                )
            else:
                framework_select.set_options([("None/Other", "other")])

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id

        if button_id == "skip-button":
            self.action_skip()
        elif button_id == "back-button":
            self._go_back()
        elif button_id == "next-button":
            self._go_next()
        elif button_id == "finish-button":
            self._finish()

    def action_skip(self) -> None:
        """Skip the wizard with minimal config."""
        # Create minimal config
        minimal_config = ProjectConfig(domain="general")

        try:
            self.config_manager.save(minimal_config)
        except Exception as e:
            logger.warning(f"Failed to save minimal config: {e}")

        self.post_message(WizardSkipped())
        self.dismiss()

    def action_next(self) -> None:
        """Handle Enter key - proceed to next step."""
        if self._step < self._max_steps:
            self._go_next()
        else:
            self._finish()

    def _go_back(self) -> None:
        """Go to previous step."""
        if self._step > 1:
            self._step -= 1
            self._update_step_display()

    def _clear_error(self) -> None:
        """Clear the error label."""
        error_label = self.query_one("#error-label", Label)
        error_label.update("")

    def _show_error(self, message: str) -> None:
        """Show an error message."""
        error_label = self.query_one("#error-label", Label)
        error_label.update(message)

    def _validate_step(self) -> str | None:
        """Validate the current step. Returns error message or None if valid."""
        if self._step == 1:
            domain_select = self.query_one("#domain-select", Select)
            if domain_select.value == Select.BLANK:
                return "Please select a domain"

        elif self._step == 2:
            domain = self.config.domain
            if domain == "automotive":
                make_select = self.query_one("#auto-make", Select)
                if make_select.value == Select.BLANK:
                    return "Please select a make"
            elif domain == "software":
                lang_select = self.query_one("#soft-language", Select)
                if lang_select.value == Select.BLANK:
                    return "Please select a language"

        return None

    def _go_next(self) -> None:
        """Go to next step after collecting current step data."""
        # Collect data from current step
        self._collect_step_data()

        # Validate current step
        error = self._validate_step()
        if error:
            self._show_error(error)
            return

        # Clear any previous error
        self._clear_error()

        if self._step < self._max_steps:
            self._step += 1
            self._update_step_display()

    def _collect_step_data(self) -> None:
        """Collect form data from the current step."""
        if self._step == 1:
            # Domain already captured in on_select_changed
            pass

        elif self._step == 2:
            # Collect domain-specific data
            domain = self.config.domain

            if domain == "automotive":
                make_select = self.query_one("#auto-make", Select)
                if make_select.value != Select.BLANK:
                    self.config.make = str(make_select.value)

                self.config.model = self.query_one("#auto-model", Input).value or None

                year_value = self.query_one("#auto-year", Input).value
                if year_value and year_value.isdigit():
                    self.config.year = int(year_value)

                self.config.engine_code = self.query_one("#auto-engine", Input).value or None

                mileage = self.query_one("#auto-mileage", Input).value
                if mileage and mileage.isdigit():
                    self.config.mileage = int(mileage)

                self.config.nickname = self.query_one("#auto-nickname", Input).value or None

            elif domain == "electronics":
                self.config.device_type = self.query_one("#elec-device-type", Input).value or None
                self.config.device_model = self.query_one("#elec-model", Input).value or None

                platform_select = self.query_one("#elec-platform", Select)
                if platform_select.value != Select.BLANK:
                    self.config.platform = str(platform_select.value)

                self.config.symptoms = self.query_one("#elec-symptoms", TextArea).text or None
                self.config.tools_available = self.query_one("#elec-tools", Input).value or None

            elif domain == "software":
                lang_select = self.query_one("#soft-language", Select)
                if lang_select.value != Select.BLANK:
                    self.config.language = str(lang_select.value)

                framework_select = self.query_one("#soft-framework", Select)
                if framework_select.value != Select.BLANK:
                    self.config.framework = str(framework_select.value)

                self.config.version = self.query_one("#soft-version", Input).value or None
                self.config.platform_target = self.query_one("#soft-platform", Input).value or None
                self.config.description = self.query_one("#soft-description", TextArea).text or None

            elif domain == "home":
                type_select = self.query_one("#home-type", Select)
                if type_select.value != Select.BLANK:
                    self.config.project_type = str(type_select.value)

                self.config.location = self.query_one("#home-location", Input).value or None

                skill_select = self.query_one("#home-skill", Select)
                if skill_select.value != Select.BLANK:
                    self.config.skill_level = str(skill_select.value)

        elif self._step == 3:
            # Collect common fields
            self.config.current_issues = self.query_one("#common-issues", TextArea).text or None
            self.config.goals = self.query_one("#common-goals", TextArea).text or None

    def _finish(self) -> None:
        """Finish the wizard and save configuration."""
        # Collect final step data
        self._collect_step_data()

        # Validate current step
        error = self._validate_step()
        if error:
            self._show_error(error)
            return

        # Clear any previous error
        self._clear_error()

        # Save configuration
        try:
            self.config_manager.save(self.config)
            self.notify("Project configuration saved", severity="information")
        except Exception as e:
            self.notify(f"Failed to save config: {e}", severity="error")
            logger.error(f"Failed to save project config: {e}")
            return

        # Post completion message and dismiss
        self.post_message(WizardComplete(self.config))
        self.dismiss()


__all__ = [
    "ProjectWizard",
    "ProjectConfig",
    "ProjectConfigManager",
    "WizardComplete",
    "WizardSkipped",
]
