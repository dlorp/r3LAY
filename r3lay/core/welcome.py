"""Dynamic welcome message based on project state and type detection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import R3LayState


@dataclass
class WelcomeProjectContext:
    """Detected project context from path and structure."""

    path: Path
    project_type: str  # automotive, electronics, software, workshop, home, general
    name: str
    metadata: dict  # type-specific: make, model, year, board, language, etc.


class ProjectDetector:
    """Detect project type and extract metadata from folder paths."""

    AUTOMOTIVE_KEYWORDS = {"garage", "automotive", "car", "vehicle", "cars"}
    ELECTRONICS_KEYWORDS = {"electronics", "iot", "embedded", "arduino", "esp32", "raspberry"}
    SOFTWARE_KEYWORDS = {"dev", "src", "code", "projects", "repos"}
    WORKSHOP_KEYWORDS = {"workshop", "woodworking", "cnc", "3dprint", "printing", "maker"}
    HOME_KEYWORDS = {"home", "hvac", "plumbing", "diy", "house", "renovation"}

    VEHICLE_MAKES = {
        "subaru", "toyota", "honda", "ford", "chevy", "chevrolet", "bmw",
        "mercedes", "audi", "vw", "volkswagen", "mazda", "nissan", "jeep",
        "dodge", "ram", "gmc", "hyundai", "kia", "tesla", "lexus", "acura"
    }

    BOARD_TYPES = {
        "esp32", "esp8266", "arduino", "raspberry", "rpi", "pico",
        "stm32", "teensy", "feather", "wemos", "nodemcu"
    }

    def detect(self, path: Path) -> WelcomeProjectContext:
        """Detect project type and extract metadata from path."""

        parts = [p.lower() for p in path.parts]

        # Check for project type indicators
        if self._matches_keywords(parts, self.AUTOMOTIVE_KEYWORDS) or self._has_vehicle_make(parts):
            return self._detect_automotive(path, parts)

        if self._matches_keywords(parts, self.ELECTRONICS_KEYWORDS) or self._has_board_type(parts):
            return self._detect_electronics(path, parts)

        has_sw_markers = self._has_software_markers(path)
        if self._matches_keywords(parts, self.SOFTWARE_KEYWORDS) or has_sw_markers:
            return self._detect_software(path, parts)

        if self._matches_keywords(parts, self.WORKSHOP_KEYWORDS):
            return self._detect_workshop(path, parts)

        if self._matches_keywords(parts, self.HOME_KEYWORDS):
            return self._detect_home(path, parts)

        # Default to general
        return WelcomeProjectContext(
            path=path,
            project_type="general",
            name=path.name,
            metadata={}
        )

    def _detect_automotive(self, path: Path, parts: list[str]) -> WelcomeProjectContext:
        """Extract vehicle metadata from path."""

        metadata = {}

        # Find make
        for part in parts:
            if part in self.VEHICLE_MAKES:
                metadata["make"] = part.title()
                break

        # Find year (4 digits, 1980-2030 range)
        year_pattern = re.compile(r'\b(19[89]\d|20[0-3]\d)\b')
        for part in parts:
            match = year_pattern.search(part)
            if match:
                metadata["year"] = match.group(1)
                break

        # Find model (segment after make, or last meaningful segment)
        model_candidates = []
        for part in parts:
            # Skip keywords and make
            if part in self.AUTOMOTIVE_KEYWORDS | self.VEHICLE_MAKES:
                continue
            # Skip year-only segments
            if re.match(r'^\d{4}$', part):
                continue
            # Clean up model name
            cleaned = re.sub(r'^\d{4}_?', '', part)  # Remove leading year
            cleaned = re.sub(r'_', ' ', cleaned)
            if cleaned:
                model_candidates.append(cleaned)

        if model_candidates:
            metadata["model"] = model_candidates[-1].title()

        # Project name
        name_parts = []
        if "year" in metadata:
            name_parts.append(metadata["year"])
        if "make" in metadata:
            name_parts.append(metadata["make"])
        if "model" in metadata:
            name_parts.append(metadata["model"])

        name = " ".join(name_parts) if name_parts else path.name

        return WelcomeProjectContext(
            path=path,
            project_type="automotive",
            name=name,
            metadata=metadata
        )

    def _detect_electronics(self, path: Path, parts: list[str]) -> WelcomeProjectContext:
        """Extract electronics project metadata."""

        metadata = {}

        # Find board type
        for part in parts:
            for board in self.BOARD_TYPES:
                if board in part:
                    metadata["board"] = board.upper()
                    break

        # Project name from last segment, cleaned
        name = path.name.replace("_", " ").replace("-", " ").title()

        return WelcomeProjectContext(
            path=path,
            project_type="electronics",
            name=name,
            metadata=metadata
        )

    def _detect_software(self, path: Path, parts: list[str]) -> WelcomeProjectContext:
        """Extract software project metadata."""

        metadata = {}

        # Detect language from project files
        if (path / "pyproject.toml").exists() or (path / "setup.py").exists():
            metadata["language"] = "Python"
        elif (path / "package.json").exists():
            metadata["language"] = "JavaScript/Node"
        elif (path / "Cargo.toml").exists():
            metadata["language"] = "Rust"
        elif (path / "go.mod").exists():
            metadata["language"] = "Go"
        elif (path / "pom.xml").exists() or (path / "build.gradle").exists():
            metadata["language"] = "Java"

        return WelcomeProjectContext(
            path=path,
            project_type="software",
            name=path.name,
            metadata=metadata
        )

    def _detect_workshop(self, path: Path, parts: list[str]) -> WelcomeProjectContext:
        """Extract workshop project metadata."""

        metadata = {}

        if "cnc" in parts:
            metadata["category"] = "CNC"
        elif "3dprint" in parts or "printing" in parts:
            metadata["category"] = "3D Printing"
        elif "woodworking" in parts:
            metadata["category"] = "Woodworking"

        return WelcomeProjectContext(
            path=path,
            project_type="workshop",
            name=path.name.replace("_", " ").title(),
            metadata=metadata
        )

    def _detect_home(self, path: Path, parts: list[str]) -> WelcomeProjectContext:
        """Extract home project metadata."""

        metadata = {}

        if "hvac" in parts:
            metadata["category"] = "HVAC"
        elif "plumbing" in parts:
            metadata["category"] = "Plumbing"
        elif "electrical" in parts:
            metadata["category"] = "Electrical"

        return WelcomeProjectContext(
            path=path,
            project_type="home",
            name=path.name.replace("_", " ").title(),
            metadata=metadata
        )

    def _matches_keywords(self, parts: list[str], keywords: set[str]) -> bool:
        return bool(set(parts) & keywords)

    def _has_vehicle_make(self, parts: list[str]) -> bool:
        return bool(set(parts) & self.VEHICLE_MAKES)

    def _has_board_type(self, parts: list[str]) -> bool:
        return any(board in part for part in parts for board in self.BOARD_TYPES)

    def _has_software_markers(self, path: Path) -> bool:
        markers = ["pyproject.toml", "package.json", "Cargo.toml", "go.mod", ".git"]
        return any((path / m).exists() for m in markers)


class WelcomeMessage:
    """Generate dynamic welcome message based on system state."""

    def __init__(
        self,
        project: WelcomeProjectContext | None,
        index_chunks: int,
        index_updated: datetime | None,
        models_status: dict,  # {"text": "qwen2.5-7b", "vision": None, "embed": "loaded"}
        registry: dict | None,  # Project registry data
    ):
        self.project = project
        self.index_chunks = index_chunks
        self.index_updated = index_updated
        self.models = models_status
        self.registry = registry or {}

    def render(self) -> str:
        """Render the welcome message.

        Uses double-space at end of lines for Markdown soft line breaks,
        ensuring each status item stays on its own line.
        """
        # Each line ends with two spaces for Markdown line break
        lines = ["**r3LAY** — Knowledge Relay System  "]

        # Core status (compact, with line breaks)
        for line in self._render_status():
            lines.append(line + "  ")  # Two spaces = soft line break

        # Project-specific info
        if self.project and self.registry:
            extra = self._render_project_specific()
            if extra:
                for line in extra:
                    lines.append(line + "  ")

        # Footer (no trailing spaces needed on last line)
        lines.append("`/help` — commands • `/status` — system info")

        return "\n".join(lines)

    def _render_status(self) -> list[str]:
        """Render core status lines (compact, no blank lines)."""

        lines = []

        # Project line with type badge highlighted using backticks
        if self.project:
            project_str = self.project.name
            badge = ""
            if self.project.metadata:
                ptype = self.project.project_type
                meta = self.project.metadata
                if ptype == "electronics" and "board" in meta:
                    badge = f" `{meta['board']}`"
                elif ptype == "software" and "language" in meta:
                    badge = f" `{meta['language']}`"
                elif self.project.project_type == "automotive":
                    badge = " `AUTO`"
                elif self.project.project_type == "workshop":
                    cat = self.project.metadata.get("category", "WORKSHOP")
                    badge = f" `{cat}`"
                elif self.project.project_type == "home":
                    cat = self.project.metadata.get("category", "HOME")
                    badge = f" `{cat}`"
            lines.append(f"`Project`   {project_str}{badge}")
        else:
            lines.append("`Project`   —")

        # Index line
        if self.index_chunks > 0:
            age = self._format_age(self.index_updated) if self.index_updated else ""
            index_str = f"{self.index_chunks:,} chunks{f' ({age})' if age else ''}"
        else:
            index_str = "not indexed"
        lines.append(f"`Index`     {index_str}")

        # Models line
        model_parts = []
        if self.models.get("text"):
            model_parts.append(f"text: {self.models['text']}")
        if self.models.get("vision"):
            model_parts.append(f"vision: {self.models['vision']}")
        if self.models.get("embed"):
            model_parts.append("embed: ready")

        if model_parts:
            models_str = " • ".join(model_parts)
            lines.append(f"`Models`    {models_str}")
        else:
            lines.append("`Models`    select from Models tab")

        return lines

    def _render_project_specific(self) -> list[str]:
        """Render project-type-specific information."""

        if not self.project:
            return []

        if self.project.project_type == "automotive":
            return self._render_automotive_info()
        elif self.project.project_type == "electronics":
            return self._render_electronics_info()
        elif self.project.project_type == "software":
            return self._render_software_info()
        elif self.project.project_type == "workshop":
            return self._render_workshop_info()
        elif self.project.project_type == "home":
            return self._render_home_info()

        return []

    def _render_automotive_info(self) -> list[str]:
        """Render automotive-specific status."""

        lines = []

        # Odometer
        if "odometer" in self.registry:
            lines.append(f"`Odometer`  {self.registry['odometer']:,} mi")

        # Alerts
        alerts = self._get_maintenance_alerts()
        if alerts:
            lines.append(f"`Alerts`    {alerts[0]}")
            for alert in alerts[1:3]:  # Max 3 alerts
                lines.append(f"            {alert}")

        return lines

    def _render_electronics_info(self) -> list[str]:
        """Render electronics-specific status."""

        lines = []

        if "last_flash" in self.registry:
            lines.append(f"`Flashed`   {self.registry['last_flash']}")

        if "firmware_version" in self.registry:
            lines.append(f"`Firmware`  {self.registry['firmware_version']}")

        return lines

    def _render_software_info(self) -> list[str]:
        """Render software-specific status."""

        lines = []

        if "version" in self.registry:
            lines.append(f"`Version`   {self.registry['version']}")

        if "last_test" in self.registry:
            status = self.registry.get("test_status", "unknown")
            lines.append(f"`Tests`     {status} ({self.registry['last_test']})")

        return lines

    def _render_workshop_info(self) -> list[str]:
        """Render workshop-specific status."""

        lines = []

        if "status" in self.registry:
            lines.append(f"`Status`    {self.registry['status']}")

        if "materials" in self.registry:
            lines.append(f"`Materials` {', '.join(self.registry['materials'][:3])}")

        return lines

    def _render_home_info(self) -> list[str]:
        """Render home project-specific status."""

        lines = []

        if "status" in self.registry:
            lines.append(f"`Status`    {self.registry['status']}")

        if "budget" in self.registry:
            spent = self.registry.get("spent", 0)
            lines.append(f"`Budget`    ${spent:,} / ${self.registry['budget']:,}")

        return lines

    def _get_maintenance_alerts(self) -> list[str]:
        """Calculate maintenance alerts from registry."""

        alerts = []
        odometer = self.registry.get("odometer", 0)

        # Oil change
        if "oil_change_interval" in self.registry and "last_oil_change_miles" in self.registry:
            next_oil = self.registry["last_oil_change_miles"] + self.registry["oil_change_interval"]
            remaining = next_oil - odometer
            if remaining <= 500:
                status = "overdue" if remaining < 0 else "due"
                alerts.append(f"Oil change {status} ({abs(remaining):,} mi)")
            elif remaining <= 1000:
                alerts.append(f"Oil change due in {remaining:,} mi")

        # Timing belt
        if "timing_belt_due_miles" in self.registry:
            remaining = self.registry["timing_belt_due_miles"] - odometer
            if remaining <= 5000:
                alerts.append(f"Timing belt due in {remaining:,} mi")

        # Custom alerts from registry
        if "alerts" in self.registry:
            alerts.extend(self.registry["alerts"])

        return alerts

    def _format_age(self, dt: datetime) -> str:
        """Format datetime as relative age string."""

        if not dt:
            return ""

        delta = datetime.now() - dt

        if delta.days > 30:
            return f"{delta.days // 30}mo ago"
        elif delta.days > 0:
            return f"{delta.days}d ago"
        elif delta.seconds > 3600:
            return f"{delta.seconds // 3600}h ago"
        elif delta.seconds > 60:
            return f"{delta.seconds // 60}m ago"
        else:
            return "just now"


def get_welcome_message(state: "R3LayState") -> str:
    """Generate welcome message from current state."""

    detector = ProjectDetector()

    project = detector.detect(state.project_path) if state.project_path else None

    # Get index stats
    index_chunks = 0
    index_updated = None
    if state.index is not None:
        stats = state.index.get_stats()
        index_chunks = stats.get("count", 0)
        # index_updated would come from index metadata if available

    # Get model status
    models_status = {
        "text": None,
        "vision": None,
        "embed": None,
    }

    if state.current_model:
        # Shorten model name for display
        model_short = state.current_model.split("/")[-1]
        if len(model_short) > 20:
            model_short = model_short[:17] + "..."
        models_status["text"] = model_short

    if state.text_embedder is not None and state.text_embedder.is_loaded:
        models_status["embed"] = "loaded"

    # Registry would come from state.registry if implemented
    registry = None

    welcome = WelcomeMessage(
        project=project,
        index_chunks=index_chunks,
        index_updated=index_updated,
        models_status=models_status,
        registry=registry,
    )

    return welcome.render()


__all__ = [
    "ProjectDetector",
    "WelcomeProjectContext",
    "WelcomeMessage",
    "get_welcome_message",
]
