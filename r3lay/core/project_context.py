"""Project context extraction for personalized RAG responses.

r3LAY supports multiple project types for hobbyists and tinkerers:
- Automotive: vehicles, maintenance, repairs
- Electronics: circuits, microcontrollers, IoT
- Software: code projects, documentation
- Workshop: woodworking, metalworking, 3D printing
- Home: home automation, repairs, improvements
- Research: general research and documentation
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ProjectContext:
    """Extracted project context for personalized citations.

    Supports multiple project types with type-specific metadata.
    """

    raw_path: Path
    project_name: str
    project_type: str = "general"

    # Type-specific metadata (dict for flexibility)
    metadata: dict[str, str | None] = field(default_factory=dict)

    # Automotive-specific (kept for backward compatibility)
    vehicle_make: str | None = None
    vehicle_model: str | None = None
    vehicle_year: str | None = None
    vehicle_nickname: str | None = None

    @property
    def project_reference(self) -> str:
        """Get preferred project reference for citations.

        Adapts to project type:
        - Automotive: "your Outback", "the Brighton"
        - Electronics: "the Arduino project", "the ESP32 build"
        - Software: "this codebase", "the project"
        - General: "this project"
        """
        if self.project_type == "automotive":
            if self.vehicle_nickname:
                return f"the {self.vehicle_nickname}"
            if self.vehicle_model:
                return f"your {self.vehicle_model}"
            if self.vehicle_make:
                return f"your {self.vehicle_make}"
            return "your vehicle"
        elif self.project_type == "electronics":
            board = self.metadata.get("board")
            if board:
                return f"the {board} project"
            return "this electronics project"
        elif self.project_type == "software":
            return "this codebase"
        elif self.project_type == "workshop":
            return "this build"
        elif self.project_type == "home":
            return "this project"
        return "this project"

    # Backward compatibility alias
    @property
    def vehicle_reference(self) -> str:
        """Alias for project_reference (backward compatibility)."""
        return self.project_reference

    @property
    def possessive(self) -> str:
        """Get possessive form (e.g., "the Brighton's", "your Impreza's")."""
        ref = self.project_reference
        if ref.endswith("s"):
            return f"{ref}'"
        return f"{ref}'s"

    @property
    def context_summary(self) -> str:
        """Get a one-line summary of the project context."""
        if self.project_type == "automotive":
            parts = []
            if self.vehicle_year:
                parts.append(self.vehicle_year)
            if self.vehicle_make:
                parts.append(self.vehicle_make)
            if self.vehicle_model:
                parts.append(self.vehicle_model)
            if self.vehicle_nickname:
                parts.append(f'"{self.vehicle_nickname}"')
            return " ".join(parts) if parts else "Automotive project"
        elif self.project_type == "electronics":
            board = self.metadata.get("board", "")
            return f"Electronics: {board}" if board else "Electronics project"
        elif self.project_type == "software":
            lang = self.metadata.get("language", "")
            return f"Software: {lang}" if lang else "Software project"
        return f"{self.project_type.title()} project: {self.project_name}"


# =============================================================================
# Project Type Detection Keywords
# =============================================================================

# Automotive makes and models
AUTOMOTIVE_MAKES: dict[str, list[str]] = {
    "subaru": ["impreza", "outback", "forester", "wrx", "sti", "legacy", "crosstrek", "brz", "ascent", "baja"],
    "toyota": ["camry", "corolla", "rav4", "tacoma", "tundra", "4runner", "highlander", "prius", "supra", "86"],
    "honda": ["civic", "accord", "cr-v", "pilot", "odyssey", "fit", "hr-v", "s2000", "nsx"],
    "ford": ["f150", "f-150", "mustang", "explorer", "escape", "bronco", "ranger", "focus"],
    "chevrolet": ["silverado", "camaro", "corvette", "tahoe", "equinox", "malibu", "colorado"],
    "nissan": ["altima", "maxima", "sentra", "rogue", "pathfinder", "frontier", "370z", "gtr"],
    "mazda": ["mazda3", "mazda6", "cx-5", "cx-9", "mx-5", "miata", "rx-7", "rx-8"],
    "bmw": ["3-series", "5-series", "x3", "x5", "m3", "m5", "z4"],
    "volkswagen": ["golf", "jetta", "passat", "tiguan", "gti", "r32"],
}

# Electronics keywords and boards
ELECTRONICS_KEYWORDS: set[str] = {
    "electronics", "circuit", "pcb", "breadboard", "iot", "embedded",
    "microcontroller", "mcu", "sensor", "actuator", "gpio",
}
ELECTRONICS_BOARDS: set[str] = {
    "arduino", "esp32", "esp8266", "raspberry", "rpi", "pico", "stm32",
    "teensy", "attiny", "atmega", "nodemcu", "wemos", "feather",
}

# Software keywords
SOFTWARE_KEYWORDS: set[str] = {
    "software", "code", "src", "source", "dev", "development",
    "app", "application", "api", "backend", "frontend", "lib", "library",
}
SOFTWARE_LANGUAGES: set[str] = {
    "python", "javascript", "typescript", "rust", "go", "java", "cpp", "c++",
}

# Workshop keywords
WORKSHOP_KEYWORDS: set[str] = {
    "workshop", "woodworking", "metalwork", "welding", "cnc", "lathe",
    "3dprint", "3d-print", "printer", "laser", "mill", "forge",
    "furniture", "cabinet", "bench", "table", "shelf",
}

# Home/DIY keywords
HOME_KEYWORDS: set[str] = {
    "home", "house", "apartment", "hvac", "plumbing", "electrical",
    "renovation", "remodel", "repair", "diy", "improvement",
    "automation", "smart-home", "smarthome",
}


def extract_project_context(project_path: Path) -> ProjectContext:
    """Extract project context from path for personalized citations.

    Detects project type from folder names and keywords:
    - Automotive: vehicles, maintenance, repairs
    - Electronics: circuits, microcontrollers, IoT
    - Software: code projects, documentation
    - Workshop: woodworking, metalworking, 3D printing
    - Home: home automation, repairs, improvements

    Examples:
        /garage/subaru/outback -> automotive, make="Subaru", model="Outback"
        /projects/esp32_weather -> electronics, board="ESP32"
        /dev/my-api -> software
        /workshop/dining_table -> workshop
    """
    path_str = str(project_path).lower()
    parts = [p for p in path_str.replace("\\", "/").split("/") if p and p != "."]

    context = ProjectContext(
        raw_path=project_path,
        project_name=project_path.name,
    )

    # Detect project type by keywords (order matters - more specific first)

    # 1. Electronics detection
    if any(kw in path_str for kw in ELECTRONICS_KEYWORDS):
        context.project_type = "electronics"
    for board in ELECTRONICS_BOARDS:
        if board in path_str:
            context.project_type = "electronics"
            context.metadata["board"] = board.upper()
            break

    # 2. Software detection
    if context.project_type == "general":
        if any(kw in path_str for kw in SOFTWARE_KEYWORDS):
            context.project_type = "software"
        for lang in SOFTWARE_LANGUAGES:
            if lang in path_str:
                context.project_type = "software"
                context.metadata["language"] = lang.title()
                break
        # Check for common software project indicators
        if (project_path / "pyproject.toml").exists():
            context.project_type = "software"
            context.metadata["language"] = "Python"
        elif (project_path / "package.json").exists():
            context.project_type = "software"
            context.metadata["language"] = "JavaScript"
        elif (project_path / "Cargo.toml").exists():
            context.project_type = "software"
            context.metadata["language"] = "Rust"

    # 3. Workshop detection
    if context.project_type == "general":
        if any(kw in path_str for kw in WORKSHOP_KEYWORDS):
            context.project_type = "workshop"

    # 4. Home detection
    if context.project_type == "general":
        if any(kw in path_str for kw in HOME_KEYWORDS):
            context.project_type = "home"

    # 5. Automotive detection (check explicit keywords first)
    if context.project_type == "general":
        if "automotive" in path_str or "car" in path_str or "garage" in path_str or "vehicle" in path_str:
            context.project_type = "automotive"

    # 6. Try to find automotive make and model
    for make, models in AUTOMOTIVE_MAKES.items():
        if make in path_str:
            context.vehicle_make = make.title()
            context.project_type = "automotive"
            for model in models:
                if model.lower() in path_str:
                    context.vehicle_model = model.title()
                    break
            break

        # Check for model without explicit make
        for model in models:
            if model.lower() in path_str:
                context.vehicle_make = make.title()
                context.vehicle_model = model.title()
                context.project_type = "automotive"
                break
        if context.vehicle_model:
            break

    # Extract year (4-digit between 1980-2030) for automotive
    if context.project_type == "automotive":
        year_match = re.search(r"(?:^|[/_\-\s])(19[89]\d|20[0-2]\d)(?:[/_\-\s]|$)", path_str)
        if year_match:
            context.vehicle_year = year_match.group(1)

        # Use folder name as nickname if not already a make/model
        folder = project_path.name.lower()
        is_make_or_model = (
            folder in AUTOMOTIVE_MAKES.keys()
            or any(folder in models for models in AUTOMOTIVE_MAKES.values())
            or any(model in folder for models in AUTOMOTIVE_MAKES.values() for model in models)
            or any(make in folder for make in AUTOMOTIVE_MAKES.keys())
        )
        is_year_only = context.vehicle_year and folder == context.vehicle_year
        is_generic_term = folder in ["car", "auto", "vehicle", "automotive", "garage"]

        if not is_make_or_model and not is_year_only and not is_generic_term and len(folder) > 2:
            context.vehicle_nickname = project_path.name.title()

    return context


__all__ = [
    "ProjectContext",
    "extract_project_context",
    "AUTOMOTIVE_MAKES",
    "ELECTRONICS_BOARDS",
    "SOFTWARE_LANGUAGES",
]
