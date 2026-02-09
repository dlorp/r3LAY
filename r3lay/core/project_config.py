"""
Project Configuration - Domain-specific research context for r3LAY.

Provides structured configuration for different research domains:
- AUTOMOTIVE: Vehicle-specific context (make, model, year, engine, etc.)
- ELECTRONICS: Device/component context (manufacturer, model, platform)
- SOFTWARE: Development context (language, framework, issue type)
- HOME_DIY: Project context (project type, location, materials)

The configuration is stored in `.r3lay/project.yaml` and provides:
- Context for LLM prompts (detailed domain knowledge)
- Search query augmentation (concise keywords)
- Wizard detection for initial setup

This aligns with r3LAY's garage hobbyist / tinkerer lens, capturing the
specific details that make research results actionable.
"""

from __future__ import annotations

import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator
from ruamel.yaml import YAML

logger = logging.getLogger(__name__)


# ============================================================================
# Domain Types
# ============================================================================


class DomainType(str, Enum):
    """Research domain classification.

    Each domain has specialized configuration fields that capture
    the context needed for effective research in that area.
    """

    AUTOMOTIVE = "automotive"
    ELECTRONICS = "electronics"
    SOFTWARE = "software"
    HOME_DIY = "home_diy"


class SkillLevel(str, Enum):
    """Skill level for HOME_DIY projects."""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


# ============================================================================
# Domain-Specific Configurations
# ============================================================================


class AutomotiveConfig(BaseModel):
    """Configuration for automotive research projects.

    Captures vehicle-specific details for parts interchange,
    repair procedures, and troubleshooting research.

    Attributes:
        make: Vehicle manufacturer (e.g., "Subaru", "Toyota")
        model: Vehicle model (e.g., "Outback", "Camry")
        year_start: First year of applicable range
        year_end: Last year of applicable range (None = single year)
        engine_code: Engine designation (e.g., "EJ25", "2GR-FE")
        transmission: Transmission type (e.g., "5MT", "CVT", "4EAT")
        mileage: Current odometer reading (km or miles)
        nickname: User's name for this vehicle
        current_issues: List of current problems being researched
        goals: What the user wants to accomplish
    """

    make: str = Field(description="Vehicle manufacturer")
    model: str = Field(description="Vehicle model name")
    year_start: int = Field(ge=1900, le=2100, description="Start year")
    year_end: int | None = Field(
        default=None,
        ge=1900,
        le=2100,
        description="End year (None = single year)",
    )
    engine_code: str | None = Field(
        default=None,
        description="Engine designation code",
    )
    transmission: str | None = Field(
        default=None,
        description="Transmission type/code",
    )
    mileage: int | None = Field(
        default=None,
        ge=0,
        description="Current odometer reading",
    )
    nickname: str | None = Field(
        default=None,
        description="User's name for this vehicle",
    )
    current_issues: list[str] = Field(
        default_factory=list,
        description="Current problems being researched",
    )
    goals: list[str] = Field(
        default_factory=list,
        description="Research/repair goals",
    )

    @field_validator("year_end", mode="after")
    @classmethod
    def validate_year_range(cls, v: int | None, info: Any) -> int | None:
        """Ensure year_end is >= year_start if provided."""
        if v is not None and "year_start" in info.data:
            if v < info.data["year_start"]:
                raise ValueError("year_end must be >= year_start")
        return v

    def get_year_range(self) -> str:
        """Get formatted year range string."""
        if self.year_end is None or self.year_end == self.year_start:
            return str(self.year_start)
        return f"{self.year_start}-{self.year_end}"

    def get_vehicle_string(self) -> str:
        """Get full vehicle description string."""
        parts = [self.get_year_range(), self.make, self.model]
        if self.engine_code:
            parts.append(f"({self.engine_code})")
        if self.transmission:
            parts.append(self.transmission)
        return " ".join(parts)


class ElectronicsConfig(BaseModel):
    """Configuration for electronics research projects.

    Captures device/component details for repair, modification,
    and troubleshooting research.

    Attributes:
        device_type: Category of device (e.g., "amplifier", "oscilloscope")
        manufacturer: Device manufacturer
        model_number: Model/part number
        platform: Hardware platform if applicable (e.g., "ESP32", "STM32")
        symptoms: Current symptoms or issues
        tools_available: List of available test equipment/tools
        goals: What the user wants to accomplish
    """

    device_type: str = Field(description="Type/category of device")
    manufacturer: str | None = Field(
        default=None,
        description="Device manufacturer",
    )
    model_number: str | None = Field(
        default=None,
        description="Model or part number",
    )
    platform: str | None = Field(
        default=None,
        description="Hardware platform (ESP32, Arduino, etc.)",
    )
    symptoms: list[str] = Field(
        default_factory=list,
        description="Current symptoms or issues",
    )
    tools_available: list[str] = Field(
        default_factory=list,
        description="Available test equipment and tools",
    )
    goals: list[str] = Field(
        default_factory=list,
        description="Research/repair goals",
    )

    def get_device_string(self) -> str:
        """Get full device description string."""
        parts = []
        if self.manufacturer:
            parts.append(self.manufacturer)
        if self.model_number:
            parts.append(self.model_number)
        parts.append(self.device_type)
        if self.platform:
            parts.append(f"({self.platform})")
        return " ".join(parts)


class SoftwareConfig(BaseModel):
    """Configuration for software development research projects.

    Captures development context for debugging, implementation,
    and troubleshooting research.

    Attributes:
        language: Primary programming language
        framework: Framework or library in use
        version: Language/framework version
        platform: Target platform (e.g., "macOS", "Linux", "embedded")
        issue_type: Category of issue (e.g., "bug", "performance", "design")
        description: Detailed description of the issue or goal
    """

    language: str = Field(description="Primary programming language")
    framework: str | None = Field(
        default=None,
        description="Framework or library",
    )
    version: str | None = Field(
        default=None,
        description="Language/framework version",
    )
    platform: str | None = Field(
        default=None,
        description="Target platform",
    )
    issue_type: str | None = Field(
        default=None,
        description="Category of issue",
    )
    description: str | None = Field(
        default=None,
        description="Detailed description",
    )

    def get_context_string(self) -> str:
        """Get full development context string."""
        parts = [self.language]
        if self.framework:
            parts.append(self.framework)
        if self.version:
            parts.append(f"v{self.version}")
        if self.platform:
            parts.append(f"on {self.platform}")
        return " ".join(parts)


class HomeDIYConfig(BaseModel):
    """Configuration for home improvement and DIY projects.

    Captures project context for home repair, renovation,
    and maker/tinkerer research.

    Attributes:
        project_type: Category of project (e.g., "plumbing", "electrical")
        location: Where the project is located (e.g., "bathroom", "garage")
        materials: List of materials involved
        skill_level: User's self-assessed skill level
        tools_available: List of available tools
        goals: What the user wants to accomplish
    """

    project_type: str = Field(description="Type/category of project")
    location: str | None = Field(
        default=None,
        description="Location of the project",
    )
    materials: list[str] = Field(
        default_factory=list,
        description="Materials involved",
    )
    skill_level: SkillLevel = Field(
        default=SkillLevel.INTERMEDIATE,
        description="Self-assessed skill level",
    )
    tools_available: list[str] = Field(
        default_factory=list,
        description="Available tools",
    )
    goals: list[str] = Field(
        default_factory=list,
        description="Project goals",
    )

    def get_project_string(self) -> str:
        """Get full project description string."""
        parts = [self.project_type]
        if self.location:
            parts.append(f"in {self.location}")
        return " ".join(parts)


# ============================================================================
# Main Project Configuration
# ============================================================================


class ProjectConfig(BaseModel):
    """Main project configuration with domain selection.

    Holds the active domain and domain-specific configuration.
    Only one domain can be active at a time.

    Attributes:
        name: Human-readable project name
        domain: Active research domain
        automotive: Automotive-specific config (if domain is AUTOMOTIVE)
        electronics: Electronics-specific config (if domain is ELECTRONICS)
        software: Software-specific config (if domain is SOFTWARE)
        home_diy: Home DIY-specific config (if domain is HOME_DIY)
        created_at: ISO timestamp when project was created
        updated_at: ISO timestamp when project was last updated
        notes: General project notes
    """

    name: str = Field(description="Project name")
    domain: DomainType = Field(description="Active research domain")

    # Domain-specific configs (only one should be populated)
    automotive: AutomotiveConfig | None = None
    electronics: ElectronicsConfig | None = None
    software: SoftwareConfig | None = None
    home_diy: HomeDIYConfig | None = None

    # Metadata
    created_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Creation timestamp",
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Last update timestamp",
    )
    notes: str | None = Field(
        default=None,
        description="General project notes",
    )

    def get_active_config(
        self,
    ) -> AutomotiveConfig | ElectronicsConfig | SoftwareConfig | HomeDIYConfig | None:
        """Get the active domain-specific configuration.

        Returns:
            The configuration for the active domain, or None if not set.
        """
        if self.domain == DomainType.AUTOMOTIVE:
            return self.automotive
        elif self.domain == DomainType.ELECTRONICS:
            return self.electronics
        elif self.domain == DomainType.SOFTWARE:
            return self.software
        elif self.domain == DomainType.HOME_DIY:
            return self.home_diy
        return None

    def get_context_for_research(self) -> str:
        """Generate formatted context for LLM prompts.

        Returns a detailed, human-readable description of the project
        context suitable for inclusion in research prompts.

        Returns:
            Formatted context string for LLM prompts.
        """
        lines: list[str] = [
            f"## Project: {self.name}",
            f"Domain: {self.domain.value.replace('_', ' ').title()}",
            "",
        ]

        if self.domain == DomainType.AUTOMOTIVE:
            if self.automotive:
                lines.extend(self._format_automotive_context(self.automotive))
        elif self.domain == DomainType.ELECTRONICS:
            if self.electronics:
                lines.extend(self._format_electronics_context(self.electronics))
        elif self.domain == DomainType.SOFTWARE:
            if self.software:
                lines.extend(self._format_software_context(self.software))
        elif self.domain == DomainType.HOME_DIY:
            if self.home_diy:
                lines.extend(self._format_home_diy_context(self.home_diy))

        if self.notes:
            lines.extend(["", "### Notes", self.notes])

        return "\n".join(lines)

    def _format_automotive_context(self, cfg: AutomotiveConfig) -> list[str]:
        """Format automotive config for LLM context."""
        lines = [
            "### Vehicle Information",
            f"- Vehicle: {cfg.get_vehicle_string()}",
        ]
        if cfg.nickname:
            lines.append(f"- Nickname: {cfg.nickname}")
        if cfg.mileage:
            lines.append(f"- Mileage: {cfg.mileage:,}")

        if cfg.current_issues:
            lines.extend(["", "### Current Issues"])
            for issue in cfg.current_issues:
                lines.append(f"- {issue}")

        if cfg.goals:
            lines.extend(["", "### Goals"])
            for goal in cfg.goals:
                lines.append(f"- {goal}")

        return lines

    def _format_electronics_context(self, cfg: ElectronicsConfig) -> list[str]:
        """Format electronics config for LLM context."""
        lines = [
            "### Device Information",
            f"- Device: {cfg.get_device_string()}",
        ]

        if cfg.symptoms:
            lines.extend(["", "### Symptoms"])
            for symptom in cfg.symptoms:
                lines.append(f"- {symptom}")

        if cfg.tools_available:
            lines.extend(["", "### Available Tools"])
            for tool in cfg.tools_available:
                lines.append(f"- {tool}")

        if cfg.goals:
            lines.extend(["", "### Goals"])
            for goal in cfg.goals:
                lines.append(f"- {goal}")

        return lines

    def _format_software_context(self, cfg: SoftwareConfig) -> list[str]:
        """Format software config for LLM context."""
        lines = [
            "### Development Context",
            f"- Stack: {cfg.get_context_string()}",
        ]
        if cfg.issue_type:
            lines.append(f"- Issue Type: {cfg.issue_type}")

        if cfg.description:
            lines.extend(["", "### Description", cfg.description])

        return lines

    def _format_home_diy_context(self, cfg: HomeDIYConfig) -> list[str]:
        """Format home DIY config for LLM context."""
        lines = [
            "### Project Information",
            f"- Project: {cfg.get_project_string()}",
            f"- Skill Level: {cfg.skill_level.value.title()}",
        ]

        if cfg.materials:
            lines.extend(["", "### Materials"])
            for material in cfg.materials:
                lines.append(f"- {material}")

        if cfg.tools_available:
            lines.extend(["", "### Available Tools"])
            for tool in cfg.tools_available:
                lines.append(f"- {tool}")

        if cfg.goals:
            lines.extend(["", "### Goals"])
            for goal in cfg.goals:
                lines.append(f"- {goal}")

        return lines

    def get_search_context(self) -> str:
        """Generate concise context for search query augmentation.

        Returns a compact string of keywords and identifiers
        suitable for augmenting search queries.

        Returns:
            Concise context string for search augmentation.
        """
        parts: list[str] = []

        if self.domain == DomainType.AUTOMOTIVE:
            if self.automotive:
                parts.append(self.automotive.get_vehicle_string())
                if self.automotive.current_issues:
                    # Add first issue as most likely search focus
                    parts.append(self.automotive.current_issues[0])

        elif self.domain == DomainType.ELECTRONICS:
            if self.electronics:
                parts.append(self.electronics.get_device_string())
                if self.electronics.symptoms:
                    parts.append(self.electronics.symptoms[0])

        elif self.domain == DomainType.SOFTWARE:
            if self.software:
                parts.append(self.software.get_context_string())
                if self.software.issue_type:
                    parts.append(self.software.issue_type)

        elif self.domain == DomainType.HOME_DIY:
            if self.home_diy:
                parts.append(self.home_diy.get_project_string())
                if self.home_diy.materials:
                    parts.append(self.home_diy.materials[0])

        return " ".join(parts)


# ============================================================================
# Configuration Manager
# ============================================================================


class ProjectConfigManager:
    """Manages project configuration with YAML persistence.

    Handles loading, saving, and accessing project configuration
    stored in `.r3lay/project.yaml`.

    The manager uses atomic writes to prevent configuration corruption.

    Attributes:
        project_path: Root path of the project
        config_file: Path to the configuration file
    """

    def __init__(self, project_path: Path) -> None:
        """Initialize the project configuration manager.

        Args:
            project_path: Root path of the project.
        """
        self.project_path = project_path
        self.r3lay_dir = project_path / ".r3lay"
        self.config_file = self.r3lay_dir / "project.yaml"

        # Configure YAML for readable output
        self.yaml = YAML()
        self.yaml.default_flow_style = False
        self.yaml.preserve_quotes = True
        self.yaml.indent(mapping=2, sequence=4, offset=2)

        # Internal cache
        self._config: ProjectConfig | None = None

    def needs_wizard(self) -> bool:
        """Check if project configuration wizard is needed.

        Returns True if no configuration file exists, indicating
        that the user should be guided through initial setup.

        Returns:
            True if no config exists and wizard should run.
        """
        return not self.config_file.exists()

    def load(self) -> ProjectConfig | None:
        """Load project configuration from disk.

        Returns:
            ProjectConfig if found and valid, None otherwise.

        Raises:
            ValueError: If configuration file is malformed.
        """
        if not self.config_file.exists():
            logger.debug("No project config file found at %s", self.config_file)
            return None

        try:
            with open(self.config_file) as f:
                data = self.yaml.load(f)

            if not data:
                logger.warning("Empty project config file")
                return None

            # Parse domain type
            data["domain"] = DomainType(data["domain"])

            # Parse skill level if present in home_diy
            if data.get("home_diy") and data["home_diy"].get("skill_level"):
                data["home_diy"]["skill_level"] = SkillLevel(
                    data["home_diy"]["skill_level"]
                )

            self._config = ProjectConfig(**data)
            logger.info("Loaded project config: %s", self._config.name)
            return self._config

        except Exception as e:
            logger.error("Failed to load project config: %s", e)
            raise ValueError(f"Invalid project configuration: {e}") from e

    def save(self, config: ProjectConfig) -> None:
        """Save project configuration to disk.

        Uses atomic writes via temp file to prevent corruption.

        Args:
            config: ProjectConfig to save.

        Raises:
            IOError: If file cannot be written.
        """
        # Ensure directory exists
        self.r3lay_dir.mkdir(parents=True, exist_ok=True)

        # Update timestamp
        config.updated_at = datetime.now().isoformat()

        # Convert to dict for YAML serialization
        data = self._config_to_dict(config)

        try:
            # Atomic write via temp file
            temp_file = self.config_file.with_suffix(".yaml.tmp")
            with open(temp_file, "w") as f:
                self.yaml.dump(data, f)
            temp_file.replace(self.config_file)

            # Update cache
            self._config = config
            logger.info("Saved project config: %s", config.name)

        except OSError as e:
            logger.error("Failed to save project config: %s", e)
            # Clean up temp file if it exists
            temp_file = self.config_file.with_suffix(".yaml.tmp")
            if temp_file.exists():
                temp_file.unlink()
            raise IOError(f"Failed to save project configuration: {e}") from e

    def _config_to_dict(self, config: ProjectConfig) -> dict[str, Any]:
        """Convert ProjectConfig to a serializable dict.

        Args:
            config: ProjectConfig to convert.

        Returns:
            Dictionary suitable for YAML serialization.
        """
        data: dict[str, Any] = {
            "name": config.name,
            "domain": config.domain.value,
            "created_at": config.created_at,
            "updated_at": config.updated_at,
        }

        if config.notes:
            data["notes"] = config.notes

        # Add domain-specific config
        if config.domain == DomainType.AUTOMOTIVE:
            if config.automotive:
                data["automotive"] = config.automotive.model_dump(
                    exclude_none=True,
                    exclude_defaults=False,
                )
        elif config.domain == DomainType.ELECTRONICS:
            if config.electronics:
                data["electronics"] = config.electronics.model_dump(
                    exclude_none=True,
                    exclude_defaults=False,
                )
        elif config.domain == DomainType.SOFTWARE:
            if config.software:
                data["software"] = config.software.model_dump(
                    exclude_none=True,
                    exclude_defaults=False,
                )
        elif config.domain == DomainType.HOME_DIY:
            if config.home_diy:
                diy_data = config.home_diy.model_dump(
                    exclude_none=True,
                    exclude_defaults=False,
                )
                # Convert skill_level enum to string
                if "skill_level" in diy_data:
                    diy_data["skill_level"] = config.home_diy.skill_level.value
                data["home_diy"] = diy_data

        return data

    def get(self) -> ProjectConfig | None:
        """Get the current project configuration.

        Loads from disk if not cached.

        Returns:
            ProjectConfig if available, None otherwise.
        """
        if self._config is None:
            self._config = self.load()
        return self._config

    def get_context_for_research(self) -> str:
        """Get formatted research context from current config.

        Convenience method that loads config and formats context.

        Returns:
            Formatted context string, or empty string if no config.
        """
        config = self.get()
        if config:
            return config.get_context_for_research()
        return ""

    def get_search_context(self) -> str:
        """Get search augmentation context from current config.

        Convenience method that loads config and formats search context.

        Returns:
            Concise context string, or empty string if no config.
        """
        config = self.get()
        if config:
            return config.get_search_context()
        return ""

    def delete(self) -> bool:
        """Delete the project configuration file.

        Returns:
            True if deleted, False if file didn't exist.
        """
        if self.config_file.exists():
            self.config_file.unlink()
            self._config = None
            logger.info("Deleted project config")
            return True
        return False

    def update(self, **kwargs: Any) -> ProjectConfig | None:
        """Update specific fields in the configuration.

        Loads existing config, updates specified fields, and saves.

        Args:
            **kwargs: Fields to update in the config.

        Returns:
            Updated ProjectConfig, or None if no config exists.

        Raises:
            ValueError: If invalid fields are specified.
        """
        config = self.get()
        if not config:
            logger.warning("Cannot update: no project config exists")
            return None

        # Create updated config
        config_dict = config.model_dump()
        config_dict.update(kwargs)

        # Re-parse to validate
        updated_config = ProjectConfig(**config_dict)
        self.save(updated_config)

        return updated_config


# ============================================================================
# Factory Functions
# ============================================================================


def create_automotive_project(
    project_path: Path,
    name: str,
    make: str,
    model: str,
    year: int,
    **kwargs: Any,
) -> ProjectConfig:
    """Create and save a new automotive project configuration.

    Convenience factory for creating automotive research projects.

    Args:
        project_path: Root path of the project.
        name: Project name.
        make: Vehicle manufacturer.
        model: Vehicle model.
        year: Vehicle year (or start year of range).
        **kwargs: Additional AutomotiveConfig fields.

    Returns:
        The created ProjectConfig.
    """
    auto_config = AutomotiveConfig(
        make=make,
        model=model,
        year_start=year,
        **kwargs,
    )

    config = ProjectConfig(
        name=name,
        domain=DomainType.AUTOMOTIVE,
        automotive=auto_config,
    )

    manager = ProjectConfigManager(project_path)
    manager.save(config)

    return config


def create_electronics_project(
    project_path: Path,
    name: str,
    device_type: str,
    **kwargs: Any,
) -> ProjectConfig:
    """Create and save a new electronics project configuration.

    Convenience factory for creating electronics research projects.

    Args:
        project_path: Root path of the project.
        name: Project name.
        device_type: Type/category of device.
        **kwargs: Additional ElectronicsConfig fields.

    Returns:
        The created ProjectConfig.
    """
    elec_config = ElectronicsConfig(
        device_type=device_type,
        **kwargs,
    )

    config = ProjectConfig(
        name=name,
        domain=DomainType.ELECTRONICS,
        electronics=elec_config,
    )

    manager = ProjectConfigManager(project_path)
    manager.save(config)

    return config


def create_software_project(
    project_path: Path,
    name: str,
    language: str,
    **kwargs: Any,
) -> ProjectConfig:
    """Create and save a new software project configuration.

    Convenience factory for creating software research projects.

    Args:
        project_path: Root path of the project.
        name: Project name.
        language: Primary programming language.
        **kwargs: Additional SoftwareConfig fields.

    Returns:
        The created ProjectConfig.
    """
    soft_config = SoftwareConfig(
        language=language,
        **kwargs,
    )

    config = ProjectConfig(
        name=name,
        domain=DomainType.SOFTWARE,
        software=soft_config,
    )

    manager = ProjectConfigManager(project_path)
    manager.save(config)

    return config


def create_home_diy_project(
    project_path: Path,
    name: str,
    project_type: str,
    **kwargs: Any,
) -> ProjectConfig:
    """Create and save a new home DIY project configuration.

    Convenience factory for creating home improvement research projects.

    Args:
        project_path: Root path of the project.
        name: Project name.
        project_type: Type/category of project.
        **kwargs: Additional HomeDIYConfig fields.

    Returns:
        The created ProjectConfig.
    """
    diy_config = HomeDIYConfig(
        project_type=project_type,
        **kwargs,
    )

    config = ProjectConfig(
        name=name,
        domain=DomainType.HOME_DIY,
        home_diy=diy_config,
    )

    manager = ProjectConfigManager(project_path)
    manager.save(config)

    return config


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Enums
    "DomainType",
    "SkillLevel",
    # Domain configs
    "AutomotiveConfig",
    "ElectronicsConfig",
    "SoftwareConfig",
    "HomeDIYConfig",
    # Main config
    "ProjectConfig",
    # Manager
    "ProjectConfigManager",
    # Factory functions
    "create_automotive_project",
    "create_electronics_project",
    "create_software_project",
    "create_home_diy_project",
]
