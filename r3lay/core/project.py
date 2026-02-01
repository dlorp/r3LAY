"""Project state management for r3LAY garage terminal.

Manages vehicle profiles and project state for maintenance tracking.
Each project folder can have a .r3lay/project.yaml file containing:
- Vehicle profile (year, make, model, engine, VIN)
- Current mileage
- Last updated timestamp
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from ruamel.yaml import YAML

logger = logging.getLogger(__name__)


class VehicleProfile(BaseModel):
    """Vehicle identification and specifications.

    Attributes:
        year: Model year (e.g., 2006)
        make: Manufacturer (e.g., "Subaru")
        model: Model name (e.g., "Outback")
        engine: Engine specification (e.g., "2.5L H4")
        vin: Vehicle Identification Number
        nickname: Friendly name for the vehicle
    """

    year: int
    make: str
    model: str
    engine: str | None = None
    vin: str | None = None
    nickname: str | None = None

    @property
    def display_name(self) -> str:
        """Get a human-readable display name.

        Returns:
            e.g., "2006 Subaru Outback" or "Brighton (2006 Subaru Outback)"
        """
        base = f"{self.year} {self.make} {self.model}"
        if self.nickname:
            return f"{self.nickname} ({base})"
        return base

    @property
    def short_name(self) -> str:
        """Get a short reference name.

        Returns:
            Nickname if set, otherwise model name.
        """
        return self.nickname or self.model


class ProjectState(BaseModel):
    """Project state including vehicle profile and mileage.

    Attributes:
        profile: Vehicle identification and specs
        current_mileage: Current odometer reading
        last_updated: When the state was last modified
    """

    profile: VehicleProfile
    current_mileage: int = 0
    last_updated: datetime = Field(default_factory=datetime.now)

    model_config = {"arbitrary_types_allowed": True}


class ProjectManager:
    """Manages project state persistence.

    Project state is stored in .r3lay/project.yaml within the project directory.
    Creates the .r3lay directory if it doesn't exist.

    Example:
        >>> pm = ProjectManager(Path("/garage/brighton"))
        >>> state = pm.load()
        >>> state.current_mileage
        156000
        >>> pm.update_mileage(157000)
    """

    PROJECT_DIR = ".r3lay"
    PROJECT_FILE = "project.yaml"

    def __init__(self, project_path: Path) -> None:
        """Initialize the project manager.

        Args:
            project_path: Path to the project directory
        """
        self.project_path = Path(project_path)
        self._state: ProjectState | None = None
        self._yaml = YAML()
        self._yaml.preserve_quotes = True
        self._yaml.default_flow_style = False

    @property
    def config_dir(self) -> Path:
        """Get the .r3lay directory path."""
        return self.project_path / self.PROJECT_DIR

    @property
    def state_file(self) -> Path:
        """Get the project.yaml file path."""
        return self.config_dir / self.PROJECT_FILE

    def _ensure_config_dir(self) -> None:
        """Create the .r3lay directory if it doesn't exist."""
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def exists(self) -> bool:
        """Check if project state file exists.

        Returns:
            True if .r3lay/project.yaml exists
        """
        return self.state_file.exists()

    def load(self) -> ProjectState | None:
        """Load project state from disk.

        Returns:
            ProjectState if file exists and is valid, None otherwise
        """
        if not self.state_file.exists():
            logger.debug(f"No project state file: {self.state_file}")
            return None

        try:
            with self.state_file.open("r") as f:
                data = self._yaml.load(f)

            if data is None:
                logger.warning(f"Empty project state file: {self.state_file}")
                return None

            # Parse the YAML data into ProjectState
            self._state = self._parse_state(data)
            logger.info(f"Loaded project state: {self._state.profile.display_name}")
            return self._state

        except Exception as e:
            logger.error(f"Failed to load project state: {e}")
            return None

    def _parse_state(self, data: dict[str, Any]) -> ProjectState:
        """Parse YAML data into ProjectState.

        Args:
            data: Raw YAML data

        Returns:
            Validated ProjectState
        """
        profile_data = data.get("profile", data.get("vehicle", {}))

        # Handle datetime parsing
        last_updated = data.get("last_updated")
        if isinstance(last_updated, str):
            last_updated = datetime.fromisoformat(last_updated)
        elif last_updated is None:
            last_updated = datetime.now()

        profile = VehicleProfile(
            year=profile_data.get("year", 0),
            make=profile_data.get("make", "Unknown"),
            model=profile_data.get("model", "Unknown"),
            engine=profile_data.get("engine"),
            vin=profile_data.get("vin"),
            nickname=profile_data.get("nickname"),
        )

        return ProjectState(
            profile=profile,
            current_mileage=data.get("current_mileage", 0),
            last_updated=last_updated,
        )

    def save(self, state: ProjectState) -> None:
        """Save project state to disk.

        Creates the .r3lay directory if needed.
        Uses atomic write (write to temp, then rename).

        Args:
            state: ProjectState to save
        """
        self._ensure_config_dir()

        # Update timestamp
        state.last_updated = datetime.now()

        # Convert to YAML-friendly dict
        data = {
            "profile": {
                "year": state.profile.year,
                "make": state.profile.make,
                "model": state.profile.model,
            },
            "current_mileage": state.current_mileage,
            "last_updated": state.last_updated.isoformat(),
        }

        # Add optional profile fields
        if state.profile.engine:
            data["profile"]["engine"] = state.profile.engine
        if state.profile.vin:
            data["profile"]["vin"] = state.profile.vin
        if state.profile.nickname:
            data["profile"]["nickname"] = state.profile.nickname

        # Atomic write
        temp_file = self.state_file.with_suffix(".yaml.tmp")
        try:
            with temp_file.open("w") as f:
                self._yaml.dump(data, f)
            temp_file.rename(self.state_file)
            self._state = state
            logger.info(f"Saved project state: {state.profile.display_name}")
        except Exception as e:
            if temp_file.exists():
                temp_file.unlink()
            raise RuntimeError(f"Failed to save project state: {e}") from e

    def update_mileage(self, mileage: int) -> ProjectState | None:
        """Update the current mileage.

        Loads current state if not loaded, updates mileage, and saves.

        Args:
            mileage: New odometer reading

        Returns:
            Updated ProjectState, or None if no project exists

        Raises:
            ValueError: If mileage is less than current (odometer rollback)
        """
        if self._state is None:
            self._state = self.load()

        if self._state is None:
            logger.warning("Cannot update mileage: no project state exists")
            return None

        if mileage < self._state.current_mileage:
            raise ValueError(
                f"Mileage cannot decrease: {mileage} < {self._state.current_mileage}"
            )

        self._state.current_mileage = mileage
        self.save(self._state)
        return self._state

    def create(
        self,
        year: int,
        make: str,
        model: str,
        *,
        engine: str | None = None,
        vin: str | None = None,
        nickname: str | None = None,
        current_mileage: int = 0,
    ) -> ProjectState:
        """Create a new project state.

        Args:
            year: Model year
            make: Manufacturer
            model: Model name
            engine: Engine specification
            vin: Vehicle Identification Number
            nickname: Friendly name
            current_mileage: Starting odometer reading

        Returns:
            Created ProjectState
        """
        profile = VehicleProfile(
            year=year,
            make=make,
            model=model,
            engine=engine,
            vin=vin,
            nickname=nickname,
        )

        state = ProjectState(
            profile=profile,
            current_mileage=current_mileage,
        )

        self.save(state)
        return state

    @property
    def state(self) -> ProjectState | None:
        """Get the current cached state (loads if needed)."""
        if self._state is None:
            self._state = self.load()
        return self._state


__all__ = [
    "VehicleProfile",
    "ProjectState",
    "ProjectManager",
]
