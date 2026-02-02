"""Registry manager for project YAML files."""

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML


class RegistryManager:
    """Manages the project registry.yaml file."""

    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.registry_path = project_path / "registry.yaml"
        self.logs_path = project_path / "logs"
        self.yaml = YAML()
        self.yaml.default_flow_style = False
        self.yaml.preserve_quotes = True
        self._data: dict[str, Any] | None = None

    @property
    def data(self) -> dict[str, Any]:
        """Get registry data, loading if necessary."""
        if self._data is None:
            self.load()
        return self._data or {}

    def exists(self) -> bool:
        """Check if registry file exists."""
        return self.registry_path.exists()

    def load(self) -> dict[str, Any]:
        """Load the registry from disk."""
        if not self.registry_path.exists():
            self._data = {}
            return self._data

        with open(self.registry_path) as f:
            self._data = self.yaml.load(f) or {}

        return self._data

    def save(self, backup: bool = True) -> None:
        """Save the registry to disk."""
        if backup and self.registry_path.exists():
            self._create_backup()

        with open(self.registry_path, "w") as f:
            self.yaml.dump(self._data, f)

    def _create_backup(self) -> Path:
        """Create a timestamped backup of the registry."""
        self.logs_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        backup_path = self.logs_path / f"registry_{timestamp}.yaml"
        shutil.copy(self.registry_path, backup_path)
        return backup_path

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the registry using dot notation."""
        keys = key.split(".")
        value = self.data

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            elif isinstance(value, list) and k.isdigit():
                idx = int(k)
                value = value[idx] if idx < len(value) else None
            else:
                return default

            if value is None:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """Set a value in the registry using dot notation."""
        keys = key.split(".")
        data = self.data

        for k in keys[:-1]:
            if k not in data:
                data[k] = {}
            data = data[k]

        data[keys[-1]] = value

    def update(self, updates: dict[str, Any]) -> None:
        """Update multiple values in the registry."""
        self._deep_update(self.data, updates)

    def _deep_update(self, base: dict, updates: dict) -> None:
        """Recursively update a dict."""
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value

    def initialize(self, theme: str, project_type: str, name: str) -> None:
        """Initialize a new registry with theme-specific template."""
        from ..config import THEMES

        theme_config = THEMES.get(theme)
        if not theme_config:
            raise ValueError(f"Unknown theme: {theme}")

        self._data = {
            "theme": theme,
            "type": project_type,
            "name": name,
            "status": "active",
            "created": datetime.now().isoformat(),
            "updated": datetime.now().isoformat(),
        }

        # Create default folders
        for folder in theme_config["default_folders"]:
            (self.project_path / folder).mkdir(parents=True, exist_ok=True)

        # Add type-specific folders
        type_folders = theme_config.get("type_folders", {}).get(project_type, [])
        for folder in type_folders:
            (self.project_path / folder).mkdir(parents=True, exist_ok=True)

        # Create hidden folders
        (self.project_path / ".signals").mkdir(exist_ok=True)
        (self.project_path / ".chromadb").mkdir(exist_ok=True)

        self.save(backup=False)

    def add_maintenance_record(
        self,
        task: str,
        date: str | None = None,
        notes: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Add a maintenance record (for vehicle/home themes)."""
        if "maintenance_log" not in self.data:
            self._data["maintenance_log"] = []

        record = {
            "task": task,
            "date": date or datetime.now().strftime("%Y-%m-%d"),
            "notes": notes,
            **kwargs,
        }

        self._data["maintenance_log"].append(record)
        self._data["updated"] = datetime.now().isoformat()

    def update_odometer(self, value: int, unit: str = "miles") -> None:
        """Update odometer reading (for vehicle theme)."""
        if "odometer" not in self.data:
            self._data["odometer"] = {}

        self._data["odometer"]["value"] = value
        self._data["odometer"]["unit"] = unit
        self._data["odometer"]["updated"] = datetime.now().isoformat()
        self._data["updated"] = datetime.now().isoformat()

    def add_known_issue(self, description: str) -> None:
        """Add a known issue."""
        if "known_issues" not in self.data:
            self._data["known_issues"] = []

        self._data["known_issues"].append(description)
        self._data["updated"] = datetime.now().isoformat()

    def resolve_issue(self, index: int, resolution: str | None = None) -> None:
        """Remove a known issue by index."""
        if "known_issues" in self.data and index < len(self.data["known_issues"]):
            issue = self._data["known_issues"].pop(index)
            if resolution:
                self.add_maintenance_record(
                    task=f"Resolved: {issue}",
                    notes=resolution,
                )
            self._data["updated"] = datetime.now().isoformat()

    def get_summary(self) -> str:
        """Get a text summary of the registry for LLM context."""
        lines = [f"# {self.get('name', 'Unknown')} Registry"]
        lines.append(f"Theme: {self.get('theme')}")
        lines.append(f"Type: {self.get('type')}")
        lines.append(f"Status: {self.get('status')}")

        # Theme-specific summaries
        if self.get("theme") == "vehicle":
            if odo := self.get("odometer"):
                lines.append(f"Odometer: {odo.get('value'):,} {odo.get('unit', 'miles')}")
            if make := self.get("make"):
                lines.append(f"Vehicle: {self.get('year')} {make} {self.get('model')}")

        if issues := self.get("known_issues"):
            lines.append(f"\nKnown Issues ({len(issues)}):")
            for i, issue in enumerate(issues):
                lines.append(f"  {i + 1}. {issue}")

        return "\n".join(lines)
