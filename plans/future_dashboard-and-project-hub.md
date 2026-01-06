# R³LAY Central Dashboard & Project Hub

**R³LAY: Retrospective Recursive Research, Linked Archive Yield**

**Status:** Future Enhancement (implement after base project is complete)  
**Created:** 2026-01-04  
**Updated:** 2026-01-05

---

## Overview

Transform R³LAY from a "point-at-any-folder" tool into a centralized project management system with a dashboard, while preserving direct-launch capability.

**User's Vision:**
- Projects live inside `~/Documents/r3LAY/` (default, configurable on first run)
- Dashboard shows all projects with metrics (index health, sessions, axioms, storage)
- Auto-discover projects by scanning for `.r3lay/` folders
- `r3lay` → dashboard, `r3lay /path` → direct to project

---

## Architecture Changes

### Current Flow
```
r3lay [path] → R3LayApp → MainScreen(project)
```

### New Flow
```
r3lay           → R3LayApp → DashboardScreen → select → MainScreen(project)
r3lay [path]    → R3LayApp → MainScreen(project)  # Skip dashboard
```

---

## New File Structure

```
~/.r3lay/                          # Global R³LAY config (NEW)
├── config.yaml                    # Global settings (includes r3lay_home path)
├── registry.yaml                  # Project registry with metadata
└── models/                        # GGUF drop folder (existing)

~/Documents/r3LAY/                 # Default project home (configurable)
├── garage/
│   └── Brighton/
│       ├── .r3lay/                # Per-project data
│       │   ├── config.yaml
│       │   ├── index/
│       │   ├── sessions/
│       │   └── axioms/
│       ├── .signals/              # Provenance data
│       └── docs/                  # User's actual files
├── printing/
│   └── ender3/
│       ├── .r3lay/
│       ├── .signals/
│       └── profiles/
├── nas/
│   └── truenas/
└── electronics/
    └── esp32-weather/
```

---

## Implementation Plan

### Phase 0: First-Run Setup (Modal Dialog)

On first launch (when `~/.r3lay/config.yaml` doesn't exist):

```python
class FirstRunDialog(ModalScreen):
    """One-time setup dialog on first R³LAY launch."""

    def compose(self) -> ComposeResult:
        with Vertical(id="setup"):
            yield Static("╔══════════════════════════════════════════╗")
            yield Static("║  Welcome to R³LAY                        ║")
            yield Static("║  Retrospective Recursive Research        ║")
            yield Static("║  Linked Archive Yield                    ║")
            yield Static("╚══════════════════════════════════════════╝")
            yield Static("")
            yield Static("Choose where to store your projects:")
            yield Input(
                value=str(Path("~/Documents/r3LAY").expanduser()),
                id="home-path"
            )
            yield Button("Browse...", id="browse")
            yield Button("Continue", id="continue", variant="primary")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "continue":
            home_path = Path(self.query_one("#home-path", Input).value)
            home_path.mkdir(parents=True, exist_ok=True)
            self.dismiss(home_path)
```

### Phase 1: Global Configuration & Registry

**New file: `r3lay/core/registry.py`**

```python
@dataclass
class ProjectEntry:
    name: str
    path: Path
    project_type: str  # automotive, printing, electronics, software, home, etc.
    last_accessed: datetime
    # Metrics (computed on load)
    index_doc_count: int = 0
    axiom_count: int = 0
    disputed_count: int = 0
    session_count: int = 0
    storage_bytes: int = 0

class ProjectRegistry:
    """Manages the global list of R³LAY projects."""

    projects: list[ProjectEntry]

    @classmethod
    def load(cls) -> "ProjectRegistry":
        """Load from ~/.r3lay/registry.yaml"""

    def save(self) -> None:
        """Save to ~/.r3lay/registry.yaml"""

    def add_project(self, path: Path) -> ProjectEntry:
        """Register a new project."""

    def remove_project(self, path: Path) -> None:
        """Unregister (doesn't delete files)."""

    def update_accessed(self, path: Path) -> None:
        """Update last_accessed timestamp."""

    def compute_metrics(self, entry: ProjectEntry) -> None:
        """Calculate index size, axiom count, session count, storage."""

    def discover_projects(self, root: Path) -> list[Path]:
        """Find all directories with .r3lay/ under root."""

    def sync_with_filesystem(self, root: Path) -> None:
        """Auto-discover projects and update registry.

        - Finds all dirs with .r3lay/ under root
        - Adds newly discovered projects
        - Marks missing projects as 'archived' (doesn't remove)
        """
```

**Modify: `r3lay/config.py`**

Add `GlobalConfig` class:
```python
class GlobalConfig(BaseModel):
    """User-level R³LAY configuration."""

    r3lay_home: Path = Path("~/Documents/r3LAY").expanduser()
    first_run_complete: bool = False  # Triggers setup wizard if False
    theme: str = "dark"
    default_embedder: str = "mlx-community/all-MiniLM-L6-v2-4bit"

    # Model discovery paths (moved from hardcoded)
    hf_cache_path: Path
    mlx_folder: Path
    gguf_folder: Path
    ollama_endpoint: str
    
    # SearXNG for research
    searxng_endpoint: str = "http://localhost:8888"

    @classmethod
    def load(cls) -> "GlobalConfig":
        """Load from ~/.r3lay/config.yaml"""

    def save(self) -> None:
        """Save to ~/.r3lay/config.yaml"""

    @classmethod
    def run_first_time_setup(cls) -> "GlobalConfig":
        """Interactive first-run setup - ask user for r3lay_home location."""
```

### Phase 2: Dashboard Screen

**New file: `r3lay/ui/screens/dashboard.py`**

```python
class DashboardScreen(Screen):
    """Project selection and management dashboard."""

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+n", "new_project", "New Project"),
        Binding("enter", "open_selected", "Open"),
        Binding("r", "refresh", "Refresh"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="dashboard"):
            yield Static(
                "╔══════════════════════════════════════════╗\n"
                "║  R³LAY Projects                          ║\n"
                "╚══════════════════════════════════════════╝",
                id="title"
            )
            yield ProjectTable(id="project-table")  # DataTable with metrics
            with Horizontal(id="actions"):
                yield Button("New Project", id="btn-new")
                yield Button("Open Folder", id="btn-browse")
                yield Button("Refresh", id="btn-refresh")
        yield Footer()

    async def on_mount(self) -> None:
        """Load registry and populate table."""
        registry = ProjectRegistry.load()
        # Compute metrics for each project
        for entry in registry.projects:
            registry.compute_metrics(entry)
        self._populate_table(registry.projects)

    def _populate_table(self, projects: list[ProjectEntry]) -> None:
        """Fill the DataTable with project info."""
        table = self.query_one(ProjectTable)
        table.clear()
        for p in projects:
            # Show disputed count if any
            axiom_str = f"{p.axiom_count}"
            if p.disputed_count > 0:
                axiom_str += f" (⚠{p.disputed_count})"
            
            table.add_row(
                p.name,
                p.project_type,
                f"{p.index_doc_count} docs",
                axiom_str,
                f"{p.session_count} sessions",
                humanize_bytes(p.storage_bytes),
                p.last_accessed.strftime("%Y-%m-%d"),
            )
```

**Dashboard Table Columns:**

| Name | Type | Index | Axioms | Sessions | Storage | Last Used |
|------|------|-------|--------|----------|---------|-----------|
| Brighton | automotive | 47 docs | 142 (⚠3) | 12 | 2.3 MB | 2026-01-03 |
| ender3 | printing | 23 docs | 67 | 5 | 1.1 MB | 2026-01-02 |
| TrueNAS | home | 15 docs | 34 | 8 | 0.8 MB | 2025-12-28 |

### Phase 3: App Integration

**Modify: `r3lay/app.py`**

```python
class R3LayApp(App):
    def __init__(self, project_path: Path | None = None):
        super().__init__()
        self.project_path = project_path  # None = show dashboard
        self.global_config = GlobalConfig.load()
        self.registry = ProjectRegistry.load()

    async def on_mount(self) -> None:
        # First run setup
        if not self.global_config.first_run_complete:
            result = await self.push_screen_wait(FirstRunDialog())
            self.global_config.r3lay_home = result
            self.global_config.first_run_complete = True
            self.global_config.save()
        
        if self.project_path:
            # Direct launch - skip dashboard
            self.registry.update_accessed(self.project_path)
            state = R3LayState(self.project_path)
            await self.push_screen(MainScreen(state))
        else:
            # No path - show dashboard
            await self.push_screen(DashboardScreen())

    async def open_project(self, path: Path) -> None:
        """Called when user selects a project from dashboard."""
        self.registry.update_accessed(path)
        self.registry.save()
        state = R3LayState(path)
        await self.push_screen(MainScreen(state))

    async def back_to_dashboard(self) -> None:
        """Return to dashboard from project."""
        await self.pop_screen()

def main():
    parser = argparse.ArgumentParser(
        description="R³LAY — Retrospective Recursive Research, Linked Archive Yield"
    )
    parser.add_argument("project_path", nargs="?", default=None,
                        help="Path to project (opens dashboard if omitted)")
    args = parser.parse_args()

    project_path = Path(args.project_path).resolve() if args.project_path else None
    app = R3LayApp(project_path)
    app.run()
```

### Phase 4: New Project Creation

**New file: `r3lay/ui/dialogs/new_project.py`**

```python
PROJECT_TYPES = [
    ("Automotive", "automotive"),
    ("3D Printing", "printing"),
    ("Electronics", "electronics"),
    ("Software", "software"),
    ("Home Lab", "home"),
    ("General", "general"),
]

class NewProjectDialog(ModalScreen):
    """Dialog for creating a new project."""

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label("Create New R³LAY Project")
            yield Input(placeholder="Project name", id="name")
            yield Select(options=PROJECT_TYPES, id="type")
            yield Static("Project will be created in:")
            yield Static("", id="path-preview")
            with Horizontal():
                yield Button("Create", id="create", variant="primary")
                yield Button("Cancel", id="cancel")

    def on_input_changed(self, event: Input.Changed) -> None:
        """Update path preview as user types."""
        if event.input.id == "name":
            ptype = self.query_one("#type", Select).value or "general"
            name = event.value
            home = self.app.global_config.r3lay_home
            path = home / ptype / name
            self.query_one("#path-preview", Static).update(str(path))

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "create":
            name = self.query_one("#name", Input).value
            ptype = self.query_one("#type", Select).value
            # Create ~/Documents/r3LAY/{type}/{name}/ with .r3lay/
            self.dismiss({"name": name, "type": ptype})
        else:
            self.dismiss(None)
```

### Phase 5: MainScreen Navigation

**Modify: `r3lay/app.py` MainScreen**

Add binding to return to dashboard:
```python
class MainScreen(Screen):
    BINDINGS = [
        # ... existing ...
        Binding("ctrl+b", "back_to_dashboard", "Dashboard", show=True),
    ]

    async def action_back_to_dashboard(self) -> None:
        """Return to project dashboard."""
        # Clean up current project state if needed
        if self.state.current_backend:
            await self.state.unload_model()
        await self.app.back_to_dashboard()
```

---

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `r3lay/core/registry.py` | CREATE | ProjectRegistry, ProjectEntry, auto-discovery |
| `r3lay/config.py` | MODIFY | Add GlobalConfig class |
| `r3lay/ui/screens/__init__.py` | CREATE | Screen exports |
| `r3lay/ui/screens/dashboard.py` | CREATE | DashboardScreen with project table |
| `r3lay/ui/dialogs/__init__.py` | CREATE | Dialog exports |
| `r3lay/ui/dialogs/first_run.py` | CREATE | FirstRunDialog (home path setup) |
| `r3lay/ui/dialogs/new_project.py` | CREATE | NewProjectDialog |
| `r3lay/app.py` | MODIFY | Dashboard integration, first-run flow, launch logic |
| `r3lay/ui/styles/dashboard.tcss` | CREATE | Dashboard styling |

---

## Metrics Calculation

```python
def compute_metrics(self, entry: ProjectEntry) -> None:
    """Calculate project metrics."""
    r3lay_dir = entry.path / ".r3lay"

    # Index doc count
    index_meta = r3lay_dir / "index" / "metadata.json"
    if index_meta.exists():
        entry.index_doc_count = json.loads(index_meta.read_text()).get("doc_count", 0)

    # Session count
    sessions_dir = r3lay_dir / "sessions"
    if sessions_dir.exists():
        entry.session_count = len(list(sessions_dir.glob("*.json")))

    # Axiom count (with disputed tracking)
    axioms_file = entry.path / "axioms" / "axioms.yaml"
    if axioms_file.exists():
        axioms = yaml.safe_load(axioms_file.read_text())
        entry.axiom_count = len(axioms)
        entry.disputed_count = sum(
            1 for a in axioms if a.get("state") == "disputed"
        )

    # Storage (entire .r3lay + .signals + axioms folders)
    total_bytes = 0
    for folder in [r3lay_dir, entry.path / ".signals", entry.path / "axioms"]:
        if folder.exists():
            total_bytes += sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())
    entry.storage_bytes = total_bytes
```

---

## Design Decisions (User-Confirmed)

1. **Where do projects live?** → `~/Documents/r3LAY/` default, prompt on first run for custom location
2. **Auto-discovery?** → Yes, scan r3lay_home for `.r3lay/` folders on dashboard load
3. **Can open arbitrary paths?** → Yes, via "Open Folder" button
4. **Dashboard on every launch?** → Only when no path argument given
5. **How to return to dashboard?** → `Ctrl+B` from MainScreen
6. **Project types?** → automotive, printing, electronics, software, home, general

---

## Future Enhancements (Out of Scope for this plan)

- Cross-project search (search axioms across all projects)
- Shared axiom libraries (e.g., common Subaru specs across multiple car projects)
- Project templates (pre-configured for common domains)
- Project archiving with compression
- Axiom export/import between projects
- Dashboard axiom health alerts (projects with unresolved disputes)

---

*Document Version: 2.0*  
*Last Updated: 2026-01-05*  
*Name: R³LAY — Retrospective Recursive Research, Linked Archive Yield*