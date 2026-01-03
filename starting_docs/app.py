"""r3LAY - TUI Research Assistant."""

import asyncio
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, TabbedContent, TabPane

from .config import AppConfig
from .core import (
    AxiomManager,
    HybridIndex,
    ModelScanner,
    RegistryManager,
    SearXNGClient,
    SessionManager,
    SignalsManager,
)
from .ui.screens.init import InitScreen
from .ui.widgets.response_pane import ResponsePane
from .ui.widgets.input_pane import InputPane
from .ui.widgets.model_panel import ModelPanel
from .ui.widgets.index_panel import IndexPanel
from .ui.widgets.axiom_panel import AxiomPanel
from .ui.widgets.session_panel import SessionPanel
from .ui.widgets.settings_panel import SettingsPanel


class R3LayState:
    """Shared application state."""
    
    def __init__(self, project_path: Path, config: AppConfig):
        self.project_path = project_path
        self.config = config
        
        # Core components
        self.registry = RegistryManager(project_path)
        self.session_manager = SessionManager(project_path)
        self.signals = SignalsManager(project_path)
        self.axioms = AxiomManager(project_path)
        
        self.model_scanner = ModelScanner(
            hf_cache_path=config.models.huggingface.cache_path,
            ollama_endpoint=config.models.ollama.endpoint,
        )
        
        self.searxng = SearXNGClient(
            endpoint=config.searxng.endpoint,
            timeout=config.searxng.timeout,
        )
        
        # Hybrid index (lazy init)
        self._index: HybridIndex | None = None
        
        # Current LLM client
        self.current_model: str | None = config.models.default_model
        self.llm_client = None
    
    @property
    def index(self) -> HybridIndex:
        """Get or initialize hybrid index."""
        if self._index is None:
            self._index = HybridIndex(
                persist_path=self.project_path,
                collection_name=self.config.index.collection_name,
                embedding_model=self.config.index.embedding_model,
                use_hybrid=self.config.index.use_hybrid_search,
                vector_weight=self.config.index.vector_weight,
                bm25_weight=self.config.index.bm25_weight,
                rrf_k=self.config.index.rrf_k,
            )
        return self._index


class MainScreen(Screen):
    """Main application screen with bento layout."""
    
    BINDINGS = [
        Binding("ctrl+n", "new_session", "New"),
        Binding("ctrl+s", "save_session", "Save"),
        Binding("ctrl+r", "refresh_index", "Reindex"),
        Binding("ctrl+e", "start_expedition", "Research"),
        Binding("ctrl+enter", "submit_input", "Send"),
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+1", "tab_models", "Models", show=False),
        Binding("ctrl+2", "tab_index", "Index", show=False),
        Binding("ctrl+3", "tab_axioms", "Axioms", show=False),
        Binding("ctrl+4", "tab_sessions", "Sessions", show=False),
        Binding("ctrl+5", "tab_settings", "Settings", show=False),
    ]
    
    def __init__(self, state: R3LayState):
        super().__init__()
        self.state = state
    
    def compose(self) -> ComposeResult:
        yield Header()
        
        with Horizontal(id="main-layout"):
            # Left: Response pane (main content area ~60%)
            yield ResponsePane(self.state, id="response-pane")
            
            # Right column (~40%): Input + Tabbed panels
            with Vertical(id="right-column"):
                # Top right: User input
                yield InputPane(self.state, id="input-pane")
                
                # Bottom right: Tabbed pane
                with TabbedContent(id="control-tabs"):
                    with TabPane("Models", id="tab-models"):
                        yield ModelPanel(self.state)
                    with TabPane("Index", id="tab-index"):
                        yield IndexPanel(self.state)
                    with TabPane("Axioms", id="tab-axioms"):
                        yield AxiomPanel(self.state)
                    with TabPane("Sessions", id="tab-sessions"):
                        yield SessionPanel(self.state)
                    with TabPane("Settings", id="tab-settings"):
                        yield SettingsPanel(self.state)
        
        yield Footer()
    
    async def on_mount(self) -> None:
        """Initialize on mount."""
        # Start a new session
        self.state.session_manager.new_session()
        
        # Load project context
        if self.state.registry.exists():
            response_pane = self.query_one(ResponsePane)
            response_pane.add_system(f"▸▸ r3LAY ◂◂  Project: {self.state.registry.get('name')}")
            response_pane.add_system(self.state.registry.get_summary())
        
        # Focus input
        self.query_one(InputPane).focus_input()
    
    async def action_new_session(self) -> None:
        """Start a new session."""
        if self.state.session_manager.current:
            self.state.session_manager.end_session()
        self.state.session_manager.new_session()
        self.query_one(ResponsePane).clear()
        self.notify("New session started")
    
    async def action_save_session(self) -> None:
        """Save current session."""
        path = self.state.session_manager.end_session()
        if path:
            self.notify(f"Session saved: {path.name}")
        self.state.session_manager.new_session()
    
    async def action_refresh_index(self) -> None:
        """Refresh the hybrid index."""
        self.notify("Reindexing...")
        asyncio.create_task(self._reindex())
    
    async def _reindex(self) -> None:
        """Background reindex task."""
        from .core import DocumentLoader
        
        loader = DocumentLoader(
            chunk_size=self.state.config.index.chunk_size,
            chunk_overlap=self.state.config.index.chunk_overlap,
        )
        
        total = 0
        folders = ["manuals", "diagrams", "docs", "datasheets", "schematics", "configs"]
        
        self.state.index.clear()
        
        for folder in folders:
            folder_path = self.state.project_path / folder
            if folder_path.exists():
                chunks = loader.load_directory(folder_path)
                if chunks:
                    self.state.index.add_chunks(chunks)
                    total += len(chunks)
        
        # Index scraped content
        scraped = self.state.project_path / "links" / "scraped"
        if scraped.exists():
            chunks = loader.load_directory(scraped)
            if chunks:
                self.state.index.add_chunks(chunks)
                total += len(chunks)
        
        self.notify(f"Indexed {total} chunks")
        
        # Update index panel
        try:
            index_panel = self.query_one(IndexPanel)
            index_panel.refresh_stats()
        except Exception:
            pass
    
    async def action_start_expedition(self) -> None:
        """Start research mode."""
        input_pane = self.query_one(InputPane)
        input_pane.set_value("/research ")
        input_pane.focus_input()
    
    async def action_submit_input(self) -> None:
        """Submit the current input."""
        input_pane = self.query_one(InputPane)
        await input_pane.submit()
    
    def action_tab_models(self) -> None:
        self.query_one("#control-tabs", TabbedContent).active = "tab-models"
    
    def action_tab_index(self) -> None:
        self.query_one("#control-tabs", TabbedContent).active = "tab-index"
    
    def action_tab_axioms(self) -> None:
        self.query_one("#control-tabs", TabbedContent).active = "tab-axioms"
    
    def action_tab_sessions(self) -> None:
        self.query_one("#control-tabs", TabbedContent).active = "tab-sessions"
    
    def action_tab_settings(self) -> None:
        self.query_one("#control-tabs", TabbedContent).active = "tab-settings"


class R3LayApp(App):
    """Main r3LAY TUI application."""
    
    CSS_PATH = "ui/styles/app.tcss"
    TITLE = "r3LAY"
    SUB_TITLE = "Research Assistant"
    
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", show=True),
        Binding("ctrl+d", "toggle_dark", "Dark Mode"),
    ]
    
    def __init__(self, project_path: Path | None = None):
        super().__init__()
        self.project_path = project_path or Path.cwd()
        self.config = AppConfig.load(self.project_path)
        self.state: R3LayState | None = None
    
    async def on_mount(self) -> None:
        """Called when app is mounted."""
        registry_path = self.project_path / "registry.yaml"
        
        if not registry_path.exists():
            await self.push_screen(InitScreen(self.project_path))
        else:
            self.state = R3LayState(self.project_path, self.config)
            await self.push_screen(MainScreen(self.state))
    
    def action_toggle_dark(self) -> None:
        """Toggle dark mode."""
        self.dark = not self.dark


def main():
    """Entry point for the application."""
    import argparse
    
    parser = argparse.ArgumentParser(description="r3LAY Research Assistant")
    parser.add_argument(
        "project_path",
        nargs="?",
        default=".",
        help="Path to project directory",
    )
    args = parser.parse_args()
    
    project_path = Path(args.project_path).resolve()
    app = R3LayApp(project_path)
    app.run()


if __name__ == "__main__":
    main()
