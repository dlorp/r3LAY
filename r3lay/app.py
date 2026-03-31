"""r3LAY - TUI Research Assistant - Phase 1 Shell."""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import asyncio  # noqa: E402
import logging  # noqa: E402
import signal  # noqa: E402
from logging.handlers import RotatingFileHandler  # noqa: E402
from pathlib import Path  # noqa: E402


def _setup_logging() -> logging.Logger:
    """Configure secure logging with rotation.

    Logs are written to ~/.r3lay/logs/ with proper permissions.
    Uses INFO level by default; set R3LAY_DEBUG=1 for DEBUG level.
    """
    # Create log directory in user's home (not world-readable /tmp)
    log_dir = Path.home() / ".r3lay" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    # Restrict directory permissions to owner only (700)
    log_dir.chmod(0o700)

    log_file = log_dir / "r3lay.log"

    # Use INFO by default, DEBUG only if explicitly requested
    log_level = logging.DEBUG if os.environ.get("R3LAY_DEBUG") else logging.INFO

    # Configure rotating file handler (5MB max, keep 3 backups)
    handler = RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(handler)

    return logging.getLogger(__name__)


logger = _setup_logging()

from textual.app import App, ComposeResult  # noqa: E402
from textual.binding import Binding  # noqa: E402
from textual.containers import Horizontal, Vertical  # noqa: E402
from textual.css.query import NoMatches  # noqa: E402
from textual.screen import Screen  # noqa: E402
from textual.widgets import Footer, TabbedContent, TabPane  # noqa: E402

from .config import AppConfig  # noqa: E402
from .core import R3LayState  # noqa: E402
from .ui.widgets import (  # noqa: E402
    AxiomPanel,
    GarageHeader,
    HistoryPanel,
    IndexPanel,
    InputPane,
    MaintenancePanel,
    ModelPanel,
    ResponsePane,
    SessionPanel,
    SettingsPanel,
    SplashScreen,
    VaultPanel,
)


class MainScreen(Screen):
    """Main application screen with bento layout."""

    BINDINGS = [
        Binding("ctrl+n", "new_session", "New"),
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+r", "reindex", "Reindex", show=True),
        Binding("ctrl+1", "tab_models", "Models", show=False),
        Binding("ctrl+2", "tab_index", "Index", show=False),
        Binding("ctrl+3", "tab_axioms", "Axioms", show=False),
        Binding("ctrl+4", "tab_log", "Log", show=False),
        Binding("ctrl+5", "tab_due", "Due", show=False),
        Binding("ctrl+6", "tab_sessions", "Sessions", show=False),
        Binding("ctrl+7", "tab_vault", "Vault", show=False),
        Binding("ctrl+8", "tab_settings", "Settings", show=False),
        Binding("ctrl+h", "tab_sessions", "History", show=False),
        Binding("ctrl+comma", "tab_settings", "Settings", show=False),
    ]

    def __init__(self, state: R3LayState):
        super().__init__()
        self.state = state

    def compose(self) -> ComposeResult:
        yield GarageHeader(self.state, id="garage-header")

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
                    with TabPane("Log", id="tab-log"):
                        yield HistoryPanel(self.state)
                    with TabPane("Due", id="tab-due"):
                        yield MaintenancePanel(
                            project_path=self.state.project_path,
                            current_mileage=None,
                        )
                    with TabPane("Sessions", id="tab-sessions"):
                        yield SessionPanel(self.state)
                    with TabPane("Vault", id="tab-vault"):
                        yield VaultPanel(self.state)
                    with TabPane("Settings", id="tab-settings"):
                        yield SettingsPanel(self.state)

        yield Footer()

    async def on_mount(self) -> None:
        """Initialize on mount."""
        # Focus input
        self.query_one(InputPane).focus_input()

        # Replay auto-restored session messages to response pane
        if self.state.session and self.state.session.messages:
            response_pane = self.query_one(ResponsePane)
            for msg in self.state.session.messages:
                if msg.role == "user":
                    response_pane.add_user(msg.content)
                elif msg.role == "assistant":
                    response_pane.add_assistant(msg.content)

    def on_model_panel_role_assigned(self, message: ModelPanel.RoleAssigned) -> None:
        """Forward model assignment to GarageHeader."""
        if message.role == "text" and message.model_name:
            header = self.query_one(GarageHeader)
            header.active_model = message.model_name

    async def action_new_session(self) -> None:
        """Start a new session."""
        self.query_one(ResponsePane).clear()
        self.notify("New session started")

    def action_tab_models(self) -> None:
        self.query_one("#control-tabs", TabbedContent).active = "tab-models"

    def action_tab_index(self) -> None:
        self.query_one("#control-tabs", TabbedContent).active = "tab-index"

    def action_tab_axioms(self) -> None:
        self.query_one("#control-tabs", TabbedContent).active = "tab-axioms"

    def action_tab_log(self) -> None:
        self.query_one("#control-tabs", TabbedContent).active = "tab-log"

    def action_tab_due(self) -> None:
        self.query_one("#control-tabs", TabbedContent).active = "tab-due"

    def action_tab_sessions(self) -> None:
        self.query_one("#control-tabs", TabbedContent).active = "tab-sessions"

    def action_tab_vault(self) -> None:
        self.query_one("#control-tabs", TabbedContent).active = "tab-vault"

    def action_tab_settings(self) -> None:
        self.query_one("#control-tabs", TabbedContent).active = "tab-settings"

    def action_reindex(self) -> None:
        """Switch to Index tab (user can click Reindex button)."""
        tabs = self.query_one("#control-tabs", TabbedContent)
        tabs.active = "tab-index"

    async def on_contradiction_badge_research_request(self, message) -> None:
        """Route badge click to input pane for R3 research."""
        from .ui.widgets.input_pane import InputPane

        input_pane = self.query_one(InputPane)
        response_pane = self.query_one(ResponsePane)
        response_pane.add_system("Starting R3 research on detected contradiction...")
        await input_pane._handle_research(message.query, response_pane)

    def on_input_pane_project_switched(self, message: InputPane.ProjectSwitched) -> None:
        """Broadcast state change to all panels after /project switch."""
        new_state = message.new_state
        self.state = new_state
        # Update app-level state so _graceful_shutdown() saves correct session
        self.app.state = new_state

        # Update all state-holding widgets
        state_widget_types = (
            GarageHeader, ModelPanel, IndexPanel,
            AxiomPanel, HistoryPanel, SessionPanel, VaultPanel, SettingsPanel,
        )
        for wtype in state_widget_types:
            try:
                widget = self.query_one(wtype)
            except NoMatches:
                continue
            try:
                widget.state = new_state
                if hasattr(widget, "on_state_updated"):
                    widget.on_state_updated()
            except Exception:
                logger.exception(
                    "Failed to update %s after project switch", wtype.__name__
                )

        # ResponsePane: update state ref only — /project handler manages display
        try:
            self.query_one(ResponsePane).state = new_state
        except NoMatches:
            pass

        # Special case: MaintenancePanel stores project_path, not state
        try:
            maint = self.query_one(MaintenancePanel)
            maint.set_project_path(new_state.project_path)
        except NoMatches:
            pass
        except Exception:
            logger.exception("Failed to update MaintenancePanel after project switch")


class R3LayApp(App):
    """Main r3LAY TUI application."""

    CSS_PATH = "ui/styles/garage.tcss"
    TITLE = "r³LAY"
    SUB_TITLE = "Garage Terminal"

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", show=True),
        Binding("ctrl+d", "toggle_dark", "Dark Mode"),
    ]

    def __init__(self, project_path: Path | None = None, show_splash: bool = True):
        super().__init__()
        self.project_path = project_path or Path.cwd()
        self.config = AppConfig.load(self.project_path)
        self.state: R3LayState | None = None
        self._show_splash = show_splash

    async def on_mount(self) -> None:
        """Called when app is mounted."""
        self.state = R3LayState(self.project_path)
        # Wire config model roles to state for model switching (Phase 5.5)
        self.state.model_roles = self.config.model_roles

        # Auto-restore last session if configured
        if self.config.auto_save_session:
            self._auto_restore_session()

        # Show splash screen first, then main screen
        if self._show_splash:
            await self.push_screen(MainScreen(self.state))
            # Push splash on top - it will dismiss itself after animation
            await self.push_screen(SplashScreen(run_animation=True, duration=3.0))
        else:
            await self.push_screen(MainScreen(self.state))

        # Register signal handlers for graceful shutdown
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(
                sig,
                lambda: asyncio.create_task(self._graceful_shutdown()),
            )

        # Auto-init embedders if configured (non-blocking)
        asyncio.create_task(self._auto_init_embedders())

    def _auto_restore_session(self) -> None:
        """Auto-restore the most recent session on startup."""
        import json as _json

        try:
            sessions_dir = self.state.get_sessions_dir()
            pointer_file = sessions_dir / "last_session.json"
            if not pointer_file.exists():
                return

            import re

            pointer_data = _json.loads(pointer_file.read_text())
            session_id = pointer_data.get("session_id")
            if not session_id:
                return

            # Validate UUID format to prevent path traversal
            uuid_re = re.compile(
                r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
            )
            if not uuid_re.match(session_id):
                logger.warning("Invalid session_id in last_session.json, ignoring")
                return

            session_file = sessions_dir / f"{session_id}.json"
            if not session_file.exists():
                return

            from .core.session import Session

            session = Session.load(session_file)
            self.state.session = session
            session._dirty = False
            logger.info("Auto-restored session %s: %s", session.id, session.title)
        except Exception as e:
            logger.warning("Failed to auto-restore session: %s", e)

    async def _auto_init_embedders(self) -> None:
        """Auto-initialize configured embedders after UI renders.

        Checks config.model_roles for configured embedders and initializes
        them in the background. Once loaded, attaches to the hybrid index
        so hybrid search is available immediately. Non-blocking to avoid
        delaying startup.
        """
        try:
            roles = self.config.model_roles

            # Init text embedder if configured
            if roles.has_text_embedder():
                embedder = await self.state.init_embedder()
                if embedder:
                    logger.info(f"Auto-loaded text embedder: {roles.text_embedder}")
                    # Attach to index so hybrid search works without manual reindex
                    self.state.init_index(with_embedder=True)
                    logger.info("Hybrid search enabled — embedder attached to index")

            # Init vision embedder if configured
            if roles.has_vision_embedder():
                vision = await self.state.init_vision_embedder(model_name=roles.vision_embedder)
                if vision:
                    logger.info(f"Auto-loaded vision embedder: {roles.vision_embedder}")

            # Init reranker if configured
            if roles.has_reranker():
                try:
                    from r3lay.core.reranker import CrossEncoderReranker

                    reranker = CrossEncoderReranker(model_name=roles.reranker)
                    await reranker.load()
                    idx = self.state.init_index()
                    idx.reranker = reranker
                    logger.info(f"Auto-loaded reranker: {roles.reranker}")
                except Exception as e:
                    logger.warning(f"Failed to auto-load reranker: {e}")

        except Exception as e:
            logger.warning(f"Failed to auto-init embedders: {e}")

    async def _graceful_shutdown(self) -> None:
        """Clean up backends and embedders before exit."""
        if hasattr(self, "state") and self.state is not None:
            # Auto-save session if configured and dirty
            if self.config.auto_save_session and self.state.session is not None:
                if self.state.session.has_unsaved_changes:
                    try:
                        import json as _json

                        sessions_dir = self.state.get_sessions_dir()
                        self.state.session.save(sessions_dir)
                        pointer = sessions_dir / "last_session.json"
                        temp_pointer = pointer.with_suffix(".json.tmp")
                        temp_pointer.write_text(
                            _json.dumps({"session_id": self.state.session.id})
                        )
                        temp_pointer.replace(pointer)
                        logger.info("Auto-saved session %s", self.state.session.id)
                    except Exception as e:
                        logger.warning("Failed to auto-save session: %s", e)

            try:
                # Unload LLM backend
                if hasattr(self.state, "current_backend") and self.state.current_backend:
                    await asyncio.wait_for(
                        self.state.unload_model(),
                        timeout=5.0,
                    )

                # Unload text embedder
                if self.state.text_embedder is not None:
                    await asyncio.wait_for(
                        self.state.unload_embedder(),
                        timeout=5.0,
                    )

                # Unload vision embedder
                if self.state.vision_embedder is not None:
                    await asyncio.wait_for(
                        self.state.unload_vision_embedder(),
                        timeout=5.0,
                    )

                # Unload reranker if attached to index
                if (
                    hasattr(self.state, "index")
                    and self.state.index is not None
                    and self.state.index.reranker is not None
                ):
                    await asyncio.wait_for(
                        self.state.index.reranker.unload(),
                        timeout=5.0,
                    )

            except Exception:
                pass

        self.exit()

    async def action_quit(self) -> None:
        """Override quit to ensure cleanup."""
        await self._graceful_shutdown()

    def action_toggle_dark(self) -> None:
        """Toggle dark mode."""
        self.theme = "textual-light" if self.theme == "textual-dark" else "textual-dark"


def main():
    """Entry point for the application.

    Supports both CLI subcommands (log, mileage, status, history) and TUI mode.
    If no subcommand is given, launches the TUI.
    """
    import sys

    from .cli import run_cli

    # Check if we have a CLI subcommand
    cli_commands = {"log", "mileage", "status", "history", "--help", "-h"}
    if len(sys.argv) > 1 and sys.argv[1] in cli_commands:
        result = run_cli()
        if result is not None:
            sys.exit(result)
        # result is None means launch TUI (shouldn't happen with these commands)

    # No CLI command = TUI mode
    import argparse

    parser = argparse.ArgumentParser(description="r3LAY Research Assistant")
    parser.add_argument(
        "project_path",
        nargs="?",
        default=".",
        help="Path to project directory",
    )
    parser.add_argument(
        "--no-splash",
        action="store_true",
        help="Skip the startup splash animation",
    )
    args = parser.parse_args()

    project_path = Path(args.project_path).resolve()
    app = R3LayApp(project_path, show_splash=not args.no_splash)
    app.run()


if __name__ == "__main__":
    main()
