"""Vault panel — git log viewer and rollback UI for the knowledge vault."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, DataTable, Label, Static

if TYPE_CHECKING:
    from ...core import R3LayState
    from ...core.vault import KnowledgeVault

logger = logging.getLogger(__name__)


class VaultPanel(Vertical):
    """Panel for knowledge vault git operations.

    Shows vault status, recent commit history, and provides
    pull/revert actions for easy rollback.
    """

    DEFAULT_CSS = """
    VaultPanel {
        width: 100%;
        height: 100%;
        padding: 1;
        overflow-y: auto;
    }

    VaultPanel .vault-section {
        margin-bottom: 1;
        padding: 1;
        border: solid $primary-lighten-1;
    }

    #vault-status {
        margin-bottom: 1;
    }

    #vault-actions {
        margin-bottom: 1;
    }

    #vault-actions Button {
        margin-right: 2;
    }

    #vault-log {
        height: auto;
        max-height: 20;
    }

    #vault-message {
        margin-top: 1;
        color: $text-muted;
    }
    """

    def __init__(self, state: "R3LayState") -> None:
        super().__init__()
        self.state = state
        self._selected_hash: str | None = None

    def compose(self) -> ComposeResult:
        # Status section
        yield Static("**Knowledge Vault**", id="vault-status", classes="vault-section")

        # Action buttons
        with Horizontal(id="vault-actions"):
            yield Button("Pull Latest", variant="primary", id="vault-pull")
            yield Button("Revert Selected", id="vault-revert")
            yield Button("Refresh", id="vault-refresh")

        # Commit log table
        with Vertical(classes="vault-section"):
            yield Label("Recent Commits:")
            table: DataTable = DataTable(id="vault-log")
            table.add_columns("Hash", "Date", "Message")
            table.cursor_type = "row"
            yield table

        # Status message
        yield Static("", id="vault-message")

    async def on_mount(self) -> None:
        """Load vault status on mount."""
        await self._refresh_all()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "vault-pull":
            self.run_worker(self._do_pull())
        elif event.button.id == "vault-revert":
            self.run_worker(self._do_revert())
        elif event.button.id == "vault-refresh":
            self.run_worker(self._refresh_all())

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Track selected commit hash."""
        table = self.query_one("#vault-log", DataTable)
        row_key = event.row_key
        row_data = table.get_row(row_key)
        if row_data:
            self._selected_hash = str(row_data[0])

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Track highlighted commit hash for keyboard navigation."""
        table = self.query_one("#vault-log", DataTable)
        row_key = event.row_key
        try:
            row_data = table.get_row(row_key)
            if row_data:
                self._selected_hash = str(row_data[0])
        except (KeyError, IndexError):
            pass  # Row may have been cleared during refresh

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    async def _refresh_all(self) -> None:
        """Refresh vault status and commit log."""
        vault = self.state.init_vault()
        status_widget = self.query_one("#vault-status", Static)

        if vault is None:
            vault_path = self.state.config.knowledge_vault_path
            if vault_path:
                status_widget.update(
                    f"**Knowledge Vault**\nPath: {vault_path}\nStatus: Directory not found"
                )
            else:
                status_widget.update(
                    "**Knowledge Vault**\nNo vault configured. Set the path in Settings."
                )
            self._clear_log()
            return

        info = await vault.status()
        branch = info.get("branch", "unknown")
        clean = "clean" if info.get("clean") else "uncommitted changes"
        last_pull = info.get("last_pull", "never")

        if info["is_repo"]:
            status_widget.update(
                f"**Knowledge Vault**\n"
                f"Path: {vault.path}\n"
                f"Branch: {branch} ({clean})\n"
                f"Last pull: {last_pull}"
            )
            await self._load_log(vault)
        else:
            status_widget.update(
                f"**Knowledge Vault**\nPath: {vault.path}\nStatus: Not a git repository"
            )
            self._clear_log()

    async def _load_log(self, vault: "KnowledgeVault") -> None:
        """Load commit log into the table."""
        table = self.query_one("#vault-log", DataTable)
        table.clear()
        self._selected_hash = None

        commits = await vault.log(limit=20)
        for commit in commits:
            # Truncate date to just date + time
            date_short = commit.date[:16] if len(commit.date) > 16 else commit.date
            table.add_row(commit.short_hash, date_short, commit.message[:60])

    def _clear_log(self) -> None:
        """Clear the commit log table."""
        table = self.query_one("#vault-log", DataTable)
        table.clear()
        self._selected_hash = None

    async def _do_pull(self) -> None:
        """Pull latest vault changes."""
        vault = self.state.init_vault()
        msg_widget = self.query_one("#vault-message", Static)

        if vault is None:
            msg_widget.update("No vault configured")
            return

        if not await vault.is_git_repo():
            msg_widget.update("Vault is not a git repository")
            return

        msg_widget.update("Pulling...")
        success, message = await vault.pull()

        if success:
            msg_widget.update(f"Pull: {message}")
            self.notify("Vault updated")
        else:
            msg_widget.update(f"Pull failed: {message}")
            self.notify(f"Pull failed: {message}", severity="error")

        await self._refresh_all()

    async def _do_revert(self) -> None:
        """Revert the selected commit (requires vault write permission)."""
        vault = self.state.init_vault()
        msg_widget = self.query_one("#vault-message", Static)

        if vault is None:
            msg_widget.update("No vault configured")
            return

        if not self._selected_hash:
            msg_widget.update("Select a commit to revert")
            self.notify("Select a commit first", severity="warning")
            return

        commit_hash = self._selected_hash

        # Show diff stat as preview
        diff = await vault.diff_stat(commit_hash)
        msg_widget.update(f"Reverting {commit_hash}...\n{diff}")

        success, message = await vault.revert(commit_hash)

        if success:
            msg_widget.update(f"Reverted {commit_hash}")
            self.notify(f"Reverted {commit_hash}")
        else:
            msg_widget.update(f"Revert failed: {message}")
            self.notify(f"Revert failed: {message}", severity="error")

        await self._refresh_all()


__all__ = ["VaultPanel"]
