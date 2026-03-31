"""Vault panel — git log viewer and rollback UI for the knowledge vault."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.timer import Timer
from textual.widgets import Button, DataTable, Label, Static

if TYPE_CHECKING:
    from ...core import R3LayState
    from ...core.vault import KnowledgeVault

logger = logging.getLogger(__name__)


class VaultPanel(Vertical):
    """Panel for knowledge vault git operations.

    Shows vault status, recent commit history, and provides
    pull/revert actions for easy rollback. Both Pull and Revert
    use two-click confirmation with 5-second auto-reset.
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
        # Revert confirmation state
        self._revert_confirm: bool = False
        self._revert_confirm_hash: str | None = None
        self._revert_reset_timer: Timer | None = None
        # Pull confirmation state
        self._pull_confirm: bool = False
        self._pull_reset_timer: Timer | None = None

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

    def on_unmount(self) -> None:
        """Cancel pending confirmation timers on widget removal."""
        if self._revert_reset_timer is not None:
            self._revert_reset_timer.stop()
            self._revert_reset_timer = None
        if self._pull_reset_timer is not None:
            self._pull_reset_timer.stop()
            self._pull_reset_timer = None

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
                new_hash = str(row_data[0])
                if new_hash != self._selected_hash and self._revert_confirm:
                    self._reset_revert_confirm()
                self._selected_hash = new_hash
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
            date_short = commit.date[:16] if len(commit.date) > 16 else commit.date
            table.add_row(commit.short_hash, date_short, commit.message[:60])

    def _clear_log(self) -> None:
        """Clear the commit log table."""
        table = self.query_one("#vault-log", DataTable)
        table.clear()
        self._selected_hash = None

    # ------------------------------------------------------------------
    # Pull with two-click confirmation
    # ------------------------------------------------------------------

    async def _do_pull(self) -> None:
        """Pull latest vault changes with two-click confirmation."""
        msg_widget = self.query_one("#vault-message", Static)

        if not self._pull_confirm:
            # First click — show confirmation
            vault = self.state.init_vault()
            if vault is None:
                msg_widget.update("No vault configured")
                return
            if not await vault.is_git_repo():
                msg_widget.update("Vault is not a git repository")
                return

            try:
                pull_btn = self.query_one("#vault-pull", Button)
                pull_btn.label = "Confirm pull?"
            except Exception:
                pass
            self._pull_confirm = True

            if self._pull_reset_timer is not None:
                self._pull_reset_timer.stop()
            self._pull_reset_timer = self.set_timer(5, self._reset_pull_confirm)
            return

        # Second click — actually pull
        self._reset_pull_confirm()

        vault = self.state.init_vault()
        if vault is None:
            msg_widget.update("No vault configured")
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

    def _reset_pull_confirm(self) -> None:
        """Reset the pull confirmation state."""
        self._pull_confirm = False
        if self._pull_reset_timer is not None:
            self._pull_reset_timer.stop()
            self._pull_reset_timer = None
        try:
            pull_btn = self.query_one("#vault-pull", Button)
            pull_btn.label = "Pull Latest"
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Revert with two-click confirmation
    # ------------------------------------------------------------------

    async def _do_revert(self) -> None:
        """Revert the selected commit with two-click confirmation."""
        msg_widget = self.query_one("#vault-message", Static)

        if not self._selected_hash:
            msg_widget.update("Select a commit to revert")
            self.notify("Select a commit first", severity="warning")
            return

        try:
            revert_btn = self.query_one("#vault-revert", Button)
        except Exception:
            revert_btn = None

        if not self._revert_confirm:
            # First click — show confirmation with diff preview
            vault = self.state.init_vault()
            if vault is None:
                msg_widget.update("No vault configured")
                return

            diff = await vault.diff_stat(self._selected_hash)
            msg_widget.update(f"Preview:\n{diff}")
            if revert_btn:
                revert_btn.label = f"Confirm revert {self._selected_hash}?"
            self._revert_confirm = True
            self._revert_confirm_hash = self._selected_hash

            if self._revert_reset_timer is not None:
                self._revert_reset_timer.stop()
            self._revert_reset_timer = self.set_timer(5, self._reset_revert_confirm)
            return

        # Second click — actually revert
        if self._selected_hash != self._revert_confirm_hash:
            msg_widget.update("Selection changed — revert cancelled")
            self._reset_revert_confirm()
            return

        vault = self.state.init_vault()
        if vault is None:
            msg_widget.update("No vault configured")
            self._reset_revert_confirm()
            return

        commit_hash = self._selected_hash
        msg_widget.update(f"Reverting {commit_hash}...")

        success, message = await vault.revert(commit_hash)

        if success:
            msg_widget.update(f"Reverted {commit_hash}")
            self.notify(f"Reverted {commit_hash}")
        else:
            msg_widget.update(f"Revert failed: {message}")
            self.notify(f"Revert failed: {message}", severity="error")

        self._reset_revert_confirm()
        await self._refresh_all()

    def _reset_revert_confirm(self) -> None:
        """Reset the revert confirmation state."""
        self._revert_confirm = False
        if self._revert_reset_timer is not None:
            self._revert_reset_timer.stop()
            self._revert_reset_timer = None
        try:
            revert_btn = self.query_one("#vault-revert", Button)
            revert_btn.label = "Revert Selected"
        except Exception:
            pass


__all__ = ["VaultPanel"]
