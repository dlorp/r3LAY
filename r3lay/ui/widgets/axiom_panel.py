"""Axiom panel - validated knowledge display."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

if TYPE_CHECKING:
    from ...core import R3LayState


class AxiomPanel(Vertical):
    """Panel for displaying validated axioms."""

    DEFAULT_CSS = """
    AxiomPanel {
        width: 100%;
        height: 100%;
        padding: 1;
    }

    #axiom-status {
        height: auto;
        margin-bottom: 1;
    }

    #axiom-list {
        height: 1fr;
        padding: 1;
        background: $surface-darken-1;
    }
    """

    def __init__(self, state: "R3LayState"):
        super().__init__()
        self.state = state

    def compose(self) -> ComposeResult:
        yield Static("No axioms", id="axiom-status")
        yield Static("Use `/axiom <statement>` to add validated knowledge.", id="axiom-list")


__all__ = ["AxiomPanel"]
