"""Response pane - main content display area."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.markdown import Markdown
from rich.syntax import Syntax
from textual.app import ComposeResult
from textual.containers import ScrollableContainer
from textual.widgets import Static

if TYPE_CHECKING:
    from ...app import R3LayState


class ResponseBlock(Static):
    """A single response block."""

    DEFAULT_CSS = """
    ResponseBlock {
        width: 100%;
        padding: 1 2;
        margin: 0 0 1 0;
    }

    ResponseBlock.user {
        background: $surface;
        border-left: thick $primary;
    }

    ResponseBlock.assistant {
        background: $surface-darken-1;
        border-left: thick $secondary;
    }

    ResponseBlock.system {
        background: $surface-darken-2;
        border-left: thick $warning;
        color: $text-muted;
    }

    ResponseBlock.code {
        background: $surface-darken-3;
        border-left: thick $success;
    }

    ResponseBlock.error {
        background: $error-darken-2;
        border-left: thick $error;
    }

    .response-header {
        text-style: bold;
        margin-bottom: 1;
    }
    """

    def __init__(self, role: str, content: str, language: str | None = None):
        super().__init__()
        self.role = role
        self.content = content
        self.language = language
        self.add_class(role)

    def compose(self) -> ComposeResult:
        labels = {
            "user": "▸ You",
            "assistant": "◂ r3LAY",
            "system": "◈ System",
            "code": "⚙ Code",
            "error": "⚠ Error",
        }
        yield Static(labels.get(self.role, self.role), classes="response-header")

        if self.role == "code" and self.language:
            yield Static(Syntax(self.content, self.language, theme="monokai"))
        elif self.role in ("assistant", "system"):
            yield Static(Markdown(self.content))
        else:
            yield Static(self.content)


class ResponsePane(ScrollableContainer):
    """Main response/output pane."""

    DEFAULT_CSS = """
    ResponsePane {
        width: 60%;
        height: 100%;
        border: solid $primary-darken-2;
        border-title-color: $primary;
        padding: 1;
    }
    """

    BORDER_TITLE = "Responses"

    def __init__(self, state: "R3LayState", **kwargs):
        super().__init__(**kwargs)
        self.state = state
        self._blocks: list[ResponseBlock] = []

    def add_user(self, content: str) -> None:
        self._add_block("user", content)
        self.state.session_manager.add_message("user", content)

    def add_assistant(self, content: str) -> None:
        self._add_block("assistant", content)
        self.state.session_manager.add_message("assistant", content)

    def add_system(self, content: str) -> None:
        self._add_block("system", content)

    def add_code(self, content: str, language: str = "python") -> None:
        self._add_block("code", content, language=language)

    def add_error(self, content: str) -> None:
        self._add_block("error", content)

    def _add_block(self, role: str, content: str, language: str | None = None) -> None:
        block = ResponseBlock(role, content, language)
        self._blocks.append(block)
        self.mount(block)
        block.scroll_visible()

    def clear(self) -> None:
        for block in self._blocks:
            block.remove()
        self._blocks = []
