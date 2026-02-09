"""Response pane - main content display area."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.markdown import Markdown
from rich.syntax import Syntax
from textual.app import ComposeResult
from textual.containers import ScrollableContainer, Vertical
from textual.widgets import Static

if TYPE_CHECKING:
    from ...core import R3LayState


class ResponseBlock(Vertical):
    """A single response block - container for header + content."""

    DEFAULT_CSS = """
    ResponseBlock {
        width: 100%;
        height: auto;
        padding: 1 2;
        margin: 1 0;
        border: round #636764;
        background: #0d0d0d;
    }

    ResponseBlock.user {
        background: #0d0d0d;
        border-left: outer #FB8B24;
    }

    ResponseBlock.assistant {
        background: #0d0d0d;
        border-left: outer #FB8B24;
    }

    ResponseBlock.system {
        background: #0d0d0d;
        border-left: outer #50D8D7;
        color: #636764;
    }

    ResponseBlock.code {
        background: #0d0d0d;
        border-left: outer #3B60E4;
    }

    ResponseBlock.error {
        background: #0d0d0d;
        border-left: outer #FB8B24;
        color: #FB8B24;
    }

    .response-header {
        text-style: bold;
        color: #FB8B24;
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
            "assistant": "▸ r3LAY",
            "system": "▸ r3LAY",
            "code": "▸ Code",
            "error": "▸ Error",
        }
        yield Static(labels.get(self.role, self.role), classes="response-header")

        if self.role == "code" and self.language:
            yield Static(Syntax(self.content, self.language, theme="monokai"))
        elif self.role in ("assistant", "system"):
            yield Static(Markdown(self.content))
        else:
            yield Static(self.content)


class StreamingBlock(Vertical):
    """A streaming response block that updates incrementally.

    Used for displaying LLM responses as they are generated token by token.
    During streaming, content is rendered as plain text for performance.
    On finish(), content is re-rendered with full Markdown formatting.

    Usage:
        block = response_pane.start_streaming()
        async for token in llm.generate():
            block.append(token)
        block.finish()
    """

    DEFAULT_CSS = """
    StreamingBlock {
        width: 100%;
        height: auto;
        padding: 1 2;
        margin: 0 0 1 0;
        background: $surface-darken-1;
        border-left: thick $secondary;
    }

    StreamingBlock .streaming-header {
        text-style: bold;
        margin-bottom: 1;
    }

    StreamingBlock .streaming-content {
        width: 100%;
    }

    /* Note: ::after pseudo-elements not supported in Textual CSS */
    """

    def __init__(self) -> None:
        """Initialize a new streaming block."""
        super().__init__()
        self._buffer: str = ""
        self._content_widget: Static | None = None
        self._is_streaming: bool = True
        self.add_class("assistant")
        self.add_class("streaming")

    def compose(self) -> ComposeResult:
        """Compose the streaming block with header and content area."""
        yield Static("< r3LAY", classes="streaming-header")
        self._content_widget = Static("", classes="streaming-content")
        yield self._content_widget

    @property
    def content(self) -> str:
        """Get the current buffer content."""
        return self._buffer

    def append(self, text: str) -> None:
        """Append text to the streaming buffer and update display.

        Args:
            text: Text chunk to append (typically a token or word).
        """
        if not self._is_streaming:
            return

        self._buffer += text

        # Update display with plain text during streaming for performance
        if self._content_widget is not None:
            self._content_widget.update(self._buffer)

        # Keep the new content visible
        self.scroll_visible()

    def finish(self) -> None:
        """Finalize the streaming block with full Markdown rendering.

        Call this when the LLM has finished generating the response.
        The content will be re-rendered with proper Markdown formatting.
        """
        self._is_streaming = False
        self.remove_class("streaming")

        # Re-render with full Markdown formatting
        if self._content_widget is not None:
            self._content_widget.update(Markdown(self._buffer))

        self.scroll_visible()

    def clear(self) -> None:
        """Clear the buffer and display."""
        self._buffer = ""
        if self._content_widget is not None:
            self._content_widget.update("")


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
        self._streaming_blocks: list[StreamingBlock] = []
        self._welcome_block: ResponseBlock | None = None

    async def on_mount(self) -> None:
        """Show dynamic welcome message on mount."""
        # Guard: only create welcome block once
        if self._welcome_block is not None:
            return

        from ...core.welcome import get_welcome_message

        welcome_text = get_welcome_message(self.state)
        self._welcome_block = ResponseBlock("system", welcome_text)
        self._welcome_block.id = "welcome-block"
        self._blocks.append(self._welcome_block)
        self.mount(self._welcome_block)

    def refresh_welcome(self) -> None:
        """Refresh the welcome message with current state."""
        from ...core.welcome import get_welcome_message

        # Generate new welcome text
        welcome_text = get_welcome_message(self.state)

        # Remove ALL existing welcome blocks (by ID and reference)
        to_remove = []
        for child in self.children:
            if getattr(child, "id", None) == "welcome-block":
                to_remove.append(child)
        for child in to_remove:
            try:
                child.remove()
            except Exception:
                pass

        # Also clean up _blocks list
        if self._welcome_block in self._blocks:
            self._blocks.remove(self._welcome_block)

        # Create new welcome block
        self._welcome_block = ResponseBlock("system", welcome_text)
        self._welcome_block.id = "welcome-block"
        self._blocks.insert(0, self._welcome_block)

        # Mount at the beginning
        children = list(self.children)
        if children:
            self.mount(self._welcome_block, before=children[0])
        else:
            self.mount(self._welcome_block)

    def add_user(self, content: str) -> None:
        self._add_block("user", content)

    def add_assistant(self, content: str) -> None:
        self._add_block("assistant", content)

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

    def start_streaming(self) -> StreamingBlock:
        """Start a new streaming response block.

        Creates and mounts a StreamingBlock that can be incrementally updated
        as tokens arrive from the LLM.

        Returns:
            StreamingBlock: A block that can be updated with append() and
                finalized with finish().

        Example:
            block = response_pane.start_streaming()
            async for token in llm.generate_stream(messages):
                block.append(token)
            block.finish()
        """
        block = StreamingBlock()
        self._streaming_blocks.append(block)
        self.mount(block)
        block.scroll_visible()
        return block

    def clear(self) -> None:
        """Clear all response blocks from the pane."""
        for block in self._blocks:
            block.remove()
        self._blocks = []

        for block in self._streaming_blocks:
            block.remove()
        self._streaming_blocks = []


__all__ = ["ResponseBlock", "StreamingBlock", "ResponsePane"]
