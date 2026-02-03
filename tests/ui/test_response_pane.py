"""Tests for r3lay.ui.widgets.response_pane module.

Tests cover:
- ResponseBlock: initialization, styling, content rendering
- StreamingBlock: streaming state, append, finish, clear
- ResponsePane: adding blocks, streaming, clearing
- Role-based styling and labels
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from r3lay.ui.widgets.response_pane import ResponseBlock, ResponsePane, StreamingBlock

# =============================================================================
# ResponseBlock Tests
# =============================================================================


class TestResponseBlockInit:
    """Tests for ResponseBlock initialization."""

    def test_creation_user_role(self) -> None:
        """Test creating a ResponseBlock with user role."""
        block = ResponseBlock(role="user", content="Hello world")

        assert block.role == "user"
        assert block.content == "Hello world"
        assert block.language is None
        assert "user" in block.classes

    def test_creation_assistant_role(self) -> None:
        """Test creating a ResponseBlock with assistant role."""
        block = ResponseBlock(role="assistant", content="I can help you.")

        assert block.role == "assistant"
        assert block.content == "I can help you."
        assert "assistant" in block.classes

    def test_creation_system_role(self) -> None:
        """Test creating a ResponseBlock with system role."""
        block = ResponseBlock(role="system", content="System message")

        assert block.role == "system"
        assert "system" in block.classes

    def test_creation_code_role_with_language(self) -> None:
        """Test creating a code block with language."""
        block = ResponseBlock(role="code", content="print('hello')", language="python")

        assert block.role == "code"
        assert block.content == "print('hello')"
        assert block.language == "python"
        assert "code" in block.classes

    def test_creation_error_role(self) -> None:
        """Test creating an error block."""
        block = ResponseBlock(role="error", content="Something went wrong")

        assert block.role == "error"
        assert "error" in block.classes


class TestResponseBlockCompose:
    """Tests for ResponseBlock compose method."""

    def test_compose_yields_header_and_content(self) -> None:
        """Test that compose yields header and content widgets."""
        block = ResponseBlock(role="user", content="Test content")
        widgets = list(block.compose())

        assert len(widgets) == 2
        # First widget is header
        assert "response-header" in widgets[0].classes
        # Second widget is content
        assert widgets[1] is not None

    def test_compose_header_is_static(self) -> None:
        """Test header widget is a Static."""
        block = ResponseBlock(role="user", content="Test")
        widgets = list(block.compose())

        from textual.widgets import Static

        header = widgets[0]
        assert isinstance(header, Static)

    def test_compose_content_is_static(self) -> None:
        """Test content widget is a Static."""
        block = ResponseBlock(role="assistant", content="Test")
        widgets = list(block.compose())

        from textual.widgets import Static

        assert isinstance(widgets[1], Static)

    def test_compose_code_block_yields_two_widgets(self) -> None:
        """Test code role yields header and content widgets."""
        block = ResponseBlock(role="code", content="def foo(): pass", language="python")
        widgets = list(block.compose())

        assert len(widgets) == 2

    def test_compose_all_roles_yield_two_widgets(self) -> None:
        """Test all roles yield exactly two widgets."""
        for role in ["user", "assistant", "system", "code", "error", "custom"]:
            block = ResponseBlock(role=role, content="Test")
            widgets = list(block.compose())
            assert len(widgets) == 2, f"Role {role} should yield 2 widgets"


class TestResponseBlockCSS:
    """Tests for ResponseBlock CSS styling."""

    def test_has_default_css(self) -> None:
        """Test that ResponseBlock has DEFAULT_CSS defined."""
        assert ResponseBlock.DEFAULT_CSS is not None
        assert "ResponseBlock" in ResponseBlock.DEFAULT_CSS

    def test_css_includes_role_classes(self) -> None:
        """Test CSS includes all role classes."""
        css = ResponseBlock.DEFAULT_CSS
        assert "ResponseBlock.user" in css
        assert "ResponseBlock.assistant" in css
        assert "ResponseBlock.system" in css
        assert "ResponseBlock.code" in css
        assert "ResponseBlock.error" in css

    def test_css_includes_header_class(self) -> None:
        """Test CSS includes response-header class."""
        assert ".response-header" in ResponseBlock.DEFAULT_CSS


# =============================================================================
# StreamingBlock Tests
# =============================================================================


class TestStreamingBlockInit:
    """Tests for StreamingBlock initialization."""

    def test_creation(self) -> None:
        """Test creating a StreamingBlock."""
        block = StreamingBlock()

        assert block._buffer == ""
        assert block._content_widget is None
        assert block._is_streaming is True
        assert "assistant" in block.classes
        assert "streaming" in block.classes

    def test_content_property_returns_buffer(self) -> None:
        """Test content property returns buffer."""
        block = StreamingBlock()
        block._buffer = "test content"

        assert block.content == "test content"


class TestStreamingBlockCompose:
    """Tests for StreamingBlock compose method."""

    def test_compose_yields_header_and_content(self) -> None:
        """Test that compose yields header and content widgets."""
        block = StreamingBlock()
        widgets = list(block.compose())

        assert len(widgets) == 2

    def test_compose_sets_content_widget(self) -> None:
        """Test that compose sets _content_widget."""
        block = StreamingBlock()
        list(block.compose())  # Trigger compose

        # After compose, _content_widget should be set
        assert block._content_widget is not None
        assert "streaming-content" in block._content_widget.classes


class TestStreamingBlockAppend:
    """Tests for StreamingBlock append method."""

    def test_append_updates_buffer(self) -> None:
        """Test append adds text to buffer."""
        block = StreamingBlock()
        # Simulate compose
        list(block.compose())

        block.append("Hello ")
        assert block._buffer == "Hello "

        block.append("World")
        assert block._buffer == "Hello World"

    def test_append_noop_after_finish(self) -> None:
        """Test append does nothing after finish."""
        block = StreamingBlock()
        list(block.compose())

        block.append("First")
        block._is_streaming = False  # Simulate finish
        block.append("Second")

        assert block._buffer == "First"

    def test_append_updates_content_widget(self) -> None:
        """Test append updates the content widget."""
        block = StreamingBlock()
        list(block.compose())

        block.append("Test")

        # Content widget should have been updated
        assert block._content_widget is not None


class TestStreamingBlockFinish:
    """Tests for StreamingBlock finish method."""

    def test_finish_sets_is_streaming_false(self) -> None:
        """Test finish sets _is_streaming to False."""
        block = StreamingBlock()
        list(block.compose())

        block._is_streaming = True
        block.finish()

        assert block._is_streaming is False

    def test_finish_removes_streaming_class(self) -> None:
        """Test finish removes streaming class."""
        block = StreamingBlock()
        list(block.compose())

        assert "streaming" in block.classes
        block.finish()
        assert "streaming" not in block.classes


class TestStreamingBlockClear:
    """Tests for StreamingBlock clear method."""

    def test_clear_resets_buffer(self) -> None:
        """Test clear resets buffer to empty string."""
        block = StreamingBlock()
        list(block.compose())

        block.append("Some content")
        block.clear()

        assert block._buffer == ""

    def test_clear_updates_widget(self) -> None:
        """Test clear updates the content widget."""
        block = StreamingBlock()
        list(block.compose())

        block.append("Some content")
        block.clear()

        # Widget should be cleared
        assert block._content_widget is not None


class TestStreamingBlockCSS:
    """Tests for StreamingBlock CSS styling."""

    def test_has_default_css(self) -> None:
        """Test that StreamingBlock has DEFAULT_CSS defined."""
        assert StreamingBlock.DEFAULT_CSS is not None
        assert "StreamingBlock" in StreamingBlock.DEFAULT_CSS

    def test_css_includes_streaming_classes(self) -> None:
        """Test CSS includes streaming-related classes."""
        css = StreamingBlock.DEFAULT_CSS
        assert ".streaming-header" in css
        assert ".streaming-content" in css


# =============================================================================
# ResponsePane Tests
# =============================================================================


@pytest.fixture
def mock_state(tmp_path: Path) -> MagicMock:
    """Create a mock R3LayState."""
    state = MagicMock()
    state.project_path = tmp_path
    return state


class TestResponsePaneInit:
    """Tests for ResponsePane initialization."""

    def test_creation(self, mock_state: MagicMock) -> None:
        """Test creating a ResponsePane."""
        pane = ResponsePane(state=mock_state)

        assert pane.state is mock_state
        assert pane._blocks == []
        assert pane._streaming_blocks == []
        assert pane._welcome_block is None


class TestResponsePaneAddMethods:
    """Tests for ResponsePane add methods."""

    def test_add_user_creates_user_block(self, mock_state: MagicMock) -> None:
        """Test add_user creates a user ResponseBlock."""
        pane = ResponsePane(state=mock_state)

        # Mock mount to avoid Textual runtime
        pane.mount = MagicMock()

        pane.add_user("User message")

        assert len(pane._blocks) == 1
        assert pane._blocks[0].role == "user"
        assert pane._blocks[0].content == "User message"
        pane.mount.assert_called_once()

    def test_add_assistant_creates_assistant_block(self, mock_state: MagicMock) -> None:
        """Test add_assistant creates an assistant ResponseBlock."""
        pane = ResponsePane(state=mock_state)
        pane.mount = MagicMock()

        pane.add_assistant("Assistant response")

        assert len(pane._blocks) == 1
        assert pane._blocks[0].role == "assistant"

    def test_add_system_creates_system_block(self, mock_state: MagicMock) -> None:
        """Test add_system creates a system ResponseBlock."""
        pane = ResponsePane(state=mock_state)
        pane.mount = MagicMock()

        pane.add_system("System notification")

        assert len(pane._blocks) == 1
        assert pane._blocks[0].role == "system"

    def test_add_code_creates_code_block(self, mock_state: MagicMock) -> None:
        """Test add_code creates a code ResponseBlock."""
        pane = ResponsePane(state=mock_state)
        pane.mount = MagicMock()

        pane.add_code("print('hi')", language="python")

        assert len(pane._blocks) == 1
        assert pane._blocks[0].role == "code"
        assert pane._blocks[0].language == "python"

    def test_add_code_default_language(self, mock_state: MagicMock) -> None:
        """Test add_code uses python as default language."""
        pane = ResponsePane(state=mock_state)
        pane.mount = MagicMock()

        pane.add_code("code here")

        assert pane._blocks[0].language == "python"

    def test_add_error_creates_error_block(self, mock_state: MagicMock) -> None:
        """Test add_error creates an error ResponseBlock."""
        pane = ResponsePane(state=mock_state)
        pane.mount = MagicMock()

        pane.add_error("Error message")

        assert len(pane._blocks) == 1
        assert pane._blocks[0].role == "error"

    def test_multiple_adds(self, mock_state: MagicMock) -> None:
        """Test adding multiple blocks."""
        pane = ResponsePane(state=mock_state)
        pane.mount = MagicMock()

        pane.add_user("Message 1")
        pane.add_assistant("Response 1")
        pane.add_user("Message 2")

        assert len(pane._blocks) == 3
        assert pane._blocks[0].role == "user"
        assert pane._blocks[1].role == "assistant"
        assert pane._blocks[2].role == "user"


class TestResponsePaneStreaming:
    """Tests for ResponsePane streaming methods."""

    def test_start_streaming_creates_block(self, mock_state: MagicMock) -> None:
        """Test start_streaming creates and returns a StreamingBlock."""
        pane = ResponsePane(state=mock_state)
        pane.mount = MagicMock()

        block = pane.start_streaming()

        assert isinstance(block, StreamingBlock)
        assert len(pane._streaming_blocks) == 1
        assert pane._streaming_blocks[0] is block
        pane.mount.assert_called_once()

    def test_multiple_streaming_blocks(self, mock_state: MagicMock) -> None:
        """Test creating multiple streaming blocks."""
        pane = ResponsePane(state=mock_state)
        pane.mount = MagicMock()

        block1 = pane.start_streaming()
        block2 = pane.start_streaming()

        assert len(pane._streaming_blocks) == 2
        assert block1 is not block2


class TestResponsePaneClear:
    """Tests for ResponsePane clear method."""

    def test_clear_removes_all_blocks(self, mock_state: MagicMock) -> None:
        """Test clear removes all response blocks."""
        pane = ResponsePane(state=mock_state)
        pane.mount = MagicMock()

        # Add some blocks
        pane.add_user("User 1")
        pane.add_assistant("Response 1")

        # Mock remove on blocks
        for block in pane._blocks:
            block.remove = MagicMock()

        pane.clear()

        assert pane._blocks == []

    def test_clear_removes_streaming_blocks(self, mock_state: MagicMock) -> None:
        """Test clear removes streaming blocks."""
        pane = ResponsePane(state=mock_state)
        pane.mount = MagicMock()

        block = pane.start_streaming()
        block.remove = MagicMock()

        pane.clear()

        assert pane._streaming_blocks == []


class TestResponsePaneCSS:
    """Tests for ResponsePane CSS styling."""

    def test_has_default_css(self) -> None:
        """Test that ResponsePane has DEFAULT_CSS defined."""
        assert ResponsePane.DEFAULT_CSS is not None
        assert "ResponsePane" in ResponsePane.DEFAULT_CSS

    def test_has_border_title(self) -> None:
        """Test ResponsePane has BORDER_TITLE."""
        assert ResponsePane.BORDER_TITLE == "Responses"


# =============================================================================
# Module Exports
# =============================================================================


class TestModuleExports:
    """Tests for module exports."""

    def test_all_exports(self) -> None:
        """Test __all__ exports."""
        from r3lay.ui.widgets import response_pane

        assert "ResponseBlock" in response_pane.__all__
        assert "StreamingBlock" in response_pane.__all__
        assert "ResponsePane" in response_pane.__all__
