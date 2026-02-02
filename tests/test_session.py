"""Comprehensive tests for r3lay.core.session module.

Tests cover:
- Message dataclass: creation, serialization, LLM format conversion
- Session dataclass: message management, title generation, token budgets
- Persistence: JSON serialization, file save/load, atomic writes
- System prompt generation: project context, source types, axioms
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from r3lay.core.session import Message, Session

# =============================================================================
# Message Tests
# =============================================================================


class TestMessage:
    """Tests for Message dataclass."""

    def test_message_creation_minimal(self):
        """Test creating a message with minimal required fields."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.images is None
        assert msg.model_used is None
        assert isinstance(msg.timestamp, datetime)
        assert msg.metadata == {}

    def test_message_creation_full(self):
        """Test creating a message with all fields."""
        images = [Path("/img/test.png")]
        timestamp = datetime(2026, 1, 31, 10, 30, 0)
        metadata = {"token_count": 150, "was_cancelled": False}

        msg = Message(
            role="assistant",
            content="Response text",
            images=images,
            model_used="llama3.2",
            timestamp=timestamp,
            metadata=metadata,
        )

        assert msg.role == "assistant"
        assert msg.content == "Response text"
        assert msg.images == images
        assert msg.model_used == "llama3.2"
        assert msg.timestamp == timestamp
        assert msg.metadata == metadata

    def test_message_roles(self):
        """Test all valid message roles."""
        for role in ["user", "assistant", "system"]:
            msg = Message(role=role, content="test")
            assert msg.role == role

    def test_to_llm_format(self):
        """Test conversion to LLM API format."""
        msg = Message(
            role="user",
            content="What is r3LAY?",
            images=[Path("/img/diagram.png")],
            model_used="ignored",
            metadata={"extra": "ignored"},
        )

        llm_fmt = msg.to_llm_format()

        assert llm_fmt == {"role": "user", "content": "What is r3LAY?"}
        assert "images" not in llm_fmt
        assert "model_used" not in llm_fmt
        assert "metadata" not in llm_fmt

    def test_to_dict_serialization(self):
        """Test serialization to dictionary."""
        timestamp = datetime(2026, 1, 31, 12, 0, 0)
        msg = Message(
            role="assistant",
            content="Here's the info",
            images=[Path("/a.png"), Path("/b.jpg")],
            model_used="qwen2.5",
            timestamp=timestamp,
            metadata={"tokens": 50},
        )

        data = msg.to_dict()

        assert data["role"] == "assistant"
        assert data["content"] == "Here's the info"
        assert data["images"] == ["/a.png", "/b.jpg"]
        assert data["model_used"] == "qwen2.5"
        assert data["timestamp"] == "2026-01-31T12:00:00"
        assert data["metadata"] == {"tokens": 50}

    def test_to_dict_none_images(self):
        """Test serialization with no images."""
        msg = Message(role="user", content="No images")
        data = msg.to_dict()
        assert data["images"] is None

    def test_from_dict_deserialization(self):
        """Test deserialization from dictionary."""
        data = {
            "role": "user",
            "content": "Test content",
            "images": ["/path/to/img.png"],
            "model_used": "test-model",
            "timestamp": "2026-01-31T15:30:00",
            "metadata": {"key": "value"},
        }

        msg = Message.from_dict(data)

        assert msg.role == "user"
        assert msg.content == "Test content"
        assert msg.images == [Path("/path/to/img.png")]
        assert msg.model_used == "test-model"
        assert msg.timestamp == datetime(2026, 1, 31, 15, 30, 0)
        assert msg.metadata == {"key": "value"}

    def test_from_dict_minimal(self):
        """Test deserialization with minimal fields."""
        data = {
            "role": "system",
            "content": "You are helpful",
            "timestamp": "2026-01-01T00:00:00",
        }

        msg = Message.from_dict(data)

        assert msg.role == "system"
        assert msg.content == "You are helpful"
        assert msg.images is None
        assert msg.model_used is None
        assert msg.metadata == {}

    def test_roundtrip_serialization(self):
        """Test that to_dict/from_dict roundtrip preserves data."""
        original = Message(
            role="assistant",
            content="Complex response",
            images=[Path("/x.png")],
            model_used="llama3.2:8b",
            timestamp=datetime(2026, 6, 15, 9, 45, 30),
            metadata={"cancelled": True, "reason": "timeout"},
        )

        data = original.to_dict()
        restored = Message.from_dict(data)

        assert restored.role == original.role
        assert restored.content == original.content
        assert restored.images == original.images
        assert restored.model_used == original.model_used
        assert restored.timestamp == original.timestamp
        assert restored.metadata == original.metadata


# =============================================================================
# Session Tests - Basic Operations
# =============================================================================


class TestSessionBasic:
    """Tests for basic Session operations."""

    def test_session_creation_defaults(self):
        """Test creating a session with defaults."""
        session = Session()

        assert session.id is not None
        assert len(session.id) == 36  # UUID format
        assert session.messages == []
        assert isinstance(session.created_at, datetime)
        assert isinstance(session.updated_at, datetime)
        assert session.title is None
        assert session.project_path is None

    def test_session_creation_custom(self):
        """Test creating a session with custom values."""
        created = datetime(2026, 1, 1, 0, 0, 0)
        session = Session(
            id="custom-id-123",
            messages=[],
            created_at=created,
            updated_at=created,
            title="Test Session",
            project_path=Path("/projects/test"),
        )

        assert session.id == "custom-id-123"
        assert session.title == "Test Session"
        assert session.project_path == Path("/projects/test")

    def test_add_user_message(self):
        """Test adding a user message."""
        session = Session()
        before_update = session.updated_at

        msg = session.add_user_message("Hello, r3LAY!")

        assert len(session.messages) == 1
        assert msg.role == "user"
        assert msg.content == "Hello, r3LAY!"
        assert msg.images is None
        assert session.updated_at >= before_update

    def test_add_user_message_with_images(self):
        """Test adding a user message with images."""
        session = Session()
        images = [Path("/img/photo.jpg"), Path("/img/diagram.png")]

        msg = session.add_user_message("Check these images", images=images)

        assert msg.images == images
        assert session.messages[0].images == images

    def test_add_assistant_message(self):
        """Test adding an assistant message."""
        session = Session()

        msg = session.add_assistant_message(
            content="Here's the answer",
            model="llama3.2:8b",
            metadata={"tokens": 100},
        )

        assert len(session.messages) == 1
        assert msg.role == "assistant"
        assert msg.content == "Here's the answer"
        assert msg.model_used == "llama3.2:8b"
        assert msg.metadata == {"tokens": 100}

    def test_add_system_message(self):
        """Test adding a system message."""
        session = Session()

        msg = session.add_system_message("You are a helpful assistant")

        assert len(session.messages) == 1
        assert msg.role == "system"
        assert msg.content == "You are a helpful assistant"

    def test_auto_title_from_first_message(self):
        """Test automatic title generation from first user message."""
        session = Session()
        assert session.title is None

        session.add_user_message("How do I change the oil in my Subaru?")

        assert session.title == "How do I change the oil in my Subaru?"

    def test_auto_title_truncation(self):
        """Test that auto-generated titles are truncated at 50 chars."""
        session = Session()
        long_message = "A" * 100

        session.add_user_message(long_message)

        assert session.title == "A" * 50 + "..."
        assert len(session.title) == 53

    def test_title_not_overwritten(self):
        """Test that title isn't overwritten by subsequent messages."""
        session = Session()
        session.add_user_message("First message")
        first_title = session.title

        session.add_user_message("Second message with different content")

        assert session.title == first_title

    def test_clear_session(self):
        """Test clearing all messages."""
        session = Session()
        session.add_user_message("Message 1")
        session.add_assistant_message("Response 1", model="test")
        session.add_user_message("Message 2")

        session.clear()

        assert session.messages == []
        assert session.title is not None  # Title preserved

    def test_get_last_user_images_found(self):
        """Test getting images from most recent user message."""
        session = Session()
        session.add_user_message("No images here")
        session.add_assistant_message("Response", model="test")
        session.add_user_message("With images", images=[Path("/a.png"), Path("/b.png")])
        session.add_assistant_message("Another response", model="test")

        images = session.get_last_user_images()

        assert images == [Path("/a.png"), Path("/b.png")]

    def test_get_last_user_images_none(self):
        """Test getting images when no user messages have images."""
        session = Session()
        session.add_user_message("No images")
        session.add_assistant_message("Response", model="test")

        images = session.get_last_user_images()

        assert images == []

    def test_get_last_user_images_empty_session(self):
        """Test getting images from empty session."""
        session = Session()
        assert session.get_last_user_images() == []


# =============================================================================
# Session Tests - LLM Formatting
# =============================================================================


class TestSessionLLMFormatting:
    """Tests for Session LLM formatting and token management."""

    def test_get_messages_for_llm_basic(self):
        """Test basic LLM message formatting."""
        session = Session()
        session.add_system_message("System prompt")
        session.add_user_message("User question")
        session.add_assistant_message("Assistant answer", model="test")

        messages = session.get_messages_for_llm()

        assert len(messages) == 3
        assert messages[0] == {"role": "system", "content": "System prompt"}
        assert messages[1] == {"role": "user", "content": "User question"}
        assert messages[2] == {"role": "assistant", "content": "Assistant answer"}

    def test_get_messages_for_llm_empty(self):
        """Test LLM formatting with empty session."""
        session = Session()
        assert session.get_messages_for_llm() == []

    def test_get_messages_for_llm_excludes_system(self):
        """Test excluding system messages."""
        session = Session()
        session.add_system_message("System")
        session.add_user_message("User")

        messages = session.get_messages_for_llm(include_system=False)

        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_get_messages_for_llm_token_truncation(self):
        """Test that old messages are truncated to fit token budget."""
        session = Session()
        session.add_system_message("Short system")

        # Add many messages that exceed budget
        for i in range(20):
            session.add_user_message(f"User message {i} " * 50)
            session.add_assistant_message(f"Response {i} " * 50, model="test")

        # Very small budget - should only get system + most recent
        messages = session.get_messages_for_llm(max_tokens=200)

        # Should have system message + some recent messages
        assert len(messages) < 41  # Less than all messages
        assert messages[0]["role"] == "system"
        # Most recent messages should be present
        assert "Response 19" in messages[-1]["content"]

    def test_get_messages_for_llm_preserves_system(self):
        """Test that system message is always preserved during truncation."""
        session = Session()
        session.add_system_message("Important system prompt")

        # Add messages to exceed budget
        for i in range(10):
            session.add_user_message(f"Message {i} " * 100)

        messages = session.get_messages_for_llm(max_tokens=500)

        # System message should be first
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Important system prompt"

    def test_get_messages_for_llm_chronological_order(self):
        """Test that messages maintain chronological order."""
        session = Session()
        session.add_user_message("First")
        session.add_assistant_message("Second", model="test")
        session.add_user_message("Third")
        session.add_assistant_message("Fourth", model="test")

        messages = session.get_messages_for_llm()

        contents = [m["content"] for m in messages]
        assert contents == ["First", "Second", "Third", "Fourth"]


# =============================================================================
# Session Tests - Serialization
# =============================================================================


class TestSessionSerialization:
    """Tests for Session serialization/deserialization."""

    def test_to_dict(self):
        """Test session serialization to dictionary."""
        session = Session(
            id="test-123",
            created_at=datetime(2026, 1, 31, 10, 0, 0),
            updated_at=datetime(2026, 1, 31, 11, 0, 0),
            title="Test Session",
            project_path=Path("/projects/car"),
        )
        session.add_user_message("Question")
        session.add_assistant_message("Answer", model="llama")

        data = session.to_dict()

        assert data["id"] == "test-123"
        assert data["created_at"] == "2026-01-31T10:00:00"
        # updated_at changes when messages are added, so just verify it's a valid timestamp
        assert "updated_at" in data
        assert data["title"] == "Test Session"
        assert data["project_path"] == "/projects/car"
        assert len(data["messages"]) == 2

    def test_to_dict_none_project_path(self):
        """Test serialization with no project path."""
        session = Session()
        data = session.to_dict()
        assert data["project_path"] is None

    def test_from_dict(self):
        """Test session deserialization from dictionary."""
        data = {
            "id": "restored-456",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello",
                    "images": None,
                    "model_used": None,
                    "timestamp": "2026-01-31T12:00:00",
                    "metadata": {},
                }
            ],
            "created_at": "2026-01-31T09:00:00",
            "updated_at": "2026-01-31T12:00:00",
            "title": "Restored Session",
            "project_path": "/my/project",
        }

        session = Session.from_dict(data)

        assert session.id == "restored-456"
        assert session.title == "Restored Session"
        assert session.project_path == Path("/my/project")
        assert len(session.messages) == 1
        assert session.messages[0].content == "Hello"

    def test_from_dict_minimal(self):
        """Test deserialization with minimal required fields."""
        data = {
            "id": "min-session",
            "messages": [],
            "created_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-01T00:00:00",
        }

        session = Session.from_dict(data)

        assert session.id == "min-session"
        assert session.messages == []
        assert session.title is None
        assert session.project_path is None

    def test_roundtrip_serialization(self):
        """Test that to_dict/from_dict roundtrip preserves session."""
        original = Session(
            id="roundtrip-test",
            title="Roundtrip Test",
            project_path=Path("/test/path"),
        )
        original.add_system_message("System")
        original.add_user_message("User", images=[Path("/img.png")])
        original.add_assistant_message("Assistant", model="test", metadata={"k": "v"})

        data = original.to_dict()
        restored = Session.from_dict(data)

        assert restored.id == original.id
        assert restored.title == original.title
        assert restored.project_path == original.project_path
        assert len(restored.messages) == len(original.messages)

        for orig_msg, rest_msg in zip(original.messages, restored.messages):
            assert rest_msg.role == orig_msg.role
            assert rest_msg.content == orig_msg.content
            assert rest_msg.model_used == orig_msg.model_used


# =============================================================================
# Session Tests - File Persistence
# =============================================================================


class TestSessionPersistence:
    """Tests for Session file save/load operations."""

    def test_save_creates_file(self):
        """Test that save creates a JSON file."""
        session = Session(id="save-test")
        session.add_user_message("Test message")

        with tempfile.TemporaryDirectory() as tmpdir:
            sessions_dir = Path(tmpdir) / "sessions"
            saved_path = session.save(sessions_dir)

            assert saved_path.exists()
            assert saved_path.name == "save-test.json"
            assert saved_path.parent == sessions_dir

    def test_save_creates_directory(self):
        """Test that save creates parent directories if needed."""
        session = Session(id="mkdir-test")

        with tempfile.TemporaryDirectory() as tmpdir:
            deep_path = Path(tmpdir) / "a" / "b" / "c" / "sessions"
            session.save(deep_path)

            assert deep_path.exists()
            assert (deep_path / "mkdir-test.json").exists()

    def test_save_valid_json(self):
        """Test that saved file contains valid JSON."""
        session = Session(id="json-test")
        session.add_user_message("Hello")
        session.add_assistant_message("Hi", model="test")

        with tempfile.TemporaryDirectory() as tmpdir:
            saved_path = session.save(Path(tmpdir))

            # Should be valid JSON
            data = json.loads(saved_path.read_text())
            assert data["id"] == "json-test"
            assert len(data["messages"]) == 2

    def test_save_overwrites_existing(self):
        """Test that save overwrites existing session file."""
        session = Session(id="overwrite-test")

        with tempfile.TemporaryDirectory() as tmpdir:
            sessions_dir = Path(tmpdir)

            # First save
            session.add_user_message("First version")
            session.save(sessions_dir)

            # Modify and save again
            session.add_user_message("Second version")
            session.save(sessions_dir)

            # Load and verify
            data = json.loads((sessions_dir / "overwrite-test.json").read_text())
            assert len(data["messages"]) == 2

    def test_load_existing_file(self):
        """Test loading a session from file."""
        session_data = {
            "id": "load-test",
            "messages": [
                {
                    "role": "user",
                    "content": "Loaded message",
                    "images": None,
                    "model_used": None,
                    "timestamp": "2026-01-31T10:00:00",
                    "metadata": {},
                }
            ],
            "created_at": "2026-01-31T09:00:00",
            "updated_at": "2026-01-31T10:00:00",
            "title": "Loaded Session",
            "project_path": None,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            session_file = Path(tmpdir) / "load-test.json"
            session_file.write_text(json.dumps(session_data))

            loaded = Session.load(session_file)

            assert loaded.id == "load-test"
            assert loaded.title == "Loaded Session"
            assert len(loaded.messages) == 1
            assert loaded.messages[0].content == "Loaded message"

    def test_load_nonexistent_file(self):
        """Test that loading nonexistent file raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_path = Path(tmpdir) / "nonexistent.json"

            with pytest.raises(FileNotFoundError):
                Session.load(fake_path)

    def test_load_invalid_json(self):
        """Test that loading invalid JSON raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_file = Path(tmpdir) / "bad.json"
            bad_file.write_text("not valid json {{{")

            with pytest.raises(ValueError, match="Invalid session file format"):
                Session.load(bad_file)

    def test_load_missing_required_field(self):
        """Test that loading file with missing fields raises ValueError."""
        incomplete_data = {"id": "incomplete"}  # Missing required fields

        with tempfile.TemporaryDirectory() as tmpdir:
            bad_file = Path(tmpdir) / "incomplete.json"
            bad_file.write_text(json.dumps(incomplete_data))

            with pytest.raises(ValueError, match="missing required field"):
                Session.load(bad_file)

    def test_save_load_roundtrip(self):
        """Test complete save/load roundtrip."""
        original = Session(
            id="roundtrip-file",
            title="File Roundtrip",
            project_path=Path("/roundtrip/project"),
        )
        original.add_system_message("System prompt")
        original.add_user_message("Question", images=[Path("/q.png")])
        original.add_assistant_message("Answer", model="llama", metadata={"t": 100})

        with tempfile.TemporaryDirectory() as tmpdir:
            saved_path = original.save(Path(tmpdir))
            loaded = Session.load(saved_path)

            assert loaded.id == original.id
            assert loaded.title == original.title
            assert loaded.project_path == original.project_path
            assert len(loaded.messages) == 3

            # Verify message details
            assert loaded.messages[0].role == "system"
            assert loaded.messages[1].images == [Path("/q.png")]
            assert loaded.messages[2].model_used == "llama"
            assert loaded.messages[2].metadata == {"t": 100}


# =============================================================================
# Session Tests - System Prompt Generation
# =============================================================================


class TestSessionSystemPrompt:
    """Tests for system prompt generation with citations."""

    def test_system_prompt_basic(self):
        """Test basic system prompt generation."""
        session = Session()
        prompt = session.get_system_prompt_with_citations()

        assert "r³LAY" in prompt
        assert "Source Trust Hierarchy" in prompt
        assert "Confidence Indicators" in prompt
        assert "Do Not" in prompt

    def test_system_prompt_with_automotive_context(self):
        """Test system prompt with automotive project context."""
        session = Session()

        # Create mock project context
        mock_ctx = MagicMock()
        mock_ctx.project_name = "outback"
        mock_ctx.project_type = "automotive"
        mock_ctx.project_reference = "the Outback"
        mock_ctx.vehicle_make = "Subaru"
        mock_ctx.vehicle_model = "Outback"
        mock_ctx.vehicle_year = "2020"
        mock_ctx.possessive = "the Outback's"

        prompt = session.get_system_prompt_with_citations(project_context=mock_ctx)

        assert "Active Project Context" in prompt
        assert "automotive" in prompt
        assert "Subaru" in prompt
        assert "Outback" in prompt
        assert "2020" in prompt

    def test_system_prompt_with_electronics_context(self):
        """Test system prompt with electronics project context."""
        session = Session()

        mock_ctx = MagicMock()
        mock_ctx.project_name = "sensor-board"
        mock_ctx.project_type = "electronics"
        mock_ctx.metadata = {"board": "STM32F4"}

        prompt = session.get_system_prompt_with_citations(project_context=mock_ctx)

        assert "electronics" in prompt
        assert "STM32F4" in prompt
        assert "datasheets" in prompt
        assert "pinouts" in prompt

    def test_system_prompt_with_software_context(self):
        """Test system prompt with software project context."""
        session = Session()

        mock_ctx = MagicMock()
        mock_ctx.project_name = "my-app"
        mock_ctx.project_type = "software"
        mock_ctx.metadata = {"language": "Python"}

        prompt = session.get_system_prompt_with_citations(project_context=mock_ctx)

        assert "software" in prompt
        assert "Python" in prompt
        assert "code snippets" in prompt
        assert "API versions" in prompt

    def test_system_prompt_with_workshop_context(self):
        """Test system prompt with workshop project context."""
        session = Session()

        mock_ctx = MagicMock()
        mock_ctx.project_name = "workbench"
        mock_ctx.project_type = "workshop"
        mock_ctx.metadata = {}

        prompt = session.get_system_prompt_with_citations(project_context=mock_ctx)

        assert "workshop" in prompt
        assert "materials" in prompt
        assert "safety notes" in prompt

    def test_system_prompt_with_home_context(self):
        """Test system prompt with home project context."""
        session = Session()

        mock_ctx = MagicMock()
        mock_ctx.project_name = "bathroom-reno"
        mock_ctx.project_type = "home"
        mock_ctx.metadata = {}

        prompt = session.get_system_prompt_with_citations(project_context=mock_ctx)

        assert "home" in prompt
        assert "building codes" in prompt
        assert "licensed work" in prompt

    def test_system_prompt_with_source_types(self):
        """Test system prompt with available source types."""
        session = Session()

        # Create mock source types
        local_source = MagicMock()
        local_source.is_local = True
        local_source.is_web = False
        local_source.value = "indexed_docs"

        web_source = MagicMock()
        web_source.is_local = False
        web_source.is_web = True
        web_source.value = "forum"

        prompt = session.get_system_prompt_with_citations(
            source_types_present=[local_source, web_source]
        )

        assert "Available Sources" in prompt
        assert "indexed_docs" in prompt
        assert "forum" in prompt

    def test_system_prompt_with_axiom_manager(self):
        """Test system prompt with axiom manager context."""
        session = Session()

        mock_axiom_mgr = MagicMock()
        mock_axiom_mgr.get_context_for_llm.return_value = (
            "## Validated Knowledge\n- Axiom 1: Test axiom"
        )

        prompt = session.get_system_prompt_with_citations(axiom_manager=mock_axiom_mgr)

        assert "Validated Knowledge" in prompt
        assert "Axiom 1" in prompt
        mock_axiom_mgr.get_context_for_llm.assert_called_once_with(max_axioms=15)

    def test_system_prompt_with_all_options(self):
        """Test system prompt with all options combined."""
        session = Session()

        mock_ctx = MagicMock()
        mock_ctx.project_name = "full-test"
        mock_ctx.project_type = "general"
        mock_ctx.metadata = {}

        local_source = MagicMock()
        local_source.is_local = True
        local_source.is_web = False
        local_source.value = "manual"

        mock_axiom_mgr = MagicMock()
        mock_axiom_mgr.get_context_for_llm.return_value = "## Axioms\n- Test"

        prompt = session.get_system_prompt_with_citations(
            project_context=mock_ctx,
            source_types_present=[local_source],
            axiom_manager=mock_axiom_mgr,
        )

        assert "r³LAY" in prompt
        assert "full-test" in prompt
        assert "manual" in prompt
        assert "Axioms" in prompt


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestSessionEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_content_message(self):
        """Test handling empty content in messages."""
        session = Session()
        msg = session.add_user_message("")

        assert msg.content == ""
        assert session.title is None  # Empty content doesn't set title

    def test_unicode_content(self):
        """Test handling unicode in messages."""
        session = Session()
        unicode_content = "Hello 世界 🚗 αβγ"

        msg = session.add_user_message(unicode_content)

        assert msg.content == unicode_content

        # Test serialization roundtrip
        data = session.to_dict()
        restored = Session.from_dict(data)
        assert restored.messages[0].content == unicode_content

    def test_large_message_content(self):
        """Test handling very large message content."""
        session = Session()
        large_content = "x" * 100000  # 100KB of text

        msg = session.add_user_message(large_content)

        assert len(msg.content) == 100000

    def test_many_messages(self):
        """Test handling sessions with many messages."""
        session = Session()

        for i in range(1000):
            session.add_user_message(f"Message {i}")
            session.add_assistant_message(f"Response {i}", model="test")

        assert len(session.messages) == 2000

        # Should still serialize/deserialize
        data = session.to_dict()
        restored = Session.from_dict(data)
        assert len(restored.messages) == 2000

    def test_special_characters_in_paths(self):
        """Test handling special characters in image paths."""
        session = Session()
        special_path = Path("/path/with spaces/and-dashes/file (1).png")

        msg = session.add_user_message("Image", images=[special_path])

        # Roundtrip test
        data = session.to_dict()
        restored = Session.from_dict(data)
        assert restored.messages[0].images == [special_path]

    def test_concurrent_timestamp_updates(self):
        """Test that timestamps are properly updated on each operation."""
        session = Session()

        ts1 = session.updated_at
        session.add_user_message("First")
        ts2 = session.updated_at

        session.add_assistant_message("Second", model="test")
        ts3 = session.updated_at

        session.clear()
        ts4 = session.updated_at

        # Each operation should update the timestamp
        assert ts2 >= ts1
        assert ts3 >= ts2
        assert ts4 >= ts3
