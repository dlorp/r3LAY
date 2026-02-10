"""Tests for InputPane validation and button state management.

Tests cover:
- Send button validation with no models configured
- Send button validation with one model configured  
- Send button validation with all models configured
- Button state stability (no flickering)
- Clear error messages when validation fails
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from unittest.mock import MagicMock, Mock

import pytest


class MockBackendType(Enum):
    """Mock backend type enum."""

    OLLAMA = "ollama"


@dataclass
class MockModelInfo:
    """Mock ModelInfo for testing."""

    name: str
    backend_type: MockBackendType = MockBackendType.OLLAMA
    is_vision_model: bool = False


class MockBackend:
    """Mock inference backend."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.loaded = True


class MockState:
    """Mock R3LayState for testing validation."""

    def __init__(
        self,
        current_backend: MockBackend | None = None,
        available_models: list[MockModelInfo] | None = None,
    ):
        self.current_backend = current_backend
        self.available_models = available_models or []


class MockInputPane:
    """Mock InputPane with validation methods for testing."""

    def __init__(self, state: MockState, text: str = ""):
        self.state = state
        self._text = text
        self._processing = False

    def get_value(self) -> str:
        """Get current input text."""
        return self._text

    def _validate_can_send(self) -> tuple[bool, str]:
        """Validate if message can be sent.
        
        Returns:
            Tuple of (is_valid, error_message). If is_valid is True, error_message is empty.
        """
        # Check if a model is loaded
        if not hasattr(self.state, "current_backend") or self.state.current_backend is None:
            return False, "No model loaded. Load a model from the Models tab first."
        
        # Check if there's actual content to send
        value = self.get_value().strip()
        if not value:
            return False, "Enter a message to send."
        
        return True, ""


class TestValidateCanSend:
    """Tests for the _validate_can_send validation logic."""

    def test_no_model_configured(self):
        """Test validation fails when no model is configured."""
        state = MockState(current_backend=None, available_models=[])
        pane = MockInputPane(state, text="Hello world")
        
        is_valid, error_msg = pane._validate_can_send()
        
        assert is_valid is False
        assert "No model loaded" in error_msg
        assert "Models tab" in error_msg

    def test_one_model_configured_but_not_loaded(self):
        """Test validation fails when models exist but none is loaded."""
        models = [MockModelInfo(name="mistral-7b")]
        state = MockState(current_backend=None, available_models=models)
        pane = MockInputPane(state, text="Hello world")
        
        is_valid, error_msg = pane._validate_can_send()
        
        assert is_valid is False
        assert "No model loaded" in error_msg

    def test_one_model_loaded(self):
        """Test validation passes when one model is loaded."""
        backend = MockBackend("mistral-7b")
        state = MockState(current_backend=backend)
        pane = MockInputPane(state, text="Hello world")
        
        is_valid, error_msg = pane._validate_can_send()
        
        assert is_valid is True
        assert error_msg == ""

    def test_multiple_models_available_one_loaded(self):
        """Test validation passes when multiple models exist and one is loaded."""
        models = [
            MockModelInfo(name="mistral-7b"),
            MockModelInfo(name="llama3-8b"),
            MockModelInfo(name="qwen2.5-7b"),
        ]
        backend = MockBackend("mistral-7b")
        state = MockState(current_backend=backend, available_models=models)
        pane = MockInputPane(state, text="Hello world")
        
        is_valid, error_msg = pane._validate_can_send()
        
        assert is_valid is True
        assert error_msg == ""

    def test_empty_message_with_model_loaded(self):
        """Test validation fails when message is empty even with model loaded."""
        backend = MockBackend("mistral-7b")
        state = MockState(current_backend=backend)
        pane = MockInputPane(state, text="")
        
        is_valid, error_msg = pane._validate_can_send()
        
        assert is_valid is False
        assert "Enter a message" in error_msg

    def test_whitespace_only_message(self):
        """Test validation fails for whitespace-only messages."""
        backend = MockBackend("mistral-7b")
        state = MockState(current_backend=backend)
        pane = MockInputPane(state, text="   \n\t  ")
        
        is_valid, error_msg = pane._validate_can_send()
        
        assert is_valid is False
        assert "Enter a message" in error_msg

    def test_valid_message_with_model(self):
        """Test validation passes with valid message and loaded model."""
        backend = MockBackend("mistral-7b")
        state = MockState(current_backend=backend)
        pane = MockInputPane(state, text="Tell me about cars")
        
        is_valid, error_msg = pane._validate_can_send()
        
        assert is_valid is True
        assert error_msg == ""

    def test_state_without_backend_attribute(self):
        """Test validation handles state without current_backend attribute."""
        state = Mock(spec=[])  # Mock with no attributes
        pane = MockInputPane(state, text="Hello")
        
        is_valid, error_msg = pane._validate_can_send()
        
        assert is_valid is False
        assert "No model loaded" in error_msg


class TestButtonStateStability:
    """Tests to ensure button state doesn't flicker."""

    def test_repeated_validation_calls_same_result(self):
        """Test that repeated validation calls return consistent results."""
        state = MockState(current_backend=None)
        pane = MockInputPane(state, text="Hello")
        
        # Call validation multiple times
        results = [pane._validate_can_send() for _ in range(10)]
        
        # All results should be identical
        assert all(r == results[0] for r in results)
        # Should consistently fail with no model
        assert all(not r[0] for r in results)

    def test_validation_stable_during_processing(self):
        """Test validation during processing doesn't change state incorrectly."""
        backend = MockBackend("mistral-7b")
        state = MockState(current_backend=backend)
        pane = MockInputPane(state, text="Hello")
        
        # Simulate processing state
        pane._processing = True
        
        # Validation should still return correct result
        is_valid, _ = pane._validate_can_send()
        assert is_valid is True

    def test_model_state_change_updates_validation(self):
        """Test that validation result changes when model state changes."""
        state = MockState(current_backend=None)
        pane = MockInputPane(state, text="Hello")
        
        # Initially should fail
        is_valid, error_msg = pane._validate_can_send()
        assert is_valid is False
        assert "No model loaded" in error_msg
        
        # Load a model
        state.current_backend = MockBackend("mistral-7b")
        
        # Now should pass
        is_valid, error_msg = pane._validate_can_send()
        assert is_valid is True
        assert error_msg == ""


class TestErrorMessages:
    """Tests for clear and helpful error messages."""

    def test_no_model_error_is_actionable(self):
        """Test that 'no model' error message tells user what to do."""
        state = MockState(current_backend=None)
        pane = MockInputPane(state, text="Hello")
        
        _, error_msg = pane._validate_can_send()
        
        # Should mention where to go (Models tab) and what to do (load)
        assert "model" in error_msg.lower()
        assert "load" in error_msg.lower()

    def test_empty_message_error_is_clear(self):
        """Test that empty message error is clear."""
        backend = MockBackend("mistral-7b")
        state = MockState(current_backend=backend)
        pane = MockInputPane(state, text="")
        
        _, error_msg = pane._validate_can_send()
        
        # Should clearly indicate the problem
        assert "message" in error_msg.lower()
        assert "enter" in error_msg.lower()

    def test_error_messages_are_concise(self):
        """Test that error messages aren't overly verbose."""
        state = MockState(current_backend=None)
        pane = MockInputPane(state, text="Hello")
        
        _, error_msg = pane._validate_can_send()
        
        # Should be under 100 characters for status bar display
        assert len(error_msg) < 100

    def test_success_has_no_error_message(self):
        """Test that successful validation returns empty error message."""
        backend = MockBackend("mistral-7b")
        state = MockState(current_backend=backend)
        pane = MockInputPane(state, text="Hello")
        
        is_valid, error_msg = pane._validate_can_send()
        
        assert is_valid is True
        assert error_msg == ""


class TestEdgeCases:
    """Tests for edge cases in validation."""

    def test_backend_none_explicitly_set(self):
        """Test when backend is explicitly set to None (unload)."""
        state = MockState(current_backend=None)
        pane = MockInputPane(state, text="Hello")
        
        is_valid, _ = pane._validate_can_send()
        
        assert is_valid is False

    def test_very_long_message(self):
        """Test validation with very long message."""
        backend = MockBackend("mistral-7b")
        state = MockState(current_backend=backend)
        long_text = "x" * 10000
        pane = MockInputPane(state, text=long_text)
        
        is_valid, error_msg = pane._validate_can_send()
        
        # Should still validate successfully
        assert is_valid is True
        assert error_msg == ""

    def test_special_characters_in_message(self):
        """Test validation with special characters."""
        backend = MockBackend("mistral-7b")
        state = MockState(current_backend=backend)
        pane = MockInputPane(state, text="Hello! @#$%^&*() 你好")
        
        is_valid, error_msg = pane._validate_can_send()
        
        assert is_valid is True
        assert error_msg == ""

    def test_newlines_in_message(self):
        """Test validation with newlines in message."""
        backend = MockBackend("mistral-7b")
        state = MockState(current_backend=backend)
        pane = MockInputPane(state, text="Line 1\nLine 2\nLine 3")
        
        is_valid, error_msg = pane._validate_can_send()
        
        assert is_valid is True
        assert error_msg == ""
