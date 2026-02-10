"""Tests for ModelPanel input validation security fix.

Tests cover:
- Valid model selection
- Invalid model selection (security fix)
- Edge cases for option ID parsing
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from unittest.mock import MagicMock

from r3lay.ui.widgets.model_panel import ModelPanel


class MockBackendType(Enum):
    """Mock backend type enum."""

    OLLAMA = "ollama"
    LLAMA_CPP = "llama.cpp"
    MLX = "mlx"


class MockModelCapability(Enum):
    """Mock model capability enum."""

    TEXT = "text"
    VISION = "vision"
    TEXT_EMBEDDING = "text_embedding"
    VISION_EMBEDDING = "vision_embedding"


@dataclass
class MockModelInfo:
    """Mock ModelInfo for testing."""

    name: str
    backend_type: MockBackendType = MockBackendType.OLLAMA
    capabilities: list[MockModelCapability] = None

    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = [MockModelCapability.TEXT]


class MockState:
    """Mock R3LayState for testing."""

    def __init__(self):
        self.current_model = None
        self.scanner = None


class TestModelPanelInputValidation:
    """Tests for model panel input validation (security fix)."""

    def test_valid_model_selection(self):
        """Test that valid model selection works correctly."""
        # Setup
        state = MockState()
        panel = ModelPanel(state=state, scanner=None)

        # Add a valid model to the panel's internal state
        test_model = MockModelInfo(name="valid-model")
        panel._models = {"valid-model": test_model}

        # Mock _select_model to track if it was called
        panel._select_model = MagicMock()

        # Test valid selection
        panel._handle_selection("model:text:valid-model")

        # Verify _select_model was called with correct model name
        panel._select_model.assert_called_once_with("valid-model")

    def test_invalid_model_selection_blocked(self):
        """Test that invalid model selection is silently ignored (security fix)."""
        # Setup
        state = MockState()
        panel = ModelPanel(state=state, scanner=None)

        # Add only one valid model
        test_model = MockModelInfo(name="valid-model")
        panel._models = {"valid-model": test_model}

        # Mock _select_model to track if it was called
        panel._select_model = MagicMock()

        # Test invalid selection (model not in _models)
        panel._handle_selection("model:text:malicious-model")

        # Verify _select_model was NOT called
        panel._select_model.assert_not_called()

    def test_empty_models_dict(self):
        """Test handling when _models dict is empty."""
        # Setup
        state = MockState()
        panel = ModelPanel(state=state, scanner=None)

        # Empty models dict
        panel._models = {}

        # Mock _select_model
        panel._select_model = MagicMock()

        # Test selection with no models loaded
        panel._handle_selection("model:text:any-model")

        # Verify _select_model was NOT called
        panel._select_model.assert_not_called()

    def test_model_name_with_colons(self):
        """Test handling model names that contain colons."""
        # Setup
        state = MockState()
        panel = ModelPanel(state=state, scanner=None)

        # Model name with colons (edge case)
        model_name = "namespace:repo:model"
        test_model = MockModelInfo(name=model_name)
        panel._models = {model_name: test_model}

        # Mock _select_model
        panel._select_model = MagicMock()

        # Test selection - format is "model:role:model_name"
        # With split(":", 2), this correctly extracts the full model name
        panel._handle_selection(f"model:text:{model_name}")

        # Verify _select_model was called with correct full model name
        panel._select_model.assert_called_once_with(model_name)

    def test_malformed_option_id_short(self):
        """Test handling malformed option IDs (too few parts)."""
        # Setup
        state = MockState()
        panel = ModelPanel(state=state, scanner=None)

        test_model = MockModelInfo(name="valid-model")
        panel._models = {"valid-model": test_model}

        # Mock _select_model
        panel._select_model = MagicMock()

        # Test malformed IDs
        panel._handle_selection("model:text")  # Missing model name
        panel._handle_selection("model:")  # Only prefix

        # Verify _select_model was NOT called
        panel._select_model.assert_not_called()

    def test_none_role_selection(self):
        """Test that 'none:role' selections don't trigger model validation."""
        # Setup
        state = MockState()
        panel = ModelPanel(state=state, scanner=None)

        # Mock _clear_role
        panel._clear_role = MagicMock()

        # Test none selection
        panel._handle_selection("none:text")

        # Verify _clear_role was called (not _select_model)
        panel._clear_role.assert_called_once_with("text")

    def test_multiple_valid_models(self):
        """Test validation with multiple models in the dict."""
        # Setup
        state = MockState()
        panel = ModelPanel(state=state, scanner=None)

        # Add multiple models
        panel._models = {
            "model-a": MockModelInfo(name="model-a"),
            "model-b": MockModelInfo(name="model-b"),
            "model-c": MockModelInfo(name="model-c"),
        }

        # Mock _select_model
        panel._select_model = MagicMock()

        # Test valid selections
        panel._handle_selection("model:text:model-a")
        panel._handle_selection("model:vision:model-b")
        panel._handle_selection("model:text:model-c")

        # Verify all valid calls went through
        assert panel._select_model.call_count == 3

        # Test invalid selection
        panel._select_model.reset_mock()
        panel._handle_selection("model:text:model-d")

        # Verify invalid call was blocked
        panel._select_model.assert_not_called()

    def test_case_sensitive_validation(self):
        """Test that model name validation is case-sensitive."""
        # Setup
        state = MockState()
        panel = ModelPanel(state=state, scanner=None)

        # Add model with specific casing
        panel._models = {"ValidModel": MockModelInfo(name="ValidModel")}

        # Mock _select_model
        panel._select_model = MagicMock()

        # Test exact match (should work)
        panel._handle_selection("model:text:ValidModel")
        assert panel._select_model.call_count == 1

        # Test case mismatch (should be blocked)
        panel._select_model.reset_mock()
        panel._handle_selection("model:text:validmodel")
        panel._select_model.assert_not_called()

        panel._handle_selection("model:text:VALIDMODEL")
        panel._select_model.assert_not_called()

    def test_special_characters_in_model_name(self):
        """Test handling model names with special characters."""
        # Setup
        state = MockState()
        panel = ModelPanel(state=state, scanner=None)

        # Model names with special chars
        special_names = [
            "model-with-dashes",
            "model_with_underscores",
            "model.with.dots",
            "model@2024",
            "model[v2]",
        ]

        for name in special_names:
            panel._models = {name: MockModelInfo(name=name)}
            panel._select_model = MagicMock()

            # Test valid selection
            panel._handle_selection(f"model:text:{name}")
            panel._select_model.assert_called_once_with(name)


class TestModelPanelSecurityRegression:
    """Regression tests to ensure the security fix doesn't break existing functionality."""

    def test_legitimate_workflow(self):
        """Test a typical user workflow still works after security fix."""
        # Setup
        state = MockState()
        panel = ModelPanel(state=state, scanner=None)

        # Simulate models being loaded
        panel._models = {
            "llama3": MockModelInfo(name="llama3"),
            "qwen2-vl": MockModelInfo(name="qwen2-vl", capabilities=[MockModelCapability.VISION]),
        }

        # Mock methods
        panel._select_model = MagicMock()
        panel._clear_role = MagicMock()

        # User workflow: select text model
        panel._handle_selection("model:text:llama3")
        panel._select_model.assert_called_with("llama3")

        # User workflow: select vision model
        panel._select_model.reset_mock()
        panel._handle_selection("model:vision:qwen2-vl")
        panel._select_model.assert_called_with("qwen2-vl")

        # User workflow: clear a role
        panel._handle_selection("none:text")
        panel._clear_role.assert_called_with("text")
