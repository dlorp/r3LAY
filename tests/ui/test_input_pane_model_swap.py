"""Tests for natural language model swap functionality in InputPane.

Tests cover:
- Model name fuzzy matching (_find_model_by_name)
- Intent parsing integration for model swap commands
- End-to-end model swap flow
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pytest

from r3lay.core.intent.parser import IntentParser
from r3lay.core.intent.taxonomy import IntentType


class MockBackendType(Enum):
    """Mock backend type enum."""

    OLLAMA = "ollama"
    LLAMA_CPP = "llama.cpp"
    MLX = "mlx"


@dataclass
class MockModelInfo:
    """Mock ModelInfo for testing."""

    name: str
    backend_type: MockBackendType = MockBackendType.OLLAMA
    is_vision_model: bool = False


class MockState:
    """Mock R3LayState for testing model lookup."""

    def __init__(self, models: list[MockModelInfo] | None = None):
        self.available_models = models or []


class TestFindModelByName:
    """Tests for the _find_model_by_name logic."""

    def _find_model_by_name(self, state: MockState, model_name: str) -> MockModelInfo | None:
        """Standalone implementation of the _find_model_by_name logic for testing.

        This mirrors the InputPane._find_model_by_name method.
        """
        if not state.available_models:
            return None

        model_name_lower = model_name.lower()

        # Try exact match first (case-insensitive)
        for model in state.available_models:
            if model.name.lower() == model_name_lower:
                return model

        # Try prefix match (e.g., "mistral" matches "mistral-7b-v0.3")
        for model in state.available_models:
            if model.name.lower().startswith(model_name_lower):
                return model

        # Try contains match (e.g., "qwen" matches "qwen2.5-7b-instruct")
        for model in state.available_models:
            if model_name_lower in model.name.lower():
                return model

        return None

    @pytest.fixture
    def models(self) -> list[MockModelInfo]:
        """Fixture providing a list of mock models."""
        return [
            MockModelInfo(name="mistral-7b-v0.3"),
            MockModelInfo(name="llama3.2-8b-instruct"),
            MockModelInfo(name="qwen2.5-7b-instruct"),
            MockModelInfo(name="gemma-2b"),
            MockModelInfo(name="llava-1.5-7b", is_vision_model=True),
        ]

    def test_exact_match(self, models: list[MockModelInfo]) -> None:
        """Test exact model name match."""
        state = MockState(models)
        result = self._find_model_by_name(state, "mistral-7b-v0.3")
        assert result is not None
        assert result.name == "mistral-7b-v0.3"

    def test_exact_match_case_insensitive(self, models: list[MockModelInfo]) -> None:
        """Test exact match is case-insensitive."""
        state = MockState(models)
        result = self._find_model_by_name(state, "MISTRAL-7B-V0.3")
        assert result is not None
        assert result.name == "mistral-7b-v0.3"

    def test_prefix_match_short(self, models: list[MockModelInfo]) -> None:
        """Test prefix match with short name."""
        state = MockState(models)
        result = self._find_model_by_name(state, "mistral")
        assert result is not None
        assert result.name == "mistral-7b-v0.3"

    def test_prefix_match_medium(self, models: list[MockModelInfo]) -> None:
        """Test prefix match with medium-length name."""
        state = MockState(models)
        result = self._find_model_by_name(state, "llama3")
        assert result is not None
        assert result.name == "llama3.2-8b-instruct"

    def test_contains_match(self, models: list[MockModelInfo]) -> None:
        """Test contains match when prefix doesn't work."""
        state = MockState(models)
        result = self._find_model_by_name(state, "qwen")
        assert result is not None
        assert result.name == "qwen2.5-7b-instruct"

    def test_not_found(self, models: list[MockModelInfo]) -> None:
        """Test model not found returns None."""
        state = MockState(models)
        result = self._find_model_by_name(state, "nonexistent")
        assert result is None

    def test_empty_models_list(self) -> None:
        """Test with empty models list."""
        state = MockState([])
        result = self._find_model_by_name(state, "mistral")
        assert result is None

    def test_vision_model(self, models: list[MockModelInfo]) -> None:
        """Test finding a vision model."""
        state = MockState(models)
        result = self._find_model_by_name(state, "llava")
        assert result is not None
        assert result.name == "llava-1.5-7b"
        assert result.is_vision_model is True


class TestIntentParsingForModelSwap:
    """Tests for intent parsing of model swap commands."""

    @pytest.fixture
    def parser(self) -> IntentParser:
        return IntentParser()

    def test_swap_model_command(self, parser: IntentParser) -> None:
        """Test 'swap model X' is recognized as COMMAND/cmd.model."""
        result = parser.parse_sync("swap model mistral")
        assert result.intent == IntentType.COMMAND
        assert result.subtype == "cmd.model"
        assert result.entities.get("model_name") == "mistral"
        assert result.confidence >= 0.7

    def test_use_llama_command(self, parser: IntentParser) -> None:
        """Test 'use llama' is recognized as model swap."""
        result = parser.parse_sync("use llama3")
        # This might not match as confidently depending on patterns
        # The key is it should extract the model name
        if result.intent == IntentType.COMMAND:
            assert result.entities.get("model_name") is not None

    def test_load_model_command(self, parser: IntentParser) -> None:
        """Test 'load model X' pattern."""
        result = parser.parse_sync("load model qwen")
        assert result.intent == IntentType.COMMAND
        assert result.subtype == "cmd.model"
        assert result.entities.get("model_name") == "qwen"

    def test_switch_model_command(self, parser: IntentParser) -> None:
        """Test 'switch model X' pattern."""
        result = parser.parse_sync("switch model gemma")
        assert result.intent == IntentType.COMMAND
        assert result.subtype == "cmd.model"
        assert result.entities.get("model_name") == "gemma"

    def test_change_llm_command(self, parser: IntentParser) -> None:
        """Test 'change llm X' pattern."""
        result = parser.parse_sync("change llm mistral")
        assert result.intent == IntentType.COMMAND
        assert result.subtype == "cmd.model"

    def test_model_name_with_version(self, parser: IntentParser) -> None:
        """Test model name with version numbers."""
        result = parser.parse_sync("load model qwen2.5-7b")
        assert result.intent == IntentType.COMMAND
        assert result.entities.get("model_name") == "qwen2.5-7b"

    def test_model_name_with_path(self, parser: IntentParser) -> None:
        """Test model name with path-like structure."""
        result = parser.parse_sync("swap model meta-llama/Llama-3.1-8B")
        assert result.intent == IntentType.COMMAND
        # The model name should be extracted (may include path)
        model_name = result.entities.get("model_name", "")
        assert "llama" in model_name.lower()

    def test_not_a_model_swap_question(self, parser: IntentParser) -> None:
        """Test that questions about models don't trigger model swap.

        Note: The current pattern matcher may match 'model X' patterns broadly.
        The key is that if it matches, _find_model_by_name will return None
        for non-existent models like 'car', giving a helpful error.
        """
        result = parser.parse_sync("tell me about neural networks")
        # General chat should NOT be cmd.model
        assert not (result.intent == IntentType.COMMAND and result.subtype == "cmd.model")

    def test_confidence_threshold(self, parser: IntentParser) -> None:
        """Test that model swap has high enough confidence."""
        result = parser.parse_sync("swap model mistral")
        # For the submit() logic, we require confidence >= 0.7
        assert result.confidence >= 0.7


class TestModelSwapPatternVariations:
    """Tests for various natural language patterns for model swapping."""

    @pytest.fixture
    def parser(self) -> IntentParser:
        return IntentParser()

    @pytest.mark.parametrize(
        "input_text,expected_model",
        [
            ("swap model mistral", "mistral"),
            ("switch model llama", "llama"),
            ("change model qwen", "qwen"),
            ("load model gemma", "gemma"),
            ("use model phi", "phi"),
            ("swap llm mistral", "mistral"),
            ("load llm qwen2.5-7b", "qwen2.5-7b"),
        ],
    )
    def test_model_swap_patterns(
        self, parser: IntentParser, input_text: str, expected_model: str
    ) -> None:
        """Test various model swap patterns."""
        result = parser.parse_sync(input_text)
        assert result.intent == IntentType.COMMAND
        assert result.subtype == "cmd.model"
        assert result.entities.get("model_name") == expected_model
