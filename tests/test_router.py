"""Tests for Smart Model Router.

Tests the asymmetric threshold routing between text and vision models,
including edge cases for model switching, consecutive turn tracking,
and vision need scoring.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from r3lay.core.router import (
    IMAGE_EXTENSIONS,
    VISION_KEYWORDS,
    RouterConfig,
    RoutingDecision,
    SmartRouter,
)


class TestRouterConfig:
    """Tests for RouterConfig dataclass."""

    def test_default_config(self):
        """Test RouterConfig with minimal required args."""
        config = RouterConfig(text_model="llama-3.2-3b")
        
        assert config.text_model == "llama-3.2-3b"
        assert config.vision_model is None
        assert config.switch_to_vision_threshold == 0.6
        assert config.stay_on_vision_threshold == 0.1
        assert config.max_text_turns_on_vision == 5

    def test_full_config(self):
        """Test RouterConfig with all args specified."""
        config = RouterConfig(
            text_model="llama-3.2-3b",
            vision_model="qwen2.5-vl-7b",
            switch_to_vision_threshold=0.7,
            stay_on_vision_threshold=0.2,
            max_text_turns_on_vision=3,
        )
        
        assert config.text_model == "llama-3.2-3b"
        assert config.vision_model == "qwen2.5-vl-7b"
        assert config.switch_to_vision_threshold == 0.7
        assert config.stay_on_vision_threshold == 0.2
        assert config.max_text_turns_on_vision == 3


class TestRoutingDecision:
    """Tests for RoutingDecision dataclass."""

    def test_routing_decision_defaults(self):
        """Test RoutingDecision with default switched=False."""
        decision = RoutingDecision(
            model_type="text",
            reason="Standard text query",
            vision_score=0.0,
        )
        
        assert decision.model_type == "text"
        assert decision.reason == "Standard text query"
        assert decision.vision_score == 0.0
        assert decision.switched is False

    def test_routing_decision_with_switch(self):
        """Test RoutingDecision with switched=True."""
        decision = RoutingDecision(
            model_type="vision",
            reason="User attached image",
            vision_score=0.9,
            switched=True,
        )
        
        assert decision.model_type == "vision"
        assert decision.switched is True


class TestSmartRouterInit:
    """Tests for SmartRouter initialization."""

    def test_init_text_only(self):
        """Test router with only text model configured."""
        config = RouterConfig(text_model="llama-3.2-3b")
        router = SmartRouter(config)
        
        assert router.config == config
        assert router.has_vision is False
        assert router.current_model_type is None
        assert router.consecutive_text_turns == 0

    def test_init_with_vision(self):
        """Test router with vision model configured."""
        config = RouterConfig(
            text_model="llama-3.2-3b",
            vision_model="qwen2.5-vl-7b",
        )
        router = SmartRouter(config)
        
        assert router.has_vision is True

    def test_custom_thresholds_applied(self):
        """Test that custom thresholds from config are applied."""
        config = RouterConfig(
            text_model="llama-3.2-3b",
            switch_to_vision_threshold=0.8,
            stay_on_vision_threshold=0.3,
        )
        router = SmartRouter(config)
        
        assert router.THRESHOLD_SWITCH_TO_VISION == 0.8
        assert router.THRESHOLD_STAY_ON_VISION == 0.3


class TestSmartRouterBackends:
    """Tests for backend property setters."""

    def test_set_text_backend_loaded(self):
        """Test setting text backend that is loaded."""
        config = RouterConfig(text_model="llama-3.2-3b")
        router = SmartRouter(config)
        
        mock_backend = MagicMock()
        mock_backend.is_loaded = True
        
        router.text_backend = mock_backend
        
        assert router.text_backend == mock_backend
        assert router.current_model_type == "text"

    def test_set_vision_backend_loaded(self):
        """Test setting vision backend that is loaded."""
        config = RouterConfig(
            text_model="llama-3.2-3b",
            vision_model="qwen2.5-vl-7b",
        )
        router = SmartRouter(config)
        
        mock_backend = MagicMock()
        mock_backend.is_loaded = True
        
        router.vision_backend = mock_backend
        
        assert router.vision_backend == mock_backend
        assert router.current_model_type == "vision"

    def test_current_backend_returns_text(self):
        """Test current_backend returns text backend when on text."""
        config = RouterConfig(text_model="llama-3.2-3b")
        router = SmartRouter(config)
        
        mock_text = MagicMock()
        mock_text.is_loaded = True
        router.text_backend = mock_text
        
        assert router.current_backend == mock_text

    def test_current_backend_returns_vision(self):
        """Test current_backend returns vision backend when on vision."""
        config = RouterConfig(
            text_model="llama-3.2-3b",
            vision_model="qwen2.5-vl-7b",
        )
        router = SmartRouter(config)
        
        mock_vision = MagicMock()
        mock_vision.is_loaded = True
        router.vision_backend = mock_vision
        
        assert router.current_backend == mock_vision


class TestSmartRouterRouting:
    """Tests for the main route() method."""

    @pytest.fixture
    def vision_router(self):
        """Create a router with vision model configured."""
        config = RouterConfig(
            text_model="llama-3.2-3b",
            vision_model="qwen2.5-vl-7b",
        )
        return SmartRouter(config)

    @pytest.fixture
    def text_only_router(self):
        """Create a router without vision model."""
        config = RouterConfig(text_model="llama-3.2-3b")
        return SmartRouter(config)

    def test_route_with_image_attachment(self, vision_router):
        """Test routing to vision when image is attached."""
        decision = vision_router.route(
            message="What's in this image?",
            attachments=[Path("test.png")],
        )
        
        assert decision.model_type == "vision"
        assert decision.reason == "User attached image"
        assert decision.vision_score >= 0.9
        assert vision_router.current_model_type == "vision"
        assert vision_router.consecutive_text_turns == 0

    def test_route_image_attachment_no_vision_model(self, text_only_router):
        """Test that image attachment falls back to text without vision model."""
        decision = text_only_router.route(
            message="What's in this image?",
            attachments=[Path("test.png")],
        )
        
        # Without vision model, should route to text
        assert decision.model_type == "text"

    def test_route_stay_on_vision_high_score(self, vision_router):
        """Test staying on vision when score exceeds stay threshold."""
        vision_router.current_model_type = "vision"
        
        decision = vision_router.route(
            message="Now describe the colors in the image",
            attachments=[],
        )
        
        assert decision.model_type == "vision"
        assert decision.switched is False
        assert "Continuing with vision" in decision.reason

    def test_route_stay_on_vision_low_score_under_max_turns(self, vision_router):
        """Test staying on vision even with low score if under max turns."""
        vision_router.current_model_type = "vision"
        vision_router.consecutive_text_turns = 0
        
        # Very low vision score message
        decision = vision_router.route(
            message="What is 2 + 2?",
            attachments=[],
        )
        
        assert decision.model_type == "vision"
        assert decision.switched is False
        assert "handles text fine" in decision.reason
        assert vision_router.consecutive_text_turns == 1

    def test_route_switch_from_vision_after_max_turns(self, vision_router):
        """Test switching to text after max consecutive text turns."""
        vision_router.current_model_type = "vision"
        vision_router.consecutive_text_turns = 4  # One less than max
        
        decision = vision_router.route(
            message="What is the capital of France?",
            attachments=[],
        )
        
        assert decision.model_type == "text"
        assert decision.switched is True
        assert "Switching to text" in decision.reason
        assert vision_router.consecutive_text_turns == 0

    def test_route_switch_to_vision_high_score(self, vision_router):
        """Test switching to vision with strong vision need."""
        vision_router.current_model_type = "text"
        
        # Message with many vision keywords
        decision = vision_router.route(
            message="Show me the image and describe the diagram colors and layout",
            attachments=[],
        )
        
        assert decision.model_type == "vision"
        assert decision.switched is True
        assert "Strong vision need" in decision.reason

    def test_route_stay_on_text_low_score(self, vision_router):
        """Test staying on text with low vision score."""
        vision_router.current_model_type = "text"
        
        decision = vision_router.route(
            message="Hello, how are you?",
            attachments=[],
        )
        
        assert decision.model_type == "text"
        assert decision.switched is False
        assert "Standard text conversation" in decision.reason

    def test_route_default_to_text_when_none(self, vision_router):
        """Test defaulting to text when no model type set."""
        assert vision_router.current_model_type is None
        
        decision = vision_router.route(
            message="Hello",
            attachments=[],
        )
        
        assert decision.model_type == "text"
        assert vision_router.current_model_type == "text"


class TestVisionNeedScoring:
    """Tests for _compute_vision_need scoring."""

    @pytest.fixture
    def router(self):
        """Create a router for testing."""
        config = RouterConfig(
            text_model="llama-3.2-3b",
            vision_model="qwen2.5-vl-7b",
        )
        return SmartRouter(config)

    def test_score_no_vision_content(self, router):
        """Test score is 0 for pure text content."""
        score = router._compute_vision_need(
            message="What is machine learning?",
            attachments=[],
            retrieved_context=[],
        )
        
        assert score == 0.0

    def test_score_image_attachment(self, router):
        """Test high score for image attachment."""
        score = router._compute_vision_need(
            message="Analyze this",
            attachments=[Path("diagram.png")],
            retrieved_context=[],
        )
        
        assert score >= 0.9

    def test_score_single_keyword(self, router):
        """Test score with single vision keyword."""
        score = router._compute_vision_need(
            message="Show me an image",
            attachments=[],
            retrieved_context=[],
        )
        
        assert 0.1 <= score <= 0.2

    def test_score_multiple_keywords(self, router):
        """Test score increases with multiple keywords."""
        score = router._compute_vision_need(
            message="Show me the image with colors and shapes in the diagram",
            attachments=[],
            retrieved_context=[],
        )
        
        # Multiple keywords should give higher score
        assert score >= 0.3

    def test_score_capped_at_one(self, router):
        """Test that score is capped at 1.0."""
        # Many keywords + image attachment
        score = router._compute_vision_need(
            message="Show image picture photo chart diagram graph figure colors layout",
            attachments=[Path("test.png"), Path("another.jpg")],
            retrieved_context=[],
        )
        
        assert score == 1.0

    def test_score_image_in_retrieved_context(self, router):
        """Test score boost from image in retrieved context."""
        mock_result = MagicMock()
        mock_result.metadata = {"source": "/docs/screenshot.png"}
        
        score = router._compute_vision_need(
            message="What does this show?",
            attachments=[],
            retrieved_context=[mock_result],
        )
        
        assert score >= 0.2


class TestIsImage:
    """Tests for _is_image helper method."""

    @pytest.fixture
    def router(self):
        """Create a router for testing."""
        config = RouterConfig(text_model="llama-3.2-3b")
        return SmartRouter(config)

    @pytest.mark.parametrize("extension", IMAGE_EXTENSIONS)
    def test_is_image_valid_extensions(self, router, extension):
        """Test that all IMAGE_EXTENSIONS are recognized."""
        path = Path(f"test{extension}")
        assert router._is_image(path) is True

    @pytest.mark.parametrize("extension", IMAGE_EXTENSIONS)
    def test_is_image_case_insensitive(self, router, extension):
        """Test that image detection is case-insensitive."""
        path = Path(f"test{extension.upper()}")
        assert router._is_image(path) is True

    @pytest.mark.parametrize("extension", [".txt", ".py", ".md", ".pdf", ".mp4"])
    def test_is_image_non_image_extensions(self, router, extension):
        """Test that non-image extensions return False."""
        path = Path(f"test{extension}")
        assert router._is_image(path) is False


class TestReset:
    """Tests for reset() method."""

    def test_reset_clears_state(self):
        """Test that reset clears model type and turn counter."""
        config = RouterConfig(
            text_model="llama-3.2-3b",
            vision_model="qwen2.5-vl-7b",
        )
        router = SmartRouter(config)
        
        # Set some state
        router.current_model_type = "vision"
        router.consecutive_text_turns = 3
        
        router.reset()
        
        assert router.current_model_type is None
        assert router.consecutive_text_turns == 0


class TestImageExtensionsAndKeywords:
    """Tests for module-level constants."""

    def test_image_extensions_contains_common_formats(self):
        """Test that common image formats are included."""
        common = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
        assert common.issubset(IMAGE_EXTENSIONS)

    def test_vision_keywords_contains_essential_terms(self):
        """Test that essential vision keywords are included."""
        essential = {"image", "picture", "photo", "diagram", "chart"}
        assert essential.issubset(VISION_KEYWORDS)
