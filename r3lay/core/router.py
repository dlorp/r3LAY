"""Smart Model Router for r3LAY.

Routes between text and vision LLMs with asymmetric thresholds:
- Text -> Vision: 0.6 (high bar to switch)
- Stay on Vision: 0.1 (low bar to stay - vision handles text fine)

Key insight: Vision models (like Qwen2.5-VL) can handle text-only
queries perfectly well. The cost of staying on vision is just slightly
higher memory usage, while the cost of switching is:
1. Model unload/load time (several seconds)
2. KV cache rebuild
3. Potential context confusion

Therefore, we use asymmetric thresholds to minimize unnecessary switches.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .backends import InferenceBackend
    from .index import RetrievalResult


# Image file extensions
IMAGE_EXTENSIONS: set[str] = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff"}

# Keywords that suggest vision is needed
VISION_KEYWORDS: set[str] = {
    # Explicit visual requests
    "image", "picture", "photo", "screenshot", "diagram",
    "chart", "graph", "figure", "illustration", "drawing",
    # Actions on visual content
    "look", "see", "show", "display", "visualize",
    "describe", "analyze", "examine", "inspect",
    # Visual elements
    "color", "shape", "layout", "design", "ui", "interface",
    "button", "icon", "logo", "text in", "ocr", "read the",
    # Document types
    "pdf", "slide", "presentation", "document",
}


@dataclass
class RouterConfig:
    """Configuration for the smart router.

    Attributes:
        text_model: Name/path of the default text model
        vision_model: Name/path of the vision model (None if not available)
        switch_to_vision_threshold: Score threshold to switch from text to vision
        stay_on_vision_threshold: Score threshold to stay on vision
        max_text_turns_on_vision: Auto-switch to text after N text-only turns
    """

    text_model: str
    vision_model: str | None = None
    switch_to_vision_threshold: float = 0.6
    stay_on_vision_threshold: float = 0.1
    max_text_turns_on_vision: int = 5


@dataclass
class RoutingDecision:
    """Result of a routing decision.

    Attributes:
        model_type: Which model type to use
        reason: Human-readable explanation
        vision_score: Computed vision need score (0.0-1.0)
        switched: Whether this required a model switch
    """

    model_type: Literal["text", "vision"]
    reason: str
    vision_score: float
    switched: bool = False


class SmartRouter:
    """Smart router with asymmetric thresholds for text/vision models.

    Uses different thresholds for switching TO vision vs STAYING on vision:
    - THRESHOLD_SWITCH_TO_VISION (0.6): High bar to switch from text
    - THRESHOLD_STAY_ON_VISION (0.1): Low bar to stay on vision

    This asymmetry reflects that:
    1. Vision models handle text-only queries fine
    2. Switching has real costs (time, KV cache, memory churn)
    3. Staying on vision only costs slightly more memory

    The router tracks consecutive text turns on vision models and will
    eventually switch back to text after max_text_turns_on_vision turns
    without visual content.
    """

    # Asymmetric thresholds
    THRESHOLD_SWITCH_TO_VISION: float = 0.6  # High bar to switch
    THRESHOLD_STAY_ON_VISION: float = 0.1    # Low bar to stay

    def __init__(
        self,
        config: RouterConfig,
        text_backend: "InferenceBackend | None" = None,
        vision_backend: "InferenceBackend | None" = None,
    ) -> None:
        """Initialize the smart router.

        Args:
            config: Router configuration with model names and thresholds
            text_backend: Pre-loaded text backend (optional)
            vision_backend: Pre-loaded vision backend (optional)
        """
        self.config = config
        self._text_backend = text_backend
        self._vision_backend = vision_backend
        self.current_model_type: Literal["text", "vision"] | None = None
        self.consecutive_text_turns: int = 0

        # Apply custom thresholds from config
        self.THRESHOLD_SWITCH_TO_VISION = config.switch_to_vision_threshold
        self.THRESHOLD_STAY_ON_VISION = config.stay_on_vision_threshold

    @property
    def text_backend(self) -> "InferenceBackend | None":
        """Get the text backend."""
        return self._text_backend

    @text_backend.setter
    def text_backend(self, backend: "InferenceBackend | None") -> None:
        """Set the text backend."""
        self._text_backend = backend
        if backend is not None and backend.is_loaded:
            self.current_model_type = "text"

    @property
    def vision_backend(self) -> "InferenceBackend | None":
        """Get the vision backend."""
        return self._vision_backend

    @vision_backend.setter
    def vision_backend(self, backend: "InferenceBackend | None") -> None:
        """Set the vision backend."""
        self._vision_backend = backend
        if backend is not None and backend.is_loaded:
            self.current_model_type = "vision"

    @property
    def current_backend(self) -> "InferenceBackend | None":
        """Get the currently active backend."""
        if self.current_model_type == "vision":
            return self._vision_backend
        return self._text_backend

    @property
    def has_vision(self) -> bool:
        """Check if vision model is configured."""
        return self.config.vision_model is not None

    def route(
        self,
        message: str,
        attachments: list[Path] | None = None,
        retrieved_context: list["RetrievalResult"] | None = None,
    ) -> RoutingDecision:
        """Route to appropriate model based on content analysis.

        Uses asymmetric thresholds:
        - High bar (0.6) to switch FROM text TO vision
        - Low bar (0.1) to STAY on vision

        Args:
            message: User message text
            attachments: List of file paths attached to message
            retrieved_context: RAG results that may contain image references

        Returns:
            RoutingDecision with model_type, reason, and vision_score
        """
        attachments = attachments or []
        retrieved_context = retrieved_context or []

        # Compute vision need score
        vision_score = self._compute_vision_need(message, attachments, retrieved_context)

        # Rule 1: Explicit image attachment -> vision (if available)
        if attachments and any(self._is_image(p) for p in attachments):
            if self.has_vision:
                self.consecutive_text_turns = 0
                switched = self.current_model_type != "vision"
                self.current_model_type = "vision"
                return RoutingDecision(
                    model_type="vision",
                    reason="User attached image",
                    vision_score=vision_score,
                    switched=switched,
                )

        # Rule 2: Already on vision -> stay unless score is very low
        if self.current_model_type == "vision" and self.has_vision:
            if vision_score > self.THRESHOLD_STAY_ON_VISION:
                self.consecutive_text_turns = 0
                return RoutingDecision(
                    model_type="vision",
                    reason=f"Continuing with vision (score: {vision_score:.2f})",
                    vision_score=vision_score,
                    switched=False,
                )

            # Track consecutive text-only turns on vision
            self.consecutive_text_turns += 1

            # Stay on vision unless we hit max consecutive text turns
            if self.consecutive_text_turns < self.config.max_text_turns_on_vision:
                turns = self.consecutive_text_turns
                max_turns = self.config.max_text_turns_on_vision
                return RoutingDecision(
                    model_type="vision",
                    reason=f"Vision handles text fine ({turns}/{max_turns} text turns)",
                    vision_score=vision_score,
                    switched=False,
                )

            # Switch back to text after many text-only turns
            self.current_model_type = "text"
            self.consecutive_text_turns = 0
            max_turns = self.config.max_text_turns_on_vision
            return RoutingDecision(
                model_type="text",
                reason=f"Switching to text after {max_turns} text-only turns",
                vision_score=vision_score,
                switched=True,
            )

        # Rule 3: On text -> switch to vision only if strong need
        if vision_score > self.THRESHOLD_SWITCH_TO_VISION and self.has_vision:
            switched = self.current_model_type != "vision"
            self.current_model_type = "vision"
            self.consecutive_text_turns = 0
            return RoutingDecision(
                model_type="vision",
                reason=f"Strong vision need (score: {vision_score:.2f})",
                vision_score=vision_score,
                switched=switched,
            )

        # Default: stay on or switch to text
        switched = self.current_model_type == "vision"
        self.current_model_type = "text"
        self.consecutive_text_turns = 0
        return RoutingDecision(
            model_type="text",
            reason="Standard text conversation",
            vision_score=vision_score,
            switched=switched,
        )

    def _compute_vision_need(
        self,
        message: str,
        attachments: list[Path],
        retrieved_context: list["RetrievalResult"],
    ) -> float:
        """Compute a score indicating how much vision capability is needed.

        Score is based on:
        - Image attachments (0.9)
        - Vision keywords in message (0.1-0.5)
        - Image references in retrieved context (0.2)

        Returns:
            Float score from 0.0 (no vision need) to 1.0 (strong vision need)
        """
        score = 0.0

        # Image attachments are a strong signal
        image_attachments = [p for p in attachments if self._is_image(p)]
        if image_attachments:
            score += 0.9

        # Check for vision keywords in message
        message_lower = message.lower()
        keyword_matches = sum(1 for kw in VISION_KEYWORDS if kw in message_lower)

        # Keywords contribute up to 0.5
        if keyword_matches > 0:
            score += min(0.5, keyword_matches * 0.1)

        # Check for image references in retrieved context
        for result in retrieved_context:
            metadata = result.metadata
            source = metadata.get("source", "")

            # Check if source is an image file
            if Path(source).suffix.lower() in IMAGE_EXTENSIONS:
                score += 0.2
                break

            # Check for image-related content
            if any(ext in source.lower() for ext in [".png", ".jpg", ".jpeg"]):
                score += 0.15
                break

        # Cap at 1.0
        return min(1.0, score)

    def _is_image(self, path: Path) -> bool:
        """Check if a path points to an image file."""
        return path.suffix.lower() in IMAGE_EXTENSIONS

    def reset(self) -> None:
        """Reset router state (for new sessions)."""
        self.current_model_type = None
        self.consecutive_text_turns = 0


__all__ = [
    "RouterConfig",
    "RoutingDecision",
    "SmartRouter",
    "IMAGE_EXTENSIONS",
    "VISION_KEYWORDS",
]
