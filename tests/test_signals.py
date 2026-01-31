"""Tests for the Signals provenance tracking system."""

from r3lay.core.signals import (
    Signal,
    SignalType,
    Transmission,
    ConfidenceCalculator,
)


class TestSignalType:
    """Tests for SignalType enum and weights."""

    def test_signal_types_exist(self):
        """All expected signal types are defined."""
        expected = ["DOCUMENT", "CODE", "USER", "COMMUNITY", "WEB", "INFERENCE", "SESSION"]
        for name in expected:
            assert hasattr(SignalType, name)

    def test_signal_type_values(self):
        """Signal types have expected string values."""
        assert SignalType.DOCUMENT.value == "document"
        assert SignalType.COMMUNITY.value == "community"
        assert SignalType.SESSION.value == "session"


class TestSignal:
    """Tests for Signal dataclass."""

    def test_signal_creation(self):
        """Signals can be created with required fields."""
        signal = Signal(
            title="Test Manual",
            signal_type=SignalType.DOCUMENT,
            content="Test content",
        )
        assert signal.title == "Test Manual"
        assert signal.signal_type == SignalType.DOCUMENT
        assert signal.id.startswith("sig_")

    def test_signal_id_generation(self):
        """Each signal gets a unique ID."""
        s1 = Signal(title="A", signal_type=SignalType.WEB, content="")
        s2 = Signal(title="B", signal_type=SignalType.WEB, content="")
        assert s1.id != s2.id

    def test_signal_with_url(self):
        """Signals can include source URL."""
        signal = Signal(
            title="Forum Post",
            signal_type=SignalType.COMMUNITY,
            content="Real experience here",
            url="https://example.com/forum/123",
        )
        assert signal.url == "https://example.com/forum/123"


class TestTransmission:
    """Tests for Transmission (citation excerpts)."""

    def test_transmission_creation(self):
        """Transmissions link excerpts to signals."""
        signal = Signal(
            title="Manual",
            signal_type=SignalType.DOCUMENT,
            content="Full content here",
        )
        transmission = Transmission(
            signal_id=signal.id,
            excerpt="Relevant quote",
            confidence=0.95,
        )
        assert transmission.signal_id == signal.id
        assert transmission.excerpt == "Relevant quote"
        assert transmission.id.startswith("cite_")


class TestConfidenceCalculator:
    """Tests for confidence scoring."""

    def test_document_highest_confidence(self):
        """Document sources have highest base confidence."""
        calc = ConfidenceCalculator()
        doc_conf = calc.get_type_weight(SignalType.DOCUMENT)
        web_conf = calc.get_type_weight(SignalType.WEB)
        assert doc_conf > web_conf

    def test_confidence_ordering(self):
        """Confidence follows expected hierarchy."""
        calc = ConfidenceCalculator()
        weights = [
            calc.get_type_weight(SignalType.DOCUMENT),
            calc.get_type_weight(SignalType.CODE),
            calc.get_type_weight(SignalType.USER),
            calc.get_type_weight(SignalType.COMMUNITY),
            calc.get_type_weight(SignalType.WEB),
            calc.get_type_weight(SignalType.INFERENCE),
            calc.get_type_weight(SignalType.SESSION),
        ]
        # Should be in descending order
        assert weights == sorted(weights, reverse=True)

    def test_corroboration_boost(self):
        """Multiple sources increase confidence."""
        calc = ConfidenceCalculator()
        # Use calculate_single for single-source confidence
        single = calc.calculate_single(SignalType.WEB)
        # For multiple sources, we need transmissions
        trans1 = Transmission(signal_id="sig_1", excerpt="test", confidence=1.0)
        trans2 = Transmission(signal_id="sig_2", excerpt="test", confidence=1.0)
        multiple = calc.calculate([trans1, trans2], [SignalType.WEB, SignalType.COMMUNITY])
        assert multiple > single
