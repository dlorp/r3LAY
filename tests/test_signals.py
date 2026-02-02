"""Tests for the Signals provenance tracking system.

Covers:
- SignalType enum
- Signal dataclass
- Transmission dataclass
- Citation dataclass
- signal_type_from_source_type conversion
- ConfidenceCalculator
- SignalsManager CRUD and persistence
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from r3lay.core.signals import (
    Citation,
    ConfidenceCalculator,
    Signal,
    SignalsManager,
    SignalType,
    Transmission,
    signal_type_from_source_type,
)
from r3lay.core.sources import SourceType

# ============================================================================
# SignalType Enum Tests
# ============================================================================


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
        assert SignalType.CODE.value == "code"
        assert SignalType.USER.value == "user"
        assert SignalType.COMMUNITY.value == "community"
        assert SignalType.WEB.value == "web"
        assert SignalType.INFERENCE.value == "inference"
        assert SignalType.SESSION.value == "session"

    def test_signal_type_is_str_enum(self):
        """SignalType inherits from str."""
        assert isinstance(SignalType.DOCUMENT, str)
        assert SignalType.DOCUMENT == "document"

    def test_signal_type_from_string(self):
        """Can create SignalType from string value."""
        assert SignalType("document") == SignalType.DOCUMENT
        assert SignalType("community") == SignalType.COMMUNITY

    def test_invalid_signal_type_raises(self):
        """Invalid signal type string raises ValueError."""
        with pytest.raises(ValueError):
            SignalType("invalid_type")


# ============================================================================
# Signal Dataclass Tests
# ============================================================================


class TestSignal:
    """Tests for Signal dataclass."""

    def test_signal_creation_minimal(self):
        """Signals can be created with required fields."""
        signal = Signal(
            title="Test Manual",
            signal_type=SignalType.DOCUMENT,
        )
        assert signal.title == "Test Manual"
        assert signal.signal_type == SignalType.DOCUMENT
        assert signal.id.startswith("sig_")
        assert signal.content is None
        assert signal.path is None
        assert signal.url is None

    def test_signal_creation_full(self):
        """Signals can be created with all fields."""
        signal = Signal(
            title="Full Signal",
            signal_type=SignalType.WEB,
            id="sig_custom",
            content="Full content here",
            path="/path/to/file.txt",
            url="https://example.com",
            hash="abc123",
            indexed_at="2024-01-01T00:00:00",
            metadata={"author": "Test Author"},
        )
        assert signal.id == "sig_custom"
        assert signal.content == "Full content here"
        assert signal.path == "/path/to/file.txt"
        assert signal.url == "https://example.com"
        assert signal.hash == "abc123"
        assert signal.indexed_at == "2024-01-01T00:00:00"
        assert signal.metadata == {"author": "Test Author"}

    def test_signal_id_generation(self):
        """Each signal gets a unique ID."""
        s1 = Signal(title="A", signal_type=SignalType.WEB)
        s2 = Signal(title="B", signal_type=SignalType.WEB)
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

    def test_signal_type_property_alias(self):
        """The 'type' property aliases signal_type."""
        signal = Signal(title="Test", signal_type=SignalType.CODE)
        assert signal.type == SignalType.CODE
        assert signal.type == signal.signal_type

    def test_signal_hashable(self):
        """Signals can be used in sets and as dict keys."""
        s1 = Signal(title="A", signal_type=SignalType.WEB, id="sig_1")
        s2 = Signal(title="B", signal_type=SignalType.WEB, id="sig_2")
        signal_set = {s1, s2}
        assert len(signal_set) == 2
        assert s1 in signal_set

    def test_signal_equality(self):
        """Signal equality is based on ID."""
        s1 = Signal(title="A", signal_type=SignalType.WEB, id="sig_same")
        s2 = Signal(title="B", signal_type=SignalType.CODE, id="sig_same")
        assert s1 == s2

    def test_signal_inequality(self):
        """Signals with different IDs are not equal."""
        s1 = Signal(title="A", signal_type=SignalType.WEB, id="sig_1")
        s2 = Signal(title="A", signal_type=SignalType.WEB, id="sig_2")
        assert s1 != s2

    def test_signal_not_equal_to_non_signal(self):
        """Signal not equal to non-Signal objects."""
        signal = Signal(title="Test", signal_type=SignalType.WEB, id="sig_1")
        assert signal != "sig_1"
        assert signal != 123
        assert signal != None  # noqa: E711

    def test_signal_default_indexed_at(self):
        """Signals get auto-generated indexed_at timestamp."""
        signal = Signal(title="Test", signal_type=SignalType.WEB)
        assert signal.indexed_at is not None
        assert "T" in signal.indexed_at  # ISO format


# ============================================================================
# Transmission Dataclass Tests
# ============================================================================


class TestTransmission:
    """Tests for Transmission (citation excerpts)."""

    def test_transmission_creation_minimal(self):
        """Transmissions can be created with required fields."""
        transmission = Transmission(
            signal_id="sig_abc123",
            excerpt="Relevant quote",
        )
        assert transmission.signal_id == "sig_abc123"
        assert transmission.excerpt == "Relevant quote"
        assert transmission.id.startswith("cite_")
        assert transmission.location == ""
        assert transmission.confidence == 0.8

    def test_transmission_creation_full(self):
        """Transmissions can be created with all fields."""
        transmission = Transmission(
            signal_id="sig_abc123",
            excerpt="Relevant quote",
            id="cite_custom",
            location="page 42",
            confidence=0.95,
        )
        assert transmission.id == "cite_custom"
        assert transmission.location == "page 42"
        assert transmission.confidence == 0.95

    def test_transmission_links_to_signal(self):
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

    def test_transmission_hashable(self):
        """Transmissions can be used in sets."""
        t1 = Transmission(signal_id="sig_1", excerpt="quote 1", location="page 1")
        t2 = Transmission(signal_id="sig_2", excerpt="quote 2", location="page 2")
        trans_set = {t1, t2}
        assert len(trans_set) == 2

    def test_transmission_hash_based_on_content(self):
        """Transmission hash uses signal_id, location, and excerpt prefix."""
        t1 = Transmission(signal_id="sig_1", excerpt="quote", location="page 1")
        t2 = Transmission(signal_id="sig_1", excerpt="quote", location="page 1")
        # Same content should have same hash
        assert hash(t1) == hash(t2)


# ============================================================================
# Citation Dataclass Tests
# ============================================================================


class TestCitation:
    """Tests for Citation dataclass."""

    def test_citation_creation(self):
        """Citations can be created with required fields."""
        trans = Transmission(signal_id="sig_1", excerpt="quote")
        citation = Citation(
            id="cite_abc",
            statement="A factual claim",
            confidence=0.85,
            transmissions=[trans],
        )
        assert citation.id == "cite_abc"
        assert citation.statement == "A factual claim"
        assert citation.confidence == 0.85
        assert len(citation.transmissions) == 1

    def test_citation_default_values(self):
        """Citations have correct default values."""
        trans = Transmission(signal_id="sig_1", excerpt="quote")
        citation = Citation(
            id="cite_abc",
            statement="A factual claim",
            confidence=0.85,
            transmissions=[trans],
        )
        assert citation.created_at is not None
        assert citation.used_in == []

    def test_citation_with_used_in(self):
        """Citations can track which axioms use them."""
        trans = Transmission(signal_id="sig_1", excerpt="quote")
        citation = Citation(
            id="cite_abc",
            statement="A factual claim",
            confidence=0.85,
            transmissions=[trans],
            used_in=["axiom_1", "axiom_2"],
        )
        assert citation.used_in == ["axiom_1", "axiom_2"]

    def test_citation_hashable(self):
        """Citations can be used in sets."""
        trans = Transmission(signal_id="sig_1", excerpt="quote")
        c1 = Citation(id="cite_1", statement="A", confidence=0.8, transmissions=[trans])
        c2 = Citation(id="cite_2", statement="B", confidence=0.9, transmissions=[trans])
        citation_set = {c1, c2}
        assert len(citation_set) == 2


# ============================================================================
# signal_type_from_source_type Tests
# ============================================================================


class TestSignalTypeFromSourceType:
    """Tests for signal_type_from_source_type conversion function."""

    def test_indexed_curated_to_document(self):
        """INDEXED_CURATED maps to DOCUMENT."""
        result = signal_type_from_source_type(SourceType.INDEXED_CURATED)
        assert result == SignalType.DOCUMENT

    def test_indexed_document_to_document(self):
        """INDEXED_DOCUMENT maps to DOCUMENT."""
        result = signal_type_from_source_type(SourceType.INDEXED_DOCUMENT)
        assert result == SignalType.DOCUMENT

    def test_indexed_image_to_document(self):
        """INDEXED_IMAGE maps to DOCUMENT."""
        result = signal_type_from_source_type(SourceType.INDEXED_IMAGE)
        assert result == SignalType.DOCUMENT

    def test_indexed_code_to_code(self):
        """INDEXED_CODE maps to CODE."""
        result = signal_type_from_source_type(SourceType.INDEXED_CODE)
        assert result == SignalType.CODE

    def test_web_oe_firstparty_to_document(self):
        """WEB_OE_FIRSTPARTY maps to DOCUMENT (official = high trust)."""
        result = signal_type_from_source_type(SourceType.WEB_OE_FIRSTPARTY)
        assert result == SignalType.DOCUMENT

    def test_web_trusted_to_web(self):
        """WEB_TRUSTED maps to WEB."""
        result = signal_type_from_source_type(SourceType.WEB_TRUSTED)
        assert result == SignalType.WEB

    def test_web_community_to_community(self):
        """WEB_COMMUNITY maps to COMMUNITY."""
        result = signal_type_from_source_type(SourceType.WEB_COMMUNITY)
        assert result == SignalType.COMMUNITY

    def test_web_general_to_web(self):
        """WEB_GENERAL maps to WEB."""
        result = signal_type_from_source_type(SourceType.WEB_GENERAL)
        assert result == SignalType.WEB


# ============================================================================
# ConfidenceCalculator Tests
# ============================================================================


class TestConfidenceCalculator:
    """Tests for confidence scoring."""

    def test_document_highest_confidence(self):
        """Document sources have highest base confidence."""
        calc = ConfidenceCalculator()
        doc_conf = calc.get_type_weight(SignalType.DOCUMENT)
        web_conf = calc.get_type_weight(SignalType.WEB)
        assert doc_conf > web_conf
        assert doc_conf == 0.95

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

    def test_all_weights_defined(self):
        """All signal types have defined weights."""
        calc = ConfidenceCalculator()
        for signal_type in SignalType:
            weight = calc.get_type_weight(signal_type)
            assert 0.0 <= weight <= 1.0

    def test_base_weight_alias(self):
        """base_weight is an alias for get_type_weight."""
        calc = ConfidenceCalculator()
        for signal_type in SignalType:
            assert calc.base_weight(signal_type) == calc.get_type_weight(signal_type)

    def test_unknown_type_default_weight(self):
        """Unknown signal types return 0.5 default weight."""
        calc = ConfidenceCalculator()
        # Can't easily create an unknown type, but test the dict default
        weight = calc.SIGNAL_WEIGHTS.get("unknown", 0.5)
        assert weight == 0.5

    def test_calculate_single_full_confidence(self):
        """calculate_single with full transmission confidence."""
        calc = ConfidenceCalculator()
        result = calc.calculate_single(SignalType.DOCUMENT, 1.0)
        assert result == 0.95

    def test_calculate_single_partial_confidence(self):
        """calculate_single with partial transmission confidence."""
        calc = ConfidenceCalculator()
        result = calc.calculate_single(SignalType.DOCUMENT, 0.8)
        assert result == 0.76  # 0.95 * 0.8

    def test_calculate_single_default_confidence(self):
        """calculate_single uses 1.0 as default transmission confidence."""
        calc = ConfidenceCalculator()
        result = calc.calculate_single(SignalType.WEB)
        assert result == 0.7

    def test_calculate_simple_api_single_type(self):
        """Simple API with single signal type."""
        calc = ConfidenceCalculator()
        result = calc.calculate([SignalType.WEB])
        assert result == 0.7

    def test_calculate_simple_api_multiple_types(self):
        """Simple API with multiple signal types adds corroboration."""
        calc = ConfidenceCalculator()
        result = calc.calculate([SignalType.WEB, SignalType.COMMUNITY])
        # Base is 0.75 (community), plus 0.05 boost
        assert result == 0.8

    def test_calculate_full_api_with_transmissions(self):
        """Full API with transmissions and signal types."""
        calc = ConfidenceCalculator()
        trans = [Transmission(signal_id="sig_1", excerpt="test", confidence=0.9)]
        result = calc.calculate(trans, [SignalType.DOCUMENT])
        assert result == 0.855  # 0.95 * 0.9

    def test_calculate_empty_list_returns_zero(self):
        """Empty transmission list returns 0.0."""
        calc = ConfidenceCalculator()
        result = calc.calculate([])
        assert result == 0.0

    def test_corroboration_boost(self):
        """Multiple sources increase confidence."""
        calc = ConfidenceCalculator()
        single = calc.calculate_single(SignalType.WEB)
        trans1 = Transmission(signal_id="sig_1", excerpt="test", confidence=1.0)
        trans2 = Transmission(signal_id="sig_2", excerpt="test", confidence=1.0)
        multiple = calc.calculate([trans1, trans2], [SignalType.WEB, SignalType.COMMUNITY])
        assert multiple > single

    def test_corroboration_boost_capped_at_one(self):
        """Corroboration cannot exceed 1.0."""
        calc = ConfidenceCalculator()
        # Many high-confidence sources
        types = [SignalType.DOCUMENT] * 10
        result = calc.calculate(types)
        assert result <= 1.0

    def test_custom_corroboration_boost(self):
        """Custom corroboration boost value."""
        calc = ConfidenceCalculator(corroboration_boost=0.1)
        assert calc.corroboration_boost == 0.1

    def test_calculate_pads_signal_types(self):
        """Calculate pads signal_types with INFERENCE if too short."""
        calc = ConfidenceCalculator()
        trans = [
            Transmission(signal_id="sig_1", excerpt="test", confidence=1.0),
            Transmission(signal_id="sig_2", excerpt="test", confidence=1.0),
        ]
        # Only provide one signal type
        result = calc.calculate(trans, [SignalType.DOCUMENT])
        assert result > 0.0


# ============================================================================
# SignalsManager Tests
# ============================================================================


@pytest.fixture
def temp_project():
    """Create a temporary project directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def manager(temp_project):
    """Create a SignalsManager with temp project."""
    return SignalsManager(temp_project)


class TestSignalsManagerInit:
    """Tests for SignalsManager initialization."""

    def test_creates_signals_directory(self, temp_project):
        """Manager creates .signals directory."""
        SignalsManager(temp_project)
        assert (temp_project / ".signals").is_dir()

    def test_empty_start(self, manager):
        """Manager starts with no signals or citations."""
        assert len(manager.all_signals()) == 0
        assert len(manager.all_citations()) == 0


class TestSignalsManagerSignalCRUD:
    """Tests for SignalsManager signal operations."""

    def test_register_signal(self, manager):
        """Can register a new signal."""
        signal = manager.register_signal(
            signal_type=SignalType.DOCUMENT,
            title="Test Document",
            path="/path/to/doc.pdf",
        )
        assert signal.id.startswith("sig_")
        assert signal.title == "Test Document"
        assert signal.signal_type == SignalType.DOCUMENT

    def test_register_signal_with_metadata(self, manager):
        """Can register signal with additional metadata."""
        signal = manager.register_signal(
            signal_type=SignalType.WEB,
            title="Web Article",
            url="https://example.com/article",
            author="Test Author",
            page_count=10,
        )
        assert signal.metadata["author"] == "Test Author"
        assert signal.metadata["page_count"] == 10

    def test_register_signal_with_hash(self, manager):
        """Can register signal with pre-computed hash."""
        signal = manager.register_signal(
            signal_type=SignalType.DOCUMENT,
            title="Hashed Doc",
            content_hash="abc123def456",
        )
        assert signal.hash == "abc123def456"

    def test_get_signal(self, manager):
        """Can retrieve signal by ID."""
        created = manager.register_signal(
            signal_type=SignalType.CODE,
            title="Config File",
        )
        retrieved = manager.get_signal(created.id)
        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.title == "Config File"

    def test_get_nonexistent_signal(self, manager):
        """Getting nonexistent signal returns None."""
        result = manager.get_signal("sig_doesnotexist")
        assert result is None

    def test_delete_signal(self, manager):
        """Can delete a signal."""
        signal = manager.register_signal(
            signal_type=SignalType.WEB,
            title="To Delete",
        )
        assert manager.delete_signal(signal.id) is True
        assert manager.get_signal(signal.id) is None

    def test_delete_nonexistent_signal(self, manager):
        """Deleting nonexistent signal returns False."""
        result = manager.delete_signal("sig_doesnotexist")
        assert result is False

    def test_register_signal_from_source_type(self, manager):
        """Can register signal using SourceType."""
        signal = manager.register_signal_from_source_type(
            source_type=SourceType.WEB_COMMUNITY,
            title="Forum Post",
            url="https://forums.example.com/post/123",
        )
        assert signal.signal_type == SignalType.COMMUNITY


class TestSignalsManagerSignalSearch:
    """Tests for SignalsManager signal search operations."""

    def test_find_signals_by_path(self, manager):
        """Can find signals by file path."""
        manager.register_signal(SignalType.DOCUMENT, "Doc 1", path="/docs/a.pdf")
        manager.register_signal(SignalType.DOCUMENT, "Doc 2", path="/docs/b.pdf")
        manager.register_signal(SignalType.DOCUMENT, "Doc 3", path="/docs/a.pdf")

        results = manager.find_signals_by_path("/docs/a.pdf")
        assert len(results) == 2

    def test_find_signals_by_url(self, manager):
        """Can find signals by URL."""
        manager.register_signal(SignalType.WEB, "Page 1", url="https://a.com")
        manager.register_signal(SignalType.WEB, "Page 2", url="https://b.com")

        results = manager.find_signals_by_url("https://a.com")
        assert len(results) == 1
        assert results[0].title == "Page 1"

    def test_find_signals_by_type(self, manager):
        """Can find signals by type."""
        manager.register_signal(SignalType.DOCUMENT, "Doc")
        manager.register_signal(SignalType.WEB, "Web")
        manager.register_signal(SignalType.DOCUMENT, "Doc 2")

        results = manager.find_signals_by_type(SignalType.DOCUMENT)
        assert len(results) == 2

    def test_find_or_create_signal_finds_by_path(self, manager):
        """find_or_create_signal returns existing signal by path."""
        original = manager.register_signal(SignalType.DOCUMENT, "Original", path="/doc.pdf")
        found = manager.find_or_create_signal(
            SignalType.DOCUMENT,
            "New Title",
            path="/doc.pdf",
        )
        assert found.id == original.id

    def test_find_or_create_signal_finds_by_url(self, manager):
        """find_or_create_signal returns existing signal by URL."""
        original = manager.register_signal(
            SignalType.WEB,
            "Original",
            url="https://example.com",
        )
        found = manager.find_or_create_signal(
            SignalType.WEB,
            "New Title",
            url="https://example.com",
        )
        assert found.id == original.id

    def test_find_or_create_signal_creates_new(self, manager):
        """find_or_create_signal creates new when not found."""
        signal = manager.find_or_create_signal(
            SignalType.COMMUNITY,
            "New Signal",
            url="https://new.example.com",
        )
        assert signal.title == "New Signal"


class TestSignalsManagerCitationCRUD:
    """Tests for SignalsManager citation operations."""

    def test_add_citation(self, manager):
        """Can add a citation."""
        signal = manager.register_signal(SignalType.DOCUMENT, "Manual")
        trans = Transmission(signal_id=signal.id, excerpt="quoted text", location="page 5")
        citation = manager.add_citation(
            statement="A factual claim",
            transmissions=[trans],
        )
        assert citation.id.startswith("cite_")
        assert citation.statement == "A factual claim"
        assert len(citation.transmissions) == 1

    def test_add_citation_auto_confidence(self, manager):
        """Citation auto-calculates confidence from transmissions."""
        signal = manager.register_signal(SignalType.DOCUMENT, "Manual")
        trans = Transmission(signal_id=signal.id, excerpt="quote", confidence=1.0)
        citation = manager.add_citation(
            statement="A fact",
            transmissions=[trans],
        )
        # DOCUMENT type has 0.95 weight
        assert citation.confidence == 0.95

    def test_add_citation_override_confidence(self, manager):
        """Can override auto-calculated confidence."""
        signal = manager.register_signal(SignalType.DOCUMENT, "Manual")
        trans = Transmission(signal_id=signal.id, excerpt="quote")
        citation = manager.add_citation(
            statement="A fact",
            transmissions=[trans],
            confidence=0.5,
        )
        assert citation.confidence == 0.5

    def test_get_citation(self, manager):
        """Can retrieve citation by ID."""
        signal = manager.register_signal(SignalType.WEB, "Page")
        trans = Transmission(signal_id=signal.id, excerpt="quote")
        created = manager.add_citation("A fact", [trans])

        retrieved = manager.get_citation(created.id)
        assert retrieved is not None
        assert retrieved.id == created.id

    def test_get_nonexistent_citation(self, manager):
        """Getting nonexistent citation returns None."""
        result = manager.get_citation("cite_doesnotexist")
        assert result is None

    def test_delete_citation(self, manager):
        """Can delete a citation."""
        signal = manager.register_signal(SignalType.WEB, "Page")
        trans = Transmission(signal_id=signal.id, excerpt="quote")
        citation = manager.add_citation("A fact", [trans])

        assert manager.delete_citation(citation.id) is True
        assert manager.get_citation(citation.id) is None

    def test_delete_nonexistent_citation(self, manager):
        """Deleting nonexistent citation returns False."""
        result = manager.delete_citation("cite_doesnotexist")
        assert result is False


class TestSignalsManagerCitationSearch:
    """Tests for citation search operations."""

    def test_search_citations(self, manager):
        """Can search citations by statement text."""
        signal = manager.register_signal(SignalType.WEB, "Page")
        trans = Transmission(signal_id=signal.id, excerpt="quote")
        manager.add_citation("The timing belt interval is 100k miles", [trans])
        manager.add_citation("Oil capacity is 5 quarts", [trans])

        results = manager.search_citations("timing belt")
        assert len(results) == 1
        assert "timing belt" in results[0].statement.lower()

    def test_search_citations_case_insensitive(self, manager):
        """Citation search is case insensitive."""
        signal = manager.register_signal(SignalType.WEB, "Page")
        trans = Transmission(signal_id=signal.id, excerpt="quote")
        manager.add_citation("TIMING BELT info", [trans])

        results = manager.search_citations("timing belt")
        assert len(results) == 1

    def test_search_citations_sorted_by_confidence(self, manager):
        """Search results sorted by confidence descending."""
        signal = manager.register_signal(SignalType.WEB, "Page")
        trans = Transmission(signal_id=signal.id, excerpt="quote")
        manager.add_citation("Engine spec low", [trans], confidence=0.5)
        manager.add_citation("Engine spec high", [trans], confidence=0.9)
        manager.add_citation("Engine spec mid", [trans], confidence=0.7)

        results = manager.search_citations("Engine spec")
        confidences = [c.confidence for c in results]
        assert confidences == sorted(confidences, reverse=True)

    def test_search_citations_limit(self, manager):
        """Search respects limit parameter."""
        signal = manager.register_signal(SignalType.WEB, "Page")
        trans = Transmission(signal_id=signal.id, excerpt="quote")
        for i in range(10):
            manager.add_citation(f"Fact number {i}", [trans])

        results = manager.search_citations("Fact", limit=5)
        assert len(results) == 5


class TestSignalsManagerCitationLinks:
    """Tests for citation-axiom linking."""

    def test_link_citation_to_axiom(self, manager):
        """Can link citation to axiom."""
        signal = manager.register_signal(SignalType.WEB, "Page")
        trans = Transmission(signal_id=signal.id, excerpt="quote")
        citation = manager.add_citation("A fact", [trans])

        manager.link_citation_to_axiom(citation.id, "axiom_123")
        assert "axiom_123" in citation.used_in

    def test_link_citation_no_duplicates(self, manager):
        """Linking same axiom twice doesn't create duplicates."""
        signal = manager.register_signal(SignalType.WEB, "Page")
        trans = Transmission(signal_id=signal.id, excerpt="quote")
        citation = manager.add_citation("A fact", [trans])

        manager.link_citation_to_axiom(citation.id, "axiom_123")
        manager.link_citation_to_axiom(citation.id, "axiom_123")
        assert citation.used_in.count("axiom_123") == 1

    def test_unlink_citation_from_axiom(self, manager):
        """Can unlink citation from axiom."""
        signal = manager.register_signal(SignalType.WEB, "Page")
        trans = Transmission(signal_id=signal.id, excerpt="quote")
        citation = manager.add_citation("A fact", [trans])

        manager.link_citation_to_axiom(citation.id, "axiom_123")
        manager.unlink_citation_from_axiom(citation.id, "axiom_123")
        assert "axiom_123" not in citation.used_in


class TestSignalsManagerProvenanceChain:
    """Tests for provenance chain retrieval."""

    def test_get_citation_chain(self, manager):
        """Can get full provenance chain for citation."""
        signal = manager.register_signal(
            SignalType.DOCUMENT,
            "Service Manual",
            path="/docs/manual.pdf",
        )
        trans = Transmission(
            signal_id=signal.id,
            excerpt="Timing belt at 105k miles",
            location="page 42",
            confidence=0.95,
        )
        citation = manager.add_citation("Belt interval is 105k", [trans])

        chain = manager.get_citation_chain(citation.id)
        assert "citation" in chain
        assert chain["citation"]["id"] == citation.id
        assert "sources" in chain
        assert len(chain["sources"]) == 1
        assert chain["sources"][0]["signal"]["title"] == "Service Manual"

    def test_get_citation_chain_nonexistent(self, manager):
        """Getting chain for nonexistent citation returns empty dict."""
        chain = manager.get_citation_chain("cite_doesnotexist")
        assert chain == {}

    def test_get_axiom_provenance(self, manager):
        """Can get provenance chains for axiom."""
        signal = manager.register_signal(SignalType.DOCUMENT, "Manual")
        trans = Transmission(signal_id=signal.id, excerpt="quote")
        cite1 = manager.add_citation("Fact 1", [trans])
        cite2 = manager.add_citation("Fact 2", [trans])

        manager.link_citation_to_axiom(cite1.id, "axiom_123")
        manager.link_citation_to_axiom(cite2.id, "axiom_123")

        chains = manager.get_axiom_provenance("axiom_123")
        assert len(chains) == 2


class TestSignalsManagerPersistence:
    """Tests for YAML persistence."""

    def test_signals_file_created(self, temp_project):
        """Sources file is created on first save."""
        manager = SignalsManager(temp_project)
        manager.register_signal(SignalType.WEB, "Test")
        assert (temp_project / ".signals" / "sources.yaml").exists()

    def test_citations_file_created(self, temp_project):
        """Citations file is created on first save."""
        manager = SignalsManager(temp_project)
        signal = manager.register_signal(SignalType.WEB, "Test")
        trans = Transmission(signal_id=signal.id, excerpt="quote")
        manager.add_citation("Fact", [trans])
        assert (temp_project / ".signals" / "citations.yaml").exists()

    def test_signals_persist_across_instances(self, temp_project):
        """Signals persist when manager is recreated."""
        manager1 = SignalsManager(temp_project)
        signal = manager1.register_signal(
            SignalType.DOCUMENT,
            "Persistent Doc",
            path="/docs/persist.pdf",
        )
        signal_id = signal.id

        manager2 = SignalsManager(temp_project)
        loaded = manager2.get_signal(signal_id)

        assert loaded is not None
        assert loaded.title == "Persistent Doc"
        assert loaded.signal_type == SignalType.DOCUMENT

    def test_citations_persist_across_instances(self, temp_project):
        """Citations persist when manager is recreated."""
        manager1 = SignalsManager(temp_project)
        signal = manager1.register_signal(SignalType.WEB, "Page")
        trans = Transmission(signal_id=signal.id, excerpt="quote", location="line 5")
        citation = manager1.add_citation("Persistent fact", [trans])
        cite_id = citation.id

        manager2 = SignalsManager(temp_project)
        loaded = manager2.get_citation(cite_id)

        assert loaded is not None
        assert loaded.statement == "Persistent fact"
        assert len(loaded.transmissions) == 1
        assert loaded.transmissions[0].location == "line 5"

    def test_used_in_persists(self, temp_project):
        """Citation used_in links persist."""
        manager1 = SignalsManager(temp_project)
        signal = manager1.register_signal(SignalType.WEB, "Page")
        trans = Transmission(signal_id=signal.id, excerpt="quote")
        citation = manager1.add_citation("Fact", [trans])
        manager1.link_citation_to_axiom(citation.id, "axiom_123")
        cite_id = citation.id

        manager2 = SignalsManager(temp_project)
        loaded = manager2.get_citation(cite_id)
        assert "axiom_123" in loaded.used_in


class TestSignalsManagerStats:
    """Tests for statistics and utilities."""

    def test_get_stats_empty(self, manager):
        """Stats on empty manager."""
        stats = manager.get_stats()
        assert stats["total_signals"] == 0
        assert stats["total_citations"] == 0
        assert stats["avg_confidence"] == 0.0

    def test_get_stats_with_data(self, manager):
        """Stats with signals and citations."""
        manager.register_signal(SignalType.DOCUMENT, "Doc")
        manager.register_signal(SignalType.WEB, "Web")
        signal = manager.register_signal(SignalType.CODE, "Code")
        trans = Transmission(signal_id=signal.id, excerpt="quote")
        manager.add_citation("Fact 1", [trans], confidence=0.8)
        manager.add_citation("Fact 2", [trans], confidence=0.6)

        stats = manager.get_stats()
        assert stats["total_signals"] == 3
        assert stats["total_citations"] == 2
        assert stats["avg_confidence"] == 0.7
        assert "document" in stats["signals_by_type"]

    def test_compute_content_hash_string(self, manager):
        """Can compute hash from string content."""
        hash1 = manager.compute_content_hash("content")
        hash2 = manager.compute_content_hash("content")
        hash3 = manager.compute_content_hash("different")
        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 16

    def test_compute_content_hash_bytes(self, manager):
        """Can compute hash from bytes content."""
        hash_result = manager.compute_content_hash(b"binary content")
        assert len(hash_result) == 16

    def test_all_signals(self, manager):
        """all_signals returns all registered signals."""
        manager.register_signal(SignalType.DOCUMENT, "Doc 1")
        manager.register_signal(SignalType.WEB, "Web 1")

        signals = manager.all_signals()
        assert len(signals) == 2

    def test_all_citations(self, manager):
        """all_citations returns all citations."""
        signal = manager.register_signal(SignalType.WEB, "Page")
        trans = Transmission(signal_id=signal.id, excerpt="quote")
        manager.add_citation("Fact 1", [trans])
        manager.add_citation("Fact 2", [trans])

        citations = manager.all_citations()
        assert len(citations) == 2
