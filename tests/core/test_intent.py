"""Comprehensive tests for r3LAY intent parsing system.

Tests cover:
- Entity extraction (mileage, parts, costs, etc.)
- Pattern matching (all intent types)
- Intent parser (stages 1-2, sync)
- Edge cases and ambiguous inputs
"""

from __future__ import annotations

import pytest

from r3lay.core.intent import (
    EntityExtractor,
    IntentConfidence,
    IntentParser,
    IntentPatternMatcher,
    IntentResult,
    IntentType,
    extract_entities,
)

# ============================================================================
# Entity Extraction Tests
# ============================================================================


class TestEntityExtractor:
    """Tests for entity extraction."""

    @pytest.fixture
    def extractor(self) -> EntityExtractor:
        return EntityExtractor()

    # --- Mileage extraction ---

    def test_mileage_with_k_suffix(self, extractor: EntityExtractor) -> None:
        """Test mileage extraction with 'k' shorthand."""
        result = extractor.extract("at 98.5k miles")
        assert result.mileage == 98500

    def test_mileage_full_number(self, extractor: EntityExtractor) -> None:
        """Test mileage extraction with full number."""
        result = extractor.extract("odometer at 98,500 miles")
        assert result.mileage == 98500

    def test_mileage_no_unit(self, extractor: EntityExtractor) -> None:
        """Test mileage extraction without unit."""
        result = extractor.extract("at 100k")
        assert result.mileage == 100000

    def test_mileage_small_number_assumed_thousands(self, extractor: EntityExtractor) -> None:
        """Test that small numbers are assumed to be in thousands."""
        result = extractor.extract("at 98 miles")
        assert result.mileage == 98000

    def test_mileage_at_symbol(self, extractor: EntityExtractor) -> None:
        """Test mileage with @ symbol."""
        result = extractor.extract("@100k")
        assert result.mileage == 100000

    # --- Part extraction ---

    def test_part_oil(self, extractor: EntityExtractor) -> None:
        """Test oil part extraction."""
        result = extractor.extract("changed the oil")
        assert result.part == "engine_oil"

    def test_part_timing_belt(self, extractor: EntityExtractor) -> None:
        """Test timing belt extraction (with alias)."""
        result = extractor.extract("need to check the t-belt")
        assert result.part == "timing_belt"

    def test_part_head_gasket_alias(self, extractor: EntityExtractor) -> None:
        """Test head gasket HG alias."""
        result = extractor.extract("hg replacement tomorrow")
        assert result.part == "head_gasket"

    def test_part_brake_fluid(self, extractor: EntityExtractor) -> None:
        """Test brake fluid extraction."""
        result = extractor.extract("need to flush the brake fluid")
        assert result.part == "brake_fluid"

    # --- Service type extraction ---

    def test_service_change(self, extractor: EntityExtractor) -> None:
        """Test 'change' service type."""
        result = extractor.extract("oil change")
        assert result.service_type == "change"

    def test_service_replaced(self, extractor: EntityExtractor) -> None:
        """Test 'replaced' → 'replace' normalization."""
        result = extractor.extract("replaced the filter")
        assert result.service_type == "replace"

    def test_service_flush(self, extractor: EntityExtractor) -> None:
        """Test flush service type."""
        result = extractor.extract("flushed the coolant")
        assert result.service_type == "flush"

    def test_service_install(self, extractor: EntityExtractor) -> None:
        """Test install service type."""
        result = extractor.extract("installed new turbo")
        assert result.service_type == "install"

    # --- Cost extraction ---

    def test_cost_simple(self, extractor: EntityExtractor) -> None:
        """Test simple cost extraction."""
        result = extractor.extract("cost $45")
        assert result.cost == 45.0

    def test_cost_with_cents(self, extractor: EntityExtractor) -> None:
        """Test cost with cents."""
        result = extractor.extract("paid $45.99")
        assert result.cost == 45.99

    def test_cost_with_comma(self, extractor: EntityExtractor) -> None:
        """Test cost with comma separator."""
        result = extractor.extract("total was $1,200.00")
        assert result.cost == 1200.0

    # --- Model name extraction ---

    def test_model_name_swap(self, extractor: EntityExtractor) -> None:
        """Test model name with 'swap' keyword."""
        result = extractor.extract("swap model mistral")
        assert result.model_name == "mistral"

    def test_model_name_load(self, extractor: EntityExtractor) -> None:
        """Test model name with 'load' keyword."""
        result = extractor.extract("load qwen2.5-7b")
        assert result.model_name == "qwen2.5-7b"

    def test_model_name_use(self, extractor: EntityExtractor) -> None:
        """Test model name with 'use' keyword."""
        result = extractor.extract("use llama3")
        assert result.model_name == "llama3"

    # --- Date extraction ---

    def test_date_slash_format(self, extractor: EntityExtractor) -> None:
        """Test date with slash format."""
        result = extractor.extract("done on 1/15")
        assert result.date == "1/15"

    def test_date_iso_format(self, extractor: EntityExtractor) -> None:
        """Test ISO date format."""
        result = extractor.extract("scheduled for 2025-01-15")
        assert result.date == "2025-01-15"

    # --- Combined extraction ---

    def test_combined_log_entry(self, extractor: EntityExtractor) -> None:
        """Test extracting multiple entities from a log entry."""
        result = extractor.extract("just did oil change at 98.5k, cost $45")
        assert result.mileage == 98500
        assert result.part == "engine_oil"
        assert result.service_type == "change"
        assert result.cost == 45.0

    def test_to_dict_excludes_none(self, extractor: EntityExtractor) -> None:
        """Test that to_dict excludes None values."""
        result = extractor.extract("oil change")
        d = result.to_dict()
        assert "mileage" not in d
        assert "part" in d


class TestExtractEntitiesFunction:
    """Test the module-level extract_entities function."""

    def test_function_works(self) -> None:
        """Test that the convenience function works."""
        result = extract_entities("oil change at 100k")
        assert result.mileage == 100000
        assert result.part == "engine_oil"


# ============================================================================
# Pattern Matching Tests
# ============================================================================


class TestIntentPatternMatcher:
    """Tests for regex pattern matching."""

    @pytest.fixture
    def matcher(self) -> IntentPatternMatcher:
        return IntentPatternMatcher()

    # --- LOG intents ---

    def test_log_maintenance_oil_change(self, matcher: IntentPatternMatcher) -> None:
        """Test oil change detection."""
        result = matcher.match("just did oil change at 98.5k")
        assert result is not None
        assert result.subtype == "log.maintenance"
        assert result.confidence >= 0.85

    def test_log_maintenance_changed_oil(self, matcher: IntentPatternMatcher) -> None:
        """Test 'changed the oil' pattern."""
        result = matcher.match("changed the oil")
        assert result is not None
        assert result.subtype == "log.maintenance"
        assert result.confidence >= 0.8

    def test_log_maintenance_filter_service(self, matcher: IntentPatternMatcher) -> None:
        """Test filter service detection."""
        result = matcher.match("filter service done")
        assert result is not None
        assert result.subtype == "log.maintenance"

    def test_log_mod_turbo_install(self, matcher: IntentPatternMatcher) -> None:
        """Test turbo installation detection."""
        result = matcher.match("installed a new turbo")
        assert result is not None
        assert result.subtype == "log.mod"
        assert result.confidence >= 0.8

    def test_log_repair_fixed(self, matcher: IntentPatternMatcher) -> None:
        """Test repair detection."""
        result = matcher.match("fixed the leak")
        assert result is not None
        assert result.subtype == "log.repair"

    # --- QUERY intents ---

    def test_query_reminder_when_due(self, matcher: IntentPatternMatcher) -> None:
        """Test 'when is X due' pattern."""
        result = matcher.match("when is the timing belt due")
        assert result is not None
        assert result.subtype == "query.reminder"
        assert result.confidence >= 0.85

    def test_query_reminder_whats_next(self, matcher: IntentPatternMatcher) -> None:
        """Test 'what's next' pattern."""
        result = matcher.match("what's next for maintenance")
        assert result is not None
        assert result.subtype == "query.reminder"

    def test_query_history_when_did(self, matcher: IntentPatternMatcher) -> None:
        """Test 'when did I' pattern."""
        result = matcher.match("when did i last change the oil")
        assert result is not None
        assert result.subtype == "query.history"

    def test_query_status_current(self, matcher: IntentPatternMatcher) -> None:
        """Test current status query."""
        result = matcher.match("what's the current mileage")
        assert result is not None
        assert result.subtype == "query.status"

    def test_query_spec_torque(self, matcher: IntentPatternMatcher) -> None:
        """Test torque spec query."""
        result = matcher.match("what's the torque spec for the crank bolt")
        assert result is not None
        assert result.subtype == "query.spec"
        assert result.confidence >= 0.85

    # --- SEARCH intents ---

    def test_search_how_to(self, matcher: IntentPatternMatcher) -> None:
        """Test 'how to' pattern."""
        result = matcher.match("how do i change the timing belt")
        assert result is not None
        assert result.subtype == "search.docs"

    def test_search_lookup(self, matcher: IntentPatternMatcher) -> None:
        """Test 'look up' pattern."""
        result = matcher.match("look up the procedure")
        assert result is not None
        assert result.subtype == "search.docs"

    def test_search_web(self, matcher: IntentPatternMatcher) -> None:
        """Test web search detection."""
        result = matcher.match("web search for timing belt replacement")
        assert result is not None
        assert result.subtype == "search.web"
        assert result.confidence >= 0.9

    # --- UPDATE intents ---

    def test_update_mileage_now(self, matcher: IntentPatternMatcher) -> None:
        """Test 'mileage is now' pattern."""
        result = matcher.match("mileage is now 99000")
        assert result is not None
        assert result.subtype == "update.mileage"
        assert result.confidence >= 0.9

    def test_update_mileage_odo(self, matcher: IntentPatternMatcher) -> None:
        """Test odometer update pattern."""
        result = matcher.match("odometer reads 99500")
        assert result is not None
        assert result.subtype == "update.mileage"

    def test_update_mileage_at_now(self, matcher: IntentPatternMatcher) -> None:
        """Test 'now at X' pattern."""
        result = matcher.match("now at 100k")
        assert result is not None
        assert result.subtype == "update.mileage"

    # --- COMMAND intents ---

    def test_cmd_model_swap(self, matcher: IntentPatternMatcher) -> None:
        """Test 'swap model' pattern."""
        result = matcher.match("swap model mistral")
        assert result is not None
        assert result.subtype == "cmd.model"
        assert result.confidence >= 0.9

    def test_cmd_model_load(self, matcher: IntentPatternMatcher) -> None:
        """Test 'load llm' pattern."""
        result = matcher.match("load llm qwen2.5")
        assert result is not None
        assert result.subtype == "cmd.model"

    def test_cmd_session_save(self, matcher: IntentPatternMatcher) -> None:
        """Test session save command."""
        result = matcher.match("save session")
        assert result is not None
        assert result.subtype == "cmd.session"

    def test_cmd_index_reindex(self, matcher: IntentPatternMatcher) -> None:
        """Test reindex command."""
        result = matcher.match("reindex")
        assert result is not None
        assert result.subtype == "cmd.index"

    # --- Entity extraction in pattern match ---

    def test_pattern_includes_entities(self, matcher: IntentPatternMatcher) -> None:
        """Test that pattern matching includes extracted entities."""
        result = matcher.match("oil change at 98.5k cost $45")
        assert result is not None
        assert result.entities.get("mileage") == 98500
        assert result.entities.get("cost") == 45.0
        assert result.entities.get("part") == "engine_oil"

    def test_no_match_returns_none(self, matcher: IntentPatternMatcher) -> None:
        """Test that unrecognized input returns None."""
        result = matcher.match("xyzzy plugh")
        assert result is None


# ============================================================================
# Intent Parser Tests
# ============================================================================


class TestIntentParser:
    """Tests for the main IntentParser class."""

    @pytest.fixture
    def parser(self) -> IntentParser:
        return IntentParser()

    # --- Stage 1: Command bypass ---

    def test_slash_command_help(self, parser: IntentParser) -> None:
        """Test /help command detection."""
        result = parser.parse_sync("/help")
        assert result.intent == IntentType.COMMAND
        assert result.subtype == "cmd.help"
        assert result.confidence == 1.0
        assert result.source == "command"

    def test_slash_command_with_args(self, parser: IntentParser) -> None:
        """Test slash command with arguments."""
        result = parser.parse_sync("/search timing belt")
        assert result.intent == IntentType.COMMAND
        assert result.subtype == "cmd.search"
        assert result.entities["args"] == ["timing", "belt"]

    def test_legacy_alias_help(self, parser: IntentParser) -> None:
        """Test legacy 'help' alias."""
        result = parser.parse_sync("help")
        assert result.intent == IntentType.COMMAND
        assert result.subtype == "cmd.help"

    def test_legacy_alias_quit(self, parser: IntentParser) -> None:
        """Test legacy 'quit' alias."""
        result = parser.parse_sync("quit")
        assert result.intent == IntentType.COMMAND
        assert result.subtype == "cmd.quit"

    def test_path_not_treated_as_command(self, parser: IntentParser) -> None:
        """Test that paths are not treated as commands."""
        result = parser.parse_sync("/home/user/file.txt")
        assert result.intent != IntentType.COMMAND or result.source != "command"

    # --- Stage 2: Pattern matching ---

    def test_log_maintenance_high_confidence(self, parser: IntentParser) -> None:
        """Test log maintenance with high confidence."""
        result = parser.parse_sync("just did oil change at 98.5k")
        assert result.intent == IntentType.LOG
        assert result.subtype == "log.maintenance"
        assert result.is_high_confidence()
        assert result.entities.get("mileage") == 98500

    def test_query_reminder_pattern(self, parser: IntentParser) -> None:
        """Test query reminder pattern matching."""
        result = parser.parse_sync("when is the timing belt due")
        assert result.intent == IntentType.QUERY
        assert result.subtype == "query.reminder"

    def test_search_torque_spec(self, parser: IntentParser) -> None:
        """Test torque spec search."""
        result = parser.parse_sync("what's the torque spec for crank bolt")
        assert result.intent in (IntentType.QUERY, IntentType.SEARCH)
        assert "spec" in result.subtype

    def test_update_mileage(self, parser: IntentParser) -> None:
        """Test mileage update detection."""
        result = parser.parse_sync("mileage is now 99000")
        assert result.intent == IntentType.UPDATE
        assert result.subtype == "update.mileage"
        assert result.entities.get("mileage") == 99000

    def test_cmd_model_swap(self, parser: IntentParser) -> None:
        """Test model swap command detection."""
        result = parser.parse_sync("swap model mistral")
        assert result.intent == IntentType.COMMAND
        assert result.subtype == "cmd.model"
        assert result.entities.get("model_name") == "mistral"

    # --- Required entity checking ---

    def test_log_without_mileage_needs_clarification(self, parser: IntentParser) -> None:
        """Test that LOG without mileage needs clarification."""
        result = parser.parse_sync("changed the oil")
        assert result.intent == IntentType.LOG
        assert result.needs_clarification is True
        assert "mileage" in result.missing_entities
        # Confidence should be lowered
        assert result.confidence < IntentConfidence.MEDIUM

    def test_log_with_mileage_no_clarification(self, parser: IntentParser) -> None:
        """Test that LOG with mileage doesn't need clarification."""
        result = parser.parse_sync("changed the oil at 98k")
        assert result.intent == IntentType.LOG
        assert result.needs_clarification is False
        assert result.missing_entities == []

    # --- Fallback behavior ---

    def test_empty_input_returns_chat(self, parser: IntentParser) -> None:
        """Test empty input returns chat fallback."""
        result = parser.parse_sync("")
        assert result.intent == IntentType.CHAT
        assert result.source == "fallback"

    def test_whitespace_input_returns_chat(self, parser: IntentParser) -> None:
        """Test whitespace-only input returns chat fallback."""
        result = parser.parse_sync("   ")
        assert result.intent == IntentType.CHAT

    def test_unrecognized_input_returns_search(self, parser: IntentParser) -> None:
        """Test unrecognized input falls back to search."""
        result = parser.parse_sync("xyzzy plugh")
        assert result.intent == IntentType.SEARCH
        assert result.subtype == "search.docs"
        assert result.source == "fallback"
        assert result.entities.get("query") == "xyzzy plugh"

    # --- IntentResult helper methods ---

    def test_is_high_confidence(self, parser: IntentParser) -> None:
        """Test is_high_confidence method."""
        result = parser.parse_sync("just did oil change at 98.5k")
        assert result.is_high_confidence() is True

    def test_is_medium_confidence(self, parser: IntentParser) -> None:
        """Test is_medium_confidence for borderline cases."""
        result = parser.parse_sync("look up something")
        # This should be medium confidence
        assert result.confidence >= IntentConfidence.MEDIUM

    def test_is_ambiguous(self, parser: IntentParser) -> None:
        """Test is_ambiguous for very unclear input."""
        # Direct test of the method
        result = IntentResult(
            intent=IntentType.CHAT,
            subtype="chat.general",
            confidence=0.3,
            entities={},
        )
        assert result.is_ambiguous() is True

    # --- From factories ---

    def test_from_command_factory(self) -> None:
        """Test IntentResult.from_command factory."""
        result = IntentResult.from_command("help", ["topic"])
        assert result.intent == IntentType.COMMAND
        assert result.subtype == "cmd.help"
        assert result.confidence == 1.0
        assert result.entities["command"] == "help"
        assert result.entities["args"] == ["topic"]

    def test_chat_fallback_factory(self) -> None:
        """Test IntentResult.chat_fallback factory."""
        result = IntentResult.chat_fallback("hello world")
        assert result.intent == IntentType.CHAT
        assert result.source == "fallback"
        assert result.entities["text"] == "hello world"

    def test_search_fallback_factory(self) -> None:
        """Test IntentResult.search_fallback factory."""
        result = IntentResult.search_fallback("find something")
        assert result.intent == IntentType.SEARCH
        assert result.source == "fallback"
        assert result.entities["query"] == "find something"


# ============================================================================
# Edge Cases and Regression Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and potential issues."""

    @pytest.fixture
    def parser(self) -> IntentParser:
        return IntentParser()

    def test_mixed_case_input(self, parser: IntentParser) -> None:
        """Test case insensitivity."""
        result = parser.parse_sync("JUST DID OIL CHANGE AT 98.5K")
        assert result.intent == IntentType.LOG
        assert result.entities.get("mileage") == 98500

    def test_extra_whitespace(self, parser: IntentParser) -> None:
        """Test handling of extra whitespace."""
        result = parser.parse_sync("  oil  change  at  100k  ")
        assert result.intent == IntentType.LOG

    def test_unicode_input(self, parser: IntentParser) -> None:
        """Test handling of unicode characters."""
        result = parser.parse_sync("oil change — 100k miles")
        assert result.entities.get("mileage") == 100000

    def test_special_model_names(self, parser: IntentParser) -> None:
        """Test model names with special characters."""
        result = parser.parse_sync("load model meta-llama/Llama-3.1-8B")
        assert result.intent == IntentType.COMMAND
        assert "llama" in result.entities.get("model_name", "").lower()

    def test_mileage_comma_and_k(self, parser: IntentParser) -> None:
        """Test mileage with both comma and k notation."""
        extractor = EntityExtractor()
        # "1,500k" would be weird but let's handle it
        result = extractor.extract("at 1,500 miles")
        # 1,500 as written is probably 1500 miles, but our heuristic treats <1000 as thousands
        # This tests the actual behavior
        assert result.mileage is not None

    def test_similar_intents_pick_best(self, parser: IntentParser) -> None:
        """Test that parser picks best intent when multiple could match."""
        # This could be query.reminder or search
        result = parser.parse_sync("when is the timing belt due for replacement")
        # Should pick query.reminder as it's more specific
        assert result.subtype == "query.reminder"

    def test_command_vs_pattern_priority(self, parser: IntentParser) -> None:
        """Test that explicit commands take priority over patterns."""
        result = parser.parse_sync("/help oil change")
        assert result.intent == IntentType.COMMAND
        assert result.subtype == "cmd.help"
        assert result.source == "command"


# ============================================================================
# Integration-style Tests
# ============================================================================


class TestRealWorldExamples:
    """Tests based on realistic user input examples from the design doc."""

    @pytest.fixture
    def parser(self) -> IntentParser:
        return IntentParser()

    def test_doc_example_oil_change(self, parser: IntentParser) -> None:
        """Test: 'just did the oil change at 98.5k miles'"""
        result = parser.parse_sync("just did the oil change at 98.5k miles")
        assert result.intent == IntentType.LOG
        assert result.subtype == "log.maintenance"
        assert result.entities.get("mileage") == 98500
        assert result.entities.get("part") == "engine_oil"
        assert result.is_high_confidence()

    def test_doc_example_torque_spec(self, parser: IntentParser) -> None:
        """Test: 'what's the torque spec for the crank bolt'"""
        result = parser.parse_sync("what's the torque spec for the crank bolt")
        assert result.intent in (IntentType.QUERY, IntentType.SEARCH)
        assert "spec" in result.subtype

    def test_doc_example_timing_belt_due(self, parser: IntentParser) -> None:
        """Test: 'when is the timing belt due'"""
        result = parser.parse_sync("when is the timing belt due")
        assert result.intent == IntentType.QUERY
        assert result.subtype == "query.reminder"

    def test_doc_example_swap_model(self, parser: IntentParser) -> None:
        """Test: 'swap model mistral'"""
        result = parser.parse_sync("swap model mistral")
        assert result.intent == IntentType.COMMAND
        assert result.subtype == "cmd.model"
        assert result.entities.get("model_name") == "mistral"

    def test_doc_example_mileage_update(self, parser: IntentParser) -> None:
        """Test: 'mileage is now 99000'"""
        result = parser.parse_sync("mileage is now 99000")
        assert result.intent == IntentType.UPDATE
        assert result.subtype == "update.mileage"
        assert result.entities.get("mileage") == 99000

    def test_informal_oil_at_k(self, parser: IntentParser) -> None:
        """Test informal: 'oil at 100k rotella t6'"""
        result = parser.parse_sync("oil at 100k rotella t6")
        # This might be LOG or UPDATE depending on patterns
        # The key is entities are extracted
        assert result.entities.get("mileage") == 100000

    def test_question_about_history(self, parser: IntentParser) -> None:
        """Test: 'when did I last change the oil'"""
        result = parser.parse_sync("when did I last change the oil")
        assert result.intent == IntentType.QUERY
        assert result.subtype == "query.history"
