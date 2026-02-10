"""Tests for the ContradictionMonitor."""

import pytest

from r3lay.core.contradiction_monitor import ContradictionMonitor


class TestContradictionMonitor:
    """Test contradiction detection in user messages and LLM responses."""

    def setup_method(self):
        self.monitor = ContradictionMonitor()

    # --- User message phrase detection ---

    def test_detects_user_contradiction_phrase(self):
        msg = "That contradicts what I read in the service manual."
        signal = self.monitor.check_user_message(msg)
        assert signal is not None
        assert signal.source == "user_phrase"
        assert signal.confidence > 0

    def test_detects_but_i_read(self):
        msg = "But I read that the interval is 60k miles, not 105k."
        signal = self.monitor.check_user_message(msg)
        assert signal is not None
        assert signal.source == "user_phrase"

    def test_detects_thats_wrong(self):
        msg = "That's wrong, the torque spec is 80 ft-lbs."
        signal = self.monitor.check_user_message(msg)
        assert signal is not None

    def test_detects_are_you_sure(self):
        msg = "Are you sure about that timing?"
        signal = self.monitor.check_user_message(msg)
        assert signal is not None

    def test_detects_actually(self):
        msg = "Actually, the OEM filter is different for 2015+ models."
        signal = self.monitor.check_user_message(msg)
        assert signal is not None

    def test_no_false_positive_normal_message(self):
        msg = "What's the oil capacity for a 2015 WRX?"
        signal = self.monitor.check_user_message(msg)
        assert signal is None

    def test_no_false_positive_simple_chat(self):
        msg = "Thanks, that's helpful!"
        signal = self.monitor.check_user_message(msg)
        assert signal is None

    # --- LLM response contradiction detection ---

    def test_detects_llm_conflicting_sources(self):
        response = (
            "The timing belt interval is 105,000 miles. However, some sources "
            "suggest it should be replaced at 60,000 miles for severe conditions."
        )
        signal = self.monitor.check_llm_response(response)
        assert signal is not None
        assert signal.source == "llm_response"

    def test_detects_llm_discrepancy(self):
        response = (
            "There is a discrepancy between the official specs and community reports "
            "regarding the head gasket material used in 2006+ models."
        )
        signal = self.monitor.check_llm_response(response)
        assert signal is not None

    def test_detects_llm_inconsistency(self):
        response = (
            "The data shows an inconsistency between the factory service manual "
            "and the technical service bulletin regarding torque values."
        )
        signal = self.monitor.check_llm_response(response)
        assert signal is not None

    def test_no_false_positive_normal_response(self):
        response = (
            "The oil capacity for a 2015 WRX is 5.4 quarts with filter change. "
            "Use 0W-20 synthetic oil as recommended by Subaru."
        )
        signal = self.monitor.check_llm_response(response)
        assert signal is None

    # --- Full analysis ---

    def test_analyze_returns_highest_confidence(self):
        user_msg = "But I read that it's different."
        llm_response = (
            "There is conflicting information between the manual and forums."
        )
        signal = self.monitor.analyze(user_msg, llm_response)
        assert signal is not None
        # Should return the higher confidence one
        assert signal.confidence > 0

    def test_analyze_returns_none_when_clean(self):
        user_msg = "What's the oil capacity?"
        llm_response = "The oil capacity is 5.4 quarts with filter change."
        signal = self.monitor.analyze(user_msg, llm_response)
        assert signal is None

    def test_suggested_query_not_empty(self):
        msg = "That contradicts the factory service manual data."
        signal = self.monitor.check_user_message(msg)
        assert signal is not None
        assert len(signal.suggested_query) > 0
