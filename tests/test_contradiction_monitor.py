"""Tests for the ContradictionMonitor."""

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
        # Response must be >500 chars to pass min length threshold
        response = (
            "Based on the indexed service manual and community forum data, "
            "the timing belt interval for the EJ22 engine is specified at 105,000 miles "
            "under normal driving conditions. The factory service manual recommends "
            "inspection at every 30,000 miles with replacement at the specified interval. "
            "The belt material is HNBR (Hydrogenated Nitrile Butadiene Rubber) which "
            "provides improved heat resistance over standard NBR compounds. "
            "However, some sources suggest it should be replaced at 60,000 miles "
            "for severe conditions including frequent short trips and extreme temps."
        )
        signal = self.monitor.check_llm_response(response)
        assert signal is not None
        assert signal.source == "llm_response"
        assert signal.flagged_sentence != ""

    def test_detects_llm_discrepancy(self):
        response = (
            "The Subaru EJ25 engine used in the 2006-2009 Impreza and Legacy models "
            "has undergone several revisions to address known issues. The cylinder head "
            "gasket material was changed from composite to multi-layer steel (MLS) in "
            "certain production runs. The factory parts catalog lists part number 11044AA633 "
            "for the updated gasket. Community forums extensively document the head gasket "
            "failure patterns in pre-revision engines. There is a discrepancy between "
            "the official specs and community reports regarding the head gasket material "
            "used in 2006+ models, with some owners reporting composite gaskets in "
            "vehicles that should have received the MLS update."
        )
        signal = self.monitor.check_llm_response(response)
        assert signal is not None

    def test_detects_llm_inconsistency(self):
        response = (
            "The torque specifications for the EJ22 cylinder head bolts are critical "
            "for proper sealing. The factory service manual specifies a multi-step "
            "torque sequence starting at 22 ft-lbs, then 51 ft-lbs, with a final "
            "angle-torque of 90 degrees. The data shows an inconsistency between "
            "the factory service manual and the technical service bulletin regarding "
            "torque values. TSB 02-157-07 revised the final torque specification to "
            "80 degrees for engines with updated head bolts. Community mechanics "
            "report better sealing results with the TSB values."
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
            "The factory service manual specifies the EJ22 timing belt replacement "
            "interval at 105,000 miles under normal driving conditions. The belt uses "
            "HNBR material rated for extended service life. Routine inspection is "
            "recommended at 30,000 mile intervals to check for wear, cracking, and "
            "proper tension. The tensioner and idler pulleys should be replaced at "
            "the same interval as the belt. There is conflicting information between "
            "the manual and forums regarding whether the water pump should be replaced "
            "simultaneously, with most community mechanics recommending it."
        )
        signal = self.monitor.analyze(user_msg, llm_response)
        assert signal is not None
        # user_phrase confidence is 0.7, llm_response is 0.6 — user wins
        assert signal.source == "user_phrase"
        assert signal.confidence == 0.7

    def test_short_llm_response_not_scanned(self):
        """Regression: greetings with contradiction vocabulary should not trigger."""
        response = (
            "Hello! I can help with research, finding discrepancies in documentation, "
            "and resolving conflicting information from different sources."
        )
        signal = self.monitor.check_llm_response(response)
        assert signal is None

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
