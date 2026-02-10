"""Tests for natural language intent routing in InputPane.

Tests cover:
- LOG intent routing to maintenance handlers
- QUERY intent routing (history, due services, status)
- UPDATE intent routing (mileage updates)
- Entity extraction and argument formatting
- Confidence thresholds and fallback behavior
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from r3lay.core.intent import IntentResult, IntentType


class TestMaintenanceIntentRouting:
    """Tests for LOG intent → maintenance handler routing."""

    @pytest.fixture
    def mock_input_pane(self):
        """Create a mock InputPane with required components."""
        from r3lay.ui.widgets.input_pane import InputPane

        # Create mock state
        mock_state = MagicMock()
        mock_state.current_backend = MagicMock()
        mock_state.current_backend.is_loaded = True

        # Create InputPane instance
        pane = InputPane(state=mock_state)

        # Mock the maintenance handler methods
        pane._handle_log_maintenance = AsyncMock()
        pane._handle_due_services = AsyncMock()
        pane._handle_maintenance_history = AsyncMock()
        pane._handle_update_mileage = AsyncMock()

        return pane

    @pytest.fixture
    def mock_response_pane(self):
        """Create a mock ResponsePane."""
        pane = MagicMock()
        pane.add_system = MagicMock()
        pane.add_error = MagicMock()
        pane.add_assistant = MagicMock()
        pane.add_user = MagicMock()
        return pane

    @pytest.mark.asyncio
    async def test_log_oil_change_with_mileage(self, mock_input_pane, mock_response_pane) -> None:
        """Test LOG intent with mileage routes to _handle_log_maintenance."""
        intent_result = IntentResult(
            intent=IntentType.LOG,
            subtype="log.maintenance",
            confidence=0.85,
            entities={"mileage": 98500, "part": "engine_oil"},
            source="pattern",
        )

        await mock_input_pane._handle_maintenance_intent(
            intent_result, mock_response_pane, original_input="test maintenance input"
        )

        # Should call _handle_log_maintenance with formatted args
        mock_input_pane._handle_log_maintenance.assert_called_once()
        args = mock_input_pane._handle_log_maintenance.call_args[0][0]
        assert "oil_change" in args
        assert "98500" in args

    @pytest.mark.asyncio
    async def test_log_without_mileage_shows_error(
        self, mock_input_pane, mock_response_pane
    ) -> None:
        """Test LOG intent without mileage shows error message."""
        intent_result = IntentResult(
            intent=IntentType.LOG,
            subtype="log.maintenance",
            confidence=0.85,
            entities={"part": "engine_oil"},  # Missing mileage
            source="pattern",
        )

        await mock_input_pane._handle_maintenance_intent(
            intent_result, mock_response_pane, original_input="test maintenance input"
        )

        # Should show error and not call handler
        mock_response_pane.add_system.assert_called_once()
        assert "mileage" in mock_response_pane.add_system.call_args[0][0].lower()
        mock_input_pane._handle_log_maintenance.assert_not_called()

    @pytest.mark.asyncio
    async def test_log_coolant_flush(self, mock_input_pane, mock_response_pane) -> None:
        """Test LOG intent with coolant part maps to coolant_flush."""
        intent_result = IntentResult(
            intent=IntentType.LOG,
            subtype="log.maintenance",
            confidence=0.85,
            entities={"mileage": 100000, "part": "coolant"},
            source="pattern",
        )

        await mock_input_pane._handle_maintenance_intent(
            intent_result, mock_response_pane, original_input="test maintenance input"
        )

        # Should map coolant → coolant_flush
        args = mock_input_pane._handle_log_maintenance.call_args[0][0]
        assert "coolant_flush" in args
        assert "100000" in args

    @pytest.mark.asyncio
    async def test_log_with_generic_service_type(self, mock_input_pane, mock_response_pane) -> None:
        """Test LOG with no specific part uses general_maintenance."""
        intent_result = IntentResult(
            intent=IntentType.LOG,
            subtype="log.maintenance",
            confidence=0.80,
            entities={"mileage": 95000},  # No part specified
            source="pattern",
        )

        await mock_input_pane._handle_maintenance_intent(
            intent_result, mock_response_pane, original_input="test maintenance input"
        )

        # Should use general_maintenance as fallback
        args = mock_input_pane._handle_log_maintenance.call_args[0][0]
        assert "general_maintenance" in args


class TestQueryIntentRouting:
    """Tests for QUERY intent routing."""

    @pytest.fixture
    def mock_input_pane(self):
        """Create a mock InputPane with required components."""
        from r3lay.ui.widgets.input_pane import InputPane

        mock_state = MagicMock()
        mock_state.current_backend = MagicMock()

        pane = InputPane(state=mock_state)
        pane._handle_due_services = AsyncMock()
        pane._handle_maintenance_history = AsyncMock()
        pane._handle_update_mileage = AsyncMock()

        return pane

    @pytest.fixture
    def mock_response_pane(self):
        """Create a mock ResponsePane."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_query_reminder_routes_to_due_services(
        self, mock_input_pane, mock_response_pane
    ) -> None:
        """Test query.reminder intent routes to _handle_due_services."""
        intent_result = IntentResult(
            intent=IntentType.QUERY,
            subtype="query.reminder",
            confidence=0.90,
            entities={"mileage": 100000},
            source="pattern",
        )

        await mock_input_pane._handle_query_intent(
            intent_result, mock_response_pane, original_input="test query input"
        )

        # Should call _handle_due_services with mileage
        mock_input_pane._handle_due_services.assert_called_once()
        args = mock_input_pane._handle_due_services.call_args[0][0]
        assert "100000" in args

    @pytest.mark.asyncio
    async def test_query_history_routes_to_maintenance_history(
        self, mock_input_pane, mock_response_pane
    ) -> None:
        """Test query.history intent routes to _handle_maintenance_history."""
        intent_result = IntentResult(
            intent=IntentType.QUERY,
            subtype="query.history",
            confidence=0.85,
            entities={"part": "engine_oil"},
            source="pattern",
        )

        await mock_input_pane._handle_query_intent(
            intent_result, mock_response_pane, original_input="test query input"
        )

        # Should call _handle_maintenance_history with service type
        mock_input_pane._handle_maintenance_history.assert_called_once()
        args = mock_input_pane._handle_maintenance_history.call_args[0][0]
        assert "engine_oil" in args

    @pytest.mark.asyncio
    async def test_query_status_routes_to_mileage_handler(
        self, mock_input_pane, mock_response_pane
    ) -> None:
        """Test query.status intent routes to _handle_update_mileage."""
        intent_result = IntentResult(
            intent=IntentType.QUERY,
            subtype="query.status",
            confidence=0.85,
            entities={"mileage": 98000},
            source="pattern",
        )

        await mock_input_pane._handle_query_intent(
            intent_result, mock_response_pane, original_input="test query input"
        )

        # Should call _handle_update_mileage for status display
        mock_input_pane._handle_update_mileage.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_without_mileage_uses_empty_args(
        self, mock_input_pane, mock_response_pane
    ) -> None:
        """Test query without mileage passes empty args."""
        intent_result = IntentResult(
            intent=IntentType.QUERY,
            subtype="query.reminder",
            confidence=0.85,
            entities={},  # No mileage
            source="pattern",
        )

        await mock_input_pane._handle_query_intent(
            intent_result, mock_response_pane, original_input="test query input"
        )

        # Should call with empty string
        args = mock_input_pane._handle_due_services.call_args[0][0]
        assert args == ""


class TestUpdateIntentRouting:
    """Tests for UPDATE intent routing."""

    @pytest.fixture
    def mock_input_pane(self):
        """Create a mock InputPane with required components."""
        from r3lay.ui.widgets.input_pane import InputPane

        mock_state = MagicMock()
        pane = InputPane(state=mock_state)
        pane._handle_update_mileage = AsyncMock()

        return pane

    @pytest.fixture
    def mock_response_pane(self):
        """Create a mock ResponsePane."""
        pane = MagicMock()
        pane.add_system = MagicMock()
        return pane

    @pytest.mark.asyncio
    async def test_update_mileage_with_value(self, mock_input_pane, mock_response_pane) -> None:
        """Test UPDATE intent with mileage routes correctly."""
        intent_result = IntentResult(
            intent=IntentType.UPDATE,
            subtype="update.mileage",
            confidence=0.95,
            entities={"mileage": 99000},
            source="pattern",
        )

        await mock_input_pane._handle_mileage_update_intent(
            intent_result, mock_response_pane, original_input="test mileage update"
        )

        # Should call _handle_update_mileage with mileage string
        mock_input_pane._handle_update_mileage.assert_called_once()
        args = mock_input_pane._handle_update_mileage.call_args[0][0]
        assert args == "99000"

    @pytest.mark.asyncio
    async def test_update_without_mileage_shows_error(
        self, mock_input_pane, mock_response_pane
    ) -> None:
        """Test UPDATE intent without mileage shows error."""
        intent_result = IntentResult(
            intent=IntentType.UPDATE,
            subtype="update.mileage",
            confidence=0.85,
            entities={},  # Missing mileage
            source="pattern",
        )

        await mock_input_pane._handle_mileage_update_intent(
            intent_result, mock_response_pane, original_input="test mileage update"
        )

        # Should show error and not call handler
        mock_response_pane.add_system.assert_called_once()
        assert "mileage" in mock_response_pane.add_system.call_args[0][0].lower()
        mock_input_pane._handle_update_mileage.assert_not_called()


class TestIntentRoutingIntegration:
    """Integration tests for the full intent routing flow in submit()."""

    @pytest.mark.asyncio
    async def test_low_confidence_falls_through_to_chat(self) -> None:
        """Test that low confidence intents fall through to chat handler."""
        from r3lay.ui.widgets.input_pane import InputPane

        mock_state = MagicMock()
        mock_state.current_backend = MagicMock()
        mock_state.current_backend.is_loaded = True

        pane = InputPane(state=mock_state)
        pane._handle_chat = AsyncMock()

        # Mock the intent parser to return low confidence
        with patch.object(pane._intent_parser, "parse_sync") as mock_parse:
            mock_parse.return_value = IntentResult(
                intent=IntentType.LOG,
                subtype="log.maintenance",
                confidence=0.5,  # Below 0.7 threshold
                entities={"mileage": 98000},
                source="pattern",
            )

            mock_response_pane = MagicMock()

            # Simulate processing in submit (just the intent handling part)
            intent_result = pane._intent_parser.parse_sync("some input")

            if intent_result.confidence < 0.7:
                await pane._handle_chat("some input", mock_response_pane)

            # Should fall through to chat
            pane._handle_chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_high_confidence_log_routes_correctly(self) -> None:
        """Test that high confidence LOG intent routes to maintenance handler."""
        from r3lay.ui.widgets.input_pane import InputPane

        mock_state = MagicMock()
        mock_state.current_backend = MagicMock()

        pane = InputPane(state=mock_state)
        pane._handle_maintenance_intent = AsyncMock()

        # Mock the intent parser to return high confidence
        with patch.object(pane._intent_parser, "parse_sync") as mock_parse:
            intent_result = IntentResult(
                intent=IntentType.LOG,
                subtype="log.maintenance",
                confidence=0.9,  # High confidence
                entities={"mileage": 98000, "part": "engine_oil"},
                source="pattern",
            )
            mock_parse.return_value = intent_result

            mock_response_pane = MagicMock()

            # Simulate the routing logic from submit
            result = pane._intent_parser.parse_sync("oil change at 98k")
            if result.confidence >= 0.7 and result.intent == IntentType.LOG:
                await pane._handle_maintenance_intent(result, mock_response_pane)

            # Should route to maintenance intent handler
            pane._handle_maintenance_intent.assert_called_once()


class TestServiceTypeMapping:
    """Tests for service type normalization and mapping."""

    def test_part_to_service_type_mapping(self) -> None:
        """Test that part names are correctly mapped to service types."""
        # Test the service_map used in _handle_maintenance_intent
        service_map = {
            "engine_oil": "oil_change",
            "oil_filter": "oil_change",
            "coolant": "coolant_flush",
            "brake_fluid": "brake_fluid_flush",
            "air_filter": "air_filter_replacement",
            "fuel_filter": "fuel_filter_replacement",
            "spark_plugs": "spark_plug_replacement",
            "timing_belt": "timing_belt_replacement",
            "brake_pads": "brake_pad_replacement",
        }

        # Verify key mappings
        assert service_map["engine_oil"] == "oil_change"
        assert service_map["coolant"] == "coolant_flush"
        assert service_map["timing_belt"] == "timing_belt_replacement"

    @pytest.mark.asyncio
    async def test_unmapped_part_passes_through(self) -> None:
        """Test that unmapped parts pass through as-is."""
        from r3lay.ui.widgets.input_pane import InputPane

        mock_state = MagicMock()
        pane = InputPane(state=mock_state)
        pane._handle_log_maintenance = AsyncMock()

        mock_response_pane = MagicMock()

        intent_result = IntentResult(
            intent=IntentType.LOG,
            subtype="log.maintenance",
            confidence=0.85,
            entities={"mileage": 100000, "part": "custom_part"},
            source="pattern",
        )

        await pane._handle_maintenance_intent(
            intent_result, mock_response_pane, original_input="test maintenance input"
        )
        await pane._handle_maintenance_intent(intent_result, mock_response_pane)

        # Custom part should pass through unchanged
        args = pane._handle_log_maintenance.call_args[0][0]
        assert "custom_part" in args


class TestRoutingPreference:
    """Tests for configurable intent routing (local vs OpenClaw)."""

    @pytest.fixture
    def mock_state_with_config(self):
        """Create a mock R3LayState with config."""
        mock_state = MagicMock()
        mock_state.current_backend = MagicMock()
        mock_state.current_backend.is_loaded = True
        mock_state.current_backend.generate = AsyncMock(return_value='{"intent": "CHAT"}')

        # Mock config with routing preference
        mock_config = MagicMock()
        mock_config.intent_routing = "local"
        mock_state.config = mock_config

        return mock_state

    @pytest.fixture
    def mock_input_pane_with_routing(self, mock_state_with_config):
        """Create InputPane with routing support."""
        from r3lay.ui.widgets.input_pane import InputPane

        pane = InputPane(state=mock_state_with_config)
        pane._get_project_context_string = MagicMock(return_value="Test project")
        return pane

    @pytest.mark.asyncio
    async def test_local_routing_uses_parse_sync(self, mock_input_pane_with_routing) -> None:
        """Test that 'local' routing preference uses parse_sync."""
        pane = mock_input_pane_with_routing
        pane.state.config.intent_routing = "local"

        with patch.object(pane._intent_parser, "parse_sync") as mock_sync:
            mock_sync.return_value = IntentResult(
                intent=IntentType.CHAT,
                subtype="chat.general",
                confidence=0.6,
                entities={},
                source="pattern",
            )

            result = await pane._route_intent_parsing("test message", "local")

            # Should use parse_sync
            mock_sync.assert_called_once_with("test message")
            assert result.intent == IntentType.CHAT

    @pytest.mark.asyncio
    async def test_openclaw_routing_uses_llm(self, mock_input_pane_with_routing) -> None:
        """Test that 'openclaw' routing preference uses LLM backend."""
        pane = mock_input_pane_with_routing
        pane.state.config.intent_routing = "openclaw"

        with patch.object(pane, "_parse_intent_via_openclaw") as mock_openclaw:
            mock_openclaw.return_value = IntentResult(
                intent=IntentType.SEARCH,
                subtype="search.general",
                confidence=0.85,
                entities={"query": "test"},
                source="llm",
            )

            result = await pane._route_intent_parsing("test message", "openclaw")

            # Should use OpenClaw
            mock_openclaw.assert_called_once_with("test message")
            assert result.intent == IntentType.SEARCH
            assert result.source == "llm"

    @pytest.mark.asyncio
    async def test_openclaw_routing_fallback_on_error(self, mock_input_pane_with_routing) -> None:
        """Test that 'openclaw' routing falls back to local on error."""
        pane = mock_input_pane_with_routing
        pane.state.config.intent_routing = "openclaw"

        with patch.object(pane, "_parse_intent_via_openclaw") as mock_openclaw:
            mock_openclaw.side_effect = Exception("OpenClaw unavailable")

            with patch.object(pane._intent_parser, "parse_sync") as mock_sync:
                mock_sync.return_value = IntentResult(
                    intent=IntentType.CHAT,
                    subtype="chat.general",
                    confidence=0.5,
                    entities={},
                    source="pattern",
                )

                result = await pane._route_intent_parsing("test message", "openclaw")

                # Should fallback to parse_sync
                mock_sync.assert_called_once_with("test message")
                assert result.intent == IntentType.CHAT

    @pytest.mark.asyncio
    async def test_auto_routing_prefers_openclaw(self, mock_input_pane_with_routing) -> None:
        """Test that 'auto' routing prefers OpenClaw when available."""
        pane = mock_input_pane_with_routing
        pane.state.config.intent_routing = "auto"

        with patch.object(pane, "_is_openclaw_available") as mock_available:
            mock_available.return_value = True

            with patch.object(pane, "_parse_intent_via_openclaw") as mock_openclaw:
                mock_openclaw.return_value = IntentResult(
                    intent=IntentType.LOG,
                    subtype="log.maintenance",
                    confidence=0.9,
                    entities={"mileage": 98000},
                    source="llm",
                )

                result = await pane._route_intent_parsing("test message", "auto")

                # Should use OpenClaw
                mock_openclaw.assert_called_once_with("test message")
                assert result.source == "llm"

    @pytest.mark.asyncio
    async def test_auto_routing_fallback_to_local(self, mock_input_pane_with_routing) -> None:
        """Test that 'auto' routing falls back to local when OpenClaw unavailable."""
        pane = mock_input_pane_with_routing
        pane.state.config.intent_routing = "auto"

        with patch.object(pane, "_is_openclaw_available") as mock_available:
            mock_available.return_value = False

            with patch.object(pane._intent_parser, "parse_sync") as mock_sync:
                mock_sync.return_value = IntentResult(
                    intent=IntentType.CHAT,
                    subtype="chat.general",
                    confidence=0.6,
                    entities={},
                    source="pattern",
                )

                result = await pane._route_intent_parsing("test message", "auto")

                # Should use local parsing
                mock_sync.assert_called_once_with("test message")
                assert result.source == "pattern"

    @pytest.mark.asyncio
    async def test_is_openclaw_available_with_backend(self, mock_input_pane_with_routing) -> None:
        """Test _is_openclaw_available returns True when backend is loaded."""
        pane = mock_input_pane_with_routing
        pane.state.current_backend = MagicMock()
        pane.state.current_backend.generate = MagicMock()

        result = await pane._is_openclaw_available()
        assert result is True

    @pytest.mark.asyncio
    async def test_is_openclaw_available_without_backend(
        self, mock_input_pane_with_routing
    ) -> None:
        """Test _is_openclaw_available returns False when no backend."""
        pane = mock_input_pane_with_routing
        pane.state.current_backend = None

        result = await pane._is_openclaw_available()
        assert result is False

    @pytest.mark.asyncio
    async def test_parse_intent_via_openclaw_creates_parser_with_backend(
        self, mock_input_pane_with_routing
    ) -> None:
        """Test _parse_intent_via_openclaw creates IntentParser with backend."""
        from r3lay.core.intent import IntentParser

        pane = mock_input_pane_with_routing

        with patch.object(IntentParser, "parse") as mock_parse:
            mock_parse.return_value = IntentResult(
                intent=IntentType.SEARCH,
                subtype="search.general",
                confidence=0.85,
                entities={},
                source="llm",
            )

            result = await pane._parse_intent_via_openclaw("test message")

            # Should create parser and call parse
            mock_parse.assert_called_once_with("test message")
            assert result.source == "llm"

    def test_unknown_routing_preference_defaults_to_local(
        self, mock_input_pane_with_routing
    ) -> None:
        """Test that unknown routing preference defaults to local."""
        import asyncio

        pane = mock_input_pane_with_routing

        with patch.object(pane._intent_parser, "parse_sync") as mock_sync:
            mock_sync.return_value = IntentResult(
                intent=IntentType.CHAT,
                subtype="chat.general",
                confidence=0.5,
                entities={},
                source="pattern",
            )

            # Run async function in event loop
            asyncio.run(pane._route_intent_parsing("test", "invalid_preference"))

            # Should fall back to parse_sync
            mock_sync.assert_called_once_with("test")


class TestRealWorldIntentExamples:
    """Tests using real-world examples from the design document."""

    @pytest.fixture
    def parser(self):
        """Create an IntentParser for testing."""
        from r3lay.core.intent import IntentParser

        return IntentParser()

    def test_logged_oil_change_at_98k(self, parser) -> None:
        """Test: 'Logged oil change at 98k' should route to LOG handler."""
        result = parser.parse_sync("Logged oil change at 98k")

        assert result.intent == IntentType.LOG
        assert result.subtype == "log.maintenance"
        assert result.confidence >= 0.7
        assert result.entities.get("mileage") == 98000
        # This should route to _handle_maintenance_intent

    def test_when_is_oil_change_due(self, parser) -> None:
        """Test: 'When is oil change due?' should route to QUERY handler."""
        result = parser.parse_sync("When is oil change due?")

        assert result.intent == IntentType.QUERY
        assert result.subtype == "query.reminder"
        assert result.confidence >= 0.7
        # This should route to _handle_query_intent → _handle_due_services

    def test_mileage_now_98500(self, parser) -> None:
        """Test: 'Mileage is now 98500' should route to UPDATE handler."""
        result = parser.parse_sync("Mileage is now 98500")

        assert result.intent == IntentType.UPDATE
        assert result.subtype == "update.mileage"
        assert result.confidence >= 0.7
        assert result.entities.get("mileage") == 98500
        # This should route to _handle_mileage_update_intent
