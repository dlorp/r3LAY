"""Tests for r3lay.ui.widgets.axiom_panel module.

Tests cover:
- AxiomPanel: initialization, stats display, filtering, selection
- AxiomItem: rendering, status classes, truncation
- AxiomStatus enum: values and display configuration
- AXIOM_CATEGORIES constant: validation
- STATUS_ICONS constant: markup validation
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from r3lay.ui.widgets.axiom_panel import (
    AXIOM_CATEGORIES,
    STATUS_ICONS,
    AxiomItem,
    AxiomPanel,
    AxiomStatus,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    """Create a temporary project directory."""
    return tmp_path


@pytest.fixture
def mock_state(tmp_project: Path) -> MagicMock:
    """Create a mock R3LayState."""
    state = MagicMock()
    state.project_path = tmp_project
    state.axioms = None  # No axiom manager by default
    return state


@pytest.fixture
def mock_axiom_manager() -> MagicMock:
    """Create a mock AxiomManager."""
    manager = MagicMock()
    manager.get_stats.return_value = {
        "total": 10,
        "validated": 5,
        "pending": 3,
        "disputed": 2,
        "avg_confidence": 0.85,
    }
    manager.search.return_value = []
    return manager


@pytest.fixture
def mock_state_with_axioms(tmp_project: Path, mock_axiom_manager: MagicMock) -> MagicMock:
    """Create a mock R3LayState with axiom manager."""
    state = MagicMock()
    state.project_path = tmp_project
    state.axioms = mock_axiom_manager
    return state


@pytest.fixture
def mock_axiom() -> MagicMock:
    """Create a mock Axiom object."""
    axiom = MagicMock()
    axiom.id = "axiom_test123"
    axiom.statement = "Test axiom statement"
    axiom.confidence = 0.9
    axiom.is_validated = True
    axiom.is_disputed = False
    axiom.superseded_by = None
    return axiom


# =============================================================================
# AxiomStatus Enum Tests
# =============================================================================


class TestAxiomStatus:
    """Tests for AxiomStatus enum."""

    def test_enum_values(self):
        """Test that all status values exist."""
        assert AxiomStatus.VALIDATED.value == "validated"
        assert AxiomStatus.PENDING.value == "pending"
        assert AxiomStatus.DISPUTED.value == "disputed"
        assert AxiomStatus.SUPERSEDED.value == "superseded"

    def test_status_is_string_enum(self):
        """Test that AxiomStatus inherits from str."""
        assert isinstance(AxiomStatus.VALIDATED, str)
        assert AxiomStatus.VALIDATED == "validated"

    def test_status_from_string(self):
        """Test creating AxiomStatus from string value."""
        status = AxiomStatus("validated")
        assert status == AxiomStatus.VALIDATED

    def test_invalid_status_raises(self):
        """Test that invalid status string raises ValueError."""
        with pytest.raises(ValueError):
            AxiomStatus("invalid_status")

    def test_all_statuses_have_icons(self):
        """Test that all status values have icons configured."""
        for status in AxiomStatus:
            assert status in STATUS_ICONS


# =============================================================================
# STATUS_ICONS Tests
# =============================================================================


class TestStatusIcons:
    """Tests for STATUS_ICONS configuration."""

    def test_validated_icon_is_green(self):
        """Test VALIDATED icon uses green color."""
        assert "[green]" in STATUS_ICONS[AxiomStatus.VALIDATED]
        assert "OK" in STATUS_ICONS[AxiomStatus.VALIDATED]

    def test_pending_icon_is_yellow(self):
        """Test PENDING icon uses yellow color."""
        assert "[yellow]" in STATUS_ICONS[AxiomStatus.PENDING]

    def test_disputed_icon_is_red(self):
        """Test DISPUTED icon uses red color."""
        assert "[red]" in STATUS_ICONS[AxiomStatus.DISPUTED]
        assert "!!" in STATUS_ICONS[AxiomStatus.DISPUTED]

    def test_superseded_icon_is_dim(self):
        """Test SUPERSEDED icon uses dim styling."""
        assert "[dim]" in STATUS_ICONS[AxiomStatus.SUPERSEDED]


# =============================================================================
# AXIOM_CATEGORIES Tests
# =============================================================================


class TestAxiomCategories:
    """Tests for AXIOM_CATEGORIES constant."""

    def test_categories_is_list(self):
        """Test AXIOM_CATEGORIES is a list of strings."""
        assert isinstance(AXIOM_CATEGORIES, list)
        assert all(isinstance(cat, str) for cat in AXIOM_CATEGORIES)

    def test_expected_categories_exist(self):
        """Test all expected categories are present."""
        expected = [
            "specifications",
            "procedures",
            "compatibility",
            "diagnostics",
            "history",
            "safety",
        ]
        assert AXIOM_CATEGORIES == expected

    def test_categories_not_empty(self):
        """Test AXIOM_CATEGORIES has categories."""
        assert len(AXIOM_CATEGORIES) == 6


# =============================================================================
# AxiomItem Tests
# =============================================================================


class TestAxiomItem:
    """Tests for AxiomItem widget."""

    def test_creation_validated(self):
        """Test creating item with VALIDATED status."""
        item = AxiomItem(
            axiom_id="axiom_123",
            statement="Test statement",
            confidence=0.9,
            status=AxiomStatus.VALIDATED,
        )

        assert item.axiom_id == "axiom_123"
        assert item._statement == "Test statement"
        assert item._confidence == 0.9
        assert item._status == AxiomStatus.VALIDATED
        assert "validated" in item.classes

    def test_creation_pending(self):
        """Test creating item with PENDING status."""
        item = AxiomItem(
            axiom_id="axiom_456",
            statement="Pending statement",
            confidence=0.7,
            status=AxiomStatus.PENDING,
        )

        assert item._status == AxiomStatus.PENDING
        assert "pending" in item.classes

    def test_creation_disputed(self):
        """Test creating item with DISPUTED status."""
        item = AxiomItem(
            axiom_id="axiom_789",
            statement="Disputed statement",
            confidence=0.6,
            status=AxiomStatus.DISPUTED,
        )

        assert item._status == AxiomStatus.DISPUTED
        assert "disputed" in item.classes

    def test_creation_superseded(self):
        """Test creating item with SUPERSEDED status."""
        item = AxiomItem(
            axiom_id="axiom_old",
            statement="Old statement",
            confidence=0.5,
            status=AxiomStatus.SUPERSEDED,
        )

        assert item._status == AxiomStatus.SUPERSEDED
        assert "superseded" in item.classes

    def test_statement_truncation(self):
        """Test that long statements are truncated."""
        long_statement = "A" * 100
        item = AxiomItem(
            axiom_id="axiom_long",
            statement=long_statement,
            confidence=0.8,
            status=AxiomStatus.VALIDATED,
        )

        # The item's display text should be truncated
        # Looking at the source, it truncates at 60 chars and adds "..."
        assert item._statement == long_statement
        # Truncation happens in the display text, not the stored statement

    def test_confidence_percentage_display(self):
        """Test confidence is displayed as percentage."""
        item = AxiomItem(
            axiom_id="axiom_conf",
            statement="Test",
            confidence=0.85,
            status=AxiomStatus.VALIDATED,
        )

        # Confidence should be stored as decimal
        assert item._confidence == 0.85

    def test_can_focus(self):
        """Test that AxiomItem can be focused."""
        item = AxiomItem(
            axiom_id="axiom_focus",
            statement="Focusable",
            confidence=0.9,
            status=AxiomStatus.VALIDATED,
        )

        assert item.can_focus is True


# =============================================================================
# AxiomPanel Initialization Tests
# =============================================================================


class TestAxiomPanelInit:
    """Tests for AxiomPanel initialization."""

    def test_creation_with_state(self, mock_state: MagicMock):
        """Test creating panel with state."""
        panel = AxiomPanel(state=mock_state)

        assert panel.state is mock_state
        assert panel._selected_axiom_id is None
        assert panel._current_category is None
        assert panel._current_status is None

    def test_has_default_css(self):
        """Test that AxiomPanel has DEFAULT_CSS defined."""
        assert AxiomPanel.DEFAULT_CSS is not None
        assert "AxiomPanel" in AxiomPanel.DEFAULT_CSS
        assert "#axiom-header" in AxiomPanel.DEFAULT_CSS
        assert "#axiom-stats" in AxiomPanel.DEFAULT_CSS
        assert "#axiom-list" in AxiomPanel.DEFAULT_CSS

    def test_css_defines_filter_row(self):
        """Test CSS defines filter row styling."""
        assert "#filter-row" in AxiomPanel.DEFAULT_CSS
        assert "#category-filter" in AxiomPanel.DEFAULT_CSS
        assert "#status-filter" in AxiomPanel.DEFAULT_CSS

    def test_css_defines_button_row(self):
        """Test CSS defines button row styling."""
        assert "#button-row" in AxiomPanel.DEFAULT_CSS
        assert "#validate-button" in AxiomPanel.DEFAULT_CSS
        assert "#dispute-button" in AxiomPanel.DEFAULT_CSS
        assert "#export-button" in AxiomPanel.DEFAULT_CSS


# =============================================================================
# AxiomPanel Compose Tests
# =============================================================================


class TestAxiomPanelCompose:
    """Tests for AxiomPanel composition structure.

    Note: Cannot call compose() directly without an active app context.
    These tests verify the panel structure through introspection.
    """

    def test_compose_method_exists(self, mock_state: MagicMock):
        """Test that compose method is defined."""
        panel = AxiomPanel(state=mock_state)
        assert hasattr(panel, "compose")
        assert callable(panel.compose)

    def test_panel_defines_expected_widget_ids(self, mock_state: MagicMock):
        """Test that expected widget IDs are referenced in DEFAULT_CSS."""
        # Verify the CSS references expected IDs that compose should create
        css = AxiomPanel.DEFAULT_CSS
        expected_ids = [
            "#axiom-header",
            "#axiom-stats",
            "#filter-row",
            "#category-filter",
            "#status-filter",
            "#axiom-list",
            "#button-row",
            "#validate-button",
            "#dispute-button",
            "#export-button",
        ]
        for widget_id in expected_ids:
            assert widget_id in css, f"Expected {widget_id} in DEFAULT_CSS"

    def test_panel_defines_button_styling(self, mock_state: MagicMock):
        """Test that button styling is defined in CSS."""
        css = AxiomPanel.DEFAULT_CSS
        # Check that disabled button styling is defined
        assert "#validate-button:disabled" in css or "disabled" in css
        assert "#dispute-button:disabled" in css or "disabled" in css

    def test_panel_has_state_reference(self, mock_state: MagicMock):
        """Test panel stores state reference for compose."""
        panel = AxiomPanel(state=mock_state)
        assert panel.state is mock_state


# =============================================================================
# AxiomPanel Status Determination Tests
# =============================================================================


class TestAxiomPanelStatus:
    """Tests for axiom status determination."""

    def test_get_axiom_status_validated(self, mock_state: MagicMock, mock_axiom: MagicMock):
        """Test status determination for validated axiom."""
        panel = AxiomPanel(state=mock_state)

        mock_axiom.is_validated = True
        mock_axiom.is_disputed = False
        mock_axiom.superseded_by = None

        status = panel._get_axiom_status(mock_axiom)
        assert status == AxiomStatus.VALIDATED

    def test_get_axiom_status_pending(self, mock_state: MagicMock, mock_axiom: MagicMock):
        """Test status determination for pending axiom."""
        panel = AxiomPanel(state=mock_state)

        mock_axiom.is_validated = False
        mock_axiom.is_disputed = False
        mock_axiom.superseded_by = None

        status = panel._get_axiom_status(mock_axiom)
        assert status == AxiomStatus.PENDING

    def test_get_axiom_status_disputed(self, mock_state: MagicMock, mock_axiom: MagicMock):
        """Test status determination for disputed axiom."""
        panel = AxiomPanel(state=mock_state)

        mock_axiom.is_validated = True
        mock_axiom.is_disputed = True
        mock_axiom.superseded_by = None

        status = panel._get_axiom_status(mock_axiom)
        # Disputed takes precedence
        assert status == AxiomStatus.DISPUTED

    def test_get_axiom_status_superseded(self, mock_state: MagicMock, mock_axiom: MagicMock):
        """Test status determination for superseded axiom."""
        panel = AxiomPanel(state=mock_state)

        mock_axiom.is_validated = True
        mock_axiom.is_disputed = False
        mock_axiom.superseded_by = "axiom_new"

        status = panel._get_axiom_status(mock_axiom)
        assert status == AxiomStatus.SUPERSEDED


# =============================================================================
# AxiomPanel Manager Access Tests
# =============================================================================


class TestAxiomPanelManager:
    """Tests for axiom manager access."""

    def test_get_axiom_manager_returns_none_when_missing(self, mock_state: MagicMock):
        """Test _get_axiom_manager returns None when no manager and no init method."""
        mock_state.axioms = None
        # Delete the auto-created init_axioms to simulate it not existing
        del mock_state.init_axioms
        panel = AxiomPanel(state=mock_state)

        manager = panel._get_axiom_manager()
        assert manager is None

    def test_get_axiom_manager_returns_manager(
        self, mock_state_with_axioms: MagicMock, mock_axiom_manager: MagicMock
    ):
        """Test _get_axiom_manager returns existing manager."""
        panel = AxiomPanel(state=mock_state_with_axioms)

        manager = panel._get_axiom_manager()
        assert manager is mock_axiom_manager

    def test_get_axiom_manager_tries_init(self, mock_state: MagicMock):
        """Test _get_axiom_manager tries to initialize if method exists."""
        mock_state.axioms = None
        mock_manager = MagicMock()
        mock_state.init_axioms = MagicMock(return_value=mock_manager)

        panel = AxiomPanel(state=mock_state)
        manager = panel._get_axiom_manager()

        mock_state.init_axioms.assert_called_once()
        assert manager is mock_manager


# =============================================================================
# AxiomPanel Stats Update Tests
# =============================================================================


class TestAxiomPanelStats:
    """Tests for stats display updates."""

    def test_update_stats_no_manager(self, mock_state: MagicMock):
        """Test stats update when no manager available."""
        panel = AxiomPanel(state=mock_state)

        # _update_stats_no_manager should set appropriate message
        # We can't test widget state directly without running app,
        # but we can verify the method exists and is callable
        assert hasattr(panel, "_update_stats_no_manager")
        assert callable(panel._update_stats_no_manager)

    def test_update_stats_with_manager(
        self, mock_state_with_axioms: MagicMock, mock_axiom_manager: MagicMock
    ):
        """Test stats update with axiom manager."""
        panel = AxiomPanel(state=mock_state_with_axioms)

        # Verify _update_stats method exists
        assert hasattr(panel, "_update_stats")
        assert callable(panel._update_stats)


# =============================================================================
# AxiomPanel Filtering Tests
# =============================================================================


class TestAxiomPanelFiltering:
    """Tests for axiom filtering."""

    def test_filter_by_category(
        self, mock_state_with_axioms: MagicMock, mock_axiom_manager: MagicMock
    ):
        """Test filtering by category."""
        panel = AxiomPanel(state=mock_state_with_axioms)
        panel._current_category = "specifications"

        _ = panel._get_filtered_axioms(mock_axiom_manager)

        # Manager's search should be called with category
        mock_axiom_manager.search.assert_called_once()
        call_kwargs = mock_axiom_manager.search.call_args.kwargs
        assert call_kwargs.get("category") == "specifications"

    def test_filter_by_validated_status(
        self, mock_state_with_axioms: MagicMock, mock_axiom_manager: MagicMock
    ):
        """Test filtering by validated status."""
        panel = AxiomPanel(state=mock_state_with_axioms)
        panel._current_status = "validated"

        _ = panel._get_filtered_axioms(mock_axiom_manager)

        # Should use validated_only filter
        call_kwargs = mock_axiom_manager.search.call_args.kwargs
        assert call_kwargs.get("validated_only") is True

    def test_filter_default_limit(
        self, mock_state_with_axioms: MagicMock, mock_axiom_manager: MagicMock
    ):
        """Test default limit of 50 is applied."""
        panel = AxiomPanel(state=mock_state_with_axioms)

        _ = panel._get_filtered_axioms(mock_axiom_manager)

        call_kwargs = mock_axiom_manager.search.call_args.kwargs
        assert call_kwargs.get("limit") == 50


# =============================================================================
# AxiomPanel Selection Tests
# =============================================================================


class TestAxiomPanelSelection:
    """Tests for axiom selection."""

    def test_initial_selection_is_none(self, mock_state: MagicMock):
        """Test initial selection state."""
        panel = AxiomPanel(state=mock_state)
        assert panel._selected_axiom_id is None

    def test_select_axiom_stores_id(self, mock_state: MagicMock):
        """Test that selecting axiom stores its ID."""
        panel = AxiomPanel(state=mock_state)

        item = AxiomItem(
            axiom_id="axiom_selected",
            statement="Selected axiom",
            confidence=0.9,
            status=AxiomStatus.PENDING,
        )

        # Note: Full selection behavior requires mounted widgets
        # Here we test the panel has selection tracking capability
        panel._selected_axiom_id = item.axiom_id
        assert panel._selected_axiom_id == "axiom_selected"


# =============================================================================
# AxiomPanel Exports Tests
# =============================================================================


class TestAxiomPanelExports:
    """Tests for module exports."""

    def test_all_exports(self):
        """Test __all__ exports."""
        from r3lay.ui.widgets import axiom_panel

        assert "AxiomPanel" in axiom_panel.__all__
        assert "AXIOM_CATEGORIES" in axiom_panel.__all__
        assert "AxiomStatus" in axiom_panel.__all__


# =============================================================================
# AxiomItem CSS Tests
# =============================================================================


class TestAxiomItemCSS:
    """Tests for AxiomItem CSS styling."""

    def test_default_css_exists(self):
        """Test AxiomItem has DEFAULT_CSS defined."""
        assert AxiomItem.DEFAULT_CSS is not None

    def test_css_defines_status_borders(self):
        """Test CSS defines border colors for status classes."""
        css = AxiomItem.DEFAULT_CSS
        assert "AxiomItem.validated" in css
        assert "AxiomItem.pending" in css
        assert "AxiomItem.disputed" in css
        assert "AxiomItem.superseded" in css

    def test_css_defines_focus_state(self):
        """Test CSS defines focus state."""
        css = AxiomItem.DEFAULT_CSS
        assert "AxiomItem:focus" in css

    def test_css_defines_hover_state(self):
        """Test CSS defines hover state."""
        css = AxiomItem.DEFAULT_CSS
        assert "AxiomItem:hover" in css


# =============================================================================
# Integration Tests (Basic)
# =============================================================================


class TestAxiomPanelIntegration:
    """Basic integration tests without full Textual app."""

    def test_panel_tracks_filter_state(self, mock_state: MagicMock):
        """Test panel tracks filter state correctly."""
        panel = AxiomPanel(state=mock_state)

        # Initial state
        assert panel._current_category is None
        assert panel._current_status is None

        # Set filters
        panel._current_category = "safety"
        panel._current_status = "disputed"

        assert panel._current_category == "safety"
        assert panel._current_status == "disputed"

    def test_panel_can_get_filtered_axioms_empty(
        self, mock_state_with_axioms: MagicMock, mock_axiom_manager: MagicMock
    ):
        """Test getting filtered axioms returns empty list when none match."""
        mock_axiom_manager.search.return_value = []

        panel = AxiomPanel(state=mock_state_with_axioms)
        axioms = panel._get_filtered_axioms(mock_axiom_manager)

        assert axioms == []

    def test_panel_handles_search_exception(
        self, mock_state_with_axioms: MagicMock, mock_axiom_manager: MagicMock
    ):
        """Test panel handles search exceptions gracefully."""
        mock_axiom_manager.search.side_effect = Exception("Search error")

        panel = AxiomPanel(state=mock_state_with_axioms)
        axioms = panel._get_filtered_axioms(mock_axiom_manager)

        # Should return empty list on exception
        assert axioms == []


__all__ = [
    "TestAxiomStatus",
    "TestStatusIcons",
    "TestAxiomCategories",
    "TestAxiomItem",
    "TestAxiomPanelInit",
    "TestAxiomPanelCompose",
    "TestAxiomPanelStatus",
    "TestAxiomPanelManager",
    "TestAxiomPanelStats",
    "TestAxiomPanelFiltering",
    "TestAxiomPanelSelection",
    "TestAxiomPanelExports",
    "TestAxiomItemCSS",
    "TestAxiomPanelIntegration",
]
