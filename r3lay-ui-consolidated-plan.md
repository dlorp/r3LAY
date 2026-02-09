# r3LAY UI Consolidated Improvement Plan

> **Created:** 2026-02-08 | **Status:** Ready for execution
> **Context:** PRs #82 (stream-b) and #83 (stream-a) were closed due to duplication and test failures (AXIOM-038 violation). This plan consolidates ALL work into a single linear execution path.

## Problem Summary

Both PRs attempted overlapping changes:
- **Duplicated:** emoji removal (AXIOM-037), Slider import fix, garage.tcss overhaul, settings_panel expansion, garage_header creation
- **Both broke:** `Slider` doesn't exist in Textual (must use `Input` for temperature)
- **Neither ran tests** before marking complete
- **Pre-existing bug:** `r3lay/config.py` uses `str | None` syntax (Python 3.10+) but env is Python 3.9

## Execution Rules

1. **Run `python3 -m pytest tests/` at every checkpoint** — do NOT proceed if tests fail
2. **One file change at a time** where possible — commit after each logical unit
3. **Branch:** `feat/ui-consolidated-improvements` from `main`

---

## Phase 0: Environment Fix & Shared Cleanup

> **Goal:** Fix pre-existing test breakage, then do all shared cleanup that both PRs duplicated.

### 0.1 Fix Python 3.9 Compatibility in config.py
- **File:** `r3lay/config.py`
- **Change:** Replace `str | None` with `Optional[str]` (add `from typing import Optional`)
- Lines 35-37, 41: `text_model`, `vision_model`, `text_embedder`, `vision_embedder`
- **Checkpoint:** `python3 -m pytest tests/` — ALL tests must pass ✅

### 0.2 Emoji Removal (AXIOM-037)
- **Files:** ALL source files under `r3lay/` — search for emoji characters
- **Specifically:** Tab labels in `r3lay/app.py` must NOT have emoji (Stream A added them, Stream B removed them — removal is correct per AXIOM-037)
- Remove emoji from: tab labels, any string literals, log messages
- Do NOT touch test fixtures that intentionally test emoji handling
- **Checkpoint:** `python3 -m pytest tests/` ✅ + `grep -rP '[\x{1F300}-\x{1FFFF}]' r3lay/` returns nothing

### 0.3 Commit Phase 0
```
fix: Python 3.9 compat + AXIOM-037 emoji cleanup
```

---

## Phase 1: Core Infrastructure (New Files & Shared Widgets)

> **Goal:** Add new files/widgets that multiple features depend on. Test-driven: write/update tests first.

### 1.1 Create GarageHeader Widget
- **New file:** `r3lay/ui/widgets/garage_header.py`
- Custom header with project name, active model display, mileage readout
- Methods: `watch_active_model()`, `refresh_mileage()`
- **New test file:** `tests/ui/test_garage_header.py`
- **Write tests FIRST**, then implement until tests pass
- **Checkpoint:** `python3 -m pytest tests/ui/test_garage_header.py` ✅

### 1.2 TCSS Theme Consolidation (garage.tcss)
- **File:** `r3lay/ui/styles/garage.tcss`
- Define CSS variables for amber palette: `$amber`, `$amber-dim`, `$amber-bright`
- Add focus glow styles using amber accent
- Add styles for new widgets: `.settings-section`, `#temperature-row`, `#button-row`, `GarageHeader`
- Add response block styling (border with amber left edge, spacing)
- Add splash prompt styling (`#splash-prompt`)
- **Checkpoint:** `python3 -m pytest tests/` ✅ (CSS changes shouldn't break tests)

### 1.3 Commit Phase 1
```
feat: add GarageHeader widget + TCSS theme consolidation
```

---

## Phase 2: Visual Improvements (Stream A scope)

> **Goal:** All visual/styling changes. No new interactivity.

### 2.1 Response Block Styling
- **File:** `r3lay/ui/widgets/response_pane.py`
- Add subtle borders with amber left edge accent
- Add `▸` prefix to response headers
- Replace `>`, `<`, `*`, `#` header markers with `▸`
- **Checkpoint:** `python3 -m pytest tests/ui/test_response_pane.py` ✅

### 2.2 Splash Screen Enhancement
- **File:** `r3lay/ui/widgets/splash.py`
- Add "Press any key to continue..." prompt
- Style via `#splash-prompt` CSS (already added in Phase 1.2)
- **Checkpoint:** `python3 -m pytest tests/ui/test_splash.py` ✅

### 2.3 Retro Loading Indicators
- **File:** `r3lay/ui/effects.py`
- Add `retro_progress_bar()` — block chars `█`/`░`
- Add `amber_pulse()` — pulsing glow effect
- **New tests:** Add to `tests/ui/test_effects.py` (create if needed)
- **Write tests FIRST**
- **Checkpoint:** `python3 -m pytest tests/` ✅

### 2.4 Commit Phase 2
```
feat: visual improvements - response styling, splash, retro effects
```

---

## Phase 3: Functional Improvements (Stream B scope)

> **Goal:** All interactive/behavioral changes.

### 3.1 Settings Panel Upgrade
- **File:** `r3lay/ui/widgets/settings_panel.py`
- ⚠️ **CRITICAL:** Do NOT import `Slider` — it doesn't exist in Textual
- Use `Input` widget with type validation for temperature (0.0-2.0)
- Add `RadioSet`/`RadioButton` for model selection
- Add `Button` for save/reset
- Add keybindings reference section
- **Update imports:** `from textual.widgets import Button, Input, Label, RadioButton, RadioSet, Static`
- **File:** `tests/ui/test_settings_panel.py`
- **Update tests FIRST** to match new widget structure
- **Checkpoint:** `python3 -m pytest tests/ui/test_settings_panel.py` ✅

### 3.2 Wire GarageHeader into App
- **File:** `r3lay/app.py`
- Import and mount `GarageHeader` in `MainScreen.compose()`
- Forward `ModelPanel.RoleAssigned` → `GarageHeader.ModelChanged`
- Tab labels: plain text only (NO emoji — AXIOM-037)
- **Checkpoint:** `python3 -m pytest tests/test_app.py` ✅

### 3.3 Input Placeholder Text
- **File:** `r3lay/ui/widgets/input_pane.py`
- Add placeholder: `"Ask about automotive, electronics, software, or home projects..."`
- Show/hide on focus/blur
- **Checkpoint:** `python3 -m pytest tests/ui/test_input_pane_model_swap.py` ✅

### 3.4 Clickable Session History
- **File:** `r3lay/ui/widgets/session_panel.py`
- Make past sessions clickable with hover styling
- Load previous session transcript on click
- Highlight selected session
- **Checkpoint:** `python3 -m pytest tests/ui/test_session_panel.py` ✅

### 3.5 Index Panel Clear Confirmation
- **File:** `r3lay/ui/widgets/index_panel.py`
- Two-click confirmation before clearing index
- Show document count in confirmation
- Auto-reset after 5 seconds
- **Checkpoint:** `python3 -m pytest tests/ui/test_index_panel.py` ✅

### 3.6 Keyboard Shortcuts
- **File:** `r3lay/app.py` (bindings only)
- `Ctrl+H`: Toggle session history
- `Ctrl+,`: Open settings panel
- Document in settings keybindings reference
- **Checkpoint:** `python3 -m pytest tests/test_app.py` ✅

### 3.7 Commit Phase 3
```
feat: functional improvements - settings, sessions, shortcuts, confirm dialogs
```

---

## Phase 4: Final Validation

### 4.1 Full Test Suite
```bash
python3 -m pytest tests/ -v
```
ALL tests must pass. Zero failures, zero errors.

### 4.2 Lint Check
```bash
python3 -m ruff check r3lay/ tests/
```

### 4.3 Manual Smoke Test
```bash
python3 -m r3lay
```
Verify: app launches, tabs work, settings panel renders, no crashes.

### 4.4 Open PR
- Single PR from `feat/ui-consolidated-improvements` → `main`
- Title: `feat: consolidated UI improvements (visual + functional)`
- Reference closed PRs #82 and #83

---

## File Ownership Map (Prevents Overlap)

| File | Phase | Changes |
|------|-------|---------|
| `r3lay/config.py` | 0.1 | Optional[] fix |
| `r3lay/app.py` | 0.2, 3.2, 3.6 | Emoji removal → header wiring → shortcuts |
| `r3lay/ui/styles/garage.tcss` | 1.2 | Full theme overhaul (ONE pass) |
| `r3lay/ui/widgets/garage_header.py` | 1.1 | NEW file |
| `r3lay/ui/widgets/response_pane.py` | 2.1 | Visual only |
| `r3lay/ui/widgets/splash.py` | 2.2 | Prompt addition |
| `r3lay/ui/effects.py` | 2.3 | Retro indicators |
| `r3lay/ui/widgets/settings_panel.py` | 3.1 | Full upgrade (Input, NOT Slider) |
| `r3lay/ui/widgets/input_pane.py` | 3.3 | Placeholder |
| `r3lay/ui/widgets/session_panel.py` | 3.4 | Clickable sessions |
| `r3lay/ui/widgets/index_panel.py` | 3.5 | Clear confirmation |

## Known Pitfalls

1. **`Slider` does not exist in Textual** — use `Input` with float validation
2. **Python 3.9** — no `X | Y` union syntax, use `Optional[X]` / `Union[X, Y]`
3. **AXIOM-037** — no emoji anywhere in source code (tab labels, strings, comments)
4. **`from __future__ import annotations`** — already present in most files, keeps type hints as strings
5. **Test isolation** — settings panel tests need `async with app.run_test()` pattern, not bare widget instantiation
