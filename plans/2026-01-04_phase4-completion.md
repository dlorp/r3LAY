# Phase 4 Completion Plan

## Current Status

Phase 4 (Hybrid Index) core is complete. Remaining polish:
1. Config persistence - model roles don't persist between sessions
2. Trust indicators - search results should show source type badges
3. Source filtering - `/index manual: query` syntax

---

## Implementation

### 1. Config Persistence

**File:** `r3lay/config.py`

Implement `save()` and `load()` methods (currently stubbed):

```python
@classmethod
def load(cls, path: Path) -> "AppConfig":
    """Load configuration from .r3lay/config.yaml if it exists."""
    from ruamel.yaml import YAML

    config = cls(project_path=path)
    config_file = path / ".r3lay" / "config.yaml"

    if config_file.exists():
        yaml = YAML()
        with config_file.open() as f:
            data = yaml.load(f)

        if data and "model_roles" in data:
            roles = data["model_roles"]
            config.model_roles = ModelRoles(
                text_model=roles.get("text_model"),
                vision_model=roles.get("vision_model"),
                text_embedder=roles.get("text_embedder"),
                vision_embedder=roles.get("vision_embedder"),
            )
    return config

def save(self) -> None:
    """Save configuration to .r3lay/config.yaml."""
    from ruamel.yaml import YAML

    config_dir = self.project_path / ".r3lay"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.yaml"

    yaml = YAML()
    yaml.default_flow_style = False
    data = {
        "model_roles": {
            "text_model": self.model_roles.text_model,
            "vision_model": self.model_roles.vision_model,
            "text_embedder": self.model_roles.text_embedder,
            "vision_embedder": self.model_roles.vision_embedder,
        }
    }
    with config_file.open("w") as f:
        yaml.dump(data, f)
```

**File:** `r3lay/ui/widgets/model_panel.py`

Add `app.config.save()` after role changes in `_set_embedder_role()`.

---

### 2. Trust Indicators in Search Results

**File:** `r3lay/ui/widgets/input_pane.py`

Add badge formatting function and update `_handle_index_search()`:

```python
def _format_trust_badge(source_type: SourceType) -> str:
    badges = {
        SourceType.INDEXED_MANUAL: "[MANUAL]",
        SourceType.INDEXED_DOCUMENT: "[DOC]",
        SourceType.INDEXED_CODE: "[CODE]",
        SourceType.INDEXED_IMAGE: "[IMAGE]",
        SourceType.WEB_OE_FIRSTPARTY: "[OEM]",
        SourceType.WEB_TRUSTED: "[TRUSTED]",
        SourceType.WEB_GENERAL: "[WEB]",
        SourceType.WEB_COMMUNITY: "[FORUM]",
    }
    return badges.get(source_type, "[?]")
```

---

### 3. Source Type Filtering

**File:** `r3lay/ui/widgets/input_pane.py`

Parse query prefix for source type filter:

```python
def _parse_index_query(query: str) -> tuple[SourceType | None, str]:
    prefixes = {
        "manual:": SourceType.INDEXED_MANUAL,
        "doc:": SourceType.INDEXED_DOCUMENT,
        "code:": SourceType.INDEXED_CODE,
        "oem:": SourceType.WEB_OE_FIRSTPARTY,
        "trusted:": SourceType.WEB_TRUSTED,
        "forum:": SourceType.WEB_COMMUNITY,
    }
    for prefix, source_type in prefixes.items():
        if query.lower().startswith(prefix):
            return source_type, query[len(prefix):].strip()
    return None, query
```

**File:** `r3lay/core/index.py`

Add `source_type_filter` param to `search_async()`:

```python
async def search_async(
    self,
    query: str,
    n_results: int = 5,
    source_type_filter: SourceType | None = None,
) -> list[RetrievalResult]:
    # ... existing search ...
    if source_type_filter:
        results = [r for r in results if r.source_type == source_type_filter]
    return results[:n_results]
```

---

## Files to Modify

| File | Changes | Lines |
|------|---------|-------|
| `r3lay/config.py` | Implement `save()` / `load()` | ~40 |
| `r3lay/ui/widgets/model_panel.py` | Call `save()` on role changes | ~3 |
| `r3lay/ui/widgets/input_pane.py` | Trust badges + source filter | ~30 |
| `r3lay/core/index.py` | Add `source_type_filter` param | ~5 |

**Total:** ~80 lines of changes

---

## Testing Plan

1. **Config persistence:**
   - Set embedder in Model Panel -> quit -> restart -> verify persisted

2. **Trust badges:**
   - Run `/index oil change` -> verify badges show `[MANUAL]`, `[CODE]`, etc.

3. **Source filtering:**
   - Run `/index manual: oil change` -> verify only manual results
   - Run `/index code: async def` -> verify only code results

---

## Dependencies

- `ruamel.yaml>=0.18.0` - Already in pyproject.toml
- `SourceType` enum - Already in `r3lay/core/sources.py`
