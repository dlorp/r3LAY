# r3LAY Session Notes

> Reverse chronological development log. Newest sessions at top.
> 
> **Reading Instructions:** Start here — most recent work is immediately visible.

---

## Session: 2026-01-02 - Initial Scaffold

### Summary
Created complete project scaffold with CGRAG hybrid search patterns and HERMES research patterns integrated. Implemented bento layout TUI matching mockup design.

### Files Created
- `r3lay/app.py` - Main TUI with bento layout (response | input + tabs)
- `r3lay/config.py` - Configuration + 6 themes
- `r3lay/core/index.py` - Hybrid RAG with BM25 + Vector + RRF fusion
- `r3lay/core/llm.py` - Ollama and llama.cpp adapters
- `r3lay/core/models.py` - HuggingFace cache + Ollama model discovery
- `r3lay/core/signals.py` - Provenance tracking (6 signal types)
- `r3lay/core/axioms.py` - Validated knowledge with categories
- `r3lay/core/research.py` - Deep expeditions with convergence detection
- `r3lay/core/registry.py` - Project YAML management
- `r3lay/core/session.py` - Chat session persistence
- `r3lay/core/search.py` - SearXNG web search client
- `r3lay/core/scraper.py` - Pipet-style YAML scrapers
- `r3lay/ui/widgets/*.py` - All UI panels
- `r3lay/ui/screens/init.py` - Project initialization wizard

### Architectural Decisions

1. **Bento Layout** - Response pane (60%) left, input + tabs (40%) right
   - Matches user's mockup sketch
   - Input at top-right for easy access
   - Tabbed panels below for models/index/axioms/sessions/settings

2. **CGRAG Patterns Adopted:**
   - Hybrid search (BM25 + vector) for better code/config retrieval
   - RRF fusion with k=60 for robust score merging
   - Code-aware tokenization (CamelCase/snake_case splitting)
   - Semantic chunking (AST for Python, sections for Markdown)
   - Token budget packing for context management

3. **HERMES Patterns Adopted:**
   - Multi-cycle research expeditions
   - Convergence detection (axiom < 30%, sources < 20%)
   - Full provenance tracking via Signals
   - Axiom system for validated knowledge

4. **Patterns Rejected:**
   - PostgreSQL (too heavy — ChromaDB + YAML sufficient)
   - Multi-agent architecture (single conversational agent fits better)
   - React frontend (TUI is the goal)
   - Redis caching (in-memory LRU sufficient)

5. **Naming Convention:**
   - Project: r3LAY (relay metaphor)
   - Research: Expeditions
   - Knowledge: Index
   - Facts: Axioms
   - Sources: Signals

### Problems Encountered
None yet — initial scaffold.

### Next Steps
- [ ] Test TUI boot with `python -m r3lay.app`
- [ ] Validate pyproject.toml dependencies install
- [ ] Test Ollama model scanning
- [ ] Test HuggingFace cache discovery
- [ ] Wire up first `/index` search

### Breaking Changes
N/A — initial implementation.

---

## Template for New Sessions

Copy this template for each new session:

```markdown
## Session: YYYY-MM-DD - [Brief Title]

### Summary
[1-2 sentences describing what was accomplished]

### Files Modified
- `path/to/file.py` - [What changed] (lines X-Y)
- `path/to/other.py` - [What changed]

### Problems Encountered
1. **[Problem description]** - Solution: [How it was fixed]
2. **[Problem description]** - Solution: [How it was fixed]

### Architectural Decisions
- [Decision made and rationale]

### Next Steps
- [ ] [Task 1]
- [ ] [Task 2]

### Breaking Changes
- [API or behavior changes that affect other code]
```

---

## Quick Reference

### Key Files
| Purpose | File |
|---------|------|
| Main app | `r3lay/app.py` |
| Hybrid search | `r3lay/core/index.py` |
| LLM adapters | `r3lay/core/llm.py` |
| Research | `r3lay/core/research.py` |
| Provenance | `r3lay/core/signals.py` |
| Axioms | `r3lay/core/axioms.py` |

### Commands
| Command | Purpose |
|---------|---------|
| `/help` | Show commands |
| `/search` | Web search |
| `/index` | RAG search |
| `/research` | Deep expedition |
| `/axiom` | Add axiom |
| `/axioms` | List axioms |

### Keybindings
| Key | Action |
|-----|--------|
| `Ctrl+N` | New session |
| `Ctrl+S` | Save |
| `Ctrl+R` | Reindex |
| `Ctrl+E` | Research |
| `Ctrl+Q` | Quit |
