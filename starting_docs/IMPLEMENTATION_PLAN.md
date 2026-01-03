# r3LAY Implementation Plan

> A structured plan for vibe-coding the r3LAY TUI research assistant.

---

## Project Overview

**r3LAY** is a terminal-based research assistant that combines:
- Local LLM inference (Ollama, llama.cpp, HuggingFace)
- Hybrid RAG with CGRAG patterns (BM25 + Vector + RRF fusion)
- Deep research expeditions with HERMES-inspired convergence detection
- Full provenance tracking and axiom validation
- Theme-based project organization (vehicle, compute, code, electronics, home)

---

## Architecture Summary

```
┌─────────────────────────────────┬─────────────────────────────────┐
│                                 │  User Input (TextArea)          │
│   Response Pane                 │  40% height                     │
│   (Markdown, code, diagrams)    ├─────────────────────────────────┤
│                                 │  Tabbable Pane                  │
│   60% width                     │  [Models][Index][Axioms]        │
│                                 │  [Sessions][Settings]           │
└─────────────────────────────────┴─────────────────────────────────┘
```

### Core Components

| Component | Technology | Pattern Source |
|-----------|------------|----------------|
| TUI Framework | Textual | - |
| Vector Store | ChromaDB | CGRAG |
| Lexical Search | rank_bm25 | CGRAG |
| Score Fusion | RRF (k=60) | CGRAG |
| Embeddings | sentence-transformers | CGRAG |
| LLM Backends | Ollama, llama.cpp | - |
| Provenance | YAML-based signals | HERMES |
| Research | Multi-cycle expeditions | HERMES |
| Config | Pydantic Settings | - |

---

## Implementation Phases

### Phase 1: Core Foundation (Week 1)

**Goal:** Bootable TUI with model selection

#### Tasks

- [ ] **1.1 Project Setup**
  - Validate pyproject.toml dependencies
  - Create virtual environment
  - Test basic Textual app launch

- [ ] **1.2 Configuration System**
  - Test AppConfig loading from r3lay.yaml
  - Verify HuggingFace cache path detection
  - Test Ollama endpoint configuration

- [ ] **1.3 Model Discovery**
  - Test `huggingface-cli scan-cache --json` parsing
  - Test Ollama `/api/tags` integration
  - Implement model list caching

- [ ] **1.4 Basic TUI Shell**
  - Implement bento layout (response | input + tabs)
  - Wire up Model panel with selection
  - Test hot-swapping between models

**Deliverable:** Running TUI that can scan and select models

---

### Phase 2: Hybrid Index (Week 2)

**Goal:** CGRAG-style retrieval working

#### Tasks

- [ ] **2.1 Document Loader**
  - Implement file type detection
  - Test recursive directory scanning
  - Handle encoding edge cases

- [ ] **2.2 Semantic Chunking**
  - Implement AST-based Python chunking
  - Implement section-based Markdown chunking
  - Implement paragraph-based text chunking
  - Test code block preservation

- [ ] **2.3 Code-Aware Tokenization**
  - Implement CamelCase splitting
  - Implement snake_case splitting
  - Test on real code samples

- [ ] **2.4 Hybrid Search**
  - Wire up ChromaDB vector search
  - Wire up BM25 lexical search
  - Implement RRF fusion
  - Test with `/index` command

- [ ] **2.5 Token Budget Packing**
  - Implement greedy packing by score
  - Test context window limits

**Deliverable:** `/index <query>` returns relevant chunks with hybrid scoring

---

### Phase 3: LLM Integration (Week 3)

**Goal:** Conversational chat with RAG context

#### Tasks

- [ ] **3.1 Ollama Adapter**
  - Implement streaming chat
  - Test response metrics capture
  - Handle connection errors gracefully

- [ ] **3.2 llama.cpp Adapter**
  - Implement ChatML prompt building
  - Test streaming SSE parsing
  - Handle stop tokens

- [ ] **3.3 Context Assembly**
  - Build system prompt with project summary
  - Inject RAG context
  - Inject axiom context
  - Test token counting

- [ ] **3.4 Chat Flow**
  - Wire up InputPane → LLM → ResponsePane
  - Implement streaming display
  - Test conversation history windowing

**Deliverable:** Natural conversation with context-aware responses

---

### Phase 4: Provenance & Axioms (Week 4)

**Goal:** Source tracking and knowledge validation

#### Tasks

- [ ] **4.1 Signals Manager**
  - Implement source registration
  - Test citation creation
  - Implement confidence calculation

- [ ] **4.2 Axiom System**
  - Implement axiom CRUD
  - Test category validation
  - Implement tag filtering
  - Test supersession tracking

- [ ] **4.3 Citation Chains**
  - Link axioms to citations
  - Link citations to signals
  - Implement chain traversal

- [ ] **4.4 UI Integration**
  - Wire up Axiom panel
  - Implement `/axiom` command
  - Implement `/axioms` listing
  - Test markdown export

**Deliverable:** `/axiom` creates tracked knowledge with provenance

---

### Phase 5: Deep Research (Week 5)

**Goal:** Multi-cycle expeditions with convergence

#### Tasks

- [ ] **5.1 Web Search**
  - Wire up SearXNG client
  - Test result parsing
  - Implement error handling

- [ ] **5.2 Research Orchestrator**
  - Implement query generation
  - Implement cycle execution
  - Wire up axiom extraction

- [ ] **5.3 Convergence Detection**
  - Implement metrics recording
  - Test threshold logic
  - Test early termination

- [ ] **5.4 Synthesis**
  - Implement report generation
  - Test expedition saving
  - Implement `/research` command

**Deliverable:** `/research <query>` runs autonomous investigation

---

### Phase 6: Polish & Themes (Week 6)

**Goal:** Production-ready with all themes

#### Tasks

- [ ] **6.1 Theme Templates**
  - Validate vehicle template
  - Validate compute template
  - Validate code template
  - Validate electronics template
  - Validate home template
  - Validate projects template

- [ ] **6.2 Registry Commands**
  - Test `/update` for all themes
  - Test `/issue` for vehicle/home
  - Test `/mileage` for vehicle

- [ ] **6.3 Session Management**
  - Test session persistence
  - Implement session loading
  - Test search functionality

- [ ] **6.4 Docker Deployment**
  - Test Dockerfile build
  - Test docker-compose with SearXNG
  - Test host Ollama connectivity

- [ ] **6.5 Performance**
  - Profile index search latency
  - Optimize BM25 rebuilds
  - Test with large document sets

**Deliverable:** Fully functional r3LAY ready for daily use

---

## File Inventory

### Core Modules

| File | Purpose | Status |
|------|---------|--------|
| `r3lay/app.py` | Main TUI application | Scaffolded |
| `r3lay/config.py` | Configuration + themes | Scaffolded |
| `r3lay/core/index.py` | Hybrid RAG (CGRAG) | Scaffolded |
| `r3lay/core/llm.py` | LLM adapters | Scaffolded |
| `r3lay/core/models.py` | Model discovery | Scaffolded |
| `r3lay/core/signals.py` | Provenance tracking | Scaffolded |
| `r3lay/core/axioms.py` | Knowledge validation | Scaffolded |
| `r3lay/core/research.py` | Deep expeditions | Scaffolded |
| `r3lay/core/registry.py` | Project YAML | Scaffolded |
| `r3lay/core/session.py` | Chat persistence | Scaffolded |
| `r3lay/core/search.py` | SearXNG client | Scaffolded |
| `r3lay/core/scraper.py` | Pipet-style scrapers | Scaffolded |

### UI Widgets

| File | Purpose | Status |
|------|---------|--------|
| `r3lay/ui/widgets/response_pane.py` | Main output area | Scaffolded |
| `r3lay/ui/widgets/input_pane.py` | User input | Scaffolded |
| `r3lay/ui/widgets/model_panel.py` | Model selection | Scaffolded |
| `r3lay/ui/widgets/index_panel.py` | RAG management | Scaffolded |
| `r3lay/ui/widgets/axiom_panel.py` | Axiom viewer | Scaffolded |
| `r3lay/ui/widgets/session_panel.py` | Session history | Scaffolded |
| `r3lay/ui/widgets/settings_panel.py` | Configuration | Scaffolded |
| `r3lay/ui/screens/init.py` | Project init wizard | Scaffolded |

### Configuration

| File | Purpose |
|------|---------|
| `r3lay.yaml` | Per-project config |
| `registry.yaml` | Project metadata |
| `.signals/sources.yaml` | Registered sources |
| `.signals/citations.yaml` | Citation records |
| `axioms/axioms.yaml` | Validated knowledge |

---

## Claude Integration Guide

### SESSION_NOTES.md Structure

Create this file in the project root:

```markdown
# r3LAY Session Notes

> Reverse chronological development log. Newest sessions at top.

---

## Session: YYYY-MM-DD - [Title]

### Summary
Brief description of what was accomplished.

### Files Modified
- `r3lay/core/index.py` - Added RRF fusion (lines 180-220)
- `r3lay/ui/widgets/input_pane.py` - Fixed command parsing

### Problems Encountered
1. **BM25 index not rebuilding** - Solution: Call `_rebuild_bm25()` after `add_chunks()`
2. **Streaming cuts off** - Solution: Handle `done` field in response

### Architectural Decisions
- Decided to use ChromaDB's built-in embedding function instead of manual calls
- Moved token budget packing to retriever instead of index

### Next Steps
- [ ] Wire up axiom panel refresh
- [ ] Test with real Ollama models
- [ ] Add error boundaries to UI

### Breaking Changes
- `HybridIndex.search()` now returns `RetrievalResult` instead of raw dicts

---
```

### claude.md Content

```markdown
# r3LAY Development Guide

## Your Role & Responsibilities

You are the lead technical architect and implementer for r3LAY. Your responsibilities include:

- **Architecture decisions** - Design robust, scalable solutions following the spec
- **Code implementation** - Write production-quality Python with Textual TUI
- **System integration** - Connect local LLMs, ChromaDB, SearXNG
- **UI/UX implementation** - Build terminal-aesthetic interfaces
- **Performance optimization** - Meet retrieval latency targets (<100ms)
- **Documentation** - Provide clear inline documentation

## Getting Up to Speed

**IMPORTANT:** Before starting work, review `SESSION_NOTES.md`

The SESSION_NOTES.md file contains the complete development history with:

- Recent implementation sessions (newest first - no scrolling needed!)
- Problems encountered and solutions
- Files modified with line numbers
- Breaking changes and architectural decisions
- Next steps and pending work

### When to check SESSION_NOTES.md:

- At the start of every work session
- Before modifying recently changed files
- When encountering unexpected behavior
- To understand recent architectural decisions
- To avoid repeating solved problems

## Key Patterns

### CGRAG Patterns (from hybrid index)
- BM25 + Vector search with RRF fusion (k=60)
- Code-aware tokenization (CamelCase/snake_case)
- Semantic chunking by file type
- Token budget packing

### HERMES Patterns (from research system)
- Multi-cycle expeditions with convergence detection
- Stop when axiom generation < 30% of previous
- Stop when source discovery < 20% of previous
- Min 2 cycles, max 10 cycles

### Provenance
- Every fact needs a Signal (source)
- Every assertion needs a Citation (location + excerpt)
- Confidence = type_weight × transmission_confidence × corroboration_boost

## Testing Workflow

1. **TUI Testing:** `python -m r3lay.app /path/to/test/project`
2. **Unit Tests:** `pytest tests/`
3. **Integration:** `docker compose run --rm r3lay`

## Common Issues

| Issue | Solution |
|-------|----------|
| ChromaDB connection error | Check `.chromadb/` folder exists |
| Ollama timeout | Verify `ollama serve` is running |
| BM25 empty results | Ensure `_rebuild_bm25()` was called |
| Textual import errors | Check Python 3.11+ |
```

---

## Testing Checklist

### Unit Tests

```python
# tests/test_index.py
def test_code_aware_tokenizer():
    tokenizer = CodeAwareTokenizer()
    assert tokenizer.tokenize("getUserData") == ["get", "user", "data"]
    assert tokenizer.tokenize("user_data") == ["user", "data"]

def test_rrf_fusion():
    index = HybridIndex(...)
    # Test that RRF combines vector and BM25 ranks correctly

def test_semantic_chunking():
    chunker = SemanticChunker()
    # Test Python AST chunking
    # Test Markdown section chunking
```

### Integration Tests

```python
# tests/test_research.py
async def test_expedition_convergence():
    orchestrator = ResearchOrchestrator(...)
    events = []
    async for event in orchestrator.run("test query"):
        events.append(event)
    assert any(e["type"] == "converged" for e in events)
```

---

## Environment Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in dev mode
pip install -e ".[dev]"

# Start Ollama (if using)
ollama serve

# Start SearXNG (if using)
docker run -d -p 8080:8080 searxng/searxng

# Run r3LAY
r3lay /path/to/project
```

---

## Naming Conventions

| Concept | Name | Metaphor |
|---------|------|----------|
| Project | Relay | Communication relay station |
| Deep Research | Expedition | Exploration journey |
| Knowledge Store | Index | Library index |
| Validated Facts | Axioms | Mathematical axioms |
| Sources | Signals | Radio signals |
| Chat History | Sessions | Communication sessions |

---

## Success Criteria

### MVP (Phase 1-3)
- [ ] TUI launches and displays bento layout
- [ ] Can scan and select Ollama/HF models
- [ ] Can index documents with hybrid search
- [ ] Can chat with RAG context

### Full Release (Phase 4-6)
- [ ] Axioms persist and accumulate
- [ ] Provenance chains are complete
- [ ] Deep research converges correctly
- [ ] All themes work
- [ ] Docker deployment works

---

## Resources

- **Textual Docs:** https://textual.textualize.io/
- **ChromaDB Docs:** https://docs.trychroma.com/
- **rank_bm25:** https://github.com/dorianbrown/rank_bm25
- **CGRAG Paper:** (reference for hybrid search patterns)
- **HERMES Protocol:** (reference for research automation)

---

*Last Updated: January 2026*
*Version: 0.2.0*
