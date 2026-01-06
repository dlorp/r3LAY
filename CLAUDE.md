# r³LAY - Claude Code Instructions

## Project Identity

**r³LAY** — *Retrospective Recursive Research, Linked Archive Yield*

The name encapsulates the core philosophy:
- **r³ (Retrospective Recursive Research)**: Multi-cycle research that loops back on itself, re-evaluating earlier findings when new information contradicts or refines them
- **Linked Archive Yield**: Knowledge base emphasis — cross-referenced manuals, forums, your own service logs, all yielding actionable, provenance-tracked results

---

## Core Philosophy

r³LAY refines official documentation with community knowledge to produce verifiable, real world proven results. The system recognizes that:

1. **Official sources** (FSMs, datasheets, manufacturer docs) provide specifications and procedures
2. **Community knowledge** (forums, discussions, personal experience) reveals what actually works, common failures, and proven alternatives
3. **The gap between them** is where real value lives — r³LAY bridges this gap systematically

This applies across domains with a **garage hobbyist / tinkerer lens**:
- **Automotive**: Parts interchange, proven fixes, real torque specs vs book specs
- **Electronics**: Component substitutions, actual vs rated specs, failure modes
- **Software**: Workarounds, undocumented behavior, stack combinations that work
- **Home/DIY**: Tool recommendations, technique variations, "what I wish I knew"

### The Retrospective Revision Loop

When r³LAY finds information that contradicts an existing axiom:
1. Mark the existing axiom as **DISPUTED** with the contradicting citation
2. Generate **targeted resolution queries** to clarify the discrepancy
3. Run resolution cycles searching for the nuance (year ranges, model variants, conditions)
4. Either **SUPERSEDE** with qualified axioms or flag for manual review

This is what makes r³LAY different from linear research — it doesn't just accumulate facts, it revises earlier conclusions when new evidence demands it.

---

## Project Overview

r³LAY is a TUI-based personal research assistant with local LLM integration, hybrid RAG, and deep research capabilities. Built with Textual for Python.

```
┌─────────────────────────────────┬─────────────────────────────────┐
│                                 │  User Input (TextArea)          │
│   Response Pane                 │  Multi-line, 40% height         │
│   (Markdown, code, diagrams)    ├─────────────────────────────────┤
│                                 │  Tabbable Pane                  │
│   60% width, scrollable         │  [Models][Index][Axioms]        │
│                                 │  [Sessions][Settings]           │
└─────────────────────────────────┴─────────────────────────────────┘
```

---

## Your Role & Responsibilities

You are the lead technical architect and implementer for r³LAY. Your responsibilities include:

- **Architecture decisions** - Design robust, scalable solutions following the spec
- **Code implementation** - Write production-quality Python 3.11+ with type hints
- **System integration** - Connect local LLMs (MLX, llama.cpp, vLLM), ChromaDB, SearXNG
- **UI/UX implementation** - Build terminal-aesthetic interfaces with Textual
- **Memory management** - Handle model hot-swapping and graceful shutdown properly
- **Knowledge systems** - Implement provenance tracking and retrospective revision
- **Documentation** - Provide clear inline documentation

---

## Role-Based Development

When working on r³LAY, adopt the appropriate role:

**As Architect** (planning tasks):
- Break down features into tasks
- Design interfaces before implementing
- Consider memory constraints and async patterns

**As Backend** (core/ work):
- Focus on LLM backends, RAG, model discovery
- Strict memory cleanup patterns
- Test load/unload cycles

**As Frontend** (ui/ work):
- Focus on Textual widgets and styling
- No blocking in UI thread
- Keyboard navigation

**As Research** (signals, axioms, expeditions):
- Provenance tracking via Signals
- Knowledge validation via Axioms
- Convergence detection in expeditions
- **Retrospective revision when contradictions surface**
- Axiom state management (pending → validated → disputed → superseded)

When I say "@Backend", "@Architect", "@Frontend", or "@Research", switch focus to that role's priorities.

---

## Getting Up to Speed

**IMPORTANT**: Before starting work, review `SESSION_NOTES.md`

**Tips for Effective Sessions**

- Start small - Get one thing working before adding complexity
- Test incrementally - Run `python -m r3lay.app` after each change
- Update SESSION_NOTES.md - Document what you did and any issues
- Read the reference - `starting_docs/` has working code to adapt
- Memory matters - Test model loading/unloading cycles early

The SESSION_NOTES.md file contains the complete development history with:
- Recent implementation sessions (newest first - no scrolling needed!)
- Problems encountered and solutions
- Files modified with descriptions
- Breaking changes and architectural decisions
- Next steps and pending work

### When to check SESSION_NOTES.md:
- At the start of every work session
- Before modifying recently changed files
- When encountering unexpected behavior
- To understand recent architectural decisions
- To avoid repeating solved problems

### Reference Materials
The `starting_docs/` folder contains scaffold code and architecture docs for reference.

<use_parallel_tool_calls>
If you intend to call multiple tools and there are no dependencies
between the tool calls, make all of the independent tool calls in
parallel. Prioritize calling tools simultaneously whenever the
actions can be done in parallel rather than sequentially. For
example, when reading 3 files, run 3 tool calls in parallel to read
all 3 files into context at the same time. Maximize use of parallel
tool calls where possible to increase speed and efficiency.
However, if some tool calls depend on previous calls to inform
dependent values like the parameters, do NOT call these tools in
parallel and instead call them sequentially. Never use placeholders
or guess missing parameters in tool calls.
</use_parallel_tool_calls>

<investigate_before_answering>
Never speculate about code you have not opened. If the user
references a specific file, you MUST read the file before
answering. Make sure to investigate and read relevant files BEFORE
answering questions about the codebase. Never make any claims about
code before investigating unless you are certain of the correct
answer - give grounded and hallucination-free answers.
</investigate_before_answering>

<do_not_act_before_instructions>
Do not jump into implementation or change files unless clearly
instructed to make changes. When the user's intent is ambiguous,
default to providing information, doing research, and providing
recommendations rather than taking action. Only proceed with edits,
modifications, or implementations when the user explicitly requests
them.
</do_not_act_before_instructions>

---

## Critical Architecture Decisions

### LLM Backend Priority Stack

| Priority | Backend | Use Case | Library |
|----------|---------|----------|---------|
| 1 | **MLX** | Apple Silicon (default) | `mlx-lm`, `mlx-vlm` |
| 2 | **vLLM** | NVIDIA GPUs | `vllm` AsyncLLM |
| 3 | **llama.cpp** | Universal fallback | `llama-cpp-python` |

**Auto-detection logic:**
```python
if platform.system() == "Darwin" and platform.machine() == "arm64":
    return "mlx"
elif torch.cuda.is_available():
    return "vllm"
else:
    return "llama_cpp"
```

### Model Loading Rules

1. **TUI starts with NO models loaded** - fast startup, user selects from Models panel
2. **Hot-swap without restart** - load/unload models while TUI runs
3. **Graceful shutdown** - clean up all models on Ctrl+Q or SIGTERM
4. **Memory isolation** - use subprocess if in-process cleanup fails

### Model Sources

| Source | Path | Format |
|--------|------|--------|
| HuggingFace cache | `/Users/dperez/Documents/LLM/` | safetensors, GGUF |
| GGUF drop folder | `~/.r3lay/models/` | .gguf files |
| Ollama | `http://localhost:11434` | via API |

The model scanner should enumerate all three sources and present a unified list.

### Memory Management (Critical)

**`del model + gc.collect()` does NOT reliably free GPU/Metal memory.**

MLX cleanup pattern:
```python
import mlx.core as mx
del model, tokenizer
gc.collect()
mx.metal.clear_cache()
mx.eval(mx.zeros(1))  # Force sync
mx.metal.clear_cache()
```

llama-cpp-python cleanup:
```python
llm.close()  # Explicit cleanup
del llm
gc.collect()
```

**If memory isn't released:** Use subprocess isolation for model inference.

---

## Key Patterns

### Hybrid RAG Patterns
- BM25 + Vector search with RRF fusion (k=60)
- Code-aware tokenization (CamelCase/snake_case splitting)
- Semantic chunking (AST for code, sections for markdown)
- Token budget packing (8000 tokens default)

### R³ Deep Research Patterns
- Multi-cycle expeditions with convergence detection
- Stop when axiom generation < 30% of previous cycle
- **Never converge with unresolved disputes**
- Provenance tracking via Signals system
- Axiom categories: specifications, procedures, compatibility, diagnostics, history, safety

### Signals & Axioms Patterns

**Signal Types** (source classification):
| Type | Weight | Example |
|------|--------|---------|
| DOCUMENT | 0.95 | FSM, datasheet, official manual |
| CODE | 0.90 | Config file, source code |
| USER | 0.80 | User-provided information |
| COMMUNITY | 0.75 | Forum post, discussion thread |
| WEB | 0.70 | Web search result |
| INFERENCE | 0.60 | LLM-derived conclusion |

**Axiom States**:
```
PENDING → VALIDATED
    ↓         ↓
REJECTED   DISPUTED → SUPERSEDED
               ↓
           INVALIDATED
```

**Confidence Calculation**:
- Base confidence from signal type weight
- Corroboration boost: +0.05 per additional source
- Recency factor for time-sensitive info

### Retrospective Revision Patterns

When contradictions are detected:
1. **Detection**: Semantic similarity check against existing axioms in same category
2. **Dispute**: Mark existing axiom as DISPUTED with reason and contradicting citation
3. **Resolution Query**: Generate targeted search (e.g., "EJ22 vs EJ25 timing belt interval")
4. **Resolution Outcomes**:
   - **Supersede**: Create qualified replacement axiom(s)
   - **Confirm**: Validate original, reject new (with reason)
   - **Merge**: Synthesize both into nuanced axiom
   - **Manual**: Flag for user review if unresolvable

### Document Processing Options
| Tool | Use Case | Apple Silicon |
|------|----------|---------------|
| Marker | PDF → Markdown | ✓ MPS supported |
| Qwen2.5-VL-7B | Document OCR | ✓ via MLX 4-bit |
| ColQwen2 | Visual RAG (no OCR) | ✓ MPS supported |

---

## Code Style

```python
# Type hints required
async def generate_stream(
    self,
    messages: list[dict[str, str]],
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> AsyncGenerator[str, None]:
    ...

# Pydantic for data models
class ModelConfig(BaseModel):
    name: str
    source: Literal["mlx", "llama_cpp", "vllm", "ollama"]
    path: Path | None = None

# Async/await for I/O
async with httpx.AsyncClient() as client:
    response = await client.get(url)
```

---

## Commands Reference

| Command | Description |
|---------|-------------|
| `/help` | Show commands |
| `/search <query>` | Web search via SearXNG |
| `/index <query>` | Search hybrid RAG index |
| `/research <query>` | Deep research expedition (R³) |
| `/axiom <statement>` | Add validated knowledge |
| `/axioms [category]` | List axioms (filter by category) |
| `/axioms --disputed` | Show disputed axioms needing resolution |
| `/cite <axiom_id>` | Show provenance chain |
| `/dispute <axiom_id> <reason>` | Mark axiom as disputed |
| `/update <key> <value>` | Update registry |
| `/clear` | New session |

---

## File Structure

```
r3lay/
├── r3lay/
│   ├── __init__.py
│   ├── app.py                # Main Textual app
│   ├── config.py             # Pydantic settings + themes
│   ├── core/
│   │   ├── backends/         # MLX, llama.cpp, vLLM adapters
│   │   │   ├── base.py       # Abstract InferenceBackend
│   │   │   ├── mlx.py
│   │   │   ├── llama_cpp.py
│   │   │   └── vllm.py
│   │   ├── models.py         # Model discovery + manifest
│   │   ├── index.py          # Hybrid RAG
│   │   ├── signals.py        # Provenance tracking
│   │   ├── axioms.py         # Validated knowledge + state machine
│   │   ├── research.py       # Deep research orchestrator (R³)
│   │   ├── contradiction.py  # Contradiction detection
│   │   ├── search.py         # SearXNG client
│   │   ├── router.py         # Smart model routing
│   │   ├── registry.py       # Project YAML management
│   │   └── session.py        # Chat sessions
│   └── ui/
│       ├── widgets/
│       │   ├── response_pane.py
│       │   ├── input_pane.py
│       │   ├── model_panel.py
│       │   ├── index_panel.py
│       │   ├── axiom_panel.py
│       │   └── session_panel.py
│       └── styles/
│           └── app.tcss
├── tests/
├── plans/                    # Implementation plans
├── starting_docs/            # Reference implementations
├── pyproject.toml
├── Dockerfile
├── docker-compose.yaml
└── CLAUDE.md
```

---

## Testing Commands

```bash
# Run the app
python -m r3lay.app /path/to/project

# Run tests
pytest tests/ -v

# Type checking
mypy r3lay/

# Linting
ruff check r3lay/
```

---

## Common Tasks

### Adding a new LLM backend
1. Create adapter in `core/backends/`
2. Implement `InferenceBackend` abstract class
3. Add detection logic to `core/models.py`
4. Register in backend factory

### Adding a new command
1. Add handler in `ui/widgets/input_pane.py` → `_handle_command()`
2. Update `/help` output
3. Add to README.md commands table
4. Add to this CLAUDE.md commands table

### Adding a new panel tab
1. Create widget in `ui/widgets/`
2. Import in `ui/widgets/__init__.py`
3. Add TabPane in `app.py` → `MainScreen.compose()`
4. Add keybinding in `BINDINGS`

### Adding a new axiom state transition
1. Update `AxiomState` enum in `core/axioms.py`
2. Add transition method to `AxiomManager`
3. Update state diagram in docs
4. Add UI action in `axiom_panel.py`

---

## Hardware Context

**Primary development machine:**
- Apple M4 Pro, 24GB unified memory
- Models stored in `/Users/dperez/Documents/LLM/` (HF structure)
- Optimal: 7B models at 4-bit quantization (~6-8GB)

**Secondary target:**
- NVIDIA RTX 3080 Ti (12GB VRAM)
- Use vLLM or llama.cpp with CUDA
- 7B models at Q4_K_M fit fully on GPU

---

## Session Notes Format

**IMPORTANT**: `SESSION_NOTES.md` is a live, append-only file.

- **Add new sessions at the TOP** (below the header, above previous sessions)
- **NEVER modify or replace existing session entries**
- Newest sessions first = no scrolling needed to see recent work

After each coding session, add a new entry at the top:

```markdown
## Session: YYYY-MM-DD HH:MM - Brief Title

### Summary
Brief description of what was accomplished.

### Files Created/Modified
- `path/to/file.py` - Description of changes

### Problems Encountered
- Problem: Description
- Solution: How it was fixed

### Architectural Decisions
- Decision made and rationale

### Next Steps
- [ ] Immediate next task
- [ ] Following task

### Breaking Changes
- API or behavior changes that affect other code

---
```
