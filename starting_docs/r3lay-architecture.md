# r3LAY Architecture

## Overview

**r3LAY** is a TUI-based personal research assistant with local LLM integration, hybrid RAG (CGRAG-inspired), and deep research capabilities.

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

## Core Philosophy

| Principle | Implementation |
|-----------|----------------|
| **Local-First** | MLX (Apple Silicon), llama.cpp (fallback), vLLM (future Nvidia) |
| **Interactive + Autonomous** | Conversational chat with deep research expeditions |
| **Knowledge Accumulation** | Findings compound via provenance-tracked axioms |
| **Theme-Agnostic** | Same engine powers vehicles, code, electronics, etc. |
| **Hot-Swappable** | Switch models without restart |
| **Multi-Instance Safe** | Multiple r3LAY processes can run simultaneously |

---

## LLM Backend Architecture

### Backend Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                    Hardware Auto-Detection                       │
├─────────────────────────────────────────────────────────────────┤
│  Apple Silicon?  ──────────────────────►  MLX Backend           │
│  (M1/M2/M3/M4)                            (40-130 t/s)          │
│         │                                                        │
│         ▼                                                        │
│  Nvidia GPU?  ─────────────────────────►  vLLM Backend          │
│                                           (stubbed for future)   │
│         │                                                        │
│         ▼                                                        │
│  Fallback  ────────────────────────────►  llama.cpp Backend     │
│                                           + Speculative Decoding │
└─────────────────────────────────────────────────────────────────┘
```

### MLX Backend (Primary for Apple Silicon)

**Performance on M4 Pro (24GB, 273 GB/s bandwidth)**:

| Model Size | Quantization | Speed | Memory |
|------------|--------------|-------|--------|
| 7B-8B | Q4 | 40-60 t/s | ~5-6GB |
| 14B | Q4 | 25-40 t/s | ~9-10GB |
| 32B | Q4 | 10-15 t/s | ~18-20GB |

**Key APIs**:
```python
from mlx_lm import load, stream_generate
import mlx.core as mx

# Load
model, tokenizer = load("mlx-community/Qwen2.5-7B-Instruct-4bit")

# Generate (streaming)
for response in stream_generate(model, tokenizer, prompt):
    print(response.text, end="")

# Unload (critical for hot-swap)
del model, tokenizer
mx.metal.clear_cache()
gc.collect()
```

### llama.cpp Backend (Fallback + Speculative Decoding)

**Speculative decoding** pairs a small draft model with a larger target model for 1.5-2.5× speedups on structured outputs.

**Requirements**:
- Draft and target must share IDENTICAL tokenizers
- Best for: code, JSON, structured outputs
- Avoid for: creative writing (low draft acceptance)

**Effective pairs**:
| Target | Draft | Speedup |
|--------|-------|---------|
| Qwen 2.5 7B | Qwen 2.5 0.5B | 2.5× |
| Llama 3.1 8B | Llama 3.2 1B | 1.83× |

**Server mode**:
```bash
llama-server \
  -m qwen2.5-7b-instruct-q4_k_m.gguf \
  -md qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --draft-max 12 \
  --draft-min 4 \
  -ngl 99 \
  --port 8080
```

### vLLM Backend (Future - Nvidia)

Stubbed for future implementation. Will target:
- Nvidia GPUs with CUDA
- Continuous batching
- PagedAttention for efficient KV cache

### Critical Note: dLLM ≠ Speculative Decoding

**Important**: The dLLM repository (github.com/ZHZisZZ/dllm) implements *Diffusion Language Models* — NOT traditional speculative decoding. For draft+target speculation, use llama.cpp's native `-md` flag.

---

## Quantization Guide (24GB Memory)

| Quantization | 7B Size | 14B Size | Quality | Use Case |
|--------------|---------|----------|---------|----------|
| Q5_K_M | ~4.7GB | ~8.6GB | Best | Quality priority |
| **Q4_K_M** | **~3.8GB** | **~7.0GB** | **Good** | **Default choice** |
| Q4_K_S | ~3.6GB | ~6.5GB | Okay | Tight memory |
| Q3_K_M | ~3.1GB | ~5.6GB | Degraded | Not recommended |

**Memory formula**: Model size + KV cache (~0.13-0.5 MB/token × context length)

---

## HuggingFace Cache Scanning

**Custom cache path**: `/Users/dperez/Documents/LLM`

**Structure**: `hub/models--{org}--{name}/snapshots/{commit}/`

**Format detection**:
```python
def detect_format(path: Path) -> ModelFormat:
    files = [f.name for f in path.iterdir()]
    if any(f.endswith('.gguf') for f in files):
        return ModelFormat.GGUF
    if 'mlx-community' in str(path):
        return ModelFormat.MLX
    if any(f.endswith('.safetensors') for f in files):
        return ModelFormat.SAFETENSORS
```

---

## Multi-Instance Coordination

**Challenge**: Multiple r3LAY instances (e.g., `~/vehicles/brighton`, `~/home/server`) may run simultaneously.

**MLX**: Models cannot be shared across processes (unified memory is per-process). Each instance loads independently.

**llama-cpp**: Use server mode with port management:
```python
class PortManager:
    def allocate(self, instance_id: str) -> int:
        for port in range(8080, 8180):
            if self._is_available(port):
                self._lock(port, instance_id)
                return port
        raise RuntimeError("No available ports")
```

**File locking for model loading**:
```python
import fasteners
lock = fasteners.InterProcessLock('/tmp/r3lay_model.lock')
with lock:
    backend.load(model_path)
```

---

## Key Features

### 1. Hybrid Index (CGRAG-Inspired)

Based on research-backed patterns from CGRAG:

| Feature | Implementation |
|---------|----------------|
| **Vector + BM25** | Dual retrieval for better recall |
| **RRF Fusion** | Reciprocal Rank Fusion (k=60) |
| **Code-aware Tokenization** | CamelCase/snake_case splitting |
| **Semantic Chunking** | AST for code, section-based for markdown |
| **Token Budget Packing** | Greedy by relevance score |

```python
# Score combination
rrf_score = (
    vector_weight * (1 / (k + vector_rank)) +
    bm25_weight * (1 / (k + bm25_rank))
)
```

### 2. Deep Research (Expeditions)

Inspired by HERMES protocol:

```
User Query → Generate Queries → Search (Web + RAG)
    ↓
Cycle 1: Extract axioms → Record metrics
Cycle 2: Fill gaps → Record metrics
Cycle N: Convergence detected → Synthesize
    ↓
Final Report + Axioms + Signals
```

**Convergence Detection:**
- Stop when axiom generation < 30% of previous cycle
- Stop when source discovery < 20% of previous cycle
- Min 2 cycles, max 10 cycles

### 3. Provenance (Signals)

Every fact has full source tracking:

| Signal Type | Example |
|-------------|---------|
| `DOCUMENT` | PDF manual, datasheet |
| `WEB` | Forum post, article |
| `USER` | User-provided fact |
| `INFERENCE` | LLM-derived |
| `CODE` | Config file, source |
| `SESSION` | Chat context |

**Confidence Scoring:**
```python
confidence = base_type_weight * transmission_confidence * corroboration_boost
```

### 4. Axioms

Validated knowledge statements with categories:

- `specifications` — Torque values, capacities
- `procedures` — Repair steps, maintenance
- `compatibility` — Part interchanges
- `diagnostics` — Symptoms, causes, solutions
- `history` — Production dates, changes
- `safety` — Warnings, limits

---

## Themes

| Theme | Description | Key Folders |
|-------|-------------|-------------|
| **Vehicle** | Cars, motorcycles, boats | manuals, diagrams, parts |
| **Compute** | Servers, NAS, routers | configs, scripts, topology |
| **Code** | Software projects | src, docs, apis |
| **Electronics** | Hardware mods, builds | datasheets, schematics |
| **Home** | Property, HVAC | manuals, warranties |
| **Projects** | Miscellaneous | reference, assets |

---

## Project Structure

```
/project
├── registry.yaml           # Project metadata
├── r3lay.yaml              # Local config
├── .chromadb/              # Vector database
├── .signals/               # Provenance tracking
│   ├── sources.yaml
│   └── citations.yaml
├── axioms/
│   └── axioms.yaml
├── manuals/                # (theme-specific)
├── docs/
├── links/
│   ├── links.yaml
│   └── scraped/
├── logs/
├── plans/
├── sessions/
└── research/
    └── expedition_*/
```

---

## Commands

| Command | Description |
|---------|-------------|
| `/help` | Show commands |
| `/search <query>` | Web search via SearXNG |
| `/index <query>` | Search hybrid index |
| `/research <query>` | Start deep research expedition |
| `/axiom <statement>` | Add axiom |
| `/axioms [tags]` | List axioms |
| `/update <key> <value>` | Update registry |
| `/issue <desc>` | Add known issue |
| `/mileage <value>` | Update odometer |
| `/clear` | Clear chat |

---

## Keybindings

| Key | Action |
|-----|--------|
| `Ctrl+N` | New session |
| `Ctrl+S` | Save session |
| `Ctrl+R` | Reindex |
| `Ctrl+E` | Start research |
| `Ctrl+Enter` | Send input |
| `Ctrl+1-5` | Switch tabs |
| `Ctrl+D` | Toggle dark mode |
| `Ctrl+Q` | Quit |

---

## Configuration

```yaml
# r3lay.yaml
models:
  # Custom HuggingFace cache path
  hf_cache_path: "/Users/dperez/Documents/LLM"
  
  # Backend priority: mlx > vllm > llama_cpp
  preferred_backend: "auto"  # or "mlx", "llama_cpp", "vllm"
  
  # MLX settings (Apple Silicon)
  mlx:
    enabled: true
    default_model: "mlx-community/Qwen2.5-7B-Instruct-4bit"
  
  # llama.cpp settings (fallback)
  llama_cpp:
    enabled: true
    server_port_range: [8080, 8180]
    speculative_decoding:
      enabled: true
      draft_max: 12
      draft_min: 4
      # Draft model must share tokenizer with target
      draft_models:
        "Qwen2.5-7B": "Qwen2.5-0.5B"
        "Llama-3.1-8B": "Llama-3.2-1B"
  
  # vLLM settings (future, Nvidia)
  vllm:
    enabled: false
    tensor_parallel_size: 1

index:
  embedding_model: "all-MiniLM-L6-v2"
  chunk_size: 512
  use_hybrid_search: true
  vector_weight: 0.7
  bm25_weight: 0.3
  rrf_k: 60

searxng:
  enabled: true
  endpoint: "http://localhost:8080"

research:
  min_cycles: 2
  max_cycles: 10
  axiom_threshold: 0.3
  source_threshold: 0.2
```

---

## Docker Deployment

```yaml
# docker-compose.yaml
services:
  r3lay:
    build: .
    volumes:
      - ${PROJECT_PATH:-.}:/project
      - ${HF_CACHE_PATH}:/root/.cache/huggingface:ro
    environment:
      - R3LAY_MODELS__OLLAMA__ENDPOINT=http://host.docker.internal:11434
    extra_hosts:
      - "host.docker.internal:host-gateway"
    stdin_open: true
    tty: true

  searxng:
    image: searxng/searxng:latest
    ports: ["8080:8080"]
    profiles: ["search"]
```

**Usage:**
```bash
# Run r3LAY
PROJECT_PATH=~/vehicles/brighton docker compose run --rm r3lay

# With SearXNG
docker compose --profile search up -d searxng
PROJECT_PATH=~/vehicles/brighton docker compose run --rm r3lay
```

---

## Tech Stack

| Component | Technology | Notes |
|-----------|------------|-------|
| TUI | Textual | Async-native, CSS styling |
| **LLM Primary** | **mlx-lm** | **Apple Silicon optimized** |
| **LLM Fallback** | **llama-cpp-python** | **GGUF + speculative decoding** |
| **LLM Future** | **vLLM** | **Nvidia GPUs (stubbed)** |
| Vector DB | ChromaDB | Local persistence |
| Embeddings | sentence-transformers | all-MiniLM-L6-v2 |
| BM25 | rank_bm25 | Keyword search |
| HTTP | httpx | Async client |
| Config | Pydantic | Type-safe settings |
| YAML | ruamel.yaml | Preserves formatting |
| **Process Lock** | **fasteners** | **Multi-instance coordination** |

---

## Dependencies

```toml
[project]
dependencies = [
    # TUI
    "textual>=0.47.0",
    
    # LLM Backends
    "mlx-lm>=0.14.0",           # Apple Silicon (optional)
    "llama-cpp-python>=0.2.0",  # GGUF fallback (optional)
    
    # RAG
    "chromadb>=0.4.0",
    "sentence-transformers>=2.2.0",
    "rank-bm25>=0.2.2",
    
    # HTTP & Config
    "httpx>=0.27.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "ruamel.yaml>=0.18.0",
    
    # Utilities
    "rich>=13.0.0",
    "tiktoken>=0.5.0",
    "fasteners>=0.19",
]

[project.optional-dependencies]
mlx = ["mlx-lm>=0.14.0"]
cuda = ["vllm>=0.4.0"]
```

---

## Pattern Origins

| Pattern | Source | Notes |
|---------|--------|-------|
| Hybrid Search (BM25+Vector) | CGRAG | RRF fusion with k=60 |
| RRF Fusion | CGRAG | Robust score combination |
| Code-aware Tokenization | CGRAG | CamelCase/snake_case splitting |
| Semantic Chunking | CGRAG | AST for code, sections for markdown |
| Convergence Detection | HERMES | Stop at axiom plateau |
| Multi-cycle Research | HERMES | 2-10 cycles with metrics |
| Provenance Tracking | HERMES | Signals system |
| Axiom System | HERMES | Validated knowledge accumulation |
| **Speculative Decoding** | **llama.cpp** | **Draft+target model pairs (NOT dLLM)** |
| **MLX Inference** | **Apple MLX** | **Unified memory optimization** |

---

*Version: 0.2.0*
*Layout: Bento (response | input + tabs)*
