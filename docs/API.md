# API Reference

## Commands

### Maintenance

| Command | Description | Example |
|---------|-------------|---------|
| `/log <service> [details]` | Log maintenance entry | `/log oil_change with Mobil 1` |
| `/due` | Show upcoming/overdue services | `/due` |
| `/history [service]` | Show maintenance history | `/history brakes` |
| `/mileage <value>` | Update current odometer | `/mileage 98500` |

### Search & Research

| Command | Description | Example |
|---------|-------------|---------|
| `/index <path>` | Index files for RAG | `/index ./docs` |
| `/reindex` | Rebuild entire index | `/reindex` |
| `/search <query>` | Search indexed documents | `/search timing belt` |
| `/research <query>` | Start deep research expedition | `/research head gasket failure` |

### Knowledge Management

| Command | Description | Example |
|---------|-------------|---------|
| `/axiom [cat:] <stmt>` | Create new axiom | `/axiom torque: 72 ft-lbs` |
| `/axioms [category]` | List axioms | `/axioms torque` |
| `/axioms --disputed` | Show disputed axioms | `/axioms --disputed` |
| `/cite <id>` | Show provenance chain | `/cite ax-001` |
| `/dispute <id> <reason>` | Mark axiom as disputed | `/dispute ax-001 outdated` |

### Session

| Command | Description |
|---------|-------------|
| `/clear` | Start new session |
| `/help` | Show available commands |

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `R3LAY_OLLAMA_ENDPOINT` | Ollama API endpoint | `http://localhost:11434` |
| `R3LAY_SEARXNG_ENDPOINT` | SearXNG API endpoint | `http://localhost:8080` |
| `R3LAY_HF_CACHE_PATH` | HuggingFace model cache | Auto-detected |
| `R3LAY_MLX_FOLDER` | MLX models directory | Auto-detected |
| `R3LAY_GGUF_FOLDER` | GGUF models directory | `~/.r3lay/models` |

### Project Configuration

Location: `<project>/.r3lay/config.yaml`

```yaml
model_roles:
  text_model: Qwen2.5-7B-Instruct-mlx-4bit
  vision_model: Qwen2.5-VL-7B-Instruct-mlx-4bit
  text_embedder: mlx-community/all-MiniLM-L6-v2-4bit
  vision_embedder: openai/clip-vit-base-patch32
```

### Service Intervals

Location: `<project>/.r3lay/maintenance/intervals.yaml`

Default intervals include:
- Oil change
- Timing belt
- Brake pads/rotors
- Coolant flush
- Transmission fluid
- Spark plugs
- Air filter
- Cabin filter
- And more...

## Recommended Models

| Role | Model | Size | Notes |
|------|-------|------|-------|
| Text | Qwen2.5-7B-Instruct | ~4GB (4-bit) | Excellent instruction following |
| Vision | Qwen2-VL-7B-Instruct | ~5GB (4-bit) | Strong visual understanding |
| Embeddings | all-MiniLM-L6-v2 | ~80MB | Fast, good quality |
