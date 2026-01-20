# Docker Deployment Guide

This guide covers deploying r3LAY using Docker, including setup for different backends and platforms.

## Quick Start

### With SearXNG (Recommended)

The default profile includes SearXNG for web search functionality:

```bash
# Start services
docker compose --profile default up -d

# Run r3LAY pointing to your project directory
PROJECT_PATH=/path/to/your/project docker compose run r3lay
```

### Standalone (No Web Search)

For deployments without web search:

```bash
docker compose --profile standalone run r3lay-standalone
```

### NVIDIA GPU Support

For systems with NVIDIA GPUs:

```bash
# Build the NVIDIA image
docker compose --profile nvidia build

# Run with GPU access
docker compose --profile nvidia run r3lay-nvidia
```

## Configuration

### Environment Variables

r3LAY reads configuration from environment variables with the `R3LAY_` prefix:

| Variable | Description | Default |
|----------|-------------|---------|
| `PROJECT_PATH` | Host directory to mount as /project | Current directory |
| `HF_CACHE_PATH` | HuggingFace model cache | `~/.cache/huggingface` |
| `GGUF_PATH` | GGUF models directory | `~/.r3lay/models` |
| `R3LAY_OLLAMA_ENDPOINT` | Ollama API endpoint | `http://host.docker.internal:11434` |
| `R3LAY_SEARXNG_ENDPOINT` | SearXNG endpoint | `http://searxng:8080` |
| `SEARXNG_SECRET` | SearXNG secret key | `r3lay-secret-key` |

### Example: Custom Ollama Host

```bash
R3LAY_OLLAMA_ENDPOINT=http://my-ollama-server:11434 \
PROJECT_PATH=/my/project \
docker compose --profile standalone run r3lay-standalone
```

## Connecting to Host Services

### Ollama on Host Machine

The container uses `host.docker.internal` to reach services running on the host.

**macOS/Windows (Docker Desktop)**: Works automatically.

**Linux**: The `extra_hosts` configuration handles this in most cases. If it doesn't work:

```bash
# Find your host IP
HOST_IP=$(ip route show default | awk '/default/ {print $3}')

# Run with explicit host mapping
docker run --add-host=host.docker.internal:$HOST_IP \
  -e R3LAY_OLLAMA_ENDPOINT=http://host.docker.internal:11434 \
  -v /path/to/project:/project \
  -it r3lay
```

### Verify Ollama Connection

Before running r3LAY, ensure Ollama is accessible:

```bash
# On host
curl http://localhost:11434/api/tags

# From inside container
docker compose run --rm r3lay curl http://host.docker.internal:11434/api/tags
```

## Building Images

### Standard Build (CPU/Ollama)

```bash
docker compose build r3lay
```

### NVIDIA GPU Build

```bash
docker compose --profile nvidia build r3lay-nvidia
```

This compiles llama-cpp-python with CUDA support, which takes some time.

### Build Arguments

For custom builds:

```bash
# Build with specific Python version
docker build --build-arg PYTHON_VERSION=3.11 -t r3lay .
```

## Volume Mounts

### Project Directory

Your project directory is mounted at `/project`:

```bash
PROJECT_PATH=/home/user/my-garage-docs docker compose run r3lay
```

### HuggingFace Cache

Mount your HF cache to avoid re-downloading models:

```bash
HF_CACHE_PATH=/path/to/huggingface/cache docker compose run r3lay
```

### GGUF Models (NVIDIA)

For the NVIDIA image, mount your GGUF models:

```bash
GGUF_PATH=/path/to/gguf/models docker compose --profile nvidia run r3lay-nvidia
```

## SearXNG Configuration

SearXNG provides web search for the `/research` command.

### Start SearXNG Only

```bash
docker compose --profile search up -d searxng
```

### Verify SearXNG

```bash
curl http://localhost:8080/healthz
```

### Custom SearXNG Settings

The SearXNG data is persisted in a Docker volume. To customize:

```bash
# Access the volume
docker compose run --rm searxng cat /etc/searxng/settings.yml

# Or mount a custom settings file
# Add to docker-compose.yaml under searxng.volumes:
#   - ./searxng-settings.yml:/etc/searxng/settings.yml:ro
```

## Troubleshooting

### Cannot Connect to Ollama

1. Ensure Ollama is running: `ollama serve`
2. Check if port 11434 is accessible
3. On Linux, verify `host.docker.internal` resolves:
   ```bash
   docker compose run --rm r3lay getent hosts host.docker.internal
   ```

### SearXNG Not Responding

1. Check health: `docker compose ps` - should show "healthy"
2. View logs: `docker compose logs searxng`
3. Test endpoint: `curl http://localhost:8080/search?q=test&format=json`

### Out of Memory (NVIDIA)

1. Check available VRAM: `nvidia-smi`
2. Use smaller quantized models (Q4_K_M recommended)
3. Close other GPU applications

### TUI Rendering Issues

1. Ensure terminal supports 256 colors
2. Set `TERM=xterm-256color` before running
3. Try with `-e TERM=xterm-256color` in docker run

## Production Deployment

For production deployments:

1. **Use specific image tags** instead of `latest`
2. **Set a strong SEARXNG_SECRET**
3. **Use bind mounts** for persistent data instead of named volumes
4. **Configure resource limits** in docker-compose.yaml

Example production compose override:

```yaml
# docker-compose.prod.yaml
services:
  r3lay:
    deploy:
      resources:
        limits:
          memory: 8G
  searxng:
    environment:
      - SEARXNG_SECRET=${SEARXNG_SECRET}  # Set via .env file
```

Run with:
```bash
docker compose -f docker-compose.yaml -f docker-compose.prod.yaml up -d
```
