# OpenClaw Integration

r3LAY can use OpenClaw as an LLM backend via its OpenAI-compatible HTTP API, enabling access to various LLM providers (Anthropic, OpenAI, local models) through a unified gateway.

## Architecture

```
┌─────────────┐      HTTP API       ┌─────────────┐     Provider     ┌─────────────┐
│   r3LAY     │ ←─────────────────→ │  OpenClaw   │ ←──────────────→ │  Anthropic  │
│   (TUI)     │  localhost:18789    │  Gateway    │    API calls     │   OpenAI    │
│             │  /v1/chat/completions│             │                  │   Ollama    │
└─────────────┘                     └─────────────┘                  └─────────────┘
```

## How It Works

1. **r3LAY connects** to OpenClaw's HTTP API (`http://localhost:18789`)
2. **Sends requests** to `/v1/chat/completions` (OpenAI-compatible format)
3. **OpenClaw Gateway** routes the request to the configured LLM provider
4. **Streams response** back via Server-Sent Events (SSE)
5. **r3LAY displays** tokens in real-time

## Setup

### 1. Start OpenClaw Gateway

```bash
# Start the gateway daemon
openclaw gateway start

# Check status
openclaw gateway status
```

The gateway listens on `http://localhost:18789` by default.

### 2. Configure r3LAY Backend

In your r3LAY project config (`.r3lay/config.yaml`):

```yaml
model_roles:
  text_model: openclaw/anthropic/claude-sonnet-4-20250514
  vision_model: openclaw/anthropic/claude-sonnet-4-20250514
```

Or use environment variables:

```bash
export R3LAY_TEXT_MODEL="openclaw/anthropic/claude-sonnet-4-20250514"
export R3LAY_VISION_MODEL="openclaw/anthropic/claude-sonnet-4-20250514"
```

### 3. Model Naming Convention

Format: `openclaw/<provider>/<model-name>`

Examples:
- `openclaw/anthropic/claude-sonnet-4-20250514`
- `openclaw/openai/gpt-4`
- `openclaw/ollama/qwen2.5:7b`

The `openclaw/` prefix tells r3LAY to use the OpenClaw backend. The rest is passed to the gateway as the model name.

## API Details

### Backend Implementation

r3LAY's OpenClaw backend is an HTTP client (`r3lay/core/backends/openclaw.py`) that:

- Connects to `http://localhost:18789` (configurable via `endpoint` parameter)
- Uses the OpenAI-compatible `/v1/chat/completions` endpoint
- Streams responses via Server-Sent Events (SSE)
- Supports vision models with base64-encoded images
- Handles authentication via optional Bearer tokens

### Request Format

POST to `/v1/chat/completions`:

```json
{
  "model": "anthropic/claude-sonnet-4-20250514",
  "messages": [
    {"role": "user", "content": "What is..."}
  ],
  "stream": true,
  "max_tokens": 512,
  "temperature": 0.7
}
```

### Vision Model Support

For vision models, the last user message is converted to multimodal format:

```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "What's in this image?"},
    {
      "type": "image_url",
      "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ..."}
    }
  ]
}
```

### Response Format (SSE)

OpenClaw streams responses in OpenAI's SSE format:

```
data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"choices":[{"delta":{"content":" world"},"finish_reason":null}]}

data: {"choices":[{"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

## Configuration Options

### Custom Endpoint

```python
from r3lay.core.backends.openclaw import OpenClawBackend

backend = OpenClawBackend(
    model_name="anthropic/claude-sonnet-4-20250514",
    endpoint="http://custom-host:8080",  # Non-default endpoint
    api_key="optional-bearer-token"       # Optional authentication
)
```

### Environment Variables

You can configure the OpenClaw endpoint via environment:

```bash
export OPENCLAW_ENDPOINT="http://localhost:18789"
export OPENCLAW_API_KEY="your-token"  # Optional
```

## Benefits

- **Unified Interface** - One backend for multiple LLM providers
- **True Streaming** - Real-time token streaming via SSE
- **Vision Support** - Works with multimodal models
- **Local & Remote** - Mix local (Ollama) and remote (Anthropic/OpenAI) models
- **No Polling** - Direct HTTP requests, low latency
- **Provider Abstraction** - Switch providers without changing r3LAY config

## Troubleshooting

### Connection Refused

```
Cannot connect to OpenClaw at http://localhost:18789.
Is OpenClaw running? Try: openclaw gateway start
```

**Solution:** Start the OpenClaw gateway:
```bash
openclaw gateway start
openclaw gateway status
```

### Model Not Found

If OpenClaw can't route your model, check:
- Model name format: `<provider>/<model-id>`
- Provider credentials are configured in OpenClaw
- Model is available from the provider

### Authentication Errors

If you've configured authentication on the gateway, pass the API key:

```python
backend = OpenClawBackend(
    model_name="...",
    api_key="your-bearer-token"
)
```

## Advanced Usage

### Checking Gateway Status

```python
from r3lay.core.backends.openclaw import OpenClawBackend

# Check if gateway is running
is_running = await OpenClawBackend.is_available("http://localhost:18789")
```

### Manual Request

```bash
# Test the gateway directly
curl -X POST http://localhost:18789/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic/claude-sonnet-4-20250514",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true,
    "max_tokens": 100
  }'
```

## Comparison: File-Based vs HTTP

| Aspect | HTTP (Current) | File-Based (Old) |
|--------|---------------|------------------|
| Transport | HTTP API | JSON files |
| Latency | ~100-500ms | 5-15 seconds |
| Streaming | True SSE streaming | Simulated via polling |
| Reliability | Direct connection | Requires file watchers |
| Debugging | HTTP logs | File inspection |
| Status | ✅ Active | ❌ Deprecated |

The file-based approach described in earlier documentation was never fully implemented. The HTTP backend is the only production implementation.
