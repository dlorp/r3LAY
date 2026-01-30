# OpenClaw Integration

r3LAY can use OpenClaw agents as an LLM backend, enabling remote inference via Claude without requiring local model hosting.

## Architecture

```
┌─────────────┐     file-based      ┌─────────────┐     API      ┌─────────────┐
│   r3LAY     │ ←──────────────────→│  OpenClaw   │ ←───────────→│   Claude    │
│   (TUI)     │   ~/.r3lay/openclaw │   (agent)   │   Anthropic  │   (remote)  │
└─────────────┘                     └─────────────┘              └─────────────┘
```

## How It Works

1. **r3LAY writes queries** to `~/.r3lay/openclaw/pending/<uuid>.json`
2. **OpenClaw agent polls** for new queries (via cron or heartbeat)
3. **Agent processes** the query using its configured model (Claude)
4. **Agent writes response** to `~/.r3lay/openclaw/done/<uuid>.json`
5. **r3LAY polls** for response and displays it

## Setup

### 1. Enable OpenClaw Backend in r3LAY

In your r3LAY config (`~/.r3lay/config.yaml`):

```yaml
llm:
  backend: openclaw
  openclaw:
    poll_interval: 0.5  # seconds between polls
    timeout: 120        # max seconds to wait for response
```

### 2. Configure OpenClaw Agent

Add a cron job to process r3LAY queries:

```yaml
# In OpenClaw config
cron:
  jobs:
    - name: r3lay-processor
      schedule:
        kind: every
        everyMs: 5000  # Check every 5 seconds
      payload:
        kind: agentTurn
        message: "Check ~/.r3lay/openclaw/pending/ for queries. Process any found and write responses to ~/.r3lay/openclaw/done/"
      sessionTarget: isolated
```

Or handle via heartbeat in `HEARTBEAT.md`:

```markdown
## r3LAY Queries
- Check `~/.r3lay/openclaw/pending/` for new queries
- Process using research context
- Write responses to `~/.r3lay/openclaw/done/`
```

## Query Format

Pending queries (`pending/<uuid>.json`):

```json
{
  "request_id": "uuid",
  "timestamp": 1234567890.123,
  "messages": [
    {"role": "user", "content": "What is..."}
  ],
  "system_prompt": "You are a research assistant...",
  "temperature": 0.7,
  "max_tokens": null
}
```

## Response Format

Responses (`done/<uuid>.json`):

```json
{
  "request_id": "uuid",
  "content": "The response text...",
  "done": true,
  "timestamp": 1234567890.456,
  "tokens": 150
}
```

For streaming (optional):

```json
{
  "request_id": "uuid",
  "chunks": ["chunk1", "chunk2", "..."],
  "done": false
}
```

## Helper Script

Use `scripts/openclaw_processor.py` for manual processing:

```bash
# List pending queries
python scripts/openclaw_processor.py list

# Show a specific query
python scripts/openclaw_processor.py show <request_id>

# Respond to a query
python scripts/openclaw_processor.py respond <request_id> "Your response here"
```

## Benefits

- **No local GPU required** - Uses Claude via OpenClaw
- **Agent context** - OpenClaw agent can use its memory, tools, and research capabilities
- **Async-friendly** - Non-blocking file-based communication
- **Debuggable** - Query/response files are human-readable JSON

## Limitations

- **Latency** - 5-15 seconds typical (depends on cron interval + Claude response time)
- **No true streaming** - Streaming is simulated via chunk polling
- **Single agent** - One OpenClaw agent processes all queries (could be extended for multiple)
