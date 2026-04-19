---
name: r3-doctor
description: Run comprehensive health checks on r3LAY infrastructure — DB integrity, watcher, embedding endpoints, pipeline status. One MCP call, formatted output.
version: 2.0.0
author: r3LAY
license: MIT
metadata:
  hermes:
    tags: [r3lay, health, monitoring, diagnostics]
    related_skills: [r3-context, r3-pipeline]
---

# /r3-doctor -- system health check

Trigger: user types /r3-doctor or /doctor or "is r3LAY healthy?"

## Behavior

1. Call `mcp_r3lay_doctor(include_network=True)` (one MCP call)
   - If user says "offline" or Ollama is known-down, use `include_network=False`
2. Format the response as a compact health grid:

```
[OK] Overall: OK
  [  ok  ] db_integrity             PRAGMA integrity_check passed
  [  ok  ] db_schema                all 11 tables present
  [  ok  ] watcher_heartbeat        heartbeat 45s ago
  [ warn ] pipeline_failures        3 failures in last 60m
  ...
```

3. For any `fail` or `warn` checks:
   - Surface the `message` and `details` fields
   - Suggest concrete remediation:
     - `watcher_heartbeat` fail → "Run `r3 up` or check the tmux session"
     - `embedding_endpoint` fail → "Start Ollama: `ollama serve`"
     - `embedding_model` fail → "Pull the model: `ollama pull <model>`"
     - `pipeline_stuck` fail → "Check `r3 pipeline-status` for stuck runs"
     - `db_integrity` fail → "DB may be corrupted. Run `r3 rebuild --verify`"

4. If overall is `ok`, report briefly and move on. Don't over-explain health.

## Output

- One-screen summary, worst-first ordering
- No verbose explanations when healthy
- Clear remediation steps when unhealthy
