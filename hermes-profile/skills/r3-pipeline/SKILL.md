---
name: r3-pipeline
description: Show pipeline run summary — stuck runs, recent failures, and per-pipeline state counts. Surfaces the silent-half-failure class where work completed but bookkeeping was lost.
version: 2.0.0
author: r3LAY
license: MIT
metadata:
  hermes:
    tags: [r3lay, monitoring, pipeline, diagnostics]
    related_skills: [r3-doctor, r3-context]
---

# /r3-pipeline -- pipeline status dashboard

Trigger: user types /r3-pipeline or /pipeline or "any stuck runs?"

## Behavior

1. Call `mcp_r3lay_pipeline_status(lookback_minutes=60)` (one MCP call)
   - If user specifies a timeframe ("last 24h"), adjust lookback_minutes
2. Report three things in order of importance:

### Stuck runs (most critical)
Runs where `work_completed` has no matching `recorded` or `failed` row.
This means real work happened but the record was lost. Always surface
these prominently — they're the class of bug this system was built to catch.

### Recent failures
Failed pipeline runs with error details. Common patterns:
- `IntegrityError: FOREIGN KEY` → project not registered, auto-upsert
  should have caught it — investigate
- `ConnectionError` → Ollama or bridge unreachable during embedding
- `NoWorkCompleted` → pipeline exited without finishing real work

### Summary
Counts by (pipeline, state) — the health heartbeat. Normal looks like
equal started/recorded counts with zero stuck.

## Output format

```
Pipeline status (last 60m):
  Stuck runs: 0   (healthy)
  Failures: 2

  ingest_project:  3 started, 3 recorded, 0 failed
  watcher_ingest: 12 started, 12 recorded, 2 failed
  compile:         1 started,  1 recorded, 0 failed

  Recent failures:
    watcher_ingest  README.md    IntegrityError: FOREIGN KEY constraint failed
    watcher_ingest  notes.md     ConnectionError: Ollama unreachable
```

- If stuck > 0: warn prominently, suggest `/doctor` for full diagnosis
- If everything is zero: "Pipeline clean — no runs in last 60m" (watcher
  may be idle, not broken)
