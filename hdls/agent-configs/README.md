# HDLS Agent Configs

Reference copies of active HDLS agent configurations.
These are NOT the live configs (those live in `~/.hermes/profiles/`).
These are version-controlled snapshots for reference and recovery.

Last synced: 2026-04-07

## Agents
- [hyph4-reference.yaml](./hyph4-reference.yaml) -- coordinator, DeepSeek V3.2, hybrid thinking

## Recovery
If `~/.hermes/profiles/` is lost, use these as the restore baseline.
Run: `hermes profile import {config_file}`

## Security
These files contain **zero secrets**. All API keys, tokens, and passwords
are stored in the live configs only (`~/.hermes/profiles/`) and referenced
via environment variables (`$OPENROUTER_API_KEY`, `$R3LAY_BRIDGE_SECRET`, etc.).
