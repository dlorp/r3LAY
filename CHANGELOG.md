# Changelog

All notable changes to r3LAY.

## [2.0.0] -- 2026-04-08 (hardened through 2026-04-11)

Complete ground-up rebuild. v1 (0.x) was a Textual TUI prototype. v2 is a
fundamentally different architecture: Hermes agent profile, sqlite-vec + FTS5,
structural conflict detection, privacy model, and session management.

This v2 entry accumulated work across multiple sessions (2026-04-08 to
2026-04-11) on the `feat/v2-rebuild-pr` branch. Everything below is part
of the single v2.0.0 PR that replaces the v1 codebase.

### Added — core pipeline + bridge (2026-04-08)
- Three-stage retrieval pipeline: KNN + BM25 + graph -> RRF fusion -> cosine dedup -> MMR rerank
- FastAPI bridge on :8765 with 15 endpoints for agent access
- Structural conflict detection via spaCy NER + decisions table lookup
- Three-level privacy model (false/work/true) enforced at bridge layer
- Per-project session management: sn.md, todos.md, plans.md, open-questions.md
- Hermes agent profile with SOUL.md and 8 skill definitions
- `_ingest/` drop zone for automatic file processing
- Semver version detection and atomic bump across all project files
- CHANGELOG generation in keepachangelog format
- PR workflow helpers: branch naming, 3-agent review dispatch, PR body formatting
- Startup banners with brain/eye ASCII art (phosphor orange)
- tmux-based service launcher (window IS the lifecycle)
- `r3` CLI wrapper for Hermes skill dispatch
- macOS LaunchAgent for boot-time service start
- Pre-commit hook for secret detection

### Added — hardening + tracked paths (2026-04-10)
- Tracked external paths: `tracked_paths` table + bridge endpoints
  (`/tracked` POST/GET/DELETE, `/reindex`, `/git/check`, `/git/pull`)
  as the escape hatch for referencing folders outside the workspace
- Four new skills: track, untrack, reindex, git-status (deprecated and
  pending MCP migration as of 2026-04-11 — see Deferred section)
- `install.sh` bootstrap script with partial-bind profile install
  (symlink SOUL.md + skills/ from repo, real config.yaml and .env as
  user-local files, `__HOME__`/`__REPO_DIR__` placeholder substitution
  via sed at install time)
- Fail-closed bridge auth: auto-generated 32-byte secret at
  `~/.r3lay/api-secret` (0600), constant-time compare via
  `secrets.compare_digest()`
- Path traversal defense: allowed_roots check + symlink guard in
  `_validate_tracked_path()`, symlink skip in `scan_project_files`
- Per-path `asyncio.Lock` + `asyncio.Semaphore(4)` bounds on concurrent
  mutating ops; 409 on duplicate in-flight
- Debounce cache with periodic cleanup (2000 entries, 60s TTL)
- APSW wrapper proper transaction tracking (BEGIN/COMMIT on write ops)
- Schema init cache: `init_schema()` runs once per path per process
- Binary content sniff: NUL-byte detection + UTF-8 decode check, beyond
  extension whitelist
- Decoupled backend config: `r3lay-config.yaml` at repo root, separate
  from Hermes profile config
- `tests/conftest.py` + 30 new tests across config/ingest/bridge (36 total, all green)
- `/usr/local/bin/r3` shell-agnostic launcher via install.sh prompt
  (works in any shell, survives repo moves)

### Added — workspace + Model A (2026-04-11)
- Model A workspace convention: projects live in `~/r3LAY/<domain>/<project>/`
  rather than being referenced via tracked_paths. The r3LAY source repo
  lives at `~/r3LAY/programming/r3LAY/` alongside other active projects
  (intentional dogfooding — agent can edit its own source).
- Workspace-wide `~/r3LAY/_meta/{strategies,learnings,documentation}/`
  layout for cross-domain notes. Not scoped to any single domain.
- `.r3lay/project.yaml` extrapolation convention with `auto_init: true`
  marker + `auto_init_sources` list (auto-creation deferred — see below)
- `terminal` toolset in `platform_toolsets.cli` (was missing — required
  for the shell-wrap skills to work via the terminal tool)
- `docker_env` with `HOME`, `PATH`, `R3LAY_BRIDGE_URL` for the Hermes
  sandbox; `docker_forward_env: [R3LAY_API_KEY]` so the r3 wrapper can
  authenticate against the host bridge
- llm-wiki skill pointed at `~/Documents/knowledge_vault` via
  `skills.config.wiki.path`
- `_meta/learnings/docker-sandbox-home-env.md` — stale container + HOME
  env gotcha writeup
- `_meta/learnings/mcp-vs-skill-md-tool-hallucination.md` — MCP migration
  design + rationale
- `_meta/learnings/hrr-deferral-and-existing-coverage.md` — why HRR isn't
  being built for r3LAY
- `_meta/documentation/auto-r3lay-project-init.md` — design sketch for
  auto-creating `.r3lay/project.yaml` when new projects appear (deferred)

### Added — MCP migration + compile/distill (2026-04-11)
- `r3lay/mcp_server.py` — FastMCP stdio server exposing 11 native tools:
  `track_path`, `untrack_path`, `list_tracked`, `reindex_path`, `git_check`,
  `git_pull`, `search_chunks`, `get_project_context`, `list_active_projects`,
  `init_project`, `compile_project`.
  Runs as a host subprocess spawned by Hermes (not in the Docker sandbox),
  hits `localhost:8765` directly — no `host.docker.internal` indirection.
  Reads `~/.r3lay/api-secret` for auth (same precedence as the r3 CLI).
- `compile_project` tool + `POST /compile` endpoint: Karpathy-style
  deterministic project knowledge synthesis. Assembles metadata, session
  notes, active decisions, todos, open questions, conflicts, and file
  inventory into a single structured markdown document. Optionally persists
  to `.r3lay/compiled.md` for cold-start context loading. No LLM call —
  pure DB + file assembly.
- `mcp>=1.0.0` added to `pyproject.toml` dependencies
- `r3lay-mcp` entry point in `[project.scripts]`
- `mcp_servers.r3lay` block in `hermes-profile/config.template.yaml`
  with `__REPO_DIR__` templating — install.sh sed-substitutes the absolute
  venv python path at install time, so Hermes can spawn the subprocess
  without requiring PATH manipulation
- `tests/test_mcp_server.py` — 20 tests covering every tool's request shape,
  auth header, error paths (401/403/409/502), and FastMCP tool registration.
  Uses `httpx.MockTransport` to stub the bridge without running it.
- Tools are typed (path: str, auto_index: bool, etc.) and their docstrings
  become the MCP descriptions the agent sees in its tool list — the schema
  IS the contract, replacing SKILL.md advisory prose that the LLM was
  pattern-matching against REST-API priors instead of following literally.

### Changed
- Architecture: Hermes agent profile replaces Textual TUI runtime
- Storage: sqlite-vec + FTS5 replaces FAISS in subprocess
- Search: SearXNG removed, Hermes native web search used instead
- Backend factory: string-based `create_backend("ollama", ...)` replaces ModelInfo enum
- Primary chat model: `minimax/minimax-m2.7` (paid, tool-tuned for
  agentic workflows, ~$0.01-0.02/session). Fallback chain:
  `google/gemma-4-31b-it:free` -> `nvidia/nemotron-3-super-120b-a12b:free`
  -> local `ollama qwen3:8b`. Free fallbacks preserved for cost floor
  and offline scenarios.
- Embedding model: `qllama/multilingual-e5-large-instruct` (1024 dim)
  replaces `bge-m3` per knowledge-vault research on CoIR code retrieval
  (instruction-tuned variant, strong on asymmetric code+prose retrieval)
- Maintenance module adapted for unified SQLite database
- Example project updated with privacy field in project.yaml
- Docker volume mounts are **1:1** (host path == container path). Two
  mounts: workspace + knowledge_vault. Enables agent to pass absolute
  host paths to the bridge without translation.
- `terminal.cwd` defaults to `__HOME__/r3LAY/programming` so `pwd` on
  session start anchors the agent where active projects live
- SOUL.md Scope section rewritten for Model A + CRITICAL absolute-path
  rule + legacy `SESSION_NOTES.md` handling + pwd-discovery instruction.
  Uses `<workspace>` / `alice` placeholders so the open-source ship
  doesn't leak usernames.
- `hermes-profile/config.template.yaml` paths use `__HOME__` /
  `__REPO_DIR__` placeholders substituted by `install.sh` at install
  time (Hermes YAML loader has no runtime env substitution)
- Unified DB schema: `files.UNIQUE(project_id, path)` instead of
  `UNIQUE(path)` + `idx_files_project` for defense-in-depth against
  cross-project collisions
- `file_id` namespacing: `sha256(f"{project_id}:{relative_path}")`
  instead of unscoped hash (prevents silent overwrite of same-named
  files across projects via ON CONFLICT DO UPDATE)
- Git endpoint error responses are generic 502 — server-side logging
  keeps stderr (credentials, remote URLs) out of response bodies
- `.gitignore` uses allowlist pattern for `hermes-profile/skills/`:
  ignore everything, whitelist r3LAY's own 11 skills + curated bundled
  skills (github/*, research/llm-wiki, software-development/{plan,
  requesting-code-review,systematic-debugging,test-driven-development,
  writing-plans}, note-taking/obsidian). Future-proof against Hermes
  dropping more bundled skills.

### Fixed
- **CRITICAL: `terminal` tool was missing from `platform_toolsets.cli`**
  — the shell-wrap skills for tracked paths couldn't fire because the
  agent had no shell tool. Added `terminal` with a clarifying comment
  that Docker backend is the security boundary so exposing it is safe.
- Stale Docker containers keeping old env vars after config re-render
  (now documented as a gotcha + `docker rm -f` workflow + reminder to
  kill containers after `docker_env` / `docker_volumes` changes)
- Watcher ERROR spam on Hermes bundled_manifest tmp files: added
  `.tmp`/editor-lock/`.DS_Store`/`.bundled_manifest*` patterns to
  `_should_skip()`, downgraded `FileNotFoundError` to DEBUG log
- `check_same_thread=False` on `sqlite3.connect()` to support FastAPI
  sync generator deps + async endpoint access patterns
- `fish_user_paths` updated to the new repo location after project move
- Embed batch length + dimension validation — raises on mismatch
  instead of silently dropping chunks on partial Ollama responses
- Config rollback: `/tracked` auto-index failure rolls back the
  `tracked_paths` INSERT instead of leaving a half-tracked row
- `shutil.move` for `_processed/` moves (cross-device safe) with
  timestamp prefix; failed renames marked `.ingested` to prevent
  infinite retry loops
- `r3` CLI: proper shell quoting (`"$PROFILE"`), `join_rest()` helper
  for multi-word queries, path resolution via readlink loop

### Removed
- 4 deprecated tracked-path SKILL.md files: `hermes-profile/skills/track/`,
  `untrack/`, `reindex/`, `git-status/`. Replaced by the native MCP tool
  surface — the tool docstrings carry the per-operation contract and
  SOUL.md carries the monitor-vs-act policy. The `r3 track/untrack/reindex/
  git-check/git-pull` shell wrapper subcommands stay in place for cron
  jobs and manual CLI use (the MCP migration is for the AGENT path only,
  not the human path).
- Textual TUI (app.py, all ui/ widgets, styles)
- SearXNG integration
- OpenClaw backend
- FAISS vector store
- rank-bm25 standalone
- Docker support for the r3LAY backend itself (Dockerfile, docker-compose.yaml) —
  NOT to be confused with the Hermes profile's `terminal.backend: docker`
  which sandboxes the agent
- ChromaDB references
- Signals/provenance system (replaced by decisions table)
- Embedding subprocess workers (replaced by Ollama HTTP)
- All v1 tests (replaced by test_db.py + 30 new tests across
  config/ingest/bridge = 36 passing)
- Personal HDLS agent configs `hdls/agent-configs/` (moved to knowledge
  vault at `~/Documents/knowledge_vault/_meta/notebooks/hdls-agent-configs/`
  with DEPRECATED banners — content used hallucinated pre-audit schema
  keys)
- `launchd/r3lay-up.fish` — stale launcher (superseded by `r3lay-up.sh`)
- Hallucinated Hermes config keys (`profile:`, `role:`, `permissions:`,
  `sandbox:`) from the profile template — replaced with real schema
  (`platform_toolsets`, `terminal.backend: docker`, `command_allowlist`,
  `security.website_blocklist`, `approvals.mode`, `browser.allow_private_urls`)

### Deferred (post-v2.0.0 work)
- **HRR primitives** — indefinitely deferred; existing retrieval stack
  (dense + FTS5 + graph + RRF + MMR + decisions table) covers the
  project-brain use case (see
  `_meta/learnings/hrr-deferral-and-existing-coverage.md`)
- **PF firewall Phase B on lorpBot** — staged via scp + interactive
  `stage.sh`, activation deferred until myc3lium Pi is back online so
  the `lan_dev` allowlist table can be populated with the Pi's current IP
