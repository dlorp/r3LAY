-- r3LAY v2 unified SQLite schema
-- This is the INDEX. The filesystem is truth. If the DB is lost, rebuild from files.

PRAGMA journal_mode=WAL;
PRAGMA mmap_size=268435456;
PRAGMA page_size=8192;

-- Projects registry
CREATE TABLE IF NOT EXISTS projects (
  id           TEXT PRIMARY KEY,
  name         TEXT NOT NULL,
  path         TEXT NOT NULL,
  type         TEXT DEFAULT 'other',
  description  TEXT,
  status       TEXT DEFAULT 'active',
  privacy      TEXT DEFAULT 'false',
  created_at   TEXT DEFAULT (datetime('now')),
  updated_at   TEXT DEFAULT (datetime('now'))
);

-- Files
-- NOTE: `path` is relative to the project root, so two projects can legitimately
-- both have a README.md. The uniqueness constraint is scoped per project.
-- `id` is sha256("{project_id}:{path}") to keep it collision-free.
CREATE TABLE IF NOT EXISTS files (
  id             TEXT PRIMARY KEY,
  project_id     TEXT REFERENCES projects(id),
  path           TEXT NOT NULL,
  title          TEXT,
  content        TEXT,
  content_hash   TEXT,
  file_type      TEXT,
  provenance     TEXT DEFAULT 'human',
  quality_weight REAL DEFAULT 1.0,
  created_at     TEXT DEFAULT (datetime('now')),
  updated_at     TEXT DEFAULT (datetime('now')),
  UNIQUE(project_id, path)
);
CREATE INDEX IF NOT EXISTS idx_files_project ON files(project_id);

-- Chunks for retrieval
CREATE TABLE IF NOT EXISTS chunks (
  chunk_id      TEXT PRIMARY KEY,
  file_id       TEXT REFERENCES files(id),
  project_id    TEXT,
  content       TEXT NOT NULL,
  chunk_index   INTEGER,
  granularity   TEXT DEFAULT 'paragraph'
);

-- Vector search (NOT synced via cr-sqlite — recomputed locally)
CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
  chunk_id  TEXT PRIMARY KEY,
  embedding float[1024]
);

-- Full-text search (NOT synced — rebuilt locally)
CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
  content,
  content='chunks',
  tokenize='porter ascii'
);

-- Triggers to keep FTS in sync with chunks table
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO fts_chunks(rowid, content) VALUES (new.rowid, new.content);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
  INSERT INTO fts_chunks(fts_chunks, rowid, content) VALUES ('delete', old.rowid, old.content);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
  INSERT INTO fts_chunks(fts_chunks, rowid, content) VALUES ('delete', old.rowid, old.content);
  INSERT INTO fts_chunks(rowid, content) VALUES (new.rowid, new.content);
END;

-- Decisions / axioms (append-only, grow-only set)
CREATE TABLE IF NOT EXISTS decisions (
  id            TEXT PRIMARY KEY,
  project_id    TEXT REFERENCES projects(id),
  statement     TEXT NOT NULL,
  rationale     TEXT,
  category      TEXT,
  entities      TEXT,
  decided_at    TEXT DEFAULT (datetime('now')),
  decided_by    TEXT DEFAULT 'human',
  superseded_by TEXT,
  confidence    REAL DEFAULT 1.0,
  source        TEXT
);

-- Conflicts log
CREATE TABLE IF NOT EXISTS conflicts (
  id                 TEXT PRIMARY KEY,
  project_id         TEXT,
  proposed_change    TEXT,
  conflict_type      TEXT,
  conflict_source    TEXT,
  description        TEXT,
  severity           TEXT,
  detected_at        TEXT DEFAULT (datetime('now')),
  resolution         TEXT,
  resolved_at        TEXT
);

-- Graph edges
CREATE TABLE IF NOT EXISTS edges (
  source_id  TEXT,
  target_id  TEXT,
  edge_type  TEXT,
  weight     REAL DEFAULT 1.0,
  PRIMARY KEY (source_id, target_id, edge_type)
);
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);

-- Sessions
CREATE TABLE IF NOT EXISTS sessions (
  id         TEXT PRIMARY KEY,
  project_id TEXT,
  started_at TEXT,
  ended_at   TEXT,
  summary    TEXT,
  model_used TEXT
);

-- Maintenance log (lifted from existing r3LAY)
CREATE TABLE IF NOT EXISTS maintenance_log (
  id          TEXT PRIMARY KEY,
  project_id  TEXT REFERENCES projects(id),
  service     TEXT NOT NULL,
  mileage     INTEGER,
  date        TEXT,
  cost        REAL,
  notes       TEXT,
  logged_by   TEXT DEFAULT 'human'
);

-- Tracked external paths (folders outside ~/r3LAY/ the agent has been told to track)
-- The watcher does NOT auto-watch these — re-indexing is agent-driven with human approval.
CREATE TABLE IF NOT EXISTS tracked_paths (
  id                  TEXT PRIMARY KEY,
  path                TEXT NOT NULL UNIQUE,
  project_id          TEXT REFERENCES projects(id),
  is_git_repo         INTEGER DEFAULT 0,
  git_remote          TEXT,
  git_local_head      TEXT,
  git_remote_head     TEXT,
  last_indexed_at     TEXT,
  last_git_check_at   TEXT,
  created_at          TEXT DEFAULT (datetime('now')),
  notes               TEXT
);
CREATE INDEX IF NOT EXISTS idx_tracked_paths_project ON tracked_paths(project_id);
