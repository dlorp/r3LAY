"""SQLite connection, extension loading, and schema initialization.

Manages the unified r3LAY database at ~/r3LAY/.r3lay-global/r3lay.db.
Loads sqlite-vec for vector search and optionally cr-sqlite for CRDT sync.
The database is always rebuildable from the filesystem.

Extension Loading Strategy:
  Python's stdlib sqlite3 on macOS is often built WITHOUT SQLITE_ENABLE_LOAD_EXTENSION.
  We try stdlib first; if that fails, fall back to apsw (which always supports extensions).
  apsw connections are wrapped to behave like sqlite3.Connection for downstream code.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

import sqlite_vec

logger = logging.getLogger(__name__)

# Default database path
DEFAULT_R3LAY_ROOT = Path.home() / "r3LAY"
DEFAULT_DB_PATH = DEFAULT_R3LAY_ROOT / ".r3lay-global" / "r3lay.db"
SCHEMA_PATH = Path(__file__).parent.parent / "schema" / "schema.sql"


def _try_stdlib_connection(db_path: Path) -> sqlite3.Connection | None:
    """Try to create a stdlib sqlite3 connection with extension loading.

    Returns None if enable_load_extension is not available.

    check_same_thread=False is required because FastAPI may run sync
    dependencies (get_conn) in a thread pool worker, then hand the connection
    off to an async endpoint running in the main event loop thread. SQLite
    is thread-safe at the C level for our usage pattern (one connection per
    request, no concurrent use), so disabling the Python check is safe.
    """
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    try:
        conn.enable_load_extension(True)
        return conn
    except AttributeError:
        conn.close()
        return None


def _create_apsw_connection(db_path: Path) -> Any:
    """Create an apsw connection with extension loading support.

    apsw always supports extensions regardless of how Python was compiled.
    We wrap it in an adapter that provides execute/fetchone/fetchall/commit/close.
    """
    import apsw
    import apsw.ext

    conn = apsw.Connection(str(db_path))
    # Enable WAL and other pragmas
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA mmap_size=0")
    conn.execute("PRAGMA foreign_keys=ON")

    # Load sqlite-vec
    conn.enable_load_extension(True)
    conn.load_extension(sqlite_vec.loadable_path())
    conn.enable_load_extension(False)

    return _ApswWrapper(conn)


class _ApswWrapper:
    """Minimal adapter wrapping apsw.Connection to match sqlite3.Connection API.

    Provides: execute, commit, close, row_factory-like behavior.

    Transaction handling:
        APSW runs in SQLite autocommit mode outside of explicit transactions,
        which means one fsync per statement — 50-100x slower than stdlib sqlite3
        for bulk writes. To match stdlib semantics (all writes between commits
        batched into one transaction) we track an implicit transaction state:

        - On first execute() after commit() or open, issue BEGIN.
        - On commit(), issue COMMIT if a transaction is active.
        - close() commits any pending transaction before closing.

        This gives sqlite3.Connection-compatible performance for bulk ingest.
    """

    def __init__(self, conn: Any) -> None:
        self._conn = conn
        self._in_txn = False

    def _ensure_txn(self) -> None:
        """Begin an implicit transaction if not already in one."""
        if not self._in_txn:
            self._conn.execute("BEGIN")
            self._in_txn = True

    def execute(self, sql: str, params: tuple | list = ()) -> _ApswCursorWrapper:
        # DDL statements and pragmas shouldn't auto-open a transaction
        stripped = sql.lstrip().upper()
        ddl_prefixes = ("CREATE", "DROP", "ALTER", "PRAGMA", "BEGIN", "COMMIT", "ROLLBACK")
        is_ddl = stripped.startswith(ddl_prefixes)
        if not is_ddl:
            self._ensure_txn()

        cursor = self._conn.cursor()
        cursor.execute(sql, params)
        # get_description() fails for DDL statements (CREATE, DROP, etc.)
        try:
            desc = cursor.get_description()
        except Exception:
            desc = None
        return _ApswCursorWrapper(cursor, desc)

    def executescript(self, sql: str) -> None:
        """Execute multiple SQL statements at once (like sqlite3.executescript).

        Commits any pending transaction first — executescript runs its own
        top-level statements and can't be nested inside our implicit BEGIN.
        """
        if self._in_txn:
            self._conn.execute("COMMIT")
            self._in_txn = False
        self._conn.execute(sql)

    def commit(self) -> None:
        """Commit the current implicit transaction, if any."""
        if self._in_txn:
            self._conn.execute("COMMIT")
            self._in_txn = False

    def rollback(self) -> None:
        """Rollback the current implicit transaction, if any."""
        if self._in_txn:
            self._conn.execute("ROLLBACK")
            self._in_txn = False

    def close(self) -> None:
        # Commit any pending transaction before closing to avoid losing writes
        if self._in_txn:
            try:
                self._conn.execute("COMMIT")
            except Exception:
                pass
            self._in_txn = False
        self._conn.close()

    @property
    def row_factory(self) -> Any:
        return None

    @row_factory.setter
    def row_factory(self, value: Any) -> None:
        pass  # We handle row wrapping in the cursor wrapper


class _ApswCursorWrapper:
    """Wraps apsw cursor results to provide sqlite3.Row-like dict access."""

    def __init__(self, cursor: Any, description: Any) -> None:
        self._cursor = cursor
        self._columns = [d[0] for d in description] if description else []

    def fetchone(self) -> _DictRow | None:
        try:
            row = next(self._cursor)
            if row is None:
                return None
            return _DictRow(self._columns, row)
        except StopIteration:
            return None

    def fetchall(self) -> list[_DictRow]:
        rows = list(self._cursor)
        return [_DictRow(self._columns, row) for row in rows]

    def __iter__(self):
        for row in self._cursor:
            yield _DictRow(self._columns, row)


class _DictRow:
    """Row object that supports both index and key access (like sqlite3.Row)."""

    def __init__(self, columns: list[str], values: tuple) -> None:
        self._columns = columns
        self._values = values

    def __getitem__(self, key: int | str) -> Any:
        if isinstance(key, int):
            return self._values[key]
        if isinstance(key, str):
            try:
                idx = self._columns.index(key)
                return self._values[idx]
            except ValueError:
                raise KeyError(key) from None
        raise TypeError(f"Invalid key type: {type(key)}")

    def keys(self) -> list[str]:
        return self._columns

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


def get_connection(
    db_path: Path | None = None,
    load_crsqlite: bool = False,
) -> Any:
    """Create a new SQLite connection with extensions loaded.

    Tries stdlib sqlite3 first; falls back to apsw if extension loading unavailable.

    Args:
        db_path: Path to the database file. Defaults to ~/r3LAY/.r3lay-global/r3lay.db.
        load_crsqlite: Whether to load cr-sqlite extension for CRDT sync.

    Returns:
        Connection object (sqlite3.Connection or apsw wrapper).
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Try stdlib first
    conn = _try_stdlib_connection(db_path)
    if conn is not None:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA mmap_size=0")
        conn.execute("PRAGMA foreign_keys=ON")

        sqlite_vec.load(conn)
        logger.debug("Loaded sqlite-vec via stdlib sqlite3")

        if load_crsqlite:
            try:
                conn.load_extension("crsqlite")
                logger.debug("Loaded cr-sqlite extension")
            except sqlite3.OperationalError:
                logger.warning("cr-sqlite not available")

        conn.enable_load_extension(False)
        return conn

    # Fall back to apsw
    logger.debug("stdlib sqlite3 lacks extension loading, using apsw")
    return _create_apsw_connection(db_path)


def _prepare_schema_sql(raw_sql: str) -> str:
    """Prepare schema SQL by stripping PRAGMAs and cr-sqlite calls.

    Keeps everything else intact so multi-statement execution works
    with both sqlite3 and apsw.
    """
    lines = []
    for line in raw_sql.split("\n"):
        stripped = line.strip().upper()
        # Remove PRAGMA lines (already set in get_connection)
        if stripped.startswith("PRAGMA"):
            continue
        # Remove cr-sqlite CRDT registration calls (Phase 4)
        if "CRSQL_AS_CRR" in stripped:
            continue
        lines.append(line)
    return "\n".join(lines)


def init_schema(conn: Any) -> None:
    """Initialize the database schema from schema.sql.

    Safe to call multiple times -- all CREATE statements use IF NOT EXISTS.
    Uses executescript for multi-statement execution (handles triggers with
    semicolons inside BEGIN...END blocks).

    Args:
        conn: Active connection with extensions loaded.
    """
    raw_sql = SCHEMA_PATH.read_text()
    schema_sql = _prepare_schema_sql(raw_sql)

    # Use the appropriate multi-statement execution method
    if isinstance(conn, _ApswWrapper):
        # apsw handles multi-statement SQL natively
        conn.executescript(schema_sql)
    else:
        # stdlib sqlite3
        conn.executescript(schema_sql)

    logger.info("Database schema initialized")


# Track which DB paths have already been schema-initialized so we don't
# re-run the entire schema.sql on every get_db() call. This was a
# measurable cost under the watcher's per-event connection pattern.
_SCHEMA_INIT_PATHS: set[Path] = set()


def get_db(db_path: Path | None = None) -> Any:
    """Get a fully initialized database connection.

    Schema initialization is cached per-process per-path: init_schema() runs
    exactly once per unique db_path for the lifetime of the process.

    Args:
        db_path: Path to the database file. Defaults to ~/r3LAY/.r3lay-global/r3lay.db.

    Returns:
        Ready-to-use connection.
    """
    conn = get_connection(db_path)
    key = (db_path or DEFAULT_DB_PATH).resolve()
    if key not in _SCHEMA_INIT_PATHS:
        init_schema(conn)
        _SCHEMA_INIT_PATHS.add(key)
    return conn
