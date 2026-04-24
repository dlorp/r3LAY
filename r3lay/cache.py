"""Cache-bypass plumbing for `--fresh` / `--no-cache` flows.

In the adjacent Hermes upgrade, stale ``__pycache__`` bytecode silently kept
an old function signature in effect after a source edit — the new code never
actually ran until the cache was cleared. r3LAY has several layers of caching
(schema-init, config dict, ingest hash-match skip), and the same shape of
bug is possible: a config change doesn't take effect because the cached
config dict is still in memory; an embedding-model change doesn't take
effect because every file's content_hash still matches.

This module exposes one honest source of truth: ``cache_bypassed()``. It
reads ``R3LAY_NO_CACHE`` and a per-context override flag. Any cache that
matters should consult it and skip the cached path when it returns True.

Uses ``contextvars`` so concurrent FastAPI requests don't leak bypass
state across request boundaries. A module-global flag caused a race:
request A's ``bypass_scope(True)`` affected request B's cache decisions.

Usage::

    from .cache import cache_bypassed, bypass_scope

    if not cache_bypassed() and already_known:
        return cached_value

    # otherwise do the real work

CLI callers flip the override for the duration of one invocation via
``set_bypass(True)``; they don't need to re-export the env var. Bridge
endpoints read the ``fresh`` query parameter and call ``bypass_scope()``
around the request (see ``/ingest?fresh=true``).
"""

from __future__ import annotations

import contextvars
import os
from contextlib import contextmanager
from typing import Iterator

_override: contextvars.ContextVar[bool] = contextvars.ContextVar("cache_bypass", default=False)


def _env_truthy(val: str | None) -> bool:
    if not val:
        return False
    return val.strip().lower() in {"1", "true", "yes", "on"}


def cache_bypassed() -> bool:
    """Return True iff caches should be skipped for the current operation.

    Checks (in order):
      1. per-context override set by ``set_bypass()`` or ``bypass_scope()``
      2. ``R3LAY_NO_CACHE`` env var (truthy)
    """
    if _override.get():
        return True
    return _env_truthy(os.environ.get("R3LAY_NO_CACHE"))


def set_bypass(value: bool) -> None:
    """Set the per-context override flag.

    Callers that wrap a single request should prefer ``bypass_scope()`` so
    the flag is guaranteed to be cleared afterward.
    """
    _override.set(bool(value))


@contextmanager
def bypass_scope(bypass: bool = True) -> Iterator[None]:
    """Flip the override for the duration of a ``with`` block.

    Uses ``contextvars.Token`` for proper reset — safe under concurrent
    async requests (each coroutine gets its own context copy).
    """
    token = _override.set(bool(bypass))
    try:
        yield
    finally:
        _override.reset(token)
