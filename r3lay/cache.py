"""Cache-bypass plumbing for `--fresh` / `--no-cache` flows.

In the adjacent Hermes upgrade, stale ``__pycache__`` bytecode silently kept
an old function signature in effect after a source edit — the new code never
actually ran until the cache was cleared. r3LAY has several layers of caching
(schema-init, config dict, ingest hash-match skip), and the same shape of
bug is possible: a config change doesn't take effect because the cached
config dict is still in memory; an embedding-model change doesn't take
effect because every file's content_hash still matches.

This module exposes one honest source of truth: ``cache_bypassed()``. It
reads ``R3LAY_NO_CACHE`` and an in-process override flag. Any cache that
matters should consult it and skip the cached path when it returns True.

Usage::

    from .cache import cache_bypassed, set_bypass

    if not cache_bypassed() and already_known:
        return cached_value

    # otherwise do the real work

CLI callers flip the override for the duration of one invocation via
``set_bypass(True)``; they don't need to re-export the env var. Bridge
endpoints read the ``fresh`` query parameter and call ``set_bypass()``
around the request (see ``/ingest?fresh=true``).
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

_override: bool = False


def _env_truthy(val: str | None) -> bool:
    if not val:
        return False
    return val.strip().lower() in {"1", "true", "yes", "on"}


def cache_bypassed() -> bool:
    """Return True iff caches should be skipped for the current operation.

    Checks (in order):
      1. in-process override set by ``set_bypass()``
      2. ``R3LAY_NO_CACHE`` env var (truthy)
    """
    if _override:
        return True
    return _env_truthy(os.environ.get("R3LAY_NO_CACHE"))


def set_bypass(value: bool) -> None:
    """Set the in-process override flag.

    Callers that wrap a single request should prefer ``bypass_scope()`` so
    the flag is guaranteed to be cleared afterward.
    """
    global _override
    _override = bool(value)


@contextmanager
def bypass_scope(bypass: bool = True) -> Iterator[None]:
    """Flip the override for the duration of a ``with`` block."""
    global _override
    prior = _override
    _override = bool(bypass) or prior
    try:
        yield
    finally:
        _override = prior
