"""Tests for r3lay.cache bypass plumbing."""

from __future__ import annotations

import os

import pytest

from r3lay.cache import bypass_scope, cache_bypassed, set_bypass


@pytest.fixture(autouse=True)
def _reset_cache_state():
    """Clear env var and override flag between tests."""
    os.environ.pop("R3LAY_NO_CACHE", None)
    set_bypass(False)
    yield
    os.environ.pop("R3LAY_NO_CACHE", None)
    set_bypass(False)


def test_default_is_caching():
    assert cache_bypassed() is False


def test_env_var_enables_bypass(monkeypatch):
    monkeypatch.setenv("R3LAY_NO_CACHE", "1")
    assert cache_bypassed() is True


def test_env_falsy_does_not_bypass(monkeypatch):
    monkeypatch.setenv("R3LAY_NO_CACHE", "0")
    assert cache_bypassed() is False
    monkeypatch.setenv("R3LAY_NO_CACHE", "")
    assert cache_bypassed() is False


def test_override_flag_wins(monkeypatch):
    monkeypatch.setenv("R3LAY_NO_CACHE", "0")
    set_bypass(True)
    assert cache_bypassed() is True


def test_bypass_scope_sets_and_restores():
    assert cache_bypassed() is False
    with bypass_scope():
        assert cache_bypassed() is True
    assert cache_bypassed() is False


def test_nested_bypass_scope_restores_outer():
    """With contextvars, each scope gets a proper token-based restore.

    An inner bypass_scope(False) correctly disables bypass within its
    block without leaking that state to the outer scope after exit.
    This is the correct per-request behavior: concurrent requests
    each get their own context copy.
    """
    with bypass_scope(True):
        assert cache_bypassed() is True
        with bypass_scope(False):
            assert cache_bypassed() is False
        assert cache_bypassed() is True
    assert cache_bypassed() is False


def test_bypass_scope_restores_on_exception():
    try:
        with bypass_scope():
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert cache_bypassed() is False
