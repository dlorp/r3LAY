"""Tests for r3lay.config — backend config loading + env var resolution."""

from __future__ import annotations

import pytest


def _write_config(tmp_path, body):
    cfg_path = tmp_path / "r3lay-config.yaml"
    cfg_path.write_text(body)
    return cfg_path


def test_get_embedding_model_from_yaml(tmp_path, monkeypatch):
    """Reads embedding.model from the backend config file."""
    cfg_path = _write_config(
        tmp_path,
        "embedding:\n  model: bge-test\n  dim: 1024\n  ollama_url: http://localhost:11434\n",
    )
    import r3lay.config as cfg

    monkeypatch.setattr(cfg, "_CONFIG_PATHS", [cfg_path])
    cfg._config = None
    monkeypatch.delenv("R3LAY_EMBEDDING_MODEL", raising=False)

    assert cfg.get_embedding_model() == "bge-test"


def test_get_embedding_model_from_env(tmp_path, monkeypatch):
    """Falls back to R3LAY_EMBEDDING_MODEL env var when config missing."""
    import r3lay.config as cfg

    monkeypatch.setattr(cfg, "_CONFIG_PATHS", [tmp_path / "nonexistent.yaml"])
    cfg._config = None
    monkeypatch.setenv("R3LAY_EMBEDDING_MODEL", "env-model")

    assert cfg.get_embedding_model() == "env-model"


def test_get_embedding_model_raises_if_unconfigured(tmp_path, monkeypatch):
    """Raises RuntimeError with helpful message when nothing is configured."""
    import r3lay.config as cfg

    monkeypatch.setattr(cfg, "_CONFIG_PATHS", [tmp_path / "nonexistent.yaml"])
    cfg._config = None
    monkeypatch.delenv("R3LAY_EMBEDDING_MODEL", raising=False)

    with pytest.raises(RuntimeError, match="No embedding model configured"):
        cfg.get_embedding_model()


def test_get_tracked_path_allowed_roots_defaults(tmp_path, monkeypatch):
    """Returns sensible defaults when the key is missing."""
    cfg_path = _write_config(tmp_path, "embedding:\n  model: x\n")
    import r3lay.config as cfg

    monkeypatch.setattr(cfg, "_CONFIG_PATHS", [cfg_path])
    cfg._config = None

    roots = cfg.get_tracked_path_allowed_roots()
    assert len(roots) == 2
    # Both defaults should be expanded/resolved
    assert all(str(r).startswith("/") for r in roots)
