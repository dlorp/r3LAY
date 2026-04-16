"""r3LAY configuration.

Single source of truth for all configurable values. Reads from
environment variables and hermes-profile/config.yaml. No hardcoded
model names — everything comes from user configuration.

First-run: if no config exists, r3LAY prints setup instructions and exits.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Config file search order — r3LAY backend config
# (separate from Hermes agent profile at ~/.hermes/profiles/r3lay/config.yaml)
_CONFIG_PATHS = [
    Path(__file__).parent.parent / "r3lay-config.yaml",
    Path(__file__).parent.parent / "r3lay-config.template.yaml",
]


def _load_yaml_config() -> dict[str, Any]:
    """Load config from the first available config file."""
    from ruamel.yaml import YAML

    yaml = YAML()
    for path in _CONFIG_PATHS:
        if path.exists():
            try:
                with open(path) as f:
                    data = yaml.load(f) or {}
                logger.debug("Config loaded from %s", path)
                return data
            except Exception as e:
                logger.warning("Failed to load config from %s: %s", path, e)
    return {}


_config: dict[str, Any] | None = None


def get_config() -> dict[str, Any]:
    """Get the loaded configuration dict. Caches after first load.

    When cache bypass is active (``R3LAY_NO_CACHE=1`` or the in-process
    flag), the YAML file is re-read on every call. This matters when the
    user edits config live and re-invokes without restarting the bridge.
    """
    from .cache import cache_bypassed

    global _config
    if _config is None or cache_bypassed():
        _config = _load_yaml_config()
    return _config


# =============================================================================
# Embedding config — used by ingest.py and search.py
# =============================================================================


def get_embedding_model() -> str:
    """Get the configured embedding model name.

    Reads from config.yaml: embedding.model
    Falls back to env var: R3LAY_EMBEDDING_MODEL
    No hardcoded default — fails with a clear message if not configured.
    """
    cfg = get_config()
    model = cfg.get("embedding", {}).get("model")
    if model:
        return model

    model = os.environ.get("R3LAY_EMBEDDING_MODEL")
    if model:
        return model

    raise RuntimeError(
        "No embedding model configured. Set one of:\n"
        "  1. embedding.model in hermes-profile/config.yaml\n"
        "  2. R3LAY_EMBEDDING_MODEL environment variable\n"
        "Example: ollama pull bge-m3 && export R3LAY_EMBEDDING_MODEL=bge-m3"
    )


def get_embedding_dim() -> int:
    """Get the embedding dimension for the configured model."""
    cfg = get_config()
    return cfg.get("embedding", {}).get("dim", 1024)


def get_ollama_url() -> str:
    """Get the Ollama API URL."""
    cfg = get_config()
    url = cfg.get("embedding", {}).get("ollama_url")
    if url:
        return url
    return os.environ.get("OLLAMA_URL", "http://localhost:11434")


# =============================================================================
# Backend model config — used by ingest, search, escalation routing
# =============================================================================


def get_default_model() -> dict[str, str]:
    """Get the default backend model config (provider + model name).

    Reads r3lay_model from r3lay-config.yaml. No hardcoded defaults.
    """
    cfg = get_config()
    model_cfg = cfg.get("r3lay_model", {})
    if not model_cfg.get("provider") or not model_cfg.get("model"):
        raise RuntimeError(
            "No backend model configured. Edit r3lay-config.yaml:\n"
            "  r3lay_model:\n"
            "    provider: ollama\n"
            "    model: your-model-name"
        )
    return {"provider": model_cfg["provider"], "model": model_cfg["model"]}


def get_escalation_model() -> dict[str, str] | None:
    """Get the escalation model config for deep work."""
    cfg = get_config()
    esc = cfg.get("r3lay_escalation", {})
    if esc.get("provider") and esc.get("model"):
        return {"provider": esc["provider"], "model": esc["model"]}
    return None


def get_tracked_path_allowed_roots() -> list[Path]:
    """Get the list of allowed root paths for /tracked operations.

    Returns expanded, resolved Path objects. The bridge will reject any
    /tracked or /ingest request whose resolved path does not live under
    one of these roots.

    Defaults to [~/r3LAY, ~/Documents/Programming] if not configured.
    """
    cfg = get_config()
    raw = cfg.get("tracked_path_allowed_roots") or [
        "~/r3LAY",
        "~/Documents/Programming",
    ]
    roots = []
    for r in raw:
        try:
            roots.append(Path(str(r)).expanduser().resolve())
        except (OSError, RuntimeError):
            logger.warning("Failed to resolve allowed root: %s", r)
    return roots
