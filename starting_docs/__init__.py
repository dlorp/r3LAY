"""
r3LAY - TUI Research Assistant

A terminal-based research assistant with:
- Local LLM integration (Ollama, llama.cpp, HuggingFace)
- Hybrid RAG (BM25 + vector search with RRF fusion)
- Deep research mode with convergence detection
- Full provenance tracking (signals, citations, axioms)
- Theme-based project organization
"""

__version__ = "0.6.1"
__author__ = "r3LAY"

from .app import R3LayApp, R3LayState, main
from .config import THEMES, AppConfig

__all__ = [
    "R3LayApp",
    "R3LayState",
    "AppConfig",
    "THEMES",
    "main",
]
