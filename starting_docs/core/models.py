"""Model discovery and management for multiple sources."""

import json
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import httpx


class ModelSource(str, Enum):
    """Supported model sources."""
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    LLAMA_CPP = "llama_cpp"
    OPENCLAW = "openclaw"  # Agent-based inference via OpenClaw


@dataclass
class ModelInfo:
    """Information about an available model."""
    name: str
    source: ModelSource
    path: str | None = None
    size: int | None = None
    format: str | None = None
    last_accessed: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def display_name(self) -> str:
        """Human-readable display name."""
        tag = {"huggingface": "HF", "ollama": "OL", "llama_cpp": "LC", "openclaw": "OC"}.get(self.source.value, "??")
        return f"[{tag}] {self.name}"
    
    @property
    def size_human(self) -> str:
        """Human-readable size."""
        if not self.size:
            return "?"
        size = self.size
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} PB"


class ModelScanner:
    """Scans and discovers models from multiple sources."""
    
    def __init__(
        self,
        hf_cache_path: str | None = None,
        ollama_endpoint: str = "http://localhost:11434",
    ):
        self.hf_cache_path = hf_cache_path
        self.ollama_endpoint = ollama_endpoint
        self._models: list[ModelInfo] = []
    
    async def scan_all(
        self,
        include_hf: bool = True,
        include_ollama: bool = True,
    ) -> list[ModelInfo]:
        """Scan all enabled sources for models."""
        self._models = []
        
        if include_hf:
            self._models.extend(self.scan_huggingface())
        
        if include_ollama:
            self._models.extend(await self.scan_ollama())
        
        # Sort: HuggingFace first, then by name
        self._models.sort(key=lambda m: (
            0 if m.source == ModelSource.HUGGINGFACE else 1,
            m.name.lower(),
        ))
        
        return self._models
    
    def scan_huggingface(self) -> list[ModelInfo]:
        """Scan HuggingFace cache for available models."""
        models = []
        
        cmd = ["huggingface-cli", "scan-cache", "--json"]
        if self.hf_cache_path:
            cmd.extend(["--dir", self.hf_cache_path])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                return models
            
            cache_info = json.loads(result.stdout)
            
            for repo in cache_info.get("repos", []):
                repo_path = Path(repo.get("repo_path", ""))
                model_format = self._detect_format(repo_path)
                
                models.append(ModelInfo(
                    name=repo.get("repo_id", "unknown"),
                    source=ModelSource.HUGGINGFACE,
                    path=str(repo_path),
                    size=repo.get("size_on_disk"),
                    format=model_format,
                    last_accessed=repo.get("last_accessed"),
                    metadata={
                        "revisions": repo.get("revisions", []),
                        "nb_files": repo.get("nb_files", 0),
                    },
                ))
        
        except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
            pass
        
        return models
    
    async def scan_ollama(self) -> list[ModelInfo]:
        """Get available Ollama models."""
        models = []
        
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{self.ollama_endpoint}/api/tags")
                resp.raise_for_status()
                data = resp.json()
                
                for model in data.get("models", []):
                    models.append(ModelInfo(
                        name=model.get("name", "unknown"),
                        source=ModelSource.OLLAMA,
                        size=model.get("size"),
                        metadata={
                            "modified_at": model.get("modified_at"),
                            "digest": model.get("digest"),
                            "details": model.get("details", {}),
                        },
                    ))
        
        except httpx.HTTPError:
            pass
        
        return models
    
    def _detect_format(self, repo_path: Path) -> str | None:
        """Detect model format from files in repo."""
        if not repo_path.exists():
            return None
        
        snapshots = repo_path / "snapshots"
        if snapshots.exists():
            for snapshot in snapshots.iterdir():
                if snapshot.is_dir():
                    for file in snapshot.iterdir():
                        suffix = file.suffix.lower()
                        if suffix == ".gguf":
                            return "gguf"
                        elif suffix == ".safetensors":
                            return "safetensors"
                        elif suffix == ".bin" and "pytorch" in file.name.lower():
                            return "pytorch"
        
        return None
    
    def get_models(self) -> list[ModelInfo]:
        """Get cached model list."""
        return self._models
    
    def get_by_source(self, source: ModelSource) -> list[ModelInfo]:
        """Filter models by source."""
        return [m for m in self._models if m.source == source]
    
    def get_by_name(self, name: str) -> ModelInfo | None:
        """Find a model by name."""
        for model in self._models:
            if model.name == name:
                return model
        return None
    
    def get_gguf_models(self) -> list[ModelInfo]:
        """Get all GGUF format models."""
        return [m for m in self._models if m.format == "gguf"]
