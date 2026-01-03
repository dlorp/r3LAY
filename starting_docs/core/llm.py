"""LLM client abstraction for multiple local backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator

import httpx

from .models import ModelInfo, ModelSource


@dataclass
class Message:
    """A chat message."""
    role: str  # "user", "assistant", "system"
    content: str


@dataclass
class ChatResponse:
    """Response from LLM."""
    content: str
    model: str
    done: bool = True
    metrics: dict[str, Any] | None = None


class LLMAdapter(ABC):
    """Abstract interface for LLM backends."""
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name."""
        pass
    
    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        """Send a chat completion request."""
        pass
    
    @abstractmethod
    async def chat_stream(
        self,
        messages: list[Message],
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream a chat completion response."""
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the backend is available."""
        pass
    
    async def close(self) -> None:
        """Clean up resources."""
        pass


class OllamaAdapter(LLMAdapter):
    """Adapter for Ollama API."""
    
    def __init__(self, model: str, endpoint: str = "http://localhost:11434"):
        self.model = model
        self.endpoint = endpoint
        self.client = httpx.AsyncClient(timeout=300)
    
    @property
    def model_name(self) -> str:
        return f"ollama:{self.model}"
    
    async def chat(
        self,
        messages: list[Message],
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        formatted = [{"role": m.role, "content": m.content} for m in messages]
        if system_prompt:
            formatted.insert(0, {"role": "system", "content": system_prompt})
        
        payload = {
            "model": self.model,
            "messages": formatted,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        
        resp = await self.client.post(f"{self.endpoint}/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        
        return ChatResponse(
            content=data.get("message", {}).get("content", ""),
            model=self.model,
            done=data.get("done", True),
            metrics={
                "total_duration": data.get("total_duration"),
                "eval_count": data.get("eval_count"),
                "eval_duration": data.get("eval_duration"),
            },
        )
    
    async def chat_stream(
        self,
        messages: list[Message],
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        formatted = [{"role": m.role, "content": m.content} for m in messages]
        if system_prompt:
            formatted.insert(0, {"role": "system", "content": system_prompt})
        
        payload = {
            "model": self.model,
            "messages": formatted,
            "stream": True,
            "options": {"temperature": temperature},
        }
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        
        async with self.client.stream("POST", f"{self.endpoint}/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line:
                    import json
                    data = json.loads(line)
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content
    
    async def is_available(self) -> bool:
        try:
            resp = await self.client.get(f"{self.endpoint}/api/tags")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False
    
    async def close(self) -> None:
        await self.client.aclose()


class LlamaCppAdapter(LLMAdapter):
    """Adapter for llama.cpp server."""
    
    def __init__(self, endpoint: str = "http://localhost:8080", model_name: str = "llama.cpp"):
        self.endpoint = endpoint
        self._model_name = model_name
        self.client = httpx.AsyncClient(timeout=300)
    
    @property
    def model_name(self) -> str:
        return f"llama.cpp:{self._model_name}"
    
    async def chat(
        self,
        messages: list[Message],
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        prompt = self._build_prompt(messages, system_prompt)
        
        payload = {
            "prompt": prompt,
            "temperature": temperature,
            "stop": ["</s>", "[INST]", "[/INST]", "<|im_end|>", "<|end|>"],
        }
        if max_tokens:
            payload["n_predict"] = max_tokens
        
        resp = await self.client.post(f"{self.endpoint}/completion", json=payload)
        resp.raise_for_status()
        data = resp.json()
        
        return ChatResponse(
            content=data.get("content", ""),
            model=self._model_name,
            done=data.get("stop", True),
            metrics=data.get("timings"),
        )
    
    async def chat_stream(
        self,
        messages: list[Message],
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        prompt = self._build_prompt(messages, system_prompt)
        
        payload = {
            "prompt": prompt,
            "temperature": temperature,
            "stream": True,
            "stop": ["</s>", "[INST]", "[/INST]", "<|im_end|>", "<|end|>"],
        }
        if max_tokens:
            payload["n_predict"] = max_tokens
        
        async with self.client.stream("POST", f"{self.endpoint}/completion", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    import json
                    data = json.loads(line[6:])
                    content = data.get("content", "")
                    if content:
                        yield content
    
    async def is_available(self) -> bool:
        try:
            resp = await self.client.get(f"{self.endpoint}/health")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False
    
    def _build_prompt(self, messages: list[Message], system_prompt: str | None) -> str:
        """Build prompt string (ChatML format)."""
        parts = []
        
        if system_prompt:
            parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>")
        
        for msg in messages:
            parts.append(f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>")
        
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)
    
    async def close(self) -> None:
        await self.client.aclose()


def create_adapter(
    model_info: ModelInfo,
    config: dict[str, Any] | None = None,
) -> LLMAdapter:
    """Factory function to create appropriate LLM adapter."""
    config = config or {}
    
    if model_info.source == ModelSource.OLLAMA:
        return OllamaAdapter(
            model=model_info.name,
            endpoint=config.get("ollama_endpoint", "http://localhost:11434"),
        )
    
    elif model_info.source == ModelSource.LLAMA_CPP:
        return LlamaCppAdapter(
            endpoint=config.get("llama_cpp_endpoint", "http://localhost:8080"),
            model_name=model_info.name,
        )
    
    elif model_info.source == ModelSource.HUGGINGFACE:
        # For HF models with GGUF format, assume llama.cpp server
        if model_info.format == "gguf":
            return LlamaCppAdapter(
                endpoint=config.get("llama_cpp_endpoint", "http://localhost:8080"),
                model_name=model_info.name,
            )
        else:
            raise ValueError(f"Unsupported HuggingFace format: {model_info.format}")
    
    else:
        raise ValueError(f"Unsupported model source: {model_info.source}")
