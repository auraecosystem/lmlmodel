import os
import httpx
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CompletionRequest(BaseModel):
    prompt: str
    temperature: float = 0.2
    max_tokens: int = 1500
    system_prompt: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None


class CompletionResponse(BaseModel):
    content: str
    model_name: str
    provider: str
    usage: Dict[str, int] = Field(default_factory=dict)


class BaseAsyncModelAdapter(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    async def generate_async(self, request: CompletionRequest) -> CompletionResponse:
        pass


class OpenAIAsyncAdapter(BaseAsyncModelAdapter):
    def __init__(self, model_name: str = "gpt-4o", api_key: Optional[str] = None):
        super().__init__(model_name)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "mock_key")
        self.base_url = "https://api.openai.com/v1/chat/completions"

    async def generate_async(self, request: CompletionRequest) -> CompletionResponse:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(self.base_url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return CompletionResponse(
            content=content,
            model_name=self.model_name,
            provider="openai",
            usage={"prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0)}
        )


class OllamaAsyncAdapter(BaseAsyncModelAdapter):
    def __init__(self, model_name: str = "llama3", base_url: str = "http://localhost:11434"):
        super().__init__(model_name)
        self.base_url = f"{base_url}/api/generate"

    async def generate_async(self, request: CompletionRequest) -> CompletionResponse:
        payload = {
            "model": self.model_name,
            "prompt": request.prompt,
            "system": request.system_prompt or "",
            "stream": False,
            "options": {"temperature": request.temperature, "num_predict": request.max_tokens}
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(self.base_url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        return CompletionResponse(
            content=data.get("response", ""),
            model_name=self.model_name,
            provider="ollama",
            usage={"eval_count": data.get("eval_count", 0)}
        )


class MultiProviderRouter(BaseAsyncModelAdapter):
    """Router with automated fallback across providers."""

    def __init__(self, adapters: List[BaseAsyncModelAdapter]):
        super().__init__(model_name="multi-provider-router")
        self.adapters = adapters

    async def generate_async(self, request: CompletionRequest) -> CompletionResponse:
        last_exception = None
        for adapter in self.adapters:
            try:
                return await adapter.generate_async(request)
            except Exception as e:
                last_exception = e
                continue
        raise RuntimeError(f"All model adapters failed. Last error: {last_exception}")
