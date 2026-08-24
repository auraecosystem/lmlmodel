from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

class CompletionRequest(BaseModel):
    prompt: str
    temperature: float = 0.2
    max_tokens: int = 1000
    tools: Optional[List[Dict[str, Any]]] = None

class CompletionResponse(BaseModel):
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    usage: Dict[str, int] = {}

class BaseModelAdapter(ABC):
    """Unified adapter interface for model-agnostic execution."""
    
    @abstractmethod
    def generate(self, request: CompletionRequest) -> CompletionResponse:
        pass
