import asyncio
from typing import Any, Callable, Dict, List, Optional
from core.state import Observation, ToolDefinition


class ToolRegistry:
    """Strongly-typed registry executing registered tools with observation tracking."""

    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._definitions: Dict[str, ToolDefinition] = {}

    def register(self, definition: ToolDefinition, func: Callable):
        self._definitions[definition.name] = definition
        self._tools[definition.name] = func

    def get_definition(self, name: str) -> Optional[ToolDefinition]:
        return self._definitions.get(name)

    def get_schemas(self) -> List[Dict[str, Any]]:
        return [defn.model_dump() for defn in self._definitions.values()]

    async def execute(self, step_id: str, tool_name: str, **kwargs) -> Observation:
        if tool_name not in self._tools:
            return Observation(
                step_id=step_id,
                tool_name=tool_name,
                success=False,
                output=None,
                error=f"Tool '{tool_name}' not registered."
            )
        try:
            func = self._tools[tool_name]
            if asyncio.iscoroutinefunction(func):
                result = await func(**kwargs)
            else:
                result = func(**kwargs)
            
            return Observation(
                step_id=step_id,
                tool_name=tool_name,
                success=True,
                output=result
            )
        except Exception as e:
            return Observation(
                step_id=step_id,
                tool_name=tool_name,
                success=False,
                output=None,
                error=str(e)
            )
