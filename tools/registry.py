import asyncio
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.governance import (
    Approval, ExecutionIntent, Mandate, MandatePolicyEngine, PolicyResult,
)
from core.state import Observation, ToolDefinition


class ToolRegistry:
    """Tool registry with an explicit mandate-governed execution entry point."""

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

    async def execute_governed(
        self,
        step_id: str,
        intent: ExecutionIntent,
        *,
        mandate: Optional[Mandate],
        policy_engine: MandatePolicyEngine,
        task_id: str,
        approval: Optional[Approval] = None,
    ) -> Tuple[Observation, PolicyResult]:
        """Authorize the exact intent before dispatching its registered tool.

        Callers must construct the intent from the planned operation, not from
        untrusted worker output. A missing mandate or any non-ALLOW decision
        prevents the tool function from being invoked.
        """
        decision = policy_engine.evaluate(
            mandate, intent, approval=approval, task_id=task_id
        )
        if decision.decision.value != "allow":
            return (
                Observation(
                    step_id=step_id,
                    tool_name=intent.tool,
                    success=False,
                    output=None,
                    error=f"Execution blocked by LMLM policy: {decision.reason}",
                ),
                decision,
            )

        observation = await self.execute(
            step_id, intent.tool, **dict(intent.parameters)
        )
        return observation, decision
