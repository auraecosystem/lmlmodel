from typing import Any, Dict
from core.state import PolicyAction, ToolDefinition


class PolicyEngine:
    """Safety policy gateway evaluated prior to tool execution."""

    def evaluate(self, tool_def: ToolDefinition, parameters: Dict[str, Any]) -> PolicyAction:
        if tool_def.dangerous:
            if tool_def.requires_confirmation:
                return PolicyAction.REQUIRE_CONFIRMATION
            return PolicyAction.DENY
        return PolicyAction.ALLOW
