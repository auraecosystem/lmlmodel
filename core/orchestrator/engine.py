import logging
from typing import Any, Dict
from models.adapters.base import BaseModelAdapter, CompletionRequest
from tools.registry import ToolRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LMLM-Core")

class LMLMOrchestrator:
    """Core runtime engine executing Perceive -> Reason -> Execute -> Verify."""

    def __init__(self, model_adapter: BaseModelAdapter, tool_registry: ToolRegistry):
        self.adapter = model_adapter
        self.tools = tool_registry
        self.working_memory: List[Dict[str, str]] = []

    def process_task(self, user_input: str) -> str:
        logger.info(f"[PERCEIVE] Ingesting task: {user_input}")
        self.working_memory.append({"role": "user", "content": user_input})

        # Phase: Retrieve & Plan
        available_tools = self.tools.get_tool_schemas()
        
        # Phase: Reason via Model Adapter
        request = CompletionRequest(
            prompt=f"Task: {user_input}\nAvailable Tools: {available_tools}",
            temperature=0.1
        )
        
        logger.info("[REASON] Routing request through model adapter layer...")
        response = self.adapter.generate(request)

        # Phase: Execute & Verify (Simplified Loop)
        if response.tool_calls:
            for call in response.tool_calls:
                tool_name = call["name"]
                tool_args = call.get("args", {})
                logger.info(f"[EXECUTE] Triggering Tool: {tool_name} with args {tool_args}")
                
                result = self.tools.execute(tool_name, **tool_args)
                logger.info(f"[VERIFY] Observation result: {result}")
                
                # Append to working memory
                self.working_memory.append({"role": "tool_result", "content": str(result)})
                return f"Task completed via {tool_name}. Result: {result}"

        # Default standard completion output
        self.working_memory.append({"role": "assistant", "content": response.content})
        logger.info("[REMEMBER] State updated in working memory.")
        return response.content
