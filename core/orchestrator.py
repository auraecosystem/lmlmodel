import asyncio
import logging
from typing import Any, List, Optional

from core.state import (
    ExecutionTrace, Observation, PolicyAction, Task, TaskPlan,
    TaskStatus, TaskStep, VerificationResult
)
from models.adapters import BaseAsyncModelAdapter, CompletionRequest
from tools.registry import ToolRegistry
from core.policies import PolicyEngine
from core.memory import MemoryManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LMLM-Core")


class ProductionLMLMOrchestrator:
    """Full 9-Phase LMLM Production Engine Engine."""

    def __init__(
        self,
        model_adapter: BaseAsyncModelAdapter,
        tool_registry: ToolRegistry,
        policy_engine: PolicyEngine,
        memory_manager: MemoryManager,
        max_consecutive_failures: int = 2
    ):
        self.adapter = model_adapter
        self.tools = tool_registry
        self.policy = policy_engine
        self.memory = memory_manager
        self.max_consecutive_failures = max_consecutive_failures

    async def execute_task(self, raw_input: str) -> ExecutionTrace:
        # 1. PERCEIVE
        task = Task(id=f"task_{int(asyncio.get_event_loop().time())}", input=raw_input)
        trace = ExecutionTrace(task_id=task.id)
        logger.info(f"[1. PERCEIVE] Task created: {task.id}")

        # 2. UNDERSTAND
        task.metadata["complexity"] = "high" if len(raw_input) > 200 else "standard"
        logger.info(f"[2. UNDERSTAND] Complexity: {task.metadata['complexity']}")

        # 3. RETRIEVE
        retrieved_memories = [m for m in self.memory.persistent_episodic if task.input in str(m)]
        logger.info(f"[3. RETRIEVE] Pulled {len(retrieved_memories)} persistent records")

        # 4. REASON & 5. PLAN
        plan = await self._generate_dag_plan(task, retrieved_memories)
        trace.steps = plan.steps
        logger.info(f"[5. PLAN] Built execution plan with {len(plan.steps)} steps")

        task.status = TaskStatus.RUNNING

        # 6. EXECUTE LOOP
        for step in plan.steps:
            if step.status == TaskStatus.BLOCKED:
                continue

            step.status = TaskStatus.RUNNING
            while step.consecutive_failures < self.max_consecutive_failures:
                tool_def = self.tools.get_definition(step.action)
                if not tool_def:
                    step.status = TaskStatus.FAILED
                    break

                # POLICY EVALUATION
                action_policy = self.policy.evaluate(tool_def, step.parameters)
                if action_policy == PolicyAction.DENY:
                    logger.error(f"[POLICY DENIED] Tool blocked: {step.action}")
                    step.status = TaskStatus.FAILED
                    break
                elif action_policy == PolicyAction.REQUIRE_CONFIRMATION:
                    logger.warning(f"[POLICY INTERRUPT] Confirmation needed for {step.action}")
                    step.status = TaskStatus.BLOCKED
                    break

                # EXECUTION
                obs = await self.tools.execute(step.id, step.action, **step.parameters)
                trace.observations.append(obs)

                # 7. VERIFY
                ver_res = self._verify_step(step, obs)
                trace.verifications.append(ver_res)

                if ver_res.verified:
                    step.status = TaskStatus.COMPLETED
                    step.result = obs.output
                    step.consecutive_failures = 0
                    logger.info(f"[7. VERIFY] Step {step.id} verified.")
                    break
                else:
                    step.consecutive_failures += 1
                    logger.warning(f"[7. VERIFY] Step {step.id} failed verification.")

            if step.status != TaskStatus.COMPLETED:
                task.status = TaskStatus.FAILED
                break

        if all(s.status == TaskStatus.COMPLETED for s in plan.steps):
            task.status = TaskStatus.COMPLETED

        # 8. REMEMBER
        self.memory.commit_to_persistent(task.id, {"status": task.status, "input": task.input})
        logger.info(f"[8. REMEMBER] State committed for task {task.id}")

        # 9. IMPROVE
        logger.info(f"[9. IMPROVE] Task lifecycle finish status: {task.status}")
        return trace

    async def _generate_dag_plan(self, task: Task, context: List[Any]) -> TaskPlan:
        step = TaskStep(
            id="step_1",
            task_id=task.id,
            action="codex.execute",
            parameters={"code": "print(sum([i * 2 for i in range(10)]))"}
        )
        return TaskPlan(task_id=task.id, steps=[step])

    def _verify_step(self, step: TaskStep, observation: Observation) -> VerificationResult:
        if not observation.success:
            return VerificationResult(verified=False, message="Unhandled exception", evidence=observation.error)
        if observation.output is None:
            return VerificationResult(verified=False, message="Empty observation", evidence=None)
        return VerificationResult(verified=True, message="Verified non-null output", evidence=observation.output)
