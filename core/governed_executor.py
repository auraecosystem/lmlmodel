"""Task-plan executor that routes every tool invocation through LMLM policy."""
from __future__ import annotations

from typing import Callable, Dict, Mapping, Optional

from core.governance import (
    Approval, ExecutionIntent, Mandate, MandatePolicyEngine,
)
from core.state import (
    ExecutionTrace, TaskPlan, TaskStatus, VerificationResult,
)
from tools.registry import ToolRegistry


class GovernedTaskExecutor:
    """Execute a supplied plan without bypassing mandate authorization.

    The caller supplies an intent factory so action/resource scope is explicit;
    the executor never guesses those security-sensitive values from tool args.
    """

    def __init__(
        self,
        tools: ToolRegistry,
        policy: MandatePolicyEngine,
    ) -> None:
        self.tools = tools
        self.policy = policy

    async def execute_plan(
        self,
        plan: TaskPlan,
        *,
        mandate: Optional[Mandate],
        intent_factory: Callable,
        approvals: Optional[Mapping[str, Approval]] = None,
    ) -> ExecutionTrace:
        trace = ExecutionTrace(task_id=plan.task_id, steps=plan.steps)
        approvals = approvals or {}
        completed = set()

        for step in plan.steps:
            if any(dependency not in completed for dependency in step.dependencies):
                step.status = TaskStatus.BLOCKED
                trace.verifications.append(VerificationResult(
                    verified=False,
                    message="Dependency not completed; execution withheld",
                    evidence={"dependencies": step.dependencies},
                ))
                continue

            try:
                intent = intent_factory(step)
                if not isinstance(intent, ExecutionIntent):
                    raise TypeError("intent_factory must return ExecutionIntent")
            except Exception as exc:
                step.status = TaskStatus.FAILED
                trace.verifications.append(VerificationResult(
                    verified=False,
                    message="Execution intent construction failed closed",
                    evidence=type(exc).__name__,
                ))
                continue

            step.status = TaskStatus.RUNNING
            observation, decision = await self.tools.execute_governed(
                step.id,
                intent,
                mandate=mandate,
                policy_engine=self.policy,
                task_id=plan.task_id,
                approval=approvals.get(step.id),
            )
            trace.observations.append(observation)

            if decision.decision.value == "require_approval":
                step.status = TaskStatus.BLOCKED
                trace.verifications.append(VerificationResult(
                    verified=False,
                    message="Independent approval required",
                    evidence=decision.reason,
                ))
                continue
            if decision.decision.value != "allow":
                step.status = TaskStatus.FAILED
                trace.verifications.append(VerificationResult(
                    verified=False,
                    message="LMLM policy denied execution",
                    evidence=decision.reason,
                ))
                continue
            if not observation.success or observation.output is None:
                step.status = TaskStatus.FAILED
                trace.verifications.append(VerificationResult(
                    verified=False,
                    message="Tool execution or output verification failed",
                    evidence=observation.error,
                ))
                continue

            step.status = TaskStatus.COMPLETED
            step.result = observation.output
            completed.add(step.id)
            trace.verifications.append(VerificationResult(
                verified=True,
                message="Authorized execution returned a non-null result",
                evidence={"policy_reason": decision.reason},
            ))

        return trace
