import unittest
from datetime import datetime, timedelta, timezone

from core.governance import ExecutionIntent, Mandate, MandatePolicyEngine
from core.governed_executor import GovernedTaskExecutor
from core.state import TaskPlan, TaskStep, TaskStatus, ToolDefinition
from tools.registry import ToolRegistry


class GovernedTaskExecutorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        now = datetime.now(timezone.utc)
        self.mandate = Mandate(
            mandate_id="m-1", subject="agent-1", issuer="lmlm",
            issued_at=now - timedelta(minutes=1),
            expires_at=now + timedelta(minutes=5),
            allowed_tools=("math.add",), allowed_actions=("calculate",),
            allowed_resources=("calculator",),
        )
        self.registry = ToolRegistry()
        self.calls = []
        self.registry.register(
            ToolDefinition(name="math.add", description="Add", input_schema={}),
            lambda left, right: self.calls.append(1) or left + right,
        )
        self.executor = GovernedTaskExecutor(
            self.registry, MandatePolicyEngine()
        )

    def intent(self, step):
        return ExecutionIntent(
            mandate_id="m-1", subject="agent-1", tool=step.action,
            action="calculate", resource="calculator",
            parameters=step.parameters,
        )

    async def test_plan_step_is_authorized_and_completed(self):
        step = TaskStep(
            id="s1", task_id="t1", action="math.add",
            parameters={"left": 4, "right": 6},
        )
        trace = await self.executor.execute_plan(
            TaskPlan(task_id="t1", steps=[step]),
            mandate=self.mandate, intent_factory=self.intent,
        )
        self.assertEqual(step.status, TaskStatus.COMPLETED)
        self.assertEqual(step.result, 10)
        self.assertEqual(len(trace.observations), 1)
        self.assertEqual(len(self.calls), 1)

    async def test_denied_step_never_calls_tool(self):
        step = TaskStep(
            id="s1", task_id="t1", action="shell.exec",
            parameters={"command": "id"},
        )
        trace = await self.executor.execute_plan(
            TaskPlan(task_id="t1", steps=[step]),
            mandate=self.mandate, intent_factory=self.intent,
        )
        self.assertEqual(step.status, TaskStatus.FAILED)
        self.assertEqual(len(self.calls), 0)
        self.assertFalse(trace.verifications[0].verified)

    async def test_missing_mandate_fails_closed(self):
        step = TaskStep(
            id="s1", task_id="t1", action="math.add",
            parameters={"left": 1, "right": 2},
        )
        await self.executor.execute_plan(
            TaskPlan(task_id="t1", steps=[step]),
            mandate=None, intent_factory=self.intent,
        )
        self.assertEqual(step.status, TaskStatus.FAILED)
        self.assertEqual(len(self.calls), 0)

    async def test_dependency_failure_blocks_downstream_step(self):
        first = TaskStep(
            id="s1", task_id="t1", action="shell.exec",
            parameters={"command": "false"},
        )
        second = TaskStep(
            id="s2", task_id="t1", action="math.add",
            parameters={"left": 1, "right": 2}, dependencies=["s1"],
        )
        await self.executor.execute_plan(
            TaskPlan(task_id="t1", steps=[first, second]),
            mandate=self.mandate, intent_factory=self.intent,
        )
        self.assertEqual(first.status, TaskStatus.FAILED)
        self.assertEqual(second.status, TaskStatus.BLOCKED)
        self.assertEqual(len(self.calls), 0)


if __name__ == "__main__":
    unittest.main()
