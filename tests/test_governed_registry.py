import unittest
from datetime import datetime, timedelta, timezone

from core.governance import (
    Approval, Decision, ExecutionIntent, Mandate, MandatePolicyEngine,
    request_digest,
)
from core.state import ToolDefinition
from tools.registry import ToolRegistry


class GovernedRegistryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.now = datetime.now(timezone.utc)
        self.mandate = Mandate(
            mandate_id="mandate-1",
            subject="agent-1",
            issuer="lmlm",
            issued_at=self.now - timedelta(minutes=1),
            expires_at=self.now + timedelta(minutes=5),
            allowed_tools=("math.add",),
            allowed_actions=("calculate",),
            allowed_resources=("calculator",),
        )
        self.intent = ExecutionIntent(
            mandate_id="mandate-1",
            subject="agent-1",
            tool="math.add",
            action="calculate",
            resource="calculator",
            parameters={"left": 2, "right": 3},
        )
        self.registry = ToolRegistry()
        self.calls = []
        self.registry.register(
            ToolDefinition(
                name="math.add",
                description="Add two numbers",
                input_schema={"type": "object"},
            ),
            lambda left, right: self.calls.append((left, right)) or left + right,
        )
        self.policy = MandatePolicyEngine()

    async def test_authorized_call_executes(self):
        observation, result = await self.registry.execute_governed(
            "step-1", self.intent, mandate=self.mandate,
            policy_engine=self.policy, task_id="task-1",
        )
        self.assertEqual(result.decision, Decision.ALLOW)
        self.assertTrue(observation.success)
        self.assertEqual(observation.output, 5)
        self.assertEqual(self.calls, [(2, 3)])

    async def test_missing_mandate_does_not_execute(self):
        observation, result = await self.registry.execute_governed(
            "step-1", self.intent, mandate=None,
            policy_engine=self.policy, task_id="task-1",
        )
        self.assertEqual(result.decision, Decision.DENY)
        self.assertFalse(observation.success)
        self.assertEqual(self.calls, [])

    async def test_out_of_scope_tool_does_not_execute(self):
        from dataclasses import replace
        intent = replace(self.intent, tool="shell.exec")
        observation, result = await self.registry.execute_governed(
            "step-1", intent, mandate=self.mandate,
            policy_engine=self.policy, task_id="task-1",
        )
        self.assertEqual(result.decision, Decision.DENY)
        self.assertFalse(observation.success)
        self.assertEqual(self.calls, [])

    async def test_approval_required_call_is_not_dispatched_without_approval(self):
        from dataclasses import replace
        mandate = replace(
            self.mandate, approval_required_actions=("calculate",)
        )
        observation, result = await self.registry.execute_governed(
            "step-1", self.intent, mandate=mandate,
            policy_engine=self.policy, task_id="task-1",
        )
        self.assertEqual(result.decision, Decision.REQUIRE_APPROVAL)
        self.assertFalse(observation.success)
        self.assertEqual(self.calls, [])

    async def test_exact_approval_allows_dispatch(self):
        from dataclasses import replace
        mandate = replace(
            self.mandate, approval_required_actions=("calculate",)
        )
        approval = Approval(
            approval_id="approval-1",
            mandate_id="mandate-1",
            task_id="task-1",
            subject="agent-1",
            request_digest=request_digest(self.intent),
            approved_by="operator-1",
            approved_at=self.now - timedelta(seconds=1),
            expires_at=self.now + timedelta(minutes=1),
        )
        observation, result = await self.registry.execute_governed(
            "step-1", self.intent, mandate=mandate, approval=approval,
            policy_engine=self.policy, task_id="task-1",
        )
        self.assertEqual(result.decision, Decision.ALLOW)
        self.assertTrue(observation.success)
        self.assertEqual(observation.output, 5)
        self.assertEqual(len(self.calls), 1)


if __name__ == "__main__":
    unittest.main()
