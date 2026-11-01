import unittest
from datetime import datetime, timedelta, timezone

from core.governance import (
    Approval, Decision, ExecutionIntent, Mandate, MandatePolicyEngine, request_digest,
)


class PolicyEngineTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 10, 31, 12, tzinfo=timezone.utc)
        self.mandate = Mandate(
            mandate_id="m-1", subject="agent-1", issuer="lmlm-control-plane",
            issued_at=self.now - timedelta(minutes=1),
            expires_at=self.now + timedelta(minutes=10),
            allowed_tools=("filesystem.read",),
            allowed_actions=("read",),
            allowed_resources=("workspace/project-a",),
            approval_required_actions=("read",),
        )
        self.intent = ExecutionIntent(
            mandate_id="m-1", subject="agent-1", tool="filesystem.read",
            action="read", resource="workspace/project-a",
            parameters={"path": "README.md"},
        )
        self.engine = MandatePolicyEngine()

    def evaluate(self, mandate=None, intent=None, approval=None, task_id="task-1"):
        return self.engine.evaluate(
            mandate or self.mandate, intent or self.intent, now=self.now,
            approval=approval, task_id=task_id,
        )

    def test_valid_scoped_intent_requires_approval(self):
        result = self.evaluate()
        self.assertEqual(result.decision, Decision.REQUIRE_APPROVAL)

    def test_missing_mandate_denies(self):
        result = self.engine.evaluate(None, self.intent, now=self.now)
        self.assertEqual(result.decision, Decision.DENY)
        self.assertEqual(result.reason, "mandate_missing")

    def test_revoked_mandate_denies(self):
        from dataclasses import replace
        result = self.evaluate(mandate=replace(self.mandate, revoked=True))
        self.assertEqual(result.decision, Decision.DENY)

    def test_expired_mandate_denies(self):
        from dataclasses import replace
        result = self.evaluate(mandate=replace(self.mandate, expires_at=self.now))
        self.assertEqual(result.reason, "mandate_expired")

    def test_scope_mismatch_denies(self):
        from dataclasses import replace
        intent = replace(self.intent, resource="workspace/project-b")
        self.assertEqual(self.evaluate(intent=intent).decision, Decision.DENY)

    def test_exact_independent_approval_allows(self):
        approval = Approval(
            approval_id="a-1", mandate_id="m-1", task_id="task-1",
            subject="agent-1", request_digest=request_digest(self.intent),
            approved_by="human-operator",
            approved_at=self.now - timedelta(seconds=1),
            expires_at=self.now + timedelta(minutes=2),
        )
        self.assertEqual(self.evaluate(approval=approval).decision, Decision.ALLOW)

    def test_approval_replay_for_another_task_denies(self):
        approval = Approval(
            approval_id="a-1", mandate_id="m-1", task_id="task-1",
            subject="agent-1", request_digest=request_digest(self.intent),
            approved_by="human-operator",
            approved_at=self.now - timedelta(seconds=1),
            expires_at=self.now + timedelta(minutes=2),
        )
        self.assertEqual(
            self.evaluate(approval=approval, task_id="task-2").decision,
            Decision.DENY,
        )

    def test_parameter_tampering_invalidates_approval(self):
        from dataclasses import replace
        approval = Approval(
            approval_id="a-1", mandate_id="m-1", task_id="task-1",
            subject="agent-1", request_digest=request_digest(self.intent),
            approved_by="human-operator",
            approved_at=self.now - timedelta(seconds=1),
            expires_at=self.now + timedelta(minutes=2),
        )
        changed = replace(self.intent, parameters={"path": "secrets.txt"})
        self.assertEqual(self.evaluate(intent=changed, approval=approval).decision, Decision.DENY)

    def test_self_approval_denies(self):
        approval = Approval(
            approval_id="a-1", mandate_id="m-1", task_id="task-1",
            subject="agent-1", request_digest=request_digest(self.intent),
            approved_by="agent-1",
            approved_at=self.now - timedelta(seconds=1),
            expires_at=self.now + timedelta(minutes=2),
        )
        self.assertEqual(self.evaluate(approval=approval).decision, Decision.DENY)

    def test_non_json_parameters_fail_closed(self):
        from dataclasses import replace
        intent = replace(self.intent, parameters={"bad": object()})
        self.assertEqual(self.evaluate(intent=intent).decision, Decision.DENY)


if __name__ == "__main__":
    unittest.main()
