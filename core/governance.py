"""Fail-closed, deterministic authorization primitives for the LMLM control plane.

This module does not execute tools or provide a process sandbox. It evaluates a
single proposed operation against an LMLM-issued mandate and (when required) a
separately issued approval record.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from typing import Any, Mapping, Optional, Tuple


class Decision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


@dataclass(frozen=True)
class Mandate:
    mandate_id: str
    subject: str
    issuer: str
    issued_at: datetime
    expires_at: datetime
    allowed_tools: Tuple[str, ...] = ()
    allowed_actions: Tuple[str, ...] = ()
    allowed_resources: Tuple[str, ...] = ()
    approval_required_tools: Tuple[str, ...] = ()
    approval_required_actions: Tuple[str, ...] = ()
    revoked: bool = False


@dataclass(frozen=True)
class ExecutionIntent:
    mandate_id: str
    subject: str
    tool: str
    action: str
    resource: str
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Approval:
    approval_id: str
    mandate_id: str
    task_id: str
    subject: str
    request_digest: str
    approved_by: str
    approved_at: datetime
    expires_at: datetime


@dataclass(frozen=True)
class PolicyResult:
    decision: Decision
    reason: str
    request_digest: str


def _aware(value: datetime) -> bool:
    return value.tzinfo is not None and value.utcoffset() is not None


def request_digest(intent: ExecutionIntent) -> str:
    """Hash the exact intent, including parameters, using canonical JSON."""
    try:
        payload = {
            "mandate_id": intent.mandate_id,
            "subject": intent.subject,
            "tool": intent.tool,
            "action": intent.action,
            "resource": intent.resource,
            "parameters": intent.parameters,
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False, default=_reject_non_json,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("execution parameters must be finite JSON values") from exc
    return hashlib.sha256(encoded).hexdigest()


def _reject_non_json(value: Any) -> Any:
    raise TypeError(f"unsupported parameter type: {type(value).__name__}")


class PolicyEngine:
    """Evaluate mandates and approvals; unknown or malformed context denies."""

    def evaluate(
        self,
        mandate: Optional[Mandate],
        intent: ExecutionIntent,
        *,
        now: Optional[datetime] = None,
        approval: Optional[Approval] = None,
        task_id: Optional[str] = None,
    ) -> PolicyResult:
        try:
            digest = request_digest(intent)
        except ValueError:
            return PolicyResult(Decision.DENY, "invalid_execution_parameters", "")

        current = now or datetime.now(timezone.utc)
        if not _aware(current):
            return PolicyResult(Decision.DENY, "evaluation_time_must_be_timezone_aware", digest)
        if mandate is None:
            return PolicyResult(Decision.DENY, "mandate_missing", digest)
        if not all(_aware(t) for t in (mandate.issued_at, mandate.expires_at)):
            return PolicyResult(Decision.DENY, "mandate_timestamps_must_be_timezone_aware", digest)
        if not mandate.mandate_id or not mandate.subject or not mandate.issuer:
            return PolicyResult(Decision.DENY, "mandate_identity_incomplete", digest)
        if mandate.revoked:
            return PolicyResult(Decision.DENY, "mandate_revoked", digest)
        if mandate.mandate_id != intent.mandate_id or mandate.subject != intent.subject:
            return PolicyResult(Decision.DENY, "mandate_binding_mismatch", digest)
        if current < mandate.issued_at:
            return PolicyResult(Decision.DENY, "mandate_not_yet_valid", digest)
        if current >= mandate.expires_at:
            return PolicyResult(Decision.DENY, "mandate_expired", digest)
        if not intent.tool or intent.tool not in mandate.allowed_tools:
            return PolicyResult(Decision.DENY, "tool_outside_mandate_scope", digest)
        if not intent.action or intent.action not in mandate.allowed_actions:
            return PolicyResult(Decision.DENY, "action_outside_mandate_scope", digest)
        if not intent.resource or intent.resource not in mandate.allowed_resources:
            return PolicyResult(Decision.DENY, "resource_outside_mandate_scope", digest)

        approval_required = (
            intent.tool in mandate.approval_required_tools
            or intent.action in mandate.approval_required_actions
        )
        if not approval_required:
            return PolicyResult(Decision.ALLOW, "mandate_scope_satisfied", digest)
        if approval is None:
            return PolicyResult(Decision.REQUIRE_APPROVAL, "explicit_approval_required", digest)
        if not all(_aware(t) for t in (approval.approved_at, approval.expires_at)):
            return PolicyResult(Decision.DENY, "approval_timestamps_must_be_timezone_aware", digest)
        if not task_id or not approval.approval_id or not approval.approved_by:
            return PolicyResult(Decision.DENY, "approval_binding_incomplete", digest)
        if approval.approved_by == intent.subject:
            return PolicyResult(Decision.DENY, "self_approval_forbidden", digest)
        if (
            approval.mandate_id != mandate.mandate_id
            or approval.subject != intent.subject
            or approval.task_id != task_id
            or approval.request_digest != digest
        ):
            return PolicyResult(Decision.DENY, "approval_binding_mismatch", digest)
        if current < approval.approved_at or current >= approval.expires_at:
            return PolicyResult(Decision.DENY, "approval_not_current", digest)
        return PolicyResult(Decision.ALLOW, "mandate_and_approval_satisfied", digest)
