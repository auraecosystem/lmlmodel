from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class PolicyAction(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_CONFIRMATION = "require_confirmation"


class ToolDefinition(BaseModel):
    name: str
    description: str
    input_schema: Dict[str, Any]
    dangerous: bool = False
    requires_confirmation: bool = False


class TaskStep(BaseModel):
    id: str
    task_id: str
    action: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    consecutive_failures: int = 0
    result: Optional[Any] = None


class Task(BaseModel):
    id: str
    input: str
    status: TaskStatus = TaskStatus.PENDING
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VerificationResult(BaseModel):
    verified: bool
    message: str
    evidence: Optional[Any] = None


class Observation(BaseModel):
    step_id: str
    tool_name: str
    success: bool
    output: Any
    error: Optional[str] = None


class ExecutionTrace(BaseModel):
    task_id: str
    steps: List[TaskStep] = Field(default_factory=list)
    observations: List[Observation] = Field(default_factory=list)
    verifications: List[VerificationResult] = Field(default_factory=list)


class TaskPlan(BaseModel):
    task_id: str
    steps: List[TaskStep]
