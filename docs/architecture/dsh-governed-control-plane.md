# LMLM Governed Execution Control Plane

Status: architecture and implementation blueprint
Target: the existing LMLM runtime in this repository
Reference pattern: policy-governed orchestration, durable work records, worker isolation, and post-commit effects

## Decision

LMLM remains the sovereign control plane. Model selection, identity, authorization, policy, task state, approvals, scheduling, verification, audit history, and recovery remain owned by LMLM. Remote workers and services are execution targets only; they receive narrowly scoped, expiring capabilities through explicit adapters.

This document specifies an additive integration. It does not import another agent runtime, delegate LMLM authority to a remote service, or replace the existing model and tool adapters.

## Current code observations

The current tree already contains:
- task and execution state models in `core/state.py`;
- a basic policy gateway in `core/policies.py`;
- tool registration and invocation in `tools/registry.py`;
- working and episodic memory in `core/memory.py`;
- orchestration entry points in `core/orchestrator.py` and `core/orchestrator/engine.py`;
- model adapter abstractions under `models/adapters/` and `lmlm/adapters/`.

The current implementation is a prototype, not a production control plane. In particular, the production orchestrator creates a fixed example step, task identifiers are based on event-loop time, policy decisions are not backed by durable approvals, and tool execution is in-process without demonstrated OS isolation. The two orchestrator implementations also need consolidation behind a stable interface.

## Target architecture

```text
User / Event / Trusted Source
          |
          v
   LMLM Intake + Identity
          |
          v
   Mandate Engine ----> immutable authorization rationale
          |
          v
   Policy Engine ------> allow / deny / require approval
          |
          v
   Task & Ticket Store <---- transactional state + outbox
          |
          v
   Agent Scheduler
          |
          v
   Execution Fabric
      |       |       |
   local   sandbox  remote adapter
   worker  worker   (DSH / other)
      |       |       |
      +-------+-------+
              |
              v
   Result Verification + Audit
              |
       Memory / State Update
```

### Control-plane invariants

1. LMLM is the authority for identity, policy, mandates, approvals, task state, and audit records.
2. A worker cannot expand its own permissions or change its mandate.
3. Every execution request carries a task ID, step ID, mandate ID, adapter identity, capability set, expiry, and correlation ID.
4. No remote execution occurs before authorization and durable task/assignment creation.
5. Approval is bound to the exact action, arguments or argument digest, target, and expiry. Material changes invalidate approval.
6. Destructive, privileged, financial, production, and credential-bearing operations default to deny or explicit human approval.
7. Results are treated as untrusted observations until independently verified.
8. Retries are bounded and idempotency-aware. Cancellation and rollback are first-class states.
9. External side effects are dispatched from a durable outbox after the state transaction commits.
10. Remote adapters are replaceable. They never become the source of truth for LMLM state.

## Required subsystems

### 1. Mandate Engine

Create a durable mandate for each accepted unit of work. A mandate records:
- issuer and authenticated principal;
- purpose and task scope;
- permitted repositories, paths, services, and environments;
- permitted tools and operations;
- explicit exclusions;
- approval evidence and policy version;
- creation time, expiry, revocation state, and audit correlation.

Mandates are immutable after issuance. Changes create a new mandate/version and preserve the previous record.

### 2. Policy Engine

Replace implicit tool flags as the sole authorization mechanism with a deterministic decision interface:

- `ALLOW`: the exact operation is within an active mandate and policy.
- `DENY`: the operation is prohibited or outside scope.
- `REQUIRE_APPROVAL`: the operation may proceed only after a bound approval.
- `DEFER`: required identity, context, or policy data is unavailable.

Policy evaluation must fail closed. It must validate tool name, argument schema, target, environment, resource scope, capability, expiry, and risk class. Policy changes are versioned and auditable.

### 3. Task / Ticket Engine

Represent work as durable tasks, tickets, and assignments:
- task: user intent and lifecycle;
- ticket: independently schedulable work item with dependencies and acceptance criteria;
- assignment: one worker's bounded attempt;
- event: append-only state transition;
- outbox record: external effect awaiting dispatch.

Use transactional transitions and optimistic concurrency. Enforce valid state transitions and idempotency keys. Keep the current in-memory models as compatibility inputs while storage is introduced.

### 4. Agent Scheduler

Schedule only authorized, ready tickets. Match worker capabilities to ticket requirements, respect dependency ordering, quotas, deadlines, and trust level, and record every assignment. Workers report progress; only LMLM changes authoritative ticket state.

### 5. Execution Fabric

Define a narrow adapter protocol with operations equivalent to:
- discover capabilities;
- submit an authorized assignment;
- read progress/result;
- cancel assignment;
- health check.

The adapter receives a short-lived execution grant derived from the mandate. It does not receive broad user credentials or unrestricted control-plane access.

Local execution must use an OS-enforced sandbox/container or equivalent isolation: unprivileged identity, read-only base filesystem, explicit writable mounts, network egress policy, process and resource limits, syscall restrictions where available, and secret isolation. Tool wrappers alone are not a security boundary.

### 6. Verification and audit

Verify outputs against explicit acceptance criteria, expected artifact digests, test results, and policy constraints. Store observations separately from verified outcomes. Record who/what authorized, assigned, executed, verified, cancelled, or rejected each action.

### 7. Memory integration

Only verified, policy-approved summaries should be committed to persistent memory. Store provenance and access scope. Do not persist raw secrets, credentials, or untrusted tool output as trusted knowledge.

## Adapter contract

A remote adapter must accept a versioned, validated envelope containing:
- `protocol_version`
- `task_id`, `ticket_id`, `assignment_id`
- `mandate_id`, `policy_version`
- `capabilities`, `resource_scope`
- `issued_at`, `expires_at`, `nonce`
- `idempotency_key`, `correlation_id`
- typed input payload and integrity digest

The adapter returns a typed acknowledgement, progress events, and a result envelope. All inbound messages are authenticated, size-limited, schema-validated, replay-protected, and treated as untrusted data. Secrets are never embedded in prompts or logs.

## Incremental implementation plan

### Phase 0 — baseline and safety
- establish a clean Python package layout and supported Python/dependency matrix;
- add a real Python test suite and CI for lint, type checks, unit tests, and security checks;
- fix the tracked `.env` configuration and add safe example configuration;
- reconcile duplicate orchestrators and define compatibility tests;
- document that in-process tools are not sandboxed.

### Phase 1 — local control-plane primitives
- add typed mandate, approval, ticket, assignment, event, and execution-grant models;
- add deterministic fail-closed policy evaluation;
- add state-transition validation and idempotency;
- add unit tests for allow, deny, approval, expiry, revocation, scope mismatch, and replay.

### Phase 2 — durable orchestration
- introduce a repository interface and transactional store;
- persist state transitions and outbox records atomically;
- implement recovery after process restart and safe retry behavior;
- preserve the existing public API through an adapter/facade.

### Phase 3 — local sandbox worker
- implement a sandbox adapter with OS-enforced limits;
- add cancellation, timeouts, resource quotas, artifact collection, and independent verification;
- prohibit production and destructive operations by default.

### Phase 4 — remote worker adapter
- implement a DSH-compatible adapter as an optional execution backend;
- use short-lived scoped grants and dedicated worker identities;
- validate signatures/identity, replay protection, and result provenance;
- keep all policy and authoritative state in LMLM.

### Phase 5 — rollout
- run in observe-only mode first;
- compare policy decisions and worker outcomes without enabling side effects;
- enable narrowly scoped low-risk actions;
- require explicit approval for elevated operations;
- expand only after security review and measured reliability.

## Acceptance criteria

- A request outside its mandate is denied before execution.
- An approval cannot be reused for altered arguments, another target, or an expired request.
- A worker cannot mutate policy, mandates, or authoritative task state.
- Duplicate submissions do not create duplicate side effects.
- Process restart recovers pending assignments and outbox events safely.
- Cancellation prevents further work where supported and records partial completion.
- Every action has a complete, queryable audit trail.
- Remote execution is disabled when authentication, policy, or adapter health is uncertain.
- Tests prove that no tool executes on deny, defer, expired grant, or invalid approval.

## Explicit non-goals

- Do not make DSH-superagent a dependency of LMLM.
- Do not move the LMLM control plane into a remote agent.
- Do not automatically execute arbitrary shell commands from model output.
- Do not treat Docker, a wrapper script, or a Python permission check alone as a complete sandbox.
- Do not enable remote execution or production changes by default.
