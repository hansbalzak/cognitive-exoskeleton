# Tracing and Observability

## 1. Purpose

This project implements local tracing to ensure auditability,
debuggability, and behavioral transparency.

Tracing exists to answer:

- What happened?
- When did it happen?
- Why did the agent behave this way?
- Which internal systems were involved?

This is not telemetry.
This is local observability.

---

## 2. Trace Model

Each user interaction is assigned a unique trace identifier (`trace_id`).

A trace records:

- Timestamp (start / end)
- Model configuration (model, temperature, token limits)
- Context size (character counts)
- Tool invocations (type, arguments, bounded output)
- Claim extraction results
- Reflection results (when enabled)
- Errors and recovery paths

Traces are written to:
traces/YYYY-MM-DD.jsonl

Each line represents a single interaction.

---

## 3. Logged Events

Typical trace entries include:

- user_input_received
- context_assembled
- model_invoked
- assistant_response_generated
- claims_extracted
- self_reflection_completed
- persistence_updated
- circuit_breaker_triggered
- error_occurred

All entries are local-only.

No trace data is transmitted externally.

---

## 4. Design Constraints

Tracing is intentionally:

- Append-only
- Human-readable (JSONL)
- Local
- Bounded in size

Old traces may be rotated or pruned manually.

---

## 5. Privacy Considerations

Traces may contain:

- User prompts
- Assistant responses
- Memory references

Users are responsible for protecting trace files.

No anonymization is performed automatically.

---

## 6. Non-Goals

Tracing is not intended to provide:

- Distributed observability
- Remote analytics
- Real-time dashboards
- Enterprise monitoring integrations

This is a single-node research system.

---

## 7. Event Journal

The system now includes an event journal (`events.jsonl`) that logs all state changes and key actions. Each event is recorded with a timestamp, trace ID, event type, and payload.

### Privacy Mode

When privacy mode is enabled, user and assistant text are logged only as hashes and lengths, never as raw content.

---

## 8. Supervisor Loop

The system now includes a supervisor loop that runs in the background and performs health checks and sanity checks. It logs supervisor actions to `events.jsonl` and `logs`.

### Health Checks

- Calls `/v1/models` to check the health of the LLM API.
- Sets a "degraded mode" flag if the API is down.
- Clears the degraded mode flag if the API is recovered.

### Sanity Checks

- Validates JSON state files (`conversation.json`, `facts.jsonl`, `claims.jsonl`, `identity.json`).
- Attempts recovery of corrupted files by renaming them and creating minimal safe defaults.

---

## 9. Fault Containment

Each subsystem call is wrapped in try/except with targeted fallbacks. On any exception, an error is logged to `events.jsonl` and `failures.jsonl`, and the system continues operation in degraded mode for that subsystem.

---

## 10. Identity Freeze

The system now includes an identity freeze configuration (`identity_freeze.json`). When enabled, the core identity fields (name, values, top-level preferences) are immutable, and only the style field can be adjusted.

---

## 11. Deterministic Reflection Scheduler

The system now uses a deterministic reflection scheduler that reflects every 8 user messages or every 60 minutes, whichever comes first. Reflection is skipped in degraded mode.

---

## 12. Transactional State Writes

State writes are now transactional, using a temp file and fsync. Critical files also have `.bak` snapshots created occasionally.

---

## 13. Commands

New commands have been added:

- `/supervisor`: Enable or disable supervisor mode.
- `/privacy_mode`: Enable or disable privacy mode.
- `/identity_freeze`: Enable or disable identity freeze.
