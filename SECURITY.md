# Security and Privacy Model

## 1. Overview

This system is designed as a local-first cognitive architecture.

No external telemetry is implemented.
No cloud storage is used.
No data is transmitted beyond the configured local LLM endpoint.

The security model assumes:

- Local machine trust
- No hostile root-level compromise
- Standard Linux user-level permissions

---

## 2. Data Handling

All state is stored locally:

- JSON files
- Log files
- Trace files
- Evaluation artifacts

No data leaves the machine unless explicitly configured otherwise.

---

## 3. Threat Model

### 3.1 In-Scope Threats

- Accidental data corruption
- Model instability
- Runaway loops
- Misconfigured tool execution
- Prompt-level hallucinations

Mitigations:

- Atomic writes
- Circuit breaker
- Capability gating
- Claim verification
- Deterministic meta-calls

---

### 3.2 Out-of-Scope Threats

The following are explicitly NOT mitigated:

- Kernel-level compromise
- Root-level malicious actors
- Hardware-level surveillance
- Side-channel attacks
- Malicious local users with file access

---

## 4. Tool Execution Safety

Tool execution is restricted by:

- Explicit capability flags
- No implicit file writes
- No arbitrary network execution
- Manual approval for write/run operations

Read-only system inspection may occur when enabled.

---

## 5. Privacy Guarantees

This project guarantees:

- No telemetry
- No analytics
- No usage reporting
- No remote tracking
- No embedded third-party services

The user is fully responsible for configuring
the local LLM endpoint securely.

---

## 6. Responsible Use

The software is experimental.

It is not certified for:

- Medical use
- Legal decision-making
- Financial advisory systems
- Safety-critical systems

Use at your own risk.
