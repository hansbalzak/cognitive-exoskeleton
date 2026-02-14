# Architecture Overview

## 1. Design Philosophy

The Local Cognitive Agent is designed as a privacy-first,
self-correcting, persistent cognitive runtime.

It is not a chatbot wrapper.
It is not an enterprise framework.
It is an experimental local system focused on:

- Behavioral persistence
- Self-correction
- Memory decay and evolution
- Deterministic state management
- Auditability

The architecture prioritizes reliability over features.

---

## 2. High-Level Components

### 2.1 Core Agent Loop

The agent operates through a bounded interaction loop:

1. User input received
2. Intent classification (optional routing)
3. Context assembly:
   - Conversation history
   - Persistent memory (facts)
   - Identity state
   - Relevant corrections (claims)
4. Model invocation (local OpenAI-compatible endpoint)
5. Post-processing:
   - Claim extraction
   - Self-reflection (probabilistic)
   - Persistence update
6. Response output

All state mutations are atomic.

---

### 2.2 Persistent State

The system maintains structured local state:

- conversation.json
- facts.jsonl
- identity.json
- profile.json
- claims.jsonl
- state.json

All writes use atomic file replacement to prevent corruption.

No external storage is used.

---

### 2.3 Counterfactual Memory

The agent extracts factual claims from its own responses.

Claims are stored with:

- unique ID
- confidence score
- topic
- status (active / corrected / retracted)
- correction history

User corrections directly modify future behavior.

This enables structured self-correction rather than ephemeral memory.

---

### 2.4 Self-Reflection

Periodic reflective calls analyze:

- response quality
- uncertainty level
- potential memory updates

Reflection is deterministic (temperature = 0) for stability.

---

### 2.5 Circuit Breaker

The agent includes a circuit breaker mechanism:

- consecutive failure tracking
- cooldown period
- graceful degradation

This prevents runaway failure loops.

---

### 2.6 Tooling Model

Tool execution is capability-gated.

Capabilities are controlled via configuration flags:

- allow_write
- allow_run
- allow_network_readonly

No destructive action is executed without explicit permission.

---

## 3. Determinism and Stability

Meta-operations (claim extraction, reflection, summarization)
use deterministic model settings.

State transitions are logged.

Traceability is prioritized over novelty.

---

## 4. Non-Goals

This system intentionally does NOT aim to:

- Provide enterprise scalability
- Offer SaaS deployment
- Replace full observability tooling
- Maximize conversational entertainment

It is a research-grade local cognitive architecture.
