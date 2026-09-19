# ADR-0001: Two-layer API (simple + enhanced)

- **Status:** Accepted
- **Date:** 2026-09-19
- **Deciders:** rbenzing (with Claude)

## Context

DroneCMD serves two audiences at once: quick scripting/prototyping ("load a
file, extract packets") and production streaming analysis (async capture,
structured configuration, typed results). A single API optimized for one hurts
the other — synchronous convenience functions do not compose with `async for`
streaming, and dataclass-config constructors are heavyweight for a throwaway
script.

## Decision

We maintain **two API levels, and both are updated when a feature is added**:

- **Simple layer** (`capture/` + top-level `__init__.py`): factory functions
  (`create_capture_manager()`, `create_protocol_classifier()`) and synchronous
  classes (`CaptureManager`, `PacketSniffer`).
- **Enhanced layer** (`core/`): async/streaming classes
  (`EnhancedLiveCapture`, `DemodulationEngine`, `EnhancedProtocolClassifier`,
  …), `async with` context managers, dataclass configs (`SDRConfig`,
  `DemodConfig`, …), and structured result objects (`detected`, `confidence`,
  `error_message`).

## Consequences

### Positive
- Progressive enhancement: start simple, adopt the async/typed layer as needs grow.
- Clear home for each style; the enhanced layer owns resource lifecycle via context managers.

### Negative / trade-offs
- Every feature must be exposed (or consciously not) at both levels — double the surface to maintain.

### Neutral / notes
- `EnhancedProtocolClassifier.classify()` operates on extracted packet bytes, not raw IQ.
