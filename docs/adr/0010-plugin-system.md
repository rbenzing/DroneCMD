# ADR-0010: Plugin system for protocol extensibility

- **Status:** Accepted
- **Date:** 2026-09-19
- **Deciders:** rbenzing (with Claude)

## Context

Drone protocols are many and manufacturer-specific (DJI, Parrot, …), and third
parties need to add their own without forking the core. A hardcoded protocol
list would not scale and would couple protocol support to core releases.

## Decision

Protocol support is provided by **discoverable plugins**:

- A plugin declares a `PluginType` (PROTOCOL, INJECTION, ANALYSIS, DECODER,
  ENCODER) and `PluginCapability` (DETECT, DECODE, ENCODE, INJECT, ANALYZE,
  VALIDATE) via a `PluginMetadata` dataclass.
- Authors subclass **`BaseProtocolPlugin`** (the convenience base with default
  `metadata`/`initialize()`/`validate()`) — the fully abstract `ProtocolPlugin`
  is also available — and implement `get_name()`, `get_version()`,
  `get_supported_protocols()`, `detect()` (→ `ProtocolDetectionResult`), and
  `parse_packet()` (→ `ProtocolParseResult`).
- Plugins register via `pyproject.toml`
  `[project.entry-points."dronecmd.plugins"]`; `registry.py` discovers them.
  `plugins/protocols/_template.py` is the starting point.

## Consequences

### Positive
- New protocols ship without core changes; discovery is automatic via entry points.

### Negative / trade-offs
- The metadata/capability contract must stay stable, or every plugin breaks.

### Neutral / notes
- Bundled plugins (DJI, Parrot, generic) use `BaseProtocolPlugin` and serve as references.
