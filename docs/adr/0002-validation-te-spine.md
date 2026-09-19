# ADR-0002: Validation & T&E spine as the correctness backbone

- **Status:** Accepted
- **Date:** 2026-09-19
- **Deciders:** rbenzing (with Claude)

## Context

Signal-processing and coding claims ("this demod works", "this code gives
gain") are easy to assert and hard to trust. The project needs correctness that
is **measured** against ground truth, reproducibly, rather than asserted — and
the same backbone must cover both the detect→classify pipeline and the later
signal/coding chain.

## Decision

We treat the `validation/` package as the empirical Test & Evaluation (T&E)
spine and the primary correctness authority:

- Synthetic signal generator + calibrated channel model (`synth/`), real-capture
  ingestion (`ingest/`), and a unified SigMF-backed `LabeledDataset`.
- An injectable detect→demod→classify `pipeline.py`.
- Metrics with bootstrap confidence intervals (`metrics.py`) — Pd/Pfa, ROC,
  accuracy-vs-SNR, and the coded-link (coded BER/FER) metric.
- Evaluation `harness.py`, JSON `report.py`, and a reproducibility manifest
  (`repro.py`), driven by `dronecmd validate`.

Every new signal or coding capability lands with a synthetic generation path and
a metric that exercises it end-to-end.

## Consequences

### Positive
- Capabilities are demonstrated by measurement (e.g. coding gain is a real Monte-Carlo crossover), not by hand-wavy tests.
- Reproducible: the repro manifest pins seeds/config.

### Negative / trade-offs
- Building the synth path + metric for each feature is upfront work before the feature "counts as done".

### Neutral / notes
- `validation/` is the CI-enforced strict-`mypy` surface (see ADR-0003).
