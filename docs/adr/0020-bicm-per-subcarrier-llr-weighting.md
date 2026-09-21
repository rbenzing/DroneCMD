# ADR-0020: Coded bit-loaded OFDM via BICM with per-subcarrier LLR weighting

- **Status:** Accepted
- **Date:** 2026-09-21
- **Deciders:** rbenzing

## Context

Two capabilities existed but had never been combined: adaptive bit-loaded OFDM
(per-subcarrier square-QAM {QPSK/16-QAM/64-QAM} via Chow loading, `core/bitloading.py`
+ `core/ofdm.py`) and the seven-family channel-coding framework with soft-decision
decoders (`core/coding.py`). The bit-loaded RX emitted only hard bits and there was
no soft square-QAM demapper, so FEC could not be run over the loaded subcarriers.

Combining them requires soft-decision demapping, and on a frequency-selective
channel each subcarrier carries a *different* QAM order and a different SNR. The
repo-wide LLR convention is `L>0 => bit 0`; the min-sum/max-log decoders are
scale-invariant to a **global** LLR factor (ADR-0013/0014/0016) but not to the
**relative** weighting between subcarriers.

## Decision

We will implement coded bit-loaded OFDM as **bit-interleaved coded modulation
(BICM)**, registry-generic so any soft-input codec works: TX is
`CRC -> encode -> interleave -> bit-loaded QAM map`; RX is
`per-subcarrier soft-LLR demap -> deinterleave -> soft FEC decode -> CRC`.

The soft square-QAM demapper (`core.bitloading.qam_soft_demap`) emits max-log
per-bit LLRs derived from the same Gray/normalization construction as `qam_map`,
and **weights each subcarrier's LLRs by its reliability `|h_k|^2 / N0`**. After
one-tap equalization the effective noise variance on subcarrier `k` is `N0/|h_k|^2`,
so `|h_k|^2/N0` is the correct max-log scale. `|h_k|^2` comes from the equalizer's
existing LTF estimate (newly exposed by `ofdm_equalized_symbols_csi`); `N0` is a
scalar estimated from pilot residuals. This **relative per-subcarrier weighting is
kept, not normalized away** — it is exactly what lets the decoder lean on strong
subcarriers and protect weak ones, which is the entire point of coding over an
adaptively-loaded channel.

We reject **trellis-coded / multilevel coded modulation (TCM/MLC)**: higher
asymptotic performance but a large departure from the BICM/registry design and far
more complexity, for no need in this phase (YAGNI).

Framing: because `modulate_ofdm_loaded` zero-pads the payload to a subcarrier
boundary, the coded stream is prefixed with a 32-bit length header (repeated 3× and
majority-voted, since it rides the uncoded bit-loaded carriers) so the RX strips the
padding before decoding.

## Consequences

### Positive
- FEC + bit-loading together: coding gain *and* spectral efficiency on
  frequency-selective channels. Measured (`test_coded_ofdm_gain.py`): coded
  bit-loaded 23/30 vs uncoded 0/30 at a 2 dB CSI deficit, ~1.8× the bits/symbol of
  coded fixed-QPSK.
- Works with every soft-input codec unchanged (convolutional/RS/LDPC/turbo/polar).
- The soft square-QAM demapper is reusable beyond OFDM.

### Negative / trade-offs
- The length prefix is uncoded (repetition-protected, not FEC-protected); at very
  low SNR a mis-read length fails the frame loudly (CRC), rather than degrading
  gracefully.
- At a CSI deficit, adaptive loading trades a little reliability for efficiency
  (higher-order QAM on carriers) vs. fixed QPSK — the benefit is goodput, not FER.

### Neutral / notes
- Scope: the core capability + validation. Wiring a coded/soft branch into
  `validation/pipeline.py::ofdm_region_to_bytes` (today uncoded/hard) and blind
  resolution of coded bit-loaded bursts are deferred (design 0015 §6).
- Related: design [0015](../design/0015-coded-bit-loaded-ofdm.md), ADR-0005
  (coding framework), ADR-0006 (soft-LLR/loud-on-failure), ADR-0013/0014/0016
  (decoder scale-invariance).
