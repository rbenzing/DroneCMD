# ADR-0018: Bit-loaded OFDM — adaptive square-QAM via Chow's rate-adaptive loading

- **Status:** Accepted
- **Date:** 2026-09-20
- **Deciders:** rbenzing (with Claude)

## Context

P4 adds **bit-loaded OFDM** (`core/bitloading.py`; `core/ofdm.py` extensions),
the first phase after the seven-family FEC sheet (P3a–P3h, ADR-0005/0007/0008/
0013/0014/0016/0017) closed out. Every prior OFDM burst (`modulate_ofdm`/
`demodulate_ofdm`) used fixed QPSK on all 48 data subcarriers regardless of
channel quality — wasteful on a frequency-selective (multipath) channel, where
some subcarriers can reliably carry far more than 2 bits/symbol and others
can carry none. Bit-loading assigns each data subcarrier a modulation order
from its individual SNR so strong subcarriers carry more bits and weak ones
fewer or none, raising delivered throughput without changing transmit power
or bandwidth.

This is a different kind of feature from every P3 family: the P3 codecs are
FEC codecs on the `core.coding` `Codec` framework, corrector algorithms
applied after a fixed modulation. Bit-loading is an **OFDM PHY-layer
modulation adaptation**, orthogonal to FEC, and — because the transmitter
needs to know the channel before it transmits — a **closed-loop CSI feature**,
unlike the blind-resolvable link profiles P2/P3 built around. Four things had
to be decided: the constellation family, the loading algorithm, how the
receiver learns the per-carrier allocation, and the scope of the
demonstration.

## Decision

- **Adaptive square-QAM, orders `{0, 2, 4, 6}` bits/symbol** (null / QPSK /
  16-QAM / 64-QAM) per data subcarrier, no other orders. Square QAM is built
  as two independent Gray-coded √M-PAM rails (I, Q) at unit average energy
  from one shared code path (`_gray_pam_levels`-style rail construction, not
  per-order duplication); order 2 reduces **exactly** to the existing
  `core.ofdm.qpsk_map`/`qpsk_demap` convention (bit0→+1 per rail, 1/√2
  scaling), so a fully-loaded QPSK-only allocation is byte-identical to the
  fixed-QPSK path. Non-square orders (BPSK, 8-QAM, 32-QAM) are out of scope —
  the standard bit-loading constellation ladder is square QAM at even orders.
- **Chow's rate-adaptive loading** (`chow_load`), not water-filling,
  Levin-Campello, or Hughes-Hartogs. Each subcarrier's SNR
  (`subcarrier_snr(h_freq, noise_var) = |H_k|^2 / noise_var`) is compared
  against the **order-o feasibility SNR** `Γ·(2^o − 1)`, where the **SNR gap
  Γ** is derived from the target BER for uncoded square QAM
  (`Γ = (1/3)·[Q^{-1}(target_ber/4)]^2`, the standard gap approximation,
  pinned in `_snr_gap`). A carrier is assigned the largest allowed order whose
  feasibility SNR it clears; carriers that cannot support even order 2 at the
  target BER are nulled (order 0). This is **rate-adaptive** (maximize total
  bits subject to each used carrier meeting the target BER), not
  margin-adaptive (no fixed total-rate target), and deterministic given
  `(snr, target_ber)`.
- **Explicit signaled bit-map header**, not blind/implicit allocation
  discovery. The 48-entry allocation (each entry a 2-bit index over
  `{0,2,4,6}`) packs into exactly 96 bits — **one fixed-QPSK OFDM data
  symbol** (`pack_allocation`/`unpack_allocation`) — always sent at QPSK so
  the receiver can read the header before it knows the allocation. This adds
  one full OFDM symbol of overhead per burst but needs no side channel, no
  receiver-side blind estimation of the allocation, and reuses the existing
  QPSK map/demap path unchanged.
- **Perfect-CSI-at-TX, uncoded demonstration scope.** The transmitter is
  given the channel frequency response it will transmit over
  (`data_channel_response`, computed from the known multipath taps at the
  data-subcarrier FFT bins — the standard idealization for demonstrating
  bit-loading potential), and the demonstration is uncoded (no FEC combined
  with the adaptive modulation this phase). Real closed-loop CSI feedback /
  channel sounding, and coded bit-loaded OFDM, are both out of scope.
- **Orthogonal to FEC: `core/coding.py` untouched.** Bit-loading is additive
  to `core/ofdm.py` (`pack_allocation`, `unpack_allocation`,
  `modulate_ofdm_loaded`, `demodulate_ofdm_loaded`, `data_channel_response`)
  and a new `core/bitloading.py` module; the existing `modulate_ofdm`/
  `demodulate_ofdm`/`ofdm_equalized_symbols`/`ofdm_soft_bits`/`qpsk_map`/
  `qpsk_demap` public functions are byte-for-byte unchanged, and no FEC
  catalog family, coded-modulation path, or codec registry entry is touched.
- **No blind/pipeline/profile integration this phase.** The parametric
  link-profile catalog and blind profile resolution (`core/profiles.py`,
  `core/blind.py`, ADR-0004) are built around receiver-side blind discovery
  of a fixed waveform; a closed-loop, TX-adapts-to-channel feature does not
  fit that model. Bit-loading is delivered as a core capability plus a T&E
  throughput demonstration, not a new catalog profile. Future pipeline/CSI-
  feedback integration is a possible follow-on, not part of this decision.

## Consequences

### Positive
- Concentrates bits where the channel actually supports them: on a
  frequency-selective multipath channel, bit-loaded OFDM delivers
  substantially higher correct **goodput** than fixed-QPSK OFDM at the same
  SNR (~2x measured on the demonstration's multipath profile at 16 dB SNR;
  see `tests/validation/test_ofdm_loaded.py`,
  `test_bitloading_beats_fixed_qpsk_goodput_selective_channel`) — the direct
  payoff of adaptive modulation over a fixed constellation.
- Fully additive: zero risk to the existing fixed-QPSK OFDM path or any P3
  FEC family; `core/coding.py`, the catalog, and blind resolution are
  untouched, so nothing already shipped changes behavior.
- Order-2 reducing exactly to `qpsk_map`/`qpsk_demap` means the new
  square-QAM machinery is validated against, and consistent with, the
  existing QPSK convention rather than introducing a second one.
- The signaled-header approach needs no new side channel or receiver-side
  guesswork: the receiver demodulates the header with the same QPSK path it
  already has, before it can demodulate anything else.

### Negative / trade-offs
- **Perfect CSI at TX** is an idealization — real systems need channel
  sounding and feedback, which is not modeled here; the measured gain is an
  upper bound on what a real closed-loop system would achieve net of
  feedback overhead and staleness.
- **No header CRC.** A corrupted allocation header silently produces a wrong
  per-carrier demap; this is still loud-on-failure in practice (the outer
  CRC-16, when the payload is CRC-framed by a caller, or the demonstration's
  own exact-match check, catches it), but there is no dedicated header
  integrity check flagging *which* symbol failed. Noted as a nice-to-have,
  not implemented.
- **Uncoded, no coded combining.** This phase does not combine bit-loading
  with any P3 FEC family; a coded bit-loaded link (soft-LLR adaptive demap
  feeding a codec) is a real future capability, not delivered here.
- **Header overhead.** One fixed-QPSK OFDM symbol per burst is pure
  overhead; for the demonstration's ~20-symbol bursts this is roughly 4.8%
  of the burst (1 header symbol in ~21), non-negligible for very short
  bursts.
- **Rate-adaptive only, square QAM only.** No margin-adaptive loading, no
  water-filling/Levin-Campello/Hughes-Hartogs, no non-square constellations,
  no adaptive power beyond the per-order assignment (equal power per used
  subcarrier at its assigned order) — a narrower ladder than a
  production adaptive-modulation system would offer.

### Neutral / notes
- Verified: `qam_map`/`qam_demap` round-trip exactly at orders 2/4/6 with
  order 2 identical to `qpsk_map`/`qpsk_demap`; unit average energy per order;
  `subcarrier_snr` correct against a known `H`; `chow_load` deterministic,
  monotone (more bits on higher-SNR carriers), all entries in `{0,2,4,6}`;
  `unpack_allocation(pack_allocation(a)) == a` (96 bits); noiseless
  `modulate_ofdm_loaded`→`demodulate_ofdm_loaded` round-trip recovers both
  payload and allocation exactly over a flat channel; the headline throughput
  test shows bit-loaded goodput exceeding fixed-QPSK goodput on the
  frequency-selective channel at the pinned 16 dB operating point; the
  existing `modulate_ofdm`/`demodulate_ofdm`/`ofdm_soft_bits` path is
  unchanged (full regression green); `mypy validation` clean. Cross-refs:
  design doc `docs/design/0014-p4-bitloading.md`; ADR-0001 (two-layer API —
  bit-loading lives in the enhanced `core/` layer); the P1 OFDM chain that
  `core/ofdm.py` extends; ADR-0004 (parametric profiles + blind resolution —
  why bit-loading is explicitly *not* integrated there this phase); ADR-0006
  (soft-LLR/loud-on-failure — bit-loading's demonstration is hard-decision
  only, no soft-LLR adaptive demap this phase).
