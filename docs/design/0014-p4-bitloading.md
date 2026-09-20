# P4 — Bit-loaded OFDM (adaptive modulation, Chow's algorithm) (Design)

**Design #:** 0014
**Phase:** P4

**Date:** 2026-09-20
**Status:** Accepted — implemented (unreleased).
**Author:** rbenzing (with Claude)
**Program:** P4 (adaptive modulation / bit-loading) — the first phase after the
seven-family FEC sheet (P3). Extends the OFDM PHY (`core/ofdm.py`, 802.11a-style)
with per-subcarrier adaptive QAM driven by a channel-aware loading algorithm.
**Predecessors:** P1 (OFDM full chain, design 0001), P2 (parametric profiles),
P3a–P3h (FEC). ADRs 0001–0017.
**Scope note:** Security-research / educational software. Lawful research /
authorized testing only.

---

## 1. Overview

### Goal

Add **bit-loaded OFDM**: instead of fixed QPSK on all 48 data subcarriers, assign
a per-subcarrier modulation order (**0/2/4/6 bits → null / QPSK / 16-QAM /
64-QAM**) from the per-subcarrier SNR, via **Chow's rate-adaptive loading
algorithm**, so strong subcarriers carry more bits and weak ones fewer or none.
The receiver learns the allocation from an **explicit signaled bit-map header**.
Demonstrated by higher correct **goodput** than fixed-QPSK OFDM on a
frequency-selective (multipath) channel.

### Why this is different from every prior phase

The P3 phases were FEC codecs on the `core.coding` `Codec` framework. Bit-loading
is an **OFDM PHY-layer modulation adaptation**, orthogonal to FEC. It does not
touch `core/coding.py`; it extends `core/ofdm.py` and adds `core/bitloading.py`.
It is a **closed-loop CSI feature** (the transmitter adapts to the channel), so
it is delivered as a core capability + a T&E demonstration rather than a
blind-resolvable link profile.

### What predecessors provide (build on)

- `core/ofdm.py`: `OFDMProfile` (64-FFT, 16-CP, 52 occupied = 48 data + 4 pilot),
  `qpsk_map`/`qpsk_demap`, `modulate_ofdm`, STF/LTF sync (`ofdm_sync_confidence`),
  **LTF least-squares channel estimation + one-tap equalization**
  (`ofdm_equalized_symbols`), `demodulate_ofdm`, `ofdm_soft_bits`,
  `_estimate_noise_var`.
- `validation.synth.channel.apply_channel` — **frequency-selective multipath**
  (`multipath_taps`) + CFO + calibrated AWGN, the channel that makes bit-loading
  show gain over fixed QPSK.
- QPSK convention (`qpsk_map`): Gray, per rail bit0→+1 / bit1→−1, scaled 1/√2.
  The new square-QAM **reduces to `qpsk_map` at order 2** (consistency).

### Scope realities (explicit)

- **Perfect CSI at TX.** The transmitter is given the channel frequency response
  it will transmit over (the standard idealization for demonstrating bit-loading
  potential). Real closed-loop CSI feedback / channel sounding is out of scope.
- **Uncoded.** P4 demonstrates the bit-loading mechanic itself; combining with
  the P3 FEC families (coded bit-loaded OFDM) is a future extension.
- **Explicit signaled map**, loud-on-failure. A corrupted bit-map header ⇒ wrong
  demap ⇒ the outer CRC-16 (`frame_with_crc`, when the payload is CRC-framed by a
  caller) or the demonstration's own exact-match check fails; a dedicated header
  CRC is a noted nice-to-have, not implemented.
- **No blind/pipeline/profile integration** this phase (closed-loop CSI does not
  fit the blind-waveform catalog). Possible future item.

### Non-goals

- Not water-filling / Levin-Campello / Hughes-Hartogs (Chow's only).
- Not margin-adaptive (rate-adaptive only).
- Not non-square / odd-bit constellations (BPSK/8-QAM/32-QAM); square QAM
  {0,2,4,6} only.
- Not soft-LLR output for the adaptive orders this phase (hard demap for the
  demonstration; per-order soft LLRs are a future extension for coded combining).
- Not CSI feedback, not adaptive power beyond the loading (equal power per used
  subcarrier at its assigned order).

---

## 2. `core/bitloading.py` — new module

### 2a. Square-QAM mapper/demapper (uniform machinery)
- `qam_map(bits, order) -> symbols` for `order ∈ {2,4,6}` (M = 2^order =
  4/16/64). Square QAM = two independent Gray-coded **√M-PAM** rails (I, Q), each
  carrying `order/2` bits. PAM levels are the odd integers `±1, ±3, …` Gray-
  mapped; the symbol is scaled to **unit average energy** (divide by
  `sqrt((2/3)(M-1))`). `order == 2` is exactly `qpsk_map` (1 bit/rail, ±1/√2).
- `qam_demap(symbols, order) -> bits`: per-rail nearest-PAM-level hard decision,
  inverse Gray, in the same interleaved I/Q bit order as `qam_map`.
- `bits_per_symbol(order) == order`; `order == 0` carries no bits (null carrier).
- A single `_gray_pam_levels(bits_per_rail)` / inverse builds all three orders
  from one code path (no per-order duplication).

### 2b. Per-subcarrier SNR from CSI
- `subcarrier_snr(h_freq, noise_var) -> snr_per_carrier`: `|H_k|^2 / noise_var`
  for each **data** subcarrier (`h_freq` = channel frequency response at the data
  bins; `noise_var` the per-subcarrier noise variance). Returns linear SNR array.

### 2c. Chow's rate-adaptive loading
- `chow_load(snr, target_ber, allowed_orders=(0,2,4,6), max_iter=...) ->
  allocation`: the classic Chow–Cioffi–Bingham algorithm.
  - SNR gap `Γ` from the target BER for QAM: `Γ = (1/3)·(erfcinv(target_ber/2))^2·2`
    (the standard gap approximation; pinned in implementation).
  - Tentative bits per carrier `b_k = log2(1 + snr_k / (Γ·10^(margin/10)))`;
    round to the nearest **allowed** even order (0/2/4/6), clamped to 6.
  - Iterate: count used carriers, recompute the system margin so the rounding is
    consistent, drop the weakest carriers to 0 when they cannot support order 2 at
    the target BER. Rate-adaptive: no fixed total-rate target — maximize Σ b_k
    subject to each used carrier meeting the target BER.
  - Returns an `int` array of length `len(data_carriers)`, entries in {0,2,4,6}.
- Deterministic given `(snr, target_ber)`.

## 3. `core/ofdm.py` extensions

### 3a. Bit-map header
- The allocation is 48 orders, each ∈ {0,2,4,6} → a 2-bit index {0,1,2,3} ↦
  {0,2,4,6}. 48 × 2 = **96 bits = exactly one fixed-QPSK OFDM data symbol** (48
  data carriers × 2 bits). `pack_allocation(allocation) -> 96 bits` /
  `unpack_allocation(bits) -> allocation`.

### 3b. Adaptive modulator
- `modulate_ofdm_loaded(payload_bits, allocation, profile=DEFAULT) -> iq`:
  1. STF + LTF preamble (unchanged).
  2. **Header symbol**: `qpsk_map(pack_allocation(allocation))` on the 48 data
     carriers (pilots as usual) — always QPSK, so the RX can read it before it
     knows the allocation.
  3. **Data symbols**: consume `payload_bits` across carriers per `allocation`
     (carrier k takes its `order_k` bits; nulls skipped), packing
     `Σ order_k` bits per OFDM symbol; last symbol zero-padded. Each carrier's
     bits → `qam_map(·, order_k)`; assemble with pilots; IFFT + CP (reuse
     `_data_symbol_time`).
  - Unit-average-power normalization consistent with `modulate_ofdm`.

### 3c. Adaptive demodulator
- `demodulate_ofdm_loaded(rx, profile=DEFAULT) -> payload_bits`: reuse the STF/LTF
  sync + channel estimate + one-tap equalize + pilot-CPE chain
  (`ofdm_equalized_symbols`); the **first** equalized data symbol is the header →
  `qpsk_demap` → `unpack_allocation` → allocation; each subsequent data symbol's
  carriers are demapped at their `order_k` (`qam_demap`), nulls skipped,
  concatenated → payload bits. Empty on sync failure (below `OFDM_SYNC_THRESHOLD`).

## 4. Demonstration (the gain metric)

`validation` test: on a **frequency-selective multipath channel** at a fixed SNR,
with **perfect CSI at TX** (compute `h_freq` at the data bins from the known
multipath taps; `noise_var` from the SNR), compare per burst:
- **fixed-QPSK OFDM** (`modulate_ofdm` / `demodulate_ofdm`): 96 payload bits/symbol,
  a fixed count, but errors on weak subcarriers.
- **bit-loaded OFDM** (`chow_load` → `modulate_ofdm_loaded` /
  `demodulate_ofdm_loaded`): variable bits/symbol, weak carriers nulled/QPSK,
  strong carriers 16/64-QAM.

Metric = **correct goodput** (payload bits delivered with zero bit errors, or
matching a fixed payload) at the target BER. Assert bit-loaded goodput >
fixed-QPSK goodput on the selective channel (bit-loading concentrates bits where
the channel supports them). Payload/channel/SNR pinned in implementation so the
margin is genuine (a small multipath-profile sweep backing the operating point,
per the FEC-phase precedent).

## 5. Testing strategy

1. **QAM round-trip**: `qam_map`→`qam_demap` recovers bits for orders 2/4/6;
   `order==2` equals `qpsk_map`/`qpsk_demap` exactly; unit average energy per order
   (mean |sym|² ≈ 1).
2. **QAM noisy demap**: at high SNR the hard demap is error-free; Gray coding gives
   single-bit errors for nearest-neighbor symbol errors.
3. **SNR + Chow's**: `subcarrier_snr` correct on a known `H`; `chow_load` assigns
   more bits to higher-SNR carriers, nulls carriers below order-2 capability,
   deterministic, all entries in {0,2,4,6}; total bits monotone in channel quality.
4. **Header pack/unpack**: `unpack_allocation(pack_allocation(a)) == a`; 96 bits.
5. **Loaded round-trip (noiseless)**: `modulate_ofdm_loaded`→`demodulate_ofdm_loaded`
   recovers the payload + the allocation exactly, over a flat channel.
6. **Loaded over a selective channel**: with perfect-CSI loading, recovers the
   payload at the target BER; a corrupted header ⇒ wrong demap ⇒ mismatch (loud).
7. **Throughput gain (headline)**: bit-loaded correct goodput > fixed-QPSK goodput
   on the frequency-selective channel (§4).
8. **Regression**: existing OFDM path (`modulate_ofdm`/`demodulate_ofdm`/
   `ofdm_soft_bits`) byte-for-byte unchanged; full P1–P3h suite green; `mypy
   validation` clean.

## 6. Global constraints (bind every task)

- `mypy validation` clean (no `Any`; `numpy.typing`); minimal diffs; lint touched
  test files (black/isort/flake8).
- complex128 in the OFDM PHY (matching `core/ofdm.py`); bits uint8; SNR/energy
  float64. Deterministic modulate given (payload, allocation).
- **Loud on failure**: a corrupted header / undecodable burst yields a mismatch,
  never a silently-wrong payload claimed correct.
- Additive: `core/coding.py` untouched; existing `core/ofdm.py` public functions
  (`modulate_ofdm`, `demodulate_ofdm`, `ofdm_equalized_symbols`, `ofdm_soft_bits`,
  `qpsk_map`, `qpsk_demap`) byte-for-byte unchanged — only NEW functions added.
- Branch `feature/p4-bitloading` (from `main`); no PR/tag this phase (merge/version
  autonomous on a green closing gate, per the established pattern).
- Stall guard: implementers run only targeted test files; controller runs the
  full-suite closing gate; never background a slow run silently.

## 7. Out of scope / future

- Water-filling / Levin-Campello / Hughes-Hartogs; margin-adaptive loading;
  non-square constellations; adaptive power (beyond per-order); soft-LLR adaptive
  demap; coded bit-loaded OFDM (P4 × P3 FEC); CSI feedback / sounding; blind /
  pipeline / profile integration; a dedicated header CRC.
- Docs on completion: mark this spec Accepted (implemented); add **ADR-0018**
  (bit-loaded OFDM — Chow's rate-adaptive loading, square-QAM {0,2,4,6}, explicit
  signaled map, perfect-CSI-at-TX demonstration scope); update ADR + design
  indices, README/CLAUDE module maps (add `core/bitloading.py`; note the adaptive
  OFDM path).
