# 0015 — Coded bit-loaded OFDM (P4×P3, BICM)

- **Status:** Proposed
- **Phase:** P5 (combines P4 bit-loaded OFDM with the P3 FEC framework)
- **Depends on:** [0014 P4 bit-loaded OFDM](0014-p4-bitloading.md), the channel-coding
  framework (ADR-0005) and the soft-LLR/loud-on-failure contract (ADR-0006).

## 1. Context & goal

Two capabilities exist but have never been combined:

- **Bit-loaded OFDM** (P4): adaptive per-subcarrier square-QAM {QPSK/16-QAM/64-QAM}
  via Chow loading on per-subcarrier CSI (`core/bitloading.py`, `core/ofdm.py`
  `modulate_ofdm_loaded`/`demodulate_ofdm_loaded`). It raises goodput on
  frequency-selective channels but is **uncoded** and its RX emits **hard bits**.
- **Channel coding** (P3): a 7-family FEC framework with soft-decision decoders
  (`core/coding.py`; convolutional/RS/LDPC/turbo/polar accept LLRs), CRC framing,
  and a block interleaver.

The goal is **coded bit-loaded OFDM**: run FEC over the adaptively-loaded
subcarriers with **soft-decision** demapping, so the link gets coding gain *and*
spectral efficiency on a frequency-selective channel — the natural culmination of
P3 and P4.

The design is **bit-interleaved coded modulation (BICM)**: encode → interleave →
adaptively map to per-subcarrier QAM; on RX, per-subcarrier **soft-LLR** demap →
deinterleave → soft FEC decode. BICM is the standard adaptive-OFDM construction
(802.11/LTE/DVB) and plugs directly into the existing `Codec` registry.

## 2. Approach (decision + alternatives)

**Chosen: BICM at the Codec-registry level (generic).** The only genuinely new
DSP is a soft square-QAM demapper; everything else reuses the existing encode /
interleave / soft-decode / CRC machinery, so *any* soft-input codec
(`conv_*`, `rs_*`, `ldpc_*`, `turbo_*`, `polar_*`) works unchanged.

Alternatives considered and rejected:
- **LDPC-specific coded OFDM (802.11-style).** Simpler but throws away the
  7-family framework; no reason to special-case one codec. Rejected (YAGNI).
- **Trellis-coded / multilevel coded modulation (TCM/MLC).** Higher asymptotic
  performance by jointly designing code + constellation, but a large departure
  from the BICM/registry design and far more complex. Rejected for this phase.

## 3. Components (exact signatures)

All bit arrays are `Bits = NDArray[uint8]` (MSB-first); all soft values are
`LLRs = NDArray[float64]` with the repo-wide convention **`L>0 ⇒ bit 0`**
(`L = log P(bit=0)/P(bit=1)`).

### 3a. Soft square-QAM demapper — `core/bitloading.py`

```python
def qam_soft_demap(symbols: Complex, order: int, weight: Real | float = 1.0) -> LLRs
```

- Inverse of `qam_map`: emits `order` LLRs per symbol (`L>0 ⇒ bit 0`), matching
  `qam_map`'s Gray/MSB rail layout and unit-average-energy normalization exactly.
- `order ∈ {2,4,6}`; `weight` is the per-symbol reliability `|h_k|²/N0` (scalar or
  a per-symbol `Real` array broadcast over symbols).
- **Max-log LLR**, computed per Gray-coded √M-PAM rail (I then Q, `order//2` bits
  each): for received rail value `r` and each bit position,
  `LLR = weight · (min_{levels: bit=1}(r−ℓ)² − min_{levels: bit=0}(r−ℓ)²)`.
  Sign check: when bit 0 is nearer, `min_{bit=0}` is smaller ⇒ `LLR>0 ⇒ bit 0`. ✓
- Vectorized over the input symbol array; the PAM level table + Gray labels are
  derived from the same construction as `qam_map` so the two cannot drift.

**Test pin:** hard-slicing the soft output (`llr < 0`) reproduces `qam_demap` on
noiseless input, for every order and every bit pattern.

### 3b. Per-subcarrier CSI exposure — `core/ofdm.py`

```python
def ofdm_equalized_symbols_csi(
    rx: Complex, profile: OFDMProfile = DEFAULT_OFDM_PROFILE
) -> tuple[Complex, Real, float]
    # -> (equalized_data_symbols, per_data_carrier_gain_sq |h_k|², noise_var N0)
```

- Additive sibling of `ofdm_equalized_symbols` (which is left unchanged and
  refactored to delegate to a shared core). Returns, in addition to the equalized
  data symbols: the per-data-carrier squared channel gain `|h_k|²` (from the
  existing LTF LS estimate `h[data_bins]`, tiled per data symbol) and a **scalar**
  noise variance `N0`.
- **`N0` from pilot residuals:** `N0 = mean_{k∈pilots} |h_k|² · |y_eq,k − pilot_k|²`.
  Pilots are known, so this is the pre-equalization noise power; `|h_k|²` then gives
  each data carrier its own reliability `weight_k = |h_k|²/N0`.
- Empty/short input returns `(empty, empty, nan)` and callers treat it as a sync
  failure (loud-on-failure, per ADR-0006).

### 3c. Soft loaded demodulator — `core/ofdm.py`

```python
def demodulate_ofdm_loaded_soft(
    rx: Complex, profile: OFDMProfile = DEFAULT_OFDM_PROFILE
) -> LLRs
```

- Soft counterpart of `demodulate_ofdm_loaded`. Runs `ofdm_equalized_symbols_csi`,
  hard-demaps the fixed-QPSK header symbol to recover the allocation
  (`unpack_allocation`), then for every data symbol × data carrier at its allocated
  order calls `qam_soft_demap(sym_k, order_k, weight=|h_k|²/N0)`, concatenating LLRs
  in the exact bit order `modulate_ofdm_loaded` packs (order-0 carriers contribute
  nothing). Returns the full LLR stream.

### 3d. Coded TX/RX composition — `core/ofdm.py`

```python
def modulate_coded_ofdm_loaded(
    data: bytes, allocation: NDArray[intp], coding: CodingSpec,
    profile: OFDMProfile = DEFAULT_OFDM_PROFILE,
) -> Complex

def decode_coded_ofdm_loaded(
    rx: Complex, coding: CodingSpec, profile: OFDMProfile = DEFAULT_OFDM_PROFILE,
) -> tuple[bytes, bool]   # (payload, crc_ok)
```

- **TX** reuses the T&E convention: `unpackbits(data) → frame_with_crc → make_codec(coding).encode
  → interleave(depth 8) → modulate_ofdm_loaded(coded_bits, allocation, profile)`.
- **RX**: `demodulate_ofdm_loaded_soft` (or the hard path for hard-only codecs) →
  `deinterleave(depth 8)` → `make_codec(coding).decode` → `check_and_strip_crc`.
  Soft vs hard is chosen by `coding.soft_input` exactly as
  `single_carrier_region_to_bytes` does; `crc_ok=False` returns empty (loud failure).
- `allocation` is computed by the caller from CSI
  (`chow_load(subcarrier_snr(data_channel_response(taps, profile), N0), target_ber)`),
  mirroring the existing uncoded loaded path (perfect CSI at TX).

## 4. The per-subcarrier weighting crux (scale-invariance)

Each subcarrier carries a different QAM order *and* SNR. After one-tap
equalization `y_eq,k = s_k + n_k/h_k`, the effective noise variance is `N0/|h_k|²`,
so the correct LLR reliability weight is `|h_k|²/N0`. The framework's decoders are
scale-invariant to a **global** LLR factor (ADR-0013/0014/0016), but **not** to the
**relative per-carrier** weighting — that relative weighting is exactly what lets
FEC lean on strong subcarriers and protect weak ones. Therefore the demapper keeps
`|h_k|²/N0` per carrier and never normalizes it away. This is recorded as an ADR.

## 5. Validation plan

A `@pytest.mark` (non-slow) test on a frequency-selective channel (a fixed
multi-tap FIR + AWGN at a chosen SNR, via the existing `validation.synth.channel`)
showing, over several trials, that **coded bit-loaded OFDM** delivers a lower coded
FER (via the existing coded-link metric) than **both**:
1. **uncoded** bit-loaded OFDM (same allocation) — isolates the coding gain, and
2. **coded fixed-QPSK** OFDM (same codec) — isolates the bit-loading gain.

At least one soft codec (e.g. `conv_k7_r12` or `ldpc_648_r12`) is exercised
end-to-end through `modulate_coded_ofdm_loaded` → channel → `decode_coded_ofdm_loaded`
with `crc_ok` asserted at the operating SNR. A round-trip noiseless test pins exact
recovery. The soft-demap↔hard-demap consistency test (3a) guards the LLR signs.

## 6. Scope & non-goals

**In scope:** 3a–3d + validation — the end-to-end coded bit-loaded OFDM capability
with soft LLRs, provably beating its uncoded and fixed-QPSK baselines.

**Deferred (phase 2, not this design):** wiring a coded+soft branch into
`validation/pipeline.py::ofdm_region_to_bytes` (today uncoded/hard) and blind
resolution of coded bit-loaded bursts. The core capability lands first; the blind
pipeline integration is a separate increment.

## 7. Risks

- **LLR sign/scale drift.** Mitigated by deriving the demapper's PAM/Gray table
  from the same construction as `qam_map` and the hard/soft consistency test.
- **Noise estimate bias** from few pilots (4 in the default profile) at low SNR.
  `N0` is a scalar global scale; the decoders' global scale-invariance makes the
  result robust to modest `N0` error — only the *relative* `|h_k|²` must be right.
- **Header mis-decode** cascades (wrong allocation ⇒ garbage LLRs). Same exposure as
  the existing uncoded loaded path; the fixed-QPSK header is the most robust symbol
  and CRC catches the failure loudly.

## 8. ADRs to record

- **Per-subcarrier LLR reliability weighting** (`|h_k|²/N0`) for BICM bit-loading,
  and why relative per-carrier weighting is kept despite global scale-invariance.
- **BICM over TCM/multilevel** for coded adaptive OFDM (registry-generic).
