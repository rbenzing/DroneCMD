# P3a — Channel-Coding Framework (Design)

**Design #:** 0006
**Phase:** P3a

**Status:** Accepted — implemented, shipped in v0.3.0
**Date:** 2026-09-18
**Author:** rbenzing (with Claude)
**Program:** P3 (channel coding + interleaving), decomposed into sub-phases P3a…P3h. **This spec covers P3a only** — the shared foundation every codec plugs into.
**Predecessors:** P1 (OFDM), PE (SC receivers), PH (CFO/phase), P2-SC + P2-OFDM (profile registry + blind resolution + profile-ID metric).
**Scope note:** Security-research / educational software. All work is for lawful security research and authorized testing only.

---

## 1. Overview

### Goal

Build the **channel-coding framework** that all seven FEC families (LDPC, convolutional, turbo, polar, Reed-Solomon, BCH, fountain) will plug into in later sub-phases: a coding registry modeling every family's capabilities and parameters behind a common `Codec` interface, a shared interleaver, CRC framing, a **soft-output (LLR) demod path** for the SC and OFDM chains, a **coded-BER / FER metric**, and integration with the P2 link-profile system (coding is carried by the resolved profile). P3a ships **working codecs only for `uncoded` and `repetition`** — enough to validate the whole encode→interleave→modulate→channel→soft-demod→deinterleave→decode→CRC chain end-to-end and demonstrate real coding gain; the other six families are registered as capability descriptors and raise `NotImplementedError` until their own sub-phases.

### The P3 program decomposition (context; only P3a is built here)

P3a (framework) → P3b (convolutional+Viterbi) → P3c (Reed-Solomon) → P3d (BCH) → P3e (LDPC) → P3f (turbo) → P3g (polar) → P3h (fountain/rateless). Framework first (unblocks all); convolutional before turbo (turbo's constituents); RS adjacent to BCH (shared GF/Berlekamp-Massey); fountain last (rateless breaks the fixed-(n,k) assumption). All land on `feature/ofdm-full-chain`; one final PR + one tag at the very end of the whole effort.

### Non-goals (P3a)

- **No heavy decoders.** Only `uncoded` and `repetition` codecs are implemented. `make_codec` for the other six families raises `NotImplementedError` with the sub-phase name.
- **No blind coding detection.** The code is carried by the resolved link profile (P2 resolves the profile → the profile declares its code). Trial-decode blind coding detection is out of scope (a possible later extension).
- **No change to existing P2 profiles' behavior.** Coding is additive and opt-in; every existing profile defaults to uncoded.
- **No new modulation.** The soft path is additive alongside the existing hard-decision demods; it does not alter them.

---

## 2. Background — where coding sits and what enables it

Today the synth/validation chain carries **raw uncoded bits**: `payload bytes → _bits_from_bytes → modulate → PHY → hard-bit demod → bytes`. There is no coding, interleaving, or CRC in this path (the CRC in `core/parsing.py` is only for parsing real MAVLink packets).

Two facts shape P3a:

1. **Soft information is the foundational enabler.** Four of the seven families (convolutional/Viterbi, turbo, LDPC, polar) only achieve real coding gain from **soft demodulator output (log-likelihood ratios)**, not hard bits. The current demods emit hard bits. So the soft-output/LLR path is the single most cross-cutting piece and belongs in the foundation. RS and BCH are algebraic/hard-decision; fountain is rateless.

2. **P2 already produces the equalized symbols LLRs need.** `core.ofdm.ofdm_equalized_symbols(rx, profile)` returns the equalized OFDM data subcarriers, and `core.single_carrier.sc_aligned_payload_centers(rx, profile)` returns the aligned SC payload symbol centers. The soft path computes per-bit LLRs from exactly these, so P3a builds on P2's work rather than re-deriving equalization.

---

## 3. Data model & profile integration (`core/coding.py`)

New module `core/coding.py` (depends only on numpy + the existing `core` modules it wraps):

```python
class CodeFamily(Enum):
    UNCODED = "uncoded"
    REPETITION = "repetition"
    CONVOLUTIONAL = "convolutional"
    REED_SOLOMON = "reed_solomon"
    BCH = "bch"
    LDPC = "ldpc"
    TURBO = "turbo"
    POLAR = "polar"
    FOUNTAIN = "fountain"

@dataclass(frozen=True)
class CodingSpec:
    """A named code: family + rate + family-specific parameters."""
    name: str
    family: CodeFamily
    k: int                       # info bits per block (0 = rateless/streaming, e.g. fountain)
    n: int                       # coded bits per block (0 = rateless)
    params: Mapping[str, object]  # family-specific, e.g. constraint length + generator
                                  # polynomials (convolutional/turbo); GF field m + t
                                  # (RS/BCH); degree distribution / H handle (LDPC);
                                  # list size (polar); soft-input flag. (Exact per-family
                                  # param schemas are pinned when each codec's sub-phase lands.)
    @property
    def rate(self) -> float: ...   # k/n, or NaN for rateless
    @property
    def soft_input(self) -> bool: ...  # does decode() consume LLRs
```

- **`DecodeResult`** dataclass: `bits: Bits`, `crc_ok: bool`, `meta: Mapping` (iterations, list index, etc.). A failed CRC yields `crc_ok=False` and the framework treats it as a loud decode failure.
- **`Codec` protocol:** `encode(info_bits: Bits) -> Bits` and `decode(soft_or_hard) -> DecodeResult`. Soft codecs consume LLRs (`float64`); hard codecs consume hard bits (`uint8`). `CodingSpec.soft_input` disambiguates.
- **`CODING_CATALOG: Dict[str, CodingSpec]`** — a registry parallel to the P2 profile catalog. **P3a populates capability descriptors for all seven families** (representative real-world parameterizations: e.g. `conv_k7_r12` = 802.11 rate-1/2 K=7 generators (133,171)₈; `rs_255_223`; `bch_...`; `ldpc_...`; `turbo_...`; `polar_...`; `fountain_lt_...`) so the catalog is a complete capability sheet — but `make_codec` only builds `uncoded`/`repetition`.
- **`make_codec(spec: CodingSpec) -> Codec`** factory: dispatches on `family`; `UNCODED`/`REPETITION` return working codecs; the other six raise `NotImplementedError(f"{family.value}: implemented in sub-phase P3x")`.
- **Helpers** mirroring `core.profiles`: `coding_names()`, and validation that a profile's declared coding name exists in the catalog.

**Profile integration:** `core.profiles.SCProfileSpec` and the OFDM catalog entries gain an **optional** `coding: Optional[str] = None` (a `CODING_CATALOG` key). `None` = uncoded → **every existing P2 profile is byte-for-byte unchanged**. Blind resolution is untouched: P2 resolves the profile, and the profile's `coding` field names the code. (P3a assigns no real codes to existing profiles — only `uncoded`/`repetition`-coded profiles can exist until the codecs land; a demo `repetition`-coded profile may be added to exercise the metric.)

---

## 4. Soft-output (LLR) demod path

Additive soft-decision functions, alongside (never replacing) the existing hard-bit demods:

- `core.single_carrier.sc_soft_bits(rx, profile, *, noise_var=None) -> LLRs`
- `core.ofdm.ofdm_soft_bits(rx, profile, *, noise_var=None) -> LLRs`

where `LLRs = npt.NDArray[np.float64]`, one LLR per coded bit, `L(b) = log[P(b=0|y) / P(b=1|y)]` (so `L>0 ⇒ bit 0`, matching the existing hard convention `real<0 ⇒ bit 1`).

- **Symbol source:** reuse `ofdm_equalized_symbols` (OFDM) and `sc_aligned_payload_centers` (SC) — the equalized/aligned symbols P2 already produces. Empty source → empty LLRs (loud).
- **LLR computation:** Gray-mapped QPSK/BPSK → per-rail max-log LLR `L(rail) = 2·y_rail / σ²` (I rail = even bits, Q rail = odd, matching `qpsk_map`/`qpsk_demap`). FSK → per-bit LLR from the two-hypothesis metric difference. Max-log approximation (exact for QPSK in AWGN).
- **Noise-variance estimate `σ²`:** when `noise_var` is not supplied, estimate it at the receiver from the post-equalization residual to the nearest constellation point (the same EVM machinery `_ofdm_data_evm` uses) or the LTF/pilot residual. Calibrated LLRs matter for the iterative decoders (LDPC/turbo/polar) that arrive later; P3a delivers the estimator and validates it.
- **Consistency invariant:** `sign(sc_soft_bits) / sign(ofdm_soft_bits)` equals the hard-decision bits from the existing demods (a soft/hard agreement test), and `|LLR|` grows with SNR. The hard paths are unchanged.

Since P3a's only codecs are hard-input, nothing *consumes* the LLRs yet; they are validated by property tests and are ready for P3b's soft Viterbi.

---

## 5. Framing: CRC + interleaver + loud-failure / FER

**CRC (recommend CRC-16-CCITT, poly 0x1021):** frame = `payload_bits ‖ CRC16(payload)`. Cheap, standard for short drone frames. After decode, recompute the CRC over the recovered payload; **mismatch ⇒ loud decode failure** (`crc_ok=False` ⇒ the region decodes to `b""`). This defines coded loud-on-failure and the FER numerator.

**Block interleaver (shared, code-agnostic):** `interleave(bits, depth) / deinterleave(bits, depth)` — a rectangular block interleaver (write rows, read columns) with a `depth=0` identity option. Applied to coded bits before modulation and reversed after the soft-demod, to break the channel's burst errors before decode. Deinterleaving operates on LLRs as well as hard bits (permutation is value-type-agnostic).

**Transmit chain:** `payload bytes → payload bits → ‖CRC → codec.encode → interleave → symbol map (modulate)`.
**Receive chain:** `soft-demod (LLRs) → deinterleave → codec.decode → CRC-check → payload bytes` (or `b""` on CRC failure).

---

## 6. Metrics: coded BER + FER

New `CodedLinkMetrics` dataclass in `validation/types.py` (mirroring `ProfileIdMetrics`): `coded_ber: float`, `fer: float`, `ber_by_snr: Dict[float, float]`, `fer_by_snr: Dict[float, float]`, `ci: Dict[str, Tuple[float,float]]`. `RunResult` gains `coded_link: Optional[CodedLinkMetrics] = None`.

New `validation.metrics.coded_link_metrics(pairs, snr_by_pair=None) -> CodedLinkMetrics` (mirrors `profile_id_metrics`): each pair is `(truth_payload_bits, decoded_payload_bits_or_None)`. **Coded BER** = post-decode bit errors ÷ transmitted payload bits; a CRC-failed or no-lock frame counts all its payload bits as errors (the conservative, standard convention). **FER** = fraction of frames whose recovered payload ≠ truth (equivalently, CRC failed). The harness collects the pairs by comparing the pipeline's decoded bytes for each capture against `provenance["payload_hex"]`, using the existing SNR grid — enabling coding-gain curves (coded vs uncoded BER-vs-SNR). Bootstrap CIs via the existing `bootstrap_ci`.

---

## 7. Integration into synth & pipeline

- **Synth (`validation/synth/modulators.py` + `scenarios.py`):** `modulate` gains an optional `coding: Optional[CodingSpec] = None`; when set, it applies `‖CRC → encode → interleave` to the payload bits before symbol mapping (default `None` = today's uncoded output, exactly unchanged). `build_scenario` looks up the resolved profile's `coding` field → `CODING_CATALOG[name]` → passes the spec. `provenance` records `coding` (the code name) as truth for the metric.
- **Pipeline (`validation/pipeline.py`):** the region-to-bytes decode path, after producing LLRs from the soft-demod, applies `deinterleave → codec.decode → CRC` when the resolved profile declares a code; uncoded profiles keep today's hard-bit path. Loud failure (CRC fail / no lock) → `b""`.
- **Harness (`validation/harness.py`):** collects `(truth_payload, decoded_payload)` pairs and attaches `coded_link` to `RunResult`.

---

## 8. Testing strategy

TDD throughout (executed subagent-driven with per-task review, like P2). Buckets:

1. **Registry:** `CODING_CATALOG` has capability descriptors for all 7 families; `make_codec` builds `uncoded`/`repetition`, raises `NotImplementedError` (with the sub-phase name) for the other six; `coding_names()` / profile-`coding` validation.
2. **Reference codecs:** `uncoded` and `repetition` round-trip (encode→decode recovers payload, `crc_ok=True`, zero errors, noiseless).
3. **CRC loud-failure:** a corrupted coded frame → `crc_ok=False` → region decodes to `b""` (no silent wrong bytes).
4. **Interleaver:** `deinterleave(interleave(x, d), d) == x` for hard bits and for LLRs, all depths incl. identity.
5. **Soft LLR properties:** `sign(sc_soft_bits)/sign(ofdm_soft_bits)` matches the hard demod bits; `|LLR|` increases with SNR; empty region → empty LLRs.
6. **Coding gain (headline):** on the SNR sweep, `repetition` coded BER < `uncoded` BER at matched SNR (a real, measured coding-gain demonstration).
7. **Metric:** `coded_link_metrics` computes BER/FER + by-SNR + CIs; `RunResult.coded_link` populated by the harness.
8. **Profile-carried coding + zero regression:** a `repetition`-coded profile flows end-to-end (synth→pipeline→metric); existing P2 profiles are unchanged (uncoded default); full existing suite green; `mypy validation` clean.

---

## 9. Global constraints (bind every task)

- `mypy validation` is the CI gate and must stay clean (no `Any`; `numpy.typing`). New/modified `core/*` files must be `mypy`-clean and add **no new** legacy findings; legacy files get **minimal diffs**, never a whole-file `black` reformat.
- IQ at boundaries is `complex64`; internal DSP is `complex128`. LLRs and bits are `float64` / `uint8`.
- Transmit is deterministic (no RNG in encode/modulate).
- **Loud on failure:** a failed CRC (or no demod lock) yields no bytes — no silent wrong payload at normal SNR. The soft path never weakens the hard-path loud-failure guarantees.
- Additive only: existing P2 profiles default to uncoded and are byte-for-byte unchanged; hard-decision demods are untouched.
- Executed subagent-driven with TDD and per-task review; on `feature/ofdm-full-chain`; no PR/tag in this sub-phase.

---

## 10. Out of scope / future (later sub-phases)

- The six heavy codecs (convolutional/Viterbi, RS, BCH, LDPC, turbo, polar, fountain) — P3b…P3h.
- Blind coding *detection* (trial-decode) — possible later extension.
- Assigning real (non-trivial) codes to specific real-link profiles (e.g. `wifi_20` → 802.11 convolutional) — happens as the codecs land.
- Rateless-code framework extensions (variable n) — deferred to P3h (fountain).
- Docs consolidation and the P2 parked cleanups — final consolidation at the end of the whole effort.
