# ADR-0017: Raptor-style fountain code — per-symbol-CRC erasure model, GF(2)-GE decode

- **Status:** Accepted
- **Date:** 2026-09-20
- **Deciders:** rbenzing (with Claude)

## Context

P3h adds a Raptor-style fountain (rateless erasure) code (`core/fountain.py`,
`_Fountain` in `core/coding.py`), the seventh and final codec family after
uncoded, repetition, convolutional (ADR-0005/0008), RS (ADR-0007/0008), BCH
(ADR-0007/0008), LDPC (ADR-0013), turbo (ADR-0014), and polar (ADR-0016). The
framework already carried a `fountain_lt` capability descriptor (`k=0, n=0`
rateless; `{kind:"lt", c:0.03, delta:0.5}`) and `CodeFamily.FOUNTAIN`, with
`make_codec` raising `NotImplementedError` for it — this ADR records the
decisions that replaced that placeholder with a working codec.

Fountain codes are fundamentally different from every prior family here: they
are **erasure** codes, not error-correctors. Their contract is "recover the K
source symbols once enough encoded symbols arrive intact," not "correct bit
flips in a fixed-length codeword." Three things had to be decided to fit that
model onto DroneCMD's bit-error AWGN pipeline, which produces bit errors, not
clean symbol drops, and whose `Codec.encode(info) -> bits` contract needs a
deterministic output length even though fountain codes are nominally
rateless.

## Decision

- **Per-symbol CRC erasure detection, hard-input.** Each LT-encoded symbol
  carries its own small CRC (CRC-8). The receiver marks a symbol **erased**
  when its CRC fails, turning the bit-error AWGN channel into the
  symbol-erasure channel fountain codes are built for. This makes fountain a
  **hard-input** codec (`soft_input` not set / falsy), riding the pipeline's
  existing hard branch (`sc_demodulate_psk`), exactly like BCH — there is no
  soft-LLR metric and therefore no `sc_soft_bits` recalibration question
  (per the standing ADR-0006 obligation, resolved here the same way BCH
  resolved it: by not applying).
- **Raptor-style construction: systematic sparse precode + Robust-Soliton
  LT.** A plain LT code has a real error floor at small K (drone payloads are
  tens of bytes, so K is in the tens — squarely LT's weak regime). The outer
  **systematic sparse precode** (`build_precode`: `R = round(K·(1/precode_rate
  − 1))` seeded sparse parity rows over the K source symbols) adds redundant
  intermediate symbols so the combined GF(2) system over `L = K + R`
  intermediates is full-rank with high probability at small K, before the
  inner **Robust Soliton** LT layer (`robust_soliton` pmf; ideal-Soliton ρ
  plus the Robust spike τ) draws each encoded symbol's degree and neighbor
  set from a per-symbol seeded RNG (`default_rng(seed + i)`), so TX and RX
  agree without side information.
- **Rateless realized as fixed-overhead ε profiles.** `Codec.encode` needs a
  deterministic length, so true unbounded rateless generation is not used;
  instead each catalog entry fixes an overhead ε and emits
  `N = ceil((1 + ε)·K)` encoded symbols. Three profiles show the
  overhead/recovery trade-off directly: `fountain_r05` (ε=0.5), `fountain_r10`
  (ε=1.0, the profile default via `fountain_bpsk`), `fountain_r15` (ε=1.5,
  most overhead, most erasure tolerance).
- **GF(2) Gaussian elimination decode**, not belief-propagation / peeling.
  The decoder builds one incidence row per surviving (CRC-ok) LT symbol plus
  the R precode constraint rows, and row-reduces over GF(2) (`gf2_solve`),
  applying the same row XORs to the S-bit RHS vectors. A column resolves only
  when its pivot row is a clean single-bit row; anything else is a clean
  decode failure, never a silently wrong payload. Plain Gaussian elimination
  is affordable at this small-K demonstration scale; inactivation decoding
  (Raptor's usual large-K optimization) is out of scope.
- **Internal 16-bit length header for exact frame recovery.** Because the
  code is rateless-by-overhead rather than fixed-rate, there is no fixed
  trailer to key frame-length recovery off. The codec prepends a 16-bit
  length header to the pipeline's frame (`payload | CRC-16`) before
  symbolizing, so `K` is recovered from `N` and ε alone (the unique `K` with
  `ceil((1+ε)·K) == N`, since `ceil((1+ε)·.)` is non-decreasing), sizing the
  GF(2) solve; the *exact* bit length is then read from the solved header
  after decode, used only for the final trim. A corrupted header trims wrong
  and the outer CRC-16 then fails loud — never a silent short/garbage
  payload.
- **Small-K demonstration scope; interop out of scope.** Not RFC 5053 / RFC
  6330 (RaptorQ) bit-compatible, not systematic LT, not large-K/streaming/
  file-transfer, not soft/LLR-based erasure marking or genie erasures — per
  the RS/LDPC/turbo/polar precedent, a reproducible, self-consistent
  construction with a genuine erasure-recovery demonstration is the bar, not
  bit-compatibility with a published standard.
- **This completes the seven-family FEC capability sheet** (uncoded,
  repetition, convolutional, RS, BCH, LDPC, turbo, polar, fountain — nine
  catalog families in total counting the two structural ones) that P3a
  through P3h set out to build; `make_codec` now builds every
  `CODING_CATALOG` family with no `NotImplementedError` branch left reachable
  from the catalog (the fallback `raise` remains only as a guard for an
  unknown `CodeFamily`).

## Consequences

### Positive
- Fountain slots onto the existing hard-input pipeline branch unchanged
  (`sc_demodulate_psk` → deinterleave → codec decode → CRC-strip), exactly
  like BCH — no pipeline change, no new demod path, no LLR-scale question.
- The precode makes the small-K regime tractable: without it, plain-LT
  peeling stalls well before the payload-relevant K values DroneCMD frames
  actually use.
- Loud-on-failure is structural, not bolted on: a rank-deficient GF(2) solve
  is a clean `decode_ok=False` at the codec layer, and a corrupted header is
  caught by the pre-existing outer CRC-16 — no new silent-failure surface.
- Three ε profiles (r05/r10/r15) give a direct, inspectable overhead/recovery
  trade-off using one codec implementation and one catalog shape, rather than
  three different algorithms.

### Negative / trade-offs
- Not RFC 6330 (RaptorQ) bit-compatible — acceptable, per the RS/LDPC/
  turbo/polar interop-out-of-scope precedent.
- Fixed-overhead ε profiles are not true unbounded rateless generation; a
  receiver that wants "just enough" symbols still has to wait for the
  profile's full `N = ceil((1+ε)·K)` before this implementation's decode path
  runs. Acceptable for the `Codec.encode(info) -> bits` contract's fixed-
  length requirement; genuinely adaptive/incremental fountain reception is
  future work if ever needed.
- Plain GF(2) Gaussian elimination is `O(L^2 · rows)`-ish and pure Python;
  fine at this demonstration's small L, not a large-K/streaming design.
- LT has a real, non-zero residual failure probability even with the
  precode at small K (an under-full-rank draw is possible); the
  erasure-recovery test therefore measures FER (fraction of trials with the
  full payload recovered) against an uncoded baseline, not a 100%-recovery
  guarantee.

### Neutral / notes
- Verified: Robust Soliton pmf sums to 1 with the τ spike at the expected
  index; precode parity rows are XOR-consistent with their source neighbors
  by construction; noiseless round-trip recovers the payload exactly for all
  three ε profiles; erasure recovery scales monotonically with ε (higher
  overhead tolerates a higher erasure rate); the GF(2) solver cleanly fails
  (no crash, `decode_ok=False`) on a rank-deficient system and exactly
  recovers a full-rank one; end-to-end FER through the real synth → AWGN →
  `sc_demodulate_psk` (hard) → per-symbol-CRC erasures → fountain decode →
  outer CRC chain shows `fountain_r15` recovering the payload in a clear
  majority of trials at an operating point where the uncoded frame recovers
  none (`test_fountain_recovers_payload_where_uncoded_fails`, 3.0 dB / 30
  trials, 48-byte payload for adequate K); blind resolution of the
  `fountain_bpsk` profile (`sps=96`, a small unique value per the P3g
  blind-resolve-cost lesson) locks and recovers a small payload end-to-end
  (`test_fountain_bpsk_blind_end_to_end`). Cross-refs: ADR-0005 (framework),
  ADR-0006 (soft-LLR/loud-on-failure — fountain opts out, like BCH),
  ADR-0007 (reusable GF algebra; fountain's GF(2) solve is a separate,
  simpler linear-algebra path, not the shared `GF2m` field module), ADR-0008
  (FEC family choices), ADR-0016 (polar — the previous "last" family before
  fountain completed the sheet). Design doc: `docs/design/0013-p3h-fountain.md`.
