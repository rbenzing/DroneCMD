# P2-OFDM — Parametric OFDM Profiles & Blind OFDM Resolution (Design)

**Design #:** 0005
**Phase:** P2-OFDM

**Status:** Accepted — implemented, shipped in v0.3.0
**Date:** 2026-09-18
**Author:** rbenzing (with Claude)
**Predecessors:** P1 (OFDM full chain), PE (single-carrier receivers), PH (CFO/phase), P2-SC (single-carrier profile registry + blind resolution + profile-ID metric).
**Scope note:** Security-research / educational software. All work is for lawful security research and authorized testing only.

---

## 1. Overview

### Goal

Extend the blind-profile capability delivered in P2-SC to the OFDM family: a
**broader named OFDM catalog** (distinct-FFT-size profiles *plus* same-FFT-size
cyclic-prefix / subcarrier-layout variants) and a **blind OFDM resolver** that
recovers the transmitted profile from a detected region with no side
information — mirroring `core.blind.resolve_sc_profile`. Blind-resolved profiles
flow into the existing profile-ID metric so OFDM profile identification is
measured on the SNR grid exactly as single-carrier profiles already are.

### What makes this phase small

`core.ofdm` is **already fully parametric** (verified by spike): `OFDMProfile`
is a dataclass, and `modulate_ofdm` / `demodulate_ofdm` / `ofdm_sync_confidence`
all take a `profile` argument. Round-trip was confirmed clean at N = 32 / 64 /
128 (sync confidence 1.0, BER 0.0). **There is no PHY rewrite in this phase.**
The work is: catalog entries, one blind resolver, one byte-identical PHY
extraction the resolver reuses, and threading the chosen profile through the
synth modulator and the decode-routing path.

### Non-goals

- **No OFDM PHY rewrite.** `modulate_ofdm` / `demodulate_ofdm` keep their
  current algorithm and output. The only PHY change is a refactor that extracts
  the equalized-data-symbol stage into a reusable function with a regression
  test proving byte-identical bits.
- **No change to the production `core.demodulation.OFDMDemodulator`.** That
  class remains the fixed-`DEFAULT_OFDM_PROFILE` production wrapper. Blind
  resolution is a validation-spine concern that calls the `core.ofdm` PHY
  directly, exactly as `single_carrier_region_to_bytes` calls the SC PHY
  directly rather than going through the SC demodulator wrapper.
- **No new detector.** Region detection is unchanged; only decode *routing*
  and *profile resolution* change.
- **No adaptive bit-loading / coding.** Those are P3/P4.

---

## 2. Background — the two facts that shape the design

A spike established the two load-bearing facts:

1. **`ofdm_sync_confidence` is strongly FFT-size-selective but blind to
   same-N variants.** A wifi_20 (N=64) signal scores ~1.0 under an N=64 profile,
   0.216 under an N=128 profile, and 0.053 under an N=32 profile — because the
   Schmidl & Cox metric depends only on the N/2-sample STF repetition. But
   wifi_20, a long-CP N=64 variant, and an alternate-pilot-layout N=64 variant
   **all score ~1.0 for each other**: sync confidence cannot tell same-N
   profiles apart.

2. **A wrong CP or wrong layout corrupts the *demodulated constellation*, not
   the sync metric.** With the wrong CP the per-symbol FFT window drifts
   (`symbol_len = N + CP` is wrong), so channel estimation and every data symbol
   are mis-windowed → inter-carrier interference → scattered constellation. With
   the wrong pilot/data layout the LTF channel estimate is applied to the wrong
   bins and pilot CPE correction uses data-bearing bins → scattered
   constellation. Both show up as **high data-symbol EVM** (mean distance of
   equalized data subcarriers to the nearest ideal QPSK point) — a truth-free
   quality metric computed at the receiver.

**Consequence:** the resolver is a two-signal decision — a cheap
FFT-size-selective **sync gate** (rejects wrong-N and noise) followed by a
**trial-demodulation EVM tiebreak** (rejects wrong-CP and wrong-layout among the
locked, same-N candidates).

The P2-SC blind path is the structural template throughout:
`resolve_sc_profile` iterates the catalog, gates each candidate on a lock
metric, and returns `(spec_or_None, confidence)`; `single_carrier_region_to_bytes`
returns `(bytes, resolved_name)`; provenance carries truth for the profile-ID
metric. P2-OFDM adds the OFDM-family counterparts of exactly these pieces.

---

## 3. The OFDM catalog

`core.profiles.OFDM_CATALOG` grows from the single fixed `wifi_20` entry to five
profiles: three distinct-FFT-size profiles and two same-N variants (one CP
variant, one layout variant). This honors the "broader — add CP/layout variants"
scope: distinct-N profiles are separated by the sync gate; the two same-N
variants exercise the EVM tiebreak.

`OFDM_CATALOG` is retyped from `Dict[str, object]` to `Dict[str, OFDMProfile]`
(importing `OFDMProfile` from `core.ofdm`). All five values are ordinary
`OFDMProfile` instances built the same way `_build_default_profile` builds
`wifi_20`. Concrete layouts (all pilots `1+0j`, occupied = the stated symmetric
range excluding DC, data = occupied minus pilots):

| Name | FFT `N` | CP | Occupied (excl 0) | Pilots | Kind |
|------|--------:|---:|-------------------|--------|------|
| `wifi_20` | 64 | 16 | −26..26 | ±7, ±21 | baseline (== `DEFAULT_OFDM_PROFILE`) |
| `wifi_40` | 128 | 32 | −58..58 | ±11, ±25, ±53 | distinct N (wide) |
| `ofdm_nb` | 32 | 8 | −13..13 | ±3, ±11 | distinct N (narrowband) |
| `wifi_20_longcp` | 64 | 32 | −26..26 | ±7, ±21 | same-N **CP** variant |
| `wifi_20_altpilot` | 64 | 16 | −26..26 | ±11, ±25 | same-N **layout** variant |

Notes:

- `wifi_20` is unchanged and remains `DEFAULT_OFDM_PROFILE`; `family_of` already
  classifies every OFDM-catalog name as `Family.OFDM`.
- Distinct-N pair (`wifi_20` / `wifi_40` / `ofdm_nb`) → separated by the sync
  gate. Same-N pairs (`wifi_20` vs `wifi_20_longcp`; `wifi_20` vs
  `wifi_20_altpilot`) → separated by the EVM tiebreak. Every catalog entry is
  therefore blind-distinguishable by at least one of the two signals.
- All five are valid parametric `OFDMProfile`s; the spike confirmed clean
  round-trips at N = 32 / 64 / 128, so no new PHY behavior is required to
  modulate or demodulate any of them.

`all_profile_names()` and `family_of()` require no change beyond the larger
catalog. `DEFAULT_SC_PROFILE_NAME` is untouched; there is no default-profile
change in this phase.

---

## 4. The blind OFDM resolver

New function in `core.blind`, the OFDM counterpart of `resolve_sc_profile`:

```python
def resolve_ofdm_profile(iq: Complex) -> Tuple[Optional[str], float]:
    """Blindly resolve an OFDM region to a catalog profile name.

    Stage 1 (sync gate, FFT-size-selective): for each OFDM_CATALOG profile,
    compute ofdm_sync_confidence(iq, profile); discard candidates below
    OFDM_SYNC_THRESHOLD. This rejects noise and wrong-FFT-size profiles.

    Stage 2 (EVM tiebreak, CP/layout-selective): among locked candidates,
    trial-demodulate and score mean data-symbol EVM (distance of equalized
    data subcarriers to the nearest ideal QPSK point); pick the lowest EVM.
    This separates same-N CP and layout variants that Stage 1 cannot.

    Returns (name, sync_confidence_of_winner) on a lock, or (None, best_conf)
    when no candidate clears the sync gate OR the best EVM exceeds
    OFDM_EVM_MAX (loud no-lock -> the region is not trustworthy OFDM).
    """
```

### Algorithm

1. For each `(name, profile)` in `OFDM_CATALOG`:
   - `conf = ofdm_sync_confidence(iq_c128, profile)`.
   - If `conf < OFDM_SYNC_THRESHOLD`, skip (record `best_conf = max(...)`).
   - Else compute `evm = _ofdm_data_evm(iq_c128, profile)` and keep
     `(name, conf, evm)`.
2. If no candidate locked → return `(None, best_conf)` (loud no-lock).
3. Pick the locked candidate with the **lowest EVM**.
4. If that lowest EVM `> OFDM_EVM_MAX` → return `(None, best_conf)` (locked on
   sync but the constellation is not credible OFDM — e.g. a single-carrier
   region that slipped past `classify_family` and the sync gate). This preserves
   the loud-on-failure guarantee: the OFDM branch does not always emit bytes.
5. Otherwise return `(winner_name, winner_conf)`.

### The EVM quality metric

```python
def _ofdm_data_evm(iq: Complex, profile: OFDMProfile) -> float:
    """Mean distance of equalized data subcarriers to the nearest ideal
    QPSK point, over a trial demod with `profile`. Low for the true profile;
    high when a wrong CP mis-windows the FFT or a wrong layout mis-equalizes.
    Returns +inf when the region is too short to yield any data symbol."""
```

It reuses the *exact* equalization pipeline of `demodulate_ofdm` (S&C timing +
fractional CFO + LTF least-squares channel estimate + per-symbol one-tap
equalize + pilot CPE correction), then measures the equalized data subcarriers
rather than demapping them. The ideal QPSK points are `(±1 ± 1j)/sqrt(2)` (the
`qpsk_map` convention). EVM = mean over all data subcarriers of all data symbols
of `min_k |eq_symbol - ideal_k|`.

To avoid duplicating the equalization (and to keep the resolver and the
demodulator provably consistent), `demodulate_ofdm`'s per-symbol equalize/CPE
stage is extracted into a reusable function (Section 5). `_ofdm_data_evm` calls
that function; `demodulate_ofdm` demaps its output. No equalization logic is
duplicated in `core.blind`.

### Thresholds (data-driven, pinned in the plan)

- `OFDM_SYNC_THRESHOLD` — reuse `core.ofdm.OFDM_SYNC_THRESHOLD = 0.6` (already
  the calibrated genuine-OFDM-vs-misroute midpoint from P2-SC/PH; not re-derived
  here).
- `OFDM_EVM_MAX` — **new**, defined in `core.blind`. Its value is set by
  measurement in the plan's first resolver task (Section 8): measure the EVM
  distribution of genuine correct-profile OFDM across the SNR grid and of
  wrong-profile / misrouted-SC regions, and choose a threshold in the gap. Like
  `OFDM_SYNC_THRESHOLD` and `QPSK_QRAIL_THRESHOLD` before it, it carries an
  honest comment about where the distributions overlap at the noise floor.

### The one genuine risk, and its pre-registered fallback

The same-N EVM separation (`wifi_20` vs `wifi_20_longcp`; `wifi_20` vs
`wifi_20_altpilot`) is the only part not yet empirically confirmed — the
spike's inline EVM probe was buggy and could not measure it. The reasoning is
sound (a wrong CP grossly mis-windows every symbol; a wrong layout mis-equalizes
every bin), but it is verified, not assumed.

**Pre-registered ruling (binds execution):** the plan's **first resolver task**
empirically measures same-N EVM separation across the SNR grid *before* any
downstream task depends on it. If EVM reliably separates the same-N variants
down to a reasonable SNR (target: clean separation at ≥ ~10 dB, matching the
band where P2-SC's discriminators are clean), proceed with the full five-profile
catalog. **If it does not**, drop the two same-N variants
(`wifi_20_longcp`, `wifi_20_altpilot`) and ship the distinct-N catalog
(`wifi_20`, `wifi_40`, `ofdm_nb`) only — the sync gate alone fully resolves
that reduced catalog, with no unconfirmed discriminator shipped. Either outcome
is a complete, honest phase; the fallback is recorded so execution does not
stall on the measurement.

---

## 5. PHY refactor — byte-identical extraction

`demodulate_ofdm` currently interleaves equalization and demapping in its final
loop. Extract the equalization into a reusable function so the resolver can score
the constellation the demodulator actually produces:

```python
def ofdm_equalized_symbols(rx: Complex, profile: OFDMProfile = DEFAULT_OFDM_PROFILE) -> Complex:
    """Equalized data subcarriers for every data symbol, concatenated in
    subcarrier order (the pre-demap signal). Empty when rx is shorter than the
    STF+LTF preamble or yields no data symbol. Runs the same S&C timing, CFO,
    LTF channel estimate, and per-symbol equalize + pilot-CPE stages as
    demodulate_ofdm."""
```

`demodulate_ofdm` becomes `qpsk_demap(ofdm_equalized_symbols(rx, profile))` (with
the same empty-guard behavior). Because `qpsk_demap` is elementwise (sign per
rail), demapping the concatenation equals demapping per symbol and concatenating
— so bits are **byte-identical** to today. A regression test (Section 8) asserts
this against the pre-refactor output for `wifi_20` and at least one other N.

This mirrors P2-SC's additive `sc_aligned_payload_centers`, which exposed the
aligned single-carrier payload centers for `_psk_qrail_ratio` without changing
demodulation behavior.

---

## 6. Threading the profile through synth and decode routing

### 6a. Synthetic modulator (`validation/synth/modulators.py`)

`_ofdm` and `modulate` currently ignore the profile and always use
`DEFAULT_OFDM_PROFILE`. Thread an optional profile:

- `_ofdm(bits, profile: OFDMProfile = DEFAULT_OFDM_PROFILE)` → `modulate_ofdm(bits.astype(np.uint8), profile)`.
- `modulate(..., ofdm_profile: Optional[OFDMProfile] = None)` → in the
  `scheme == ModScheme.OFDM` branch, `_ofdm(bits, ofdm_profile or DEFAULT_OFDM_PROFILE)`.
  `ofdm_profile` is ignored for all single-carrier schemes (documented, like
  `sps`/`differential` already are for OFDM). Default `None` reproduces today's
  output exactly (back-compat).

### 6b. Scenario builder (`validation/synth/scenarios.py`)

`build_scenario`'s OFDM branch already resolves a profile *name* and sets
`provenance["profile"] = name`. Extend it to pass the profile object:

```python
if family_of(name) == Family.OFDM:
    clean = modulate(payload, ModScheme.OFDM, ofdm_profile=OFDM_CATALOG[name])
```

`provenance["profile"]` now carries `wifi_40` / `ofdm_nb` / `wifi_20_longcp` /
`wifi_20_altpilot` as appropriate — the truth the profile-ID metric scores
against. `provenance["scheme"]` stays `"ofdm"` for all OFDM profiles.

### 6c. Decode routing (`validation/pipeline.py`)

`ofdm_region_to_bytes` becomes blind and returns a resolved name, matching
`single_carrier_region_to_bytes`:

```python
def ofdm_region_to_bytes(iq_region: IQSamples) -> Tuple[bytes, Optional[str]]:
    """Blindly resolve the OFDM profile, then demodulate with it.
    Returns (packed_bytes, resolved_name) on a lock, or (b"", None) on no lock."""
    if len(iq_region) == 0:
        return b"", None
    from core.blind import resolve_ofdm_profile
    from core.ofdm import demodulate_ofdm
    from core.profiles import OFDM_CATALOG
    iq_c128 = iq_region.astype(np.complex128)
    name, _ = resolve_ofdm_profile(iq_c128)
    if name is None:
        return b"", None
    bits = demodulate_ofdm(iq_c128, OFDM_CATALOG[name])
    if len(bits) == 0:
        return b"", None
    return np.packbits(bits.astype(np.uint8)).tobytes(), name
```

`DetectClassifyPipeline.run`'s OFDM branch drops the hardcoded
`resolved = "wifi_20" if pkt else None` and instead unpacks the resolver's name:

```python
if family == Family.OFDM:
    pkt, resolved = ofdm_region_to_bytes(region)
else:
    pkt, resolved = single_carrier_region_to_bytes(region, ...)
```

This calls the `core.ofdm` PHY directly and does **not** touch the legacy
`core.demodulation.OFDMDemodulator` — matching how the SC blind path bypasses the
SC demodulator wrapper. Detector *selection* is unchanged (still keyed on
`provenance["scheme"]`); only decode routing and profile resolution change.

### 6d. Family classifier FFT-size set (`core/blind.py`)

`classify_family`'s default `fft_sizes` widens from `(64,)` to `(32, 64, 128)`
so the CP-autocorrelation family gate fires for the N=32 and N=128 OFDM profiles
too. `cp_ratio` stays `0.25`: the family gate only needs to detect *that* a CP
repetition exists at some candidate N (a partial CP correlation still peaks even
for the long-CP variant), not the exact CP. The PAPR co-requirement is
unchanged and still rejects the single-carrier preamble's spurious CP
self-match. Cost is three CP-autocorr passes instead of one per region —
negligible offline.

### 6e. CLI (`cli.py`)

No required change: P2-SC's `--profile` flag already accepts any
`all_profile_names()` entry, so `wifi_40` / `ofdm_nb` / the same-N variants are
reachable explicitly (`dronecmd validate synth --profile wifi_40 ...`).
`_default_synth_profile` keeps `ocusync → wifi_20`. (Optional, non-blocking: a
narrowband/wide protocol default may be added, but the explicit flag suffices and
keeps the legacy CLI diff minimal.)

---

## 7. Metric reuse — no metric changes

`validation.metrics.profile_id_metrics`, `validation.types.ProfileIdMetrics`,
`Detection.resolved_profile`, and the harness's (truth-profile,
resolved-profile) pair collection are **profile-name-agnostic** (built that way
in P2-SC). OFDM profile names now flow through the same path: truth
`provenance["profile"]` (e.g. `wifi_40`) versus `Detection.resolved_profile`
from `resolve_ofdm_profile`. Profile-ID accuracy, the confusion matrix, and
accuracy-by-SNR automatically span the union of the SC and OFDM catalogs. No
changes to `metrics.py`, `types.py`, `harness.py`, or `report.py`.

---

## 8. Testing strategy

TDD throughout (the phase is executed subagent-driven with per-task review, like
P2-SC). Test buckets:

1. **PHY refactor regression (first PHY task).** Assert
   `ofdm_equalized_symbols` → `qpsk_demap` yields **byte-identical** bits to the
   pre-refactor `demodulate_ofdm` for `wifi_20` and at least one other N, and
   that `demodulate_ofdm`'s own output is unchanged. Guards the "no PHY behavior
   change" non-goal.

2. **Catalog round-trip.** For every `OFDM_CATALOG` profile: `modulate_ofdm`
   → `demodulate_ofdm` with the *same* profile recovers the payload at high SNR
   (BER 0), and `ofdm_sync_confidence` ≈ 1.0. Confirms all five profiles are
   valid PHY parameterizations.

3. **Same-N EVM separation (first resolver task — the risk gate).** Measure
   `_ofdm_data_evm` for correct vs same-N-wrong profile across the SNR grid;
   assert the correct profile's EVM is separably lower at ≥ ~10 dB for both the
   CP variant and the layout variant. **This test sets `OFDM_EVM_MAX` and
   triggers the Section-4 fallback ruling if separation fails.**

4. **Resolver self-resolution.** `resolve_ofdm_profile` on a clean burst of each
   catalog profile returns that profile's name — including the same-N variants
   (contingent on bucket 3 / the fallback).

5. **Resolver rejection (loud failure).** `resolve_ofdm_profile` returns
   `(None, _)` on: pure noise; a burst of a *non-catalog* N; and a
   single-carrier region (GFSK/BPSK) — the last exercising `OFDM_EVM_MAX` (locks
   on sync but EVM too high). No silent wrong bytes.

6. **SNR-swept profile-ID.** Following the P2-SC lesson (a coverage gap hid a
   silent-wrong-bits defect until SNR-swept tests were added), include
   SNR-swept resolver/profile-ID tests from the start, not only high-SNR
   point tests.

7. **Blind pipeline round-trip.** `DetectClassifyPipeline` on synth OFDM
   captures of each profile routes to the OFDM branch via `classify_family`,
   resolves the correct profile blindly, and records
   `Detection.resolved_profile`; profile-ID accuracy is high at normal SNR.

8. **Family gate at new N.** `classify_family` tags N=32 and N=128 OFDM regions
   as `Family.OFDM` (CP+PAPR), and still tags single-carrier regions as
   `Family.SINGLE_CARRIER`.

9. **Zero regression.** The full existing suite (P1 / PE / PH / P2-SC, 277
   tests) stays green; `mypy validation` stays clean.

---

## 9. Risks & mitigations

- **Same-N EVM separation unconfirmed (primary).** Mitigated by the
  first-resolver-task measurement gate and the pre-registered distinct-N
  fallback (Section 4). Worst case ships a smaller but fully-honest catalog.
- **Low-SNR family/leak tail (inherited).** As documented on
  `OFDM_SYNC_THRESHOLD`, genuine-OFDM and misrouted-SC S&C distributions overlap
  at the noise floor (~0.1% leak tail below ~5 dB). `OFDM_EVM_MAX` narrows this
  (a leaked SC region now must also pass the EVM ceiling) but does not eliminate
  it; the residual is measured by the profile-ID/BER metrics, not claimed as a
  hard guarantee. The loud-failure guarantee holds at normal operating SNR.
- **PHY refactor drift.** Mitigated by the byte-identical regression test
  (bucket 1) — the refactor cannot silently change bits.
- **Cost.** The resolver trial-demodulates up to five candidates per OFDM
  region; all offline, negligible against detection/classification.

---

## 10. Global constraints (bind every task)

- `mypy validation` is the CI gate and must stay clean (no `Any`; `numpy.typing`
  annotations). New/modified `core/*` files (`core/ofdm.py`, `core/blind.py`,
  `core/profiles.py`) must be `mypy`-clean and introduce **no new** legacy
  findings; legacy files get **minimal diffs** and are **never** committed as a
  `black .` whole-file reformat.
- IQ at module boundaries is `numpy.complex64`; internal DSP math is
  `complex128` (cast at the boundary, as the pipeline already does).
- Transmit is deterministic (no RNG in modulation).
- **Loud on failure:** no silent wrong bits/bytes at normal operating SNR. The
  resolver fails closed via the sync gate *and* `OFDM_EVM_MAX`.
- The GFSK single-carrier default and all P2-SC behavior are unchanged; P2-OFDM
  is additive.
- Executed subagent-driven with TDD and per-task review, one branch
  (`feature/ofdm-full-chain`), no PR/tag in this phase (final consolidation
  handles that).

---

## 11. Out of scope / future

- Adaptive bit-loading and higher-order QAM per subcarrier (P4).
- Channel coding / interleaving (P3).
- Making `core.demodulation.OFDMDemodulator` profile-configurable for the
  *production* API (the docstring's "later phase"): deferred; the validation
  spine's blind path does not need it.
- Docs consolidation, the `DatasetSpec.sps` dead-field cleanup, and the
  `_default_synth_profile`/`_default_synth_scheme` mavlink inconsistency — all
  parked for final consolidation.
