# P2-SC — Single-Carrier Profiles + Blind Resolution — Design Spec

**Design #:** 0004
**Phase:** P2-SC

**Date:** 2026-09-18
**Status:** Accepted — implemented, shipped in v0.3.0
**Phase:** P2-SC (single-carrier half of "multiple profiles"), on `feature/ofdm-full-chain` (one final PR)

## Goal

Introduce a named profile registry and **blind, primary** profile resolution for the single-carrier family: given a detected region, a cheap family pre-classifier decides OFDM vs single-carrier, and for single-carrier a blind resolver infers the profile (sps + modulation) from lock confidence — with no reliance on provenance at decode time. Provenance keeps the *true* profile as evaluation ground truth, so blind profile-ID becomes a measured T&E metric. This also removes the `sps == 8` restriction that PE/PH deferred.

## Motivation

The single-carrier receivers built in PE/PH are hard-wired to `DEFAULT_SC_PROFILE` (sps=8, one modulation at a time chosen from provenance), and `validation.pipeline.single_carrier_region_to_bytes` raises `ValueError` for any `sps != 8`. Real drone links span several single-carrier PHYs (SiK/MAVLink GFSK, BLE 1M/2M, PSK control links) at different symbol rates. To evaluate the receivers against that diversity — and to move toward handling *real* captures that carry no provenance — the framework needs (1) a catalog of realistic profiles and (2) a receiver that discovers which profile a burst uses on its own.

## Decisions (user-approved)

- **Unified profile registry, built in halves.** One named registry spanning single-carrier and OFDM, but delivered as two phases: **P2-SC now** (single-carrier catalog + blind SC resolution + family pre-classifier + profile-ID metric + spine/CLI), **P2-OFDM later** (parametric `core/ofdm.py` + OFDM catalog + blind OFDM resolution). Until P2-OFDM, the OFDM family holds only the existing `wifi_20` profile.
- **Blind auto-detect is primary.** The decode path never reads the profile (or scheme) from provenance; it infers them from the signal. Provenance still records the true profile purely as ground truth for a new profile-ID accuracy metric, so the synthetic T&E spine stays measurable.
- **Cheap family pre-classifier, then within-family blind search.** A fast OFDM-vs-single-carrier discriminator (cyclic-prefix autocorrelation, PAPR tiebreak) runs first; blind profile search then runs only within the chosen family (avoiding cross-family false locks and wasted compute).
- **Broad real-link catalog (~4-6 single-carrier entries).** Named to real archetypes, each PHY-distinct so blind resolution is well-posed.
- **Payload-order discriminator for BPSK vs QPSK.** The Barker preamble is always BPSK, so acquisition alone cannot separate a BPSK-payload profile from a QPSK-payload profile at the same sps. Blind resolution keys on the preamble (sps + preamble family), and for the PSK case adds a cheap payload-order discriminator (post-alignment quadrature-rail energy) to resolve BPSK vs QPSK. Kept rather than banning same-sps PSK pairs, to keep the catalog realistic and profile-ID complete.

## Non-goals

- Parametric OFDM PHY / OFDM profile catalog / blind OFDM resolution — that is **P2-OFDM** (later). OFDM stays on the single fixed `wifi_20` profile here; the family classifier merely routes OFDM regions to the existing OFDM demod unchanged.
- Changing region detection (the energy/envelope detectors that find packet spans). Detection stays as-is and is still scored against `truth_regions`; only decode + profile-ID are blind.
- Coding/interleaving (P3) and bit-loading (P4).
- Any change to the OFDM (P1) chain or the differential/pilot (PH) behavior of the single-carrier receivers.

## Architecture

### 1. Profile registry (`core/profiles.py`, new)

- A lightweight `SCMod` enum — `FSK, GFSK, BPSK, QPSK` — local to `core` (avoids a `core → validation` dependency; the enum's four values mirror `validation.types.ModScheme`'s single-carrier members and map 1:1 at the boundaries).
- `Family` enum — `SINGLE_CARRIER, OFDM`.
- `@dataclass(frozen=True) SCProfileSpec`: `name: str`, `mod: SCMod`, `profile: core.single_carrier.SCProfile` (carries `sps`, `mod_index`, `bt`). A convenience `bits_per_symbol` property (BPSK→1, QPSK→2; FSK/GFSK→1) and `is_fsk`/`gfsk` helpers.
- `SC_CATALOG: dict[str, SCProfileSpec]` — the curated single-carrier catalog (below). `OFDM_CATALOG: dict[str, ...]` — just `wifi_20` (the existing `DEFAULT_OFDM_PROFILE`) for now; expanded in P2-OFDM. A `family_of(name)` / `all_profiles()` accessor.
- `core.single_carrier.DEFAULT_SC_PROFILE` stays; the catalog's `sik_gfsk` is defined to equal the framework's current GFSK default so the PH/default behavior is a registry entry, not a special case.
- `core/profiles.py` depends only on `core.single_carrier` (and, for the lone OFDM entry, `core.ofdm`) — no heavy imports, fully typed, no `Any`.

**Curated single-carrier catalog (PHY-distinct):**

| Name | Mod | sps | mod_index | BT | Archetype |
|------|-----|-----|-----------|----|-----------|
| `sik_gfsk` | GFSK | 8 | 0.7 | 0.5 | SiK/MAVLink telemetry (framework default archetype) |
| `ble_1m` | GFSK | 8 | 0.5 | 0.5 | BLE 1 Msym/s |
| `ble_2m` | GFSK | 4 | 0.5 | 0.5 | BLE 2 Msym/s (exercises sps≠8) |
| `fsk_basic` | FSK | 8 | 0.7 | — | Generic unshaped-FSK C2 |
| `psk_c2` | BPSK | 8 | — | — | PSK control link |
| `qpsk_link` | QPSK | 8 | — | — | Higher-rate PSK link |

Each entry is distinguishable by blind resolution: by preamble waveform (sps and FSK/GFSK/PSK preamble shape) for all but the BPSK↔QPSK pair, which the payload-order discriminator separates.

### 2. Family pre-classifier (`core/profiles.py` or `core/blind.py`)

- `classify_family(iq_region, *, candidate_fft_sizes=(64,)) -> tuple[Family, float]`.
- **Cyclic-prefix autocorrelation:** an OFDM symbol repeats its last `CP` samples at the front, so the normalized autocorrelation of the region at lag = FFT size `N` peaks (→ 1 over the CP overlap). Computed as a sliding metric at each candidate `N` (here `{64}`, matching `wifi_20`; P2-OFDM adds more); the peak is the CP score.
- **PAPR co-requirement (not just a tiebreak):** the CP score alone is *insufficient* — the short deterministic single-carrier Barker preamble spuriously self-matches at lag `N`, scoring as high as OFDM. But single-carrier here is near-constant-envelope (FSK/GFSK `|y|=1`; rect PSK `|y|=1` after normalization) → PAPR ~1–2, whereas OFDM is high-PAPR ~8–13. So the OFDM decision requires **both** a CP peak (`>= OFDM_FAMILY_THRESHOLD`) **and** high PAPR (`>= PAPR_OFDM_THRESHOLD`): the preamble fools CP but not PAPR; noise fails CP. Only true OFDM passes both.
- Returns `(family, cp_score)`. Anything not clearing both requirements is single-carrier (the conservative default — it then runs the SC blind search, which itself fails loudly if nothing locks).

### 3. Blind single-carrier resolver (`core/blind.py`, new)

- `resolve_sc_profile(iq_region) -> tuple[SCProfileSpec | None, float]`.
- For each `SCProfileSpec` in `SC_CATALOG`, build its preamble reference (`preamble_wave_psk` for PSK entries, `preamble_wave_fsk(..., gfsk=...)` for FSK/GFSK) at that profile's params, and run `sc_acquire` (PH's CFO-tolerant acquisition) to get a lock peak. Choose the profile with the highest `abs(peak)`. If the best peak is below `SC_SYNC_THRESHOLD`, return `(None, best)` — a **loud no-lock** (the caller reports `is_valid=False`, never a silent wrong profile).
- **Payload-order disambiguation:** PSK entries that share a preamble waveform (same sps: `psk_c2` BPSK and `qpsk_link` QPSK) tie on acquisition. After picking the best-acquisition PSK candidate, coherently align the payload (reusing the receiver's alignment) and measure the quadrature-rail energy ratio: QPSK fills I and Q rails at ~equal energy, BPSK leaves Q near zero. A `QPSK_QRAIL_THRESHOLD` on `mean(|Q|)/mean(|I|)` decides BPSK vs QPSK, selecting the matching catalog entry. FSK/GFSK entries need no disambiguation (distinct preamble shapes / sps).
- The resolver returns the resolved spec; the caller demods with it via the existing PH receiver (`sc_demodulate_psk`/`sc_demodulate_fsk`, honoring the profile's `differential`/`pilot_spacing` as already threaded).

### 4. Demod integration (`core/demodulation.py`, `validation/pipeline.py`)

- `validation.pipeline.single_carrier_region_to_bytes` **drops the `sps == 8` ValueError** and changes signature to `(iq_region, sample_rate, *, differential=False, pilot_spacing=0) -> Tuple[bytes, Optional[str]]`. It no longer takes the scheme/sps from the caller: it runs `core.blind.resolve_sc_profile` and, on a lock, demods by calling the shared PH receivers directly (`core.single_carrier.sc_demodulate_psk`/`sc_demodulate_fsk`) with the **resolved** profile's `SCProfile` and mod. It **returns** `(bytes, resolved_profile_name)`; on no-lock `(b"", None)`.
- **Scope of "blind":** only the *profile* (sps + modulation family + BPSK/QPSK order) is resolved blindly. `differential` and `pilot_spacing` are payload options that are *not* blindly detectable from the (always-BPSK, always-coherent) preamble and are orthogonal to profile PHY; they stay caller-provided decode knobs (the P2-SC catalog is entirely coherent + pilotless, so both default to off, and the PH differential/pilot capability is unchanged and still reachable by passing them explicitly). The profile-ID metric measures *profile* resolution, not differential/pilot detection.
- The core `PSKDemodulator`/`FSKDemodulator` classes and `DemodConfig` are left unchanged (they keep using `DEFAULT_SC_PROFILE` for their own non-blind callers). Blind resolution lives in `core/blind.py` + the validation pipeline, so no core public class signature changes.
- `DetectClassifyPipeline.run`: **detector selection stays as-is** (the OFDM-envelope-vs-energy detector choice from `provenance["scheme"]` — detection is a non-goal and scored separately against `truth_regions`). The **decode family routing is blind**: each detected region is classified by `classify_family` (not provenance) into OFDM (→ `ofdm_region_to_bytes`, resolved profile `"wifi_20"`) or single-carrier (→ blind `single_carrier_region_to_bytes`). `run` records the resolved profile name on each `Detection` (see §6). It still reads `provenance["differential"]`/`["pilot_spacing"]` as the caller-provided payload knobs above (default off), but never reads `provenance["scheme"]`/`["profile"]` for decode. Provenance's `profile` is read only by the metric layer as truth.

### 5. Spine threading (`validation/synth/scenarios.py`, `validation/__init__.py`)

- `DatasetSpec` gains `profile_by_protocol: Dict[str, str]` (protocol → catalog profile name) as the new primary selector. The existing `scheme_by_protocol` is **retained for back-compat**: when a protocol has no entry in `profile_by_protocol`, its scheme is mapped to a canonical catalog profile (`FSK → fsk_basic`, `GFSK → sik_gfsk`, `BPSK → psk_c2`, `QPSK → qpsk_link`, `OFDM → wifi_20`) so every existing caller keeps working unchanged. `build_scenario` resolves each protocol to an `SCProfileSpec` (profile_by_protocol first, else the scheme→canonical map), modulates with its mod+params (via `modulate`, which already accepts sps/mod_index/bt/differential/pilot_spacing), and records `provenance["profile"] = name` alongside the existing `scheme`.
- `create_synth_dataset` gains an optional `profile_by_protocol` parameter (default empty); it keeps its current `scheme_by_protocol` parameter. A protocol absent from both maps falls back to the framework default `sik_gfsk` (the GFSK default). Existing callers that pass only `scheme_by_protocol` are unaffected.
- Determinism unchanged (profiles are constants; no RNG in TX).

### 6. Profile-ID metric (`validation/types.py`, `validation/metrics.py`, `validation/harness.py`)

- `Detection` gains `resolved_profile: Optional[str] = None` — the blind-resolved profile the decode used (or `None` on no-lock).
- A new `ProfileIdMetrics` (accuracy overall, `accuracy_by_snr`, a small confusion dict) computed by comparing each detection's `resolved_profile` against the truth `provenance["profile"]` of the overlapping region. Folded into `RunResult` beside detection/classification metrics; surfaced in the JSON report.
- This makes blind profile resolution a first-class, measured capability rather than an invisible decode detail.

### 7. CLI (`cli.py`)

- Extend the synth mapping: `_default_synth_profile(protocol) -> str` (protocol → catalog profile name), analogous to `_default_synth_scheme`. Known protocols map to sensible catalog entries (e.g. `mavlink → sik_gfsk`, `ble → ble_1m`, `bpsk_link → psk_c2`, `dji → qpsk_link`, `ocusync → wifi_20`); unknown → `sik_gfsk` (the GFSK default).
- Optional `--profile NAME` override on `validate synth` to force a single profile across all protocols (for targeted single-profile datasets).
- `--differential`/`--pilot-spacing` continue to apply to the PSK profiles as before.

## Testing strategy

- **Family classifier:** OFDM (`wifi_20`) regions classify as OFDM; every SC catalog burst classifies as single-carrier; behavior is stable across a mid/high-SNR sweep.
- **Blind SC resolution:** each catalog profile, modulated and passed (with guard noise) through `resolve_sc_profile`, resolves to *itself* at high SNR; a burst of profile A is not mis-resolved to profile B; below-threshold noise returns a loud no-lock. Includes the `ble_2m` (sps=4) profile that the old `sps == 8` guard rejected.
- **BPSK/QPSK discriminator:** `psk_c2` and `qpsk_link` (same sps, same preamble) each resolve to the correct order via the quadrature-rail test; robustness characterized across SNR.
- **End-to-end pipeline:** a per-profile round-trip through the pipeline recovers the payload for every catalog profile with blind resolution (no provenance-fed scheme/sps); profile-ID accuracy is 100% at high SNR and degrades gracefully at low SNR.
- **Profile-ID metric:** `ProfileIdMetrics` computes correct accuracy/confusion on a mixed-profile dataset; appears in the JSON report.
- **Zero regression:** existing single-carrier round-trip/BER/pilot/differential tests stay green (the PH receivers are unchanged; only the *selection* of their parameters moves from provenance to blind). OFDM tests unchanged. `mypy validation` clean.

## Global constraints (carried)

- `numpy.complex64` at public boundaries, `complex128` internal; deterministic synthesis (profiles/preambles are constants; no RNG in TX).
- `mypy validation` strict-clean; new `core/profiles.py` and `core/blind.py` fully typed, no `Any`; Google docstrings; black/isort/flake8 (88). Legacy `core/*`/`cli.py` are not CI-gated — touch them with minimal diffs and introduce no new findings; never commit a wholesale reformat of a legacy file.
- Zero regression to OFDM and to the PE/PH single-carrier receiver behavior. New spine parameters default so existing datasets/tests are unaffected.
- Loud-on-failure: family/profile resolution that does not lock yields `is_valid=False` / no-lock, never a silent wrong decode.

## Risks

- **Family false-classification** — a single-carrier burst that happens to autocorrelate at lag 64, or a low-SNR OFDM burst whose CP peak is washed out. Mitigation: the conservative default (single-carrier) plus the SC resolver's own loud no-lock; characterized by the family-classifier SNR sweep. The candidate FFT set is just `{64}` here, minimizing spurious lags.
- **Low-SNR blind-discrimination tail (loud-failure caveat)** — at very low SNR (≤~5 dB) the OFDM Schmidl-Cox confidence distributions of genuine OFDM and of a misrouted single-carrier region *overlap* (both in ~0.55–0.65), so no single `OFDM_SYNC_THRESHOLD` cleanly separates them. `OFDM_SYNC_THRESHOLD = 0.6` is chosen in that overlap band: it fails closed on the common misroute cases and preserves genuine OFDM sync to ~6–8 dB, but a small residual tail (~0.1% of misrouted low-SNR SC regions, measured by Monte-Carlo) can leak through the OFDM decoder and yield garbage bits. This is inherent to blind family discrimination at SNRs where the whole chain is already unreliable; it is **measured** by the profile-ID and BER metrics rather than silently ignored, and does not occur at normal operating SNR. The loud-failure guarantee therefore holds at reasonable SNR; at the noise floor it degrades to a characterized, measured tail rather than a hard guarantee.
- **Blind mis-resolution among close profiles** — e.g. `sik_gfsk` (mod 0.7) vs `ble_1m` (mod 0.5), both GFSK sps=8. Their preamble waveforms differ (different deviation), so acquisition separates them, but the margin shrinks at low SNR. Covered by the wrong-profile-rejection tests; acceptable degradation at low SNR (everything degrades there).
- **BPSK/QPSK discriminator threshold** — `QPSK_QRAIL_THRESHOLD` too low mis-labels noisy BPSK as QPSK; too high mis-labels QPSK as BPSK. Tuned and pinned by the discriminator test across SNR.
- **Compute** — blind resolution runs acquisition once per catalog profile per region (`|SC_CATALOG|` acquisitions, each itself a CFO grid). Bounded and offline (T&E); documented. Vectorize later if it becomes a hotspot.
- **Scope creep into P2-OFDM** — the family classifier and registry are built with OFDM in mind but only `wifi_20` is wired; resist parametrizing `core/ofdm.py` here.

## Resolved decisions (user-approved)

1. **Split** into P2-SC (now) and P2-OFDM (later); OFDM stays single-profile `wifi_20` this phase.
2. **Blind is primary**; provenance profile is ground truth for the profile-ID metric only.
3. **Family pre-classifier** (CP-autocorrelation + PAPR) then **within-family** blind search.
4. **Broad, PHY-distinct SC catalog** (6 entries above).
5. **Payload-order discriminator** resolves BPSK vs QPSK (kept, not banned).
6. Profile-ID becomes a **measured harness metric** (`ProfileIdMetrics`), with `Detection.resolved_profile` as the carrier.
