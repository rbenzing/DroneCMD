from __future__ import annotations

import numpy as np
import pytest

from core.blind import resolve_sc_profile
from core.profiles import SC_CATALOG
from core.single_carrier import (  # noqa: F401  (kept for parity; not used directly)
    preamble_wave_fsk,
    preamble_wave_psk,
    sc_diff_encode,
    sc_map_psk,
)
from validation.synth.modulators import modulate
from validation.types import ModScheme

_SCMOD_TO_SCHEME = {
    "sik_gfsk": ModScheme.GFSK,
    "ble_1m": ModScheme.GFSK,
    "ble_2m": ModScheme.GFSK,
    "fsk_basic": ModScheme.FSK,
    "psk_c2": ModScheme.BPSK,
    "qpsk_link": ModScheme.QPSK,
    "rep_bpsk": ModScheme.BPSK,
    "conv_bpsk": ModScheme.BPSK,
    "rs_bpsk": ModScheme.BPSK,
    "bch_bpsk": ModScheme.BPSK,
    "ldpc_bpsk": ModScheme.BPSK,
    "turbo_bpsk": ModScheme.BPSK,
    "polar_bpsk": ModScheme.BPSK,
    "fountain_bpsk": ModScheme.BPSK,
}


def _burst(name: str) -> np.ndarray:
    spec = SC_CATALOG[name]
    data = bytes(range(24))
    iq = modulate(
        data,
        _SCMOD_TO_SCHEME[name],
        sps=spec.profile.sps,
        mod_index=spec.profile.mod_index,
        bt=spec.profile.bt,
    )
    return iq.astype(np.complex128)


def test_each_profile_resolves_to_itself() -> None:
    for name in SC_CATALOG:
        spec, conf = resolve_sc_profile(_burst(name))
        assert spec is not None, f"{name} failed to lock (conf={conf})"
        assert spec.name == name, f"{name} resolved to {spec.name}"


def test_bpsk_qpsk_disambiguated() -> None:
    # Same sps + identical BPSK preamble -> only the payload-order
    # discriminator separates these.
    assert resolve_sc_profile(_burst("psk_c2"))[0].name == "psk_c2"
    assert resolve_sc_profile(_burst("qpsk_link"))[0].name == "qpsk_link"


def test_ble_2m_sps4_resolves() -> None:
    assert resolve_sc_profile(_burst("ble_2m"))[0].name == "ble_2m"


def test_noise_returns_no_lock() -> None:
    rng = np.random.default_rng(0)
    noise = (rng.standard_normal(600) + 1j * rng.standard_normal(600)).astype(
        np.complex128
    )
    spec, conf = resolve_sc_profile(noise)
    assert spec is None
    assert conf < 0.5


def test_aligned_centers_separate_bpsk_qpsk() -> None:
    from core.single_carrier import sc_aligned_payload_centers

    bpsk = sc_aligned_payload_centers(_burst("psk_c2"), SC_CATALOG["psk_c2"].profile)
    qpsk = sc_aligned_payload_centers(
        _burst("qpsk_link"), SC_CATALOG["qpsk_link"].profile
    )
    bpsk_ratio = float(np.mean(np.abs(bpsk.imag))) / (
        float(np.mean(np.abs(bpsk.real))) + 1e-9
    )
    qpsk_ratio = float(np.mean(np.abs(qpsk.imag))) / (
        float(np.mean(np.abs(qpsk.real))) + 1e-9
    )
    assert bpsk_ratio < 0.3
    assert qpsk_ratio > 0.7


@pytest.mark.slow  # ~100 blind resolves across a 5-point SNR sweep
def test_bpsk_not_confused_as_qpsk_at_normal_snr() -> None:
    # Regression pin for QPSK_QRAIL_THRESHOLD (whole-branch review, BLOCKING):
    # a BPSK burst mis-resolved to qpsk_link returns SILENT wrong bits. The
    # Q-rail discriminator is asymmetric, so at 0.7 BPSK->QPSK confusion is
    # pushed out of the normal-SNR band.
    from core.blind import resolve_sc_profile
    from validation.repro import rng as make_rng
    from validation.synth.channel import add_awgn_at_snr
    from validation.synth.modulators import modulate
    from validation.types import ModScheme

    # 10 dB is included (in addition to the 12 dB floor) because empirically
    # (measured below) 12 dB alone is already clean even at the buggy 0.5
    # threshold on this codebase's synth defaults -- 10 dB is where the
    # pre-fix/post-fix contrast actually shows up, so it is the SNR point
    # that makes this a genuine regression pin rather than a vacuous check.
    tx = modulate(bytes(range(24)), ModScheme.BPSK, sps=8).astype(np.complex64)
    for snr in (25.0, 20.0, 15.0, 12.0, 10.0):
        confusions = 0
        for seed in range(20):
            noisy, _, _ = add_awgn_at_snr(tx, snr, make_rng(seed))
            spec, _ = resolve_sc_profile(noisy.astype(np.complex128))
            if spec is not None and spec.name == "qpsk_link":
                confusions += 1
        assert confusions == 0, f"BPSK->QPSK at {snr} dB: {confusions}/20"


@pytest.mark.slow  # ~80 blind resolves across a 4-point SNR sweep
def test_qpsk_not_confused_as_bpsk_across_snr() -> None:
    # The asymmetric fix must not cost QPSK: qpsk_link never resolves to psk_c2.
    from core.blind import resolve_sc_profile
    from validation.repro import rng as make_rng
    from validation.synth.channel import add_awgn_at_snr
    from validation.synth.modulators import modulate
    from validation.types import ModScheme

    tx = modulate(bytes(range(24)), ModScheme.QPSK, sps=8).astype(np.complex64)
    for snr in (25.0, 20.0, 15.0, 12.0):
        for seed in range(20):
            noisy, _, _ = add_awgn_at_snr(tx, snr, make_rng(seed))
            spec, _ = resolve_sc_profile(noisy.astype(np.complex128))
            if spec is not None:
                assert spec.name != "psk_c2", f"QPSK->BPSK at {snr} dB seed {seed}"


@pytest.mark.slow  # ~280 blind resolves (2 SNRs x 14 profiles x 10 seeds)
def test_profile_id_accuracy_degrades_gracefully() -> None:
    # Characterize blind profile-ID across SNR (the spec's "characterized
    # across SNR" -- the coverage gap the whole-branch review flagged).
    from core.blind import resolve_sc_profile
    from core.profiles import SC_CATALOG
    from validation.repro import rng as make_rng
    from validation.synth.channel import add_awgn_at_snr
    from validation.synth.modulators import modulate
    from validation.types import ModScheme

    scheme = {
        "sik_gfsk": ModScheme.GFSK,
        "ble_1m": ModScheme.GFSK,
        "ble_2m": ModScheme.GFSK,
        "fsk_basic": ModScheme.FSK,
        "psk_c2": ModScheme.BPSK,
        "qpsk_link": ModScheme.QPSK,
        "rep_bpsk": ModScheme.BPSK,
        "conv_bpsk": ModScheme.BPSK,
        "rs_bpsk": ModScheme.BPSK,
        "bch_bpsk": ModScheme.BPSK,
        "ldpc_bpsk": ModScheme.BPSK,
        "turbo_bpsk": ModScheme.BPSK,
        "polar_bpsk": ModScheme.BPSK,
        "fountain_bpsk": ModScheme.BPSK,
    }
    acc = {}
    for snr in (25.0, 15.0):
        correct = total = 0
        for name, sp in SC_CATALOG.items():
            p = sp.profile
            tx = modulate(
                bytes(range(24)),
                scheme[name],
                sps=p.sps,
                mod_index=p.mod_index,
                bt=p.bt,
            ).astype(np.complex64)
            for seed in range(10):
                noisy, _, _ = add_awgn_at_snr(tx, snr, make_rng(seed))
                spec, _ = resolve_sc_profile(noisy.astype(np.complex128))
                total += 1
                if spec is not None and spec.name == name:
                    correct += 1
        acc[snr] = correct / total
    assert acc[25.0] >= 0.9  # strong SNR: near-perfect blind profile-ID
    assert acc[15.0] <= acc[25.0]  # graceful (non-increasing) degradation


def _ofdm_noisy(profile_name: str, snr_db: float, seed: int) -> "np.ndarray":
    """Modulate a fixed payload with a catalog OFDM profile at a target SNR.

    The 12-element bit pattern is repeated 40x (480 bits) rather than 8x (96
    bits) so every catalog profile carries several data symbols (>=3 even for
    wifi_40's 220 data-bits/symbol, 5-20+ for the others). A single-symbol
    burst leaves no margin: a normal S&C timing-search offset at low SNR can
    push the sole data symbol past the end of the region, making even the
    correct profile's trial demod return +inf EVM (measured/documented in the
    Task 3 report and confirmed in Task 4's original BLOCKED report -- a
    test-fixture fragility, not a resolver defect). With several symbols, a
    timing-plateau offset costs at most the last symbol and the correct
    profile still yields a finite, low EVM.
    """
    from core.ofdm import modulate_ofdm
    from core.profiles import OFDM_CATALOG
    from validation.repro import rng
    from validation.synth.channel import add_awgn_at_snr

    b = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1] * 40, dtype=np.uint8)
    clean = modulate_ofdm(b, OFDM_CATALOG[profile_name]).astype(np.complex128)
    g = rng(seed)
    noisy, _, _ = add_awgn_at_snr(clean, snr_db, g)
    return noisy.astype(np.complex128)


def test_ofdm_data_evm_low_for_correct_profile() -> None:
    from core.blind import OFDM_EVM_MAX, _ofdm_data_evm
    from core.profiles import OFDM_CATALOG

    rx = _ofdm_noisy("wifi_20", 30.0, 1)
    evm = _ofdm_data_evm(rx, OFDM_CATALOG["wifi_20"])
    assert evm < 0.2
    assert evm < OFDM_EVM_MAX


def test_ofdm_data_evm_empty_region_is_inf() -> None:
    from core.blind import _ofdm_data_evm
    from core.profiles import OFDM_CATALOG

    assert _ofdm_data_evm(
        np.zeros(4, dtype=np.complex128), OFDM_CATALOG["wifi_20"]
    ) == float("inf")


def test_ofdm_evm_separates_same_n_variants() -> None:
    """RISK GATE: correct-profile EVM is separably below same-N-wrong EVM.

    Covers the CP variant (wifi_20 vs wifi_20_longcp) across the normal-SNR
    band. A same-N alternate-pilot-layout variant (wifi_20_altpilot) was also
    measured here and found NOT separably discriminable at this margin
    (shared occupied bins -> only a systematic, payload-dependent CPE bias,
    not a noise-scaled error) -- per the Fallback Ruling in the task header
    it was dropped from OFDM_CATALOG rather than kept with a weakened
    assertion."""
    from core.blind import _ofdm_data_evm
    from core.profiles import OFDM_CATALOG

    for snr in (10.0, 20.0, 30.0):
        rx20 = _ofdm_noisy("wifi_20", snr, 7)
        evm_correct = _ofdm_data_evm(rx20, OFDM_CATALOG["wifi_20"])
        evm_cp = _ofdm_data_evm(rx20, OFDM_CATALOG["wifi_20_longcp"])
        assert evm_correct + 0.2 < evm_cp, f"CP not separated @ {snr} dB"


def test_ofdm_evm_ceiling_rejects_single_carrier() -> None:
    """A misrouted single-carrier region scores above the ceiling on every
    OFDM profile (loud-failure ceiling placement)."""
    from core.blind import OFDM_EVM_MAX, _ofdm_data_evm
    from core.profiles import OFDM_CATALOG
    from validation.synth.modulators import modulate
    from validation.types import ModScheme

    gfsk = modulate(bytes(range(48)), ModScheme.GFSK, sps=8, mod_index=0.7, bt=0.5)
    y = gfsk.astype(np.complex128)
    best = min(_ofdm_data_evm(y, p) for p in OFDM_CATALOG.values())
    assert best > OFDM_EVM_MAX


def test_resolve_ofdm_self_resolves_each_profile() -> None:
    from core.blind import resolve_ofdm_profile
    from core.profiles import OFDM_CATALOG

    for name in OFDM_CATALOG:
        rx = _ofdm_noisy(name, 30.0, 3)
        got, conf = resolve_ofdm_profile(rx)
        assert got == name, f"resolved {got} expected {name}"
        assert conf >= 0.6


def test_resolve_ofdm_rejects_noise() -> None:
    from core.blind import resolve_ofdm_profile
    from validation.repro import rng

    g = rng(4)
    noise = (g.standard_normal(4 * 80) + 1j * g.standard_normal(4 * 80)).astype(
        np.complex128
    )
    got, _ = resolve_ofdm_profile(noise)
    assert got is None


def test_resolve_ofdm_rejects_noncatalog_fft_size() -> None:
    from core.blind import resolve_ofdm_profile
    from core.ofdm import OFDMProfile, modulate_ofdm

    # N=16 OFDM burst -- no catalog profile (32/64/128) locks its sync.
    p16 = OFDMProfile(
        fft_size=16,
        cp_len=4,
        data_carriers=tuple(k for k in range(-6, 7) if k not in (-5, 5) and k != 0),
        pilot_carriers=(-5, 5),
        pilot_values=(1 + 0j, 1 + 0j),
    )
    rx = modulate_ofdm(np.array([1, 0, 1, 1] * 8, dtype=np.uint8), p16).astype(
        np.complex128
    )
    got, _ = resolve_ofdm_profile(rx)
    assert got is None


def test_resolve_ofdm_rejects_single_carrier_region() -> None:
    from core.blind import resolve_ofdm_profile
    from validation.synth.modulators import modulate
    from validation.types import ModScheme

    gfsk = modulate(bytes(range(48)), ModScheme.GFSK, sps=8, mod_index=0.7, bt=0.5)
    got, _ = resolve_ofdm_profile(gfsk.astype(np.complex128))
    assert got is None


def test_resolve_ofdm_snr_sweep_accuracy() -> None:
    """SNR-swept: resolution is accurate across the normal-SNR band."""
    from core.blind import resolve_ofdm_profile
    from core.profiles import OFDM_CATALOG

    for snr in (12.0, 20.0, 30.0):
        correct = 0
        total = 0
        for seed, name in enumerate(OFDM_CATALOG):
            rx = _ofdm_noisy(name, snr, 100 + seed)
            got, _ = resolve_ofdm_profile(rx)
            correct += int(got == name)
            total += 1
        assert correct == total, f"{correct}/{total} @ {snr} dB"


def test_rs_bpsk_blind_resolves() -> None:
    import numpy as np

    from core.blind import resolve_sc_profile
    from validation.synth.modulators import modulate
    from validation.types import ModScheme

    iq = modulate(bytes(range(24)), ModScheme.BPSK, sps=64).astype(np.complex128)
    spec, _ = resolve_sc_profile(iq)
    assert spec is not None and spec.name == "rs_bpsk"


def test_bch_bpsk_blind_resolves() -> None:
    import numpy as np

    from core.blind import resolve_sc_profile
    from validation.synth.modulators import modulate
    from validation.types import ModScheme

    iq = modulate(bytes([0xA5, 0x3C]), ModScheme.BPSK, sps=128).astype(np.complex128)
    spec, _ = resolve_sc_profile(iq)
    assert spec is not None and spec.name == "bch_bpsk"


def test_ldpc_bpsk_blind_resolves() -> None:
    import numpy as np

    from core.blind import resolve_sc_profile
    from validation.synth.modulators import modulate
    from validation.types import ModScheme

    iq = modulate(bytes(range(24)), ModScheme.BPSK, sps=256).astype(np.complex128)
    spec, _ = resolve_sc_profile(iq)
    assert spec is not None and spec.name == "ldpc_bpsk"


def test_turbo_bpsk_blind_resolves() -> None:
    import numpy as np

    from core.blind import resolve_sc_profile
    from validation.synth.modulators import modulate
    from validation.types import ModScheme

    iq = modulate(bytes(range(24)), ModScheme.BPSK, sps=512).astype(np.complex128)
    spec, _ = resolve_sc_profile(iq)
    assert spec is not None and spec.name == "turbo_bpsk"


def test_polar_bpsk_blind_resolves() -> None:
    import numpy as np

    from core.blind import resolve_sc_profile
    from validation.synth.modulators import modulate
    from validation.types import ModScheme

    iq = modulate(bytes(range(8)), ModScheme.BPSK, sps=48).astype(np.complex128)
    spec, _ = resolve_sc_profile(iq)
    assert spec is not None and spec.name == "polar_bpsk"


def test_fountain_bpsk_blind_resolves() -> None:
    import numpy as np

    from core.blind import resolve_sc_profile
    from validation.synth.modulators import modulate
    from validation.types import ModScheme

    iq = modulate(bytes(range(8)), ModScheme.BPSK, sps=96).astype(np.complex128)
    spec, _ = resolve_sc_profile(iq)
    assert spec is not None and spec.name == "fountain_bpsk"
