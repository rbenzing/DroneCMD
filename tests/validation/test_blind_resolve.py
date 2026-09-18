from __future__ import annotations

import numpy as np

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
    """Modulate a fixed payload with a catalog OFDM profile at a target SNR."""
    from core.ofdm import modulate_ofdm
    from core.profiles import OFDM_CATALOG
    from validation.repro import rng
    from validation.synth.channel import add_awgn_at_snr

    b = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1] * 8, dtype=np.uint8)
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
