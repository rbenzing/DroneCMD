"""Tests for the injectable detect-then-classify pipeline."""
from __future__ import annotations

import numpy as np

from validation.pipeline import DetectClassifyPipeline
from validation.types import Detection, LabeledCapture


class StubClassifier:
    """Returns a fixed protocol with confidence proportional to region energy."""

    def __init__(self, label: str = "mavlink") -> None:
        self.label = label

    def classify(self, packet_bytes: bytes, signal_metrics: object = None) -> str:
        return self.label


def _capture_with_burst() -> LabeledCapture:
    silence = np.zeros(300, dtype=np.complex64)
    burst = (np.ones(400) * 0.9).astype(np.complex64)
    iq = np.concatenate([silence, burst, silence])
    return LabeledCapture(
        iq=iq,
        sample_rate=1e6,
        truth_regions=[(300, 700, "mavlink")],
        provenance={"source": "synth", "protocol": "mavlink"},
    )


def test_pipeline_detects_and_labels() -> None:
    pipe = DetectClassifyPipeline(StubClassifier("mavlink"), threshold=0.2, min_gap=100)
    dets = pipe.run(_capture_with_burst())
    assert len(dets) >= 1
    d = dets[0]
    assert isinstance(d, Detection)
    assert d.protocol == "mavlink"
    assert 250 <= d.start <= 350 and 650 <= d.end <= 750


def test_pipeline_no_detections_on_silence() -> None:
    cap = LabeledCapture(
        iq=np.zeros(2048, dtype=np.complex64),
        sample_rate=1e6,
        truth_regions=None,
        provenance={"source": "synth"},
    )
    pipe = DetectClassifyPipeline(StubClassifier(), threshold=0.05)
    assert pipe.run(cap) == []


def test_injectable_detector_is_used() -> None:
    calls = {"n": 0}

    def fake_detector(iq: np.ndarray, threshold: float, min_gap: int) -> list:
        calls["n"] += 1
        return [(10, 50)]

    pipe = DetectClassifyPipeline(StubClassifier("dji"), detector=fake_detector)
    dets = pipe.run(_capture_with_burst())
    assert calls["n"] == 1
    assert dets[0].protocol == "dji" and (dets[0].start, dets[0].end) == (10, 50)


def test_pipeline_routes_ofdm_scheme_to_ofdm_demod(monkeypatch) -> None:
    # Decode routing is blind (per-region classify_family), so the region
    # itself must look like OFDM (high PAPR + CP structure) for the OFDM
    # decode path to be selected -- provenance["scheme"] only selects the
    # OFDM *detector*, not the decode route. Provenance is still set to
    # "ofdm" here to exercise that detector-selection half.
    import validation.pipeline as P
    from validation.synth.modulators import modulate
    from validation.types import ModScheme

    calls = {"ofdm": 0, "sc": 0}
    monkeypatch.setattr(
        P,
        "ofdm_region_to_bytes",
        lambda iq: (
            calls.__setitem__("ofdm", calls["ofdm"] + 1) or (b"\x01", "wifi_20")
        ),
    )
    monkeypatch.setattr(
        P,
        "single_carrier_region_to_bytes",
        lambda iq, sample_rate, *, differential=False, pilot_spacing=0: (
            calls.__setitem__("sc", calls["sc"] + 1) or (b"\x02", None)
        ),
    )
    pkt = modulate(bytes(range(24)), ModScheme.OFDM)
    iq = np.concatenate([np.zeros(300, np.complex64), pkt, np.zeros(300, np.complex64)])
    cap = LabeledCapture(
        iq=iq,
        sample_rate=1e6,
        truth_regions=[(300, 300 + len(pkt), "ocusync")],
        provenance={"source": "synth", "scheme": "ofdm"},
    )
    P.DetectClassifyPipeline(StubClassifier("ocusync"), threshold=0.05, min_gap=64).run(
        cap
    )
    assert calls["ofdm"] >= 1 and calls["sc"] == 0


def test_pipeline_non_ofdm_scheme_uses_single_carrier_path(monkeypatch) -> None:
    import validation.pipeline as P

    calls = {"ofdm": 0, "sc": 0}
    monkeypatch.setattr(
        P,
        "ofdm_region_to_bytes",
        lambda iq: (
            calls.__setitem__("ofdm", calls["ofdm"] + 1) or (b"\x01", "wifi_20")
        ),
    )
    monkeypatch.setattr(
        P,
        "single_carrier_region_to_bytes",
        lambda iq, sample_rate, *, differential=False, pilot_spacing=0: (
            calls.__setitem__("sc", calls["sc"] + 1) or (b"\x02", None)
        ),
    )
    P.DetectClassifyPipeline(StubClassifier("mavlink"), threshold=0.2, min_gap=100).run(
        _capture_with_burst()  # existing helper; provenance has no "scheme"
    )
    assert calls["sc"] >= 1 and calls["ofdm"] == 0


def test_pipeline_ofdm_end_to_end_recovers_payload() -> None:
    from validation.synth.modulators import modulate
    from validation.types import ModScheme

    payload = bytes(range(24))
    pkt = modulate(payload, ModScheme.OFDM)
    iq = np.concatenate([np.zeros(300, np.complex64), pkt, np.zeros(300, np.complex64)])
    captured = {}

    class Spy:
        def classify(self, packet_bytes, signal_metrics=None):
            captured["b"] = packet_bytes
            return "ocusync"

    cap = LabeledCapture(
        iq=iq,
        sample_rate=1e6,
        truth_regions=[(300, 300 + len(pkt), "ocusync")],
        provenance={"source": "synth", "scheme": "ofdm"},
    )
    dets = DetectClassifyPipeline(Spy(), threshold=0.05, min_gap=64).run(cap)
    assert len(dets) >= 1
    assert captured["b"][: len(payload)] == payload


def _run_single_carrier_e2e(scheme, payload: bytes, differential: bool = False):
    """Build a synth capture for ``scheme``, run the pipeline, return recovered bytes.

    Guard-pads the burst with silence on both sides so the amplitude detector
    (constant-modulus preamble + payload) brackets it cleanly.
    """
    from validation.synth.modulators import modulate

    pkt = modulate(payload, scheme, differential=differential)
    iq = np.concatenate([np.zeros(300, np.complex64), pkt, np.zeros(300, np.complex64)])
    captured: dict = {}

    class Spy:
        def classify(self, packet_bytes, signal_metrics=None):
            captured["b"] = packet_bytes
            return "mavlink"

    cap = LabeledCapture(
        iq=iq,
        sample_rate=1e6,
        truth_regions=[(300, 300 + len(pkt), "mavlink")],
        provenance={
            "source": "synth",
            "scheme": scheme.value,
            "differential": differential,
        },
    )
    dets = DetectClassifyPipeline(Spy(), threshold=0.05, min_gap=64).run(cap)
    assert len(dets) >= 1
    return captured.get("b", b"")


def test_pipeline_fsk_end_to_end_recovers_payload() -> None:
    from validation.types import ModScheme

    payload = bytes(range(16))
    recovered = _run_single_carrier_e2e(ModScheme.FSK, payload)
    assert recovered[: len(payload)] == payload


def test_pipeline_bpsk_end_to_end_recovers_payload() -> None:
    from validation.types import ModScheme

    payload = bytes(range(16))
    recovered = _run_single_carrier_e2e(ModScheme.BPSK, payload)
    assert recovered[: len(payload)] == payload


def test_pipeline_qpsk_end_to_end_recovers_payload() -> None:
    from validation.types import ModScheme

    payload = bytes(range(16))
    recovered = _run_single_carrier_e2e(ModScheme.QPSK, payload)
    assert recovered[: len(payload)] == payload


def test_pipeline_differential_qpsk_end_to_end_recovers_payload() -> None:
    from validation.types import ModScheme

    payload = bytes(range(16))
    recovered = _run_single_carrier_e2e(ModScheme.QPSK, payload, differential=True)
    assert recovered[: len(payload)] == payload


def test_pipeline_roundtrip_with_pilots() -> None:
    # A piloted QPSK capture demodulates back to its payload through the
    # pipeline's single-carrier path (pilot_spacing read from provenance).
    from validation.pipeline import single_carrier_region_to_bytes
    from validation.synth.modulators import modulate
    from validation.types import ModScheme

    data = bytes(range(24))
    rx = modulate(data, ModScheme.QPSK, sps=8, pilot_spacing=8)
    out, profile = single_carrier_region_to_bytes(rx, 2_048_000.0, pilot_spacing=8)
    assert profile == "qpsk_link"
    assert out[: len(data)] == data


def test_single_carrier_region_blind_resolves_and_decodes() -> None:
    from validation.pipeline import single_carrier_region_to_bytes
    from validation.synth.modulators import modulate
    from validation.types import ModScheme

    data = bytes(range(24))
    # Blindly resolve + decode a GFSK burst with NO scheme/sps hint.
    rx = modulate(data, ModScheme.GFSK, sps=8, mod_index=0.7, bt=0.5)
    out, profile = single_carrier_region_to_bytes(rx, 2_048_000.0)
    assert profile == "sik_gfsk"
    assert out[: len(data)] == data


def test_single_carrier_region_blind_sps4() -> None:
    from validation.pipeline import single_carrier_region_to_bytes
    from validation.synth.modulators import modulate
    from validation.types import ModScheme

    data = bytes(range(24))
    rx = modulate(data, ModScheme.GFSK, sps=4, mod_index=0.5, bt=0.5)  # ble_2m
    out, profile = single_carrier_region_to_bytes(rx, 2_048_000.0)
    assert profile == "ble_2m"  # the old sps==8 guard would have raised here
    assert out[: len(data)] == data


def test_single_carrier_region_noise_no_lock() -> None:
    from validation.pipeline import single_carrier_region_to_bytes

    rng = np.random.default_rng(0)
    noise = (rng.standard_normal(600) + 1j * rng.standard_normal(600)).astype(
        np.complex64
    )
    out, profile = single_carrier_region_to_bytes(noise, 2_048_000.0)
    assert out == b""
    assert profile is None


def test_detection_carries_resolved_profile() -> None:
    from validation.types import Detection

    d = Detection(start=0, end=10, protocol="x", confidence=1.0)
    assert d.resolved_profile is None  # default
    d2 = Detection(
        start=0, end=10, protocol="x", confidence=1.0, resolved_profile="ble_1m"
    )
    assert d2.resolved_profile == "ble_1m"


def test_ofdm_decode_fails_closed_on_misrouted_sc_region() -> None:
    # A LOW-SNR single-carrier region can be blindly misrouted to the OFDM
    # decoder by classify_family (its PAPR rises with noise). It MUST fail
    # closed -- no bytes -- never silent wrong bits. (A CLEAN single-carrier
    # burst is routed to SINGLE_CARRIER by classify_family and never reaches
    # the OFDM decoder, so the realistic risk is only this low-SNR case.)
    from validation.pipeline import ofdm_region_to_bytes
    from validation.repro import rng
    from validation.synth.channel import add_awgn_at_snr
    from validation.synth.modulators import modulate
    from validation.types import ModScheme

    sc = modulate(bytes(range(24)), ModScheme.FSK, sps=8, mod_index=0.7)
    noisy, _, _ = add_awgn_at_snr(sc.astype(np.complex64), 5.0, rng(0))
    out, resolved = ofdm_region_to_bytes(noisy.astype(np.complex64))
    assert out == b""  # OFDM demod fails closed on a misrouted SC region
    assert resolved is None


def test_ofdm_region_to_bytes_blind_returns_name() -> None:
    import numpy as np

    from core.ofdm import modulate_ofdm
    from core.profiles import OFDM_CATALOG
    from validation.pipeline import ofdm_region_to_bytes

    for name in ("wifi_20", "wifi_40"):
        rx = modulate_ofdm(
            np.array([1, 0, 1, 1, 0, 0, 1, 0] * 12, dtype=np.uint8), OFDM_CATALOG[name]
        )
        pkt, resolved = ofdm_region_to_bytes(rx.astype(np.complex64))
        assert resolved == name
        assert len(pkt) > 0


def test_ofdm_region_to_bytes_noise_is_loud_none() -> None:
    import numpy as np

    from validation.pipeline import ofdm_region_to_bytes
    from validation.repro import rng

    g = rng(9)
    noise = (g.standard_normal(4 * 80) + 1j * g.standard_normal(4 * 80)).astype(
        np.complex64
    )
    pkt, resolved = ofdm_region_to_bytes(noise)
    assert pkt == b"" and resolved is None


def test_coded_region_decodes_and_crc_loud_fail() -> None:
    from core.coding import CODING_CATALOG
    from validation.pipeline import single_carrier_region_to_bytes
    from validation.synth.modulators import modulate
    from validation.types import ModScheme

    payload = bytes(range(20))
    iq = modulate(payload, ModScheme.BPSK, sps=16, coding=CODING_CATALOG["rep3"])
    out, name = single_carrier_region_to_bytes(iq.astype(np.complex64), 1e6)
    assert name == "rep_bpsk"
    assert out[: len(payload)] == payload  # coded round-trip recovers the payload

    # Corrupt the payload region heavily so rep3 cannot correct -> CRC fails -> loud (b"", None)
    bad = iq.copy()
    body = bad[26 * 16 :]  # skip the Barker-13x2 preamble (26*sps samples)
    body[: (2 * body.size) // 3] = 0.0  # wipe 2/3 of the payload
    out2, name2 = single_carrier_region_to_bytes(bad.astype(np.complex64), 1e6)
    assert out2 == b"" and name2 is None


def test_conv_coded_region_decodes_via_soft_path() -> None:
    from core.coding import CODING_CATALOG
    from validation.pipeline import single_carrier_region_to_bytes
    from validation.synth.modulators import modulate
    from validation.types import ModScheme

    payload = bytes(range(16))
    iq = modulate(payload, ModScheme.BPSK, sps=32, coding=CODING_CATALOG["conv_k7_r12"])
    out, name = single_carrier_region_to_bytes(iq.astype(np.complex64), 1e6)
    assert name == "conv_bpsk"
    assert out[: len(payload)] == payload


def test_conv_coded_region_crc_loud_fail() -> None:
    from core.coding import CODING_CATALOG
    from validation.pipeline import single_carrier_region_to_bytes
    from validation.synth.modulators import modulate
    from validation.types import ModScheme

    iq = modulate(
        bytes(range(16)), ModScheme.BPSK, sps=32, coding=CODING_CATALOG["conv_k7_r12"]
    ).copy()
    body = iq[26 * 32 :]
    body[
        : (3 * body.size) // 4
    ] = 0.0  # wipe most of the payload -> beyond Viterbi -> CRC fail
    out, name = single_carrier_region_to_bytes(iq.astype(np.complex64), 1e6)
    assert out == b"" and name is None


def test_rs_bpsk_blind_end_to_end() -> None:
    from core.coding import CODING_CATALOG
    from validation.pipeline import single_carrier_region_to_bytes
    from validation.synth.modulators import modulate
    from validation.types import ModScheme

    payload = bytes(range(24))
    iq = modulate(
        payload, ModScheme.BPSK, sps=64, coding=CODING_CATALOG["rs_255_239"]
    ).astype(np.complex128)
    rng = np.random.default_rng(1)
    noise = (rng.standard_normal(iq.size) + 1j * rng.standard_normal(iq.size)) * 0.1
    rx = (iq + noise).astype(np.complex64)
    out, name = single_carrier_region_to_bytes(rx, 1e6)
    assert name == "rs_bpsk"
    assert out[: len(payload)] == payload


def test_bch_bpsk_blind_end_to_end() -> None:
    from core.coding import CODING_CATALOG
    from validation.pipeline import single_carrier_region_to_bytes
    from validation.synth.modulators import modulate
    from validation.types import ModScheme

    payload = bytes(range(24))
    iq = modulate(
        payload, ModScheme.BPSK, sps=128, coding=CODING_CATALOG["bch_255_223"]
    ).astype(np.complex128)
    rng = np.random.default_rng(1)
    noise = (rng.standard_normal(iq.size) + 1j * rng.standard_normal(iq.size)) * 0.1
    rx = (iq + noise).astype(np.complex64)
    out, name = single_carrier_region_to_bytes(rx, 1e6)
    assert name == "bch_bpsk"
    assert out[: len(payload)] == payload


def test_ldpc_bpsk_blind_end_to_end() -> None:
    from core.coding import CODING_CATALOG
    from validation.pipeline import single_carrier_region_to_bytes
    from validation.synth.modulators import modulate
    from validation.types import ModScheme

    payload = bytes(range(24))
    iq = modulate(
        payload, ModScheme.BPSK, sps=256, coding=CODING_CATALOG["ldpc_648_r12"]
    ).astype(np.complex128)
    rng = np.random.default_rng(1)
    noise = (rng.standard_normal(iq.size) + 1j * rng.standard_normal(iq.size)) * 0.1
    rx = (iq + noise).astype(np.complex64)
    out, name = single_carrier_region_to_bytes(rx, 1e6)
    assert name == "ldpc_bpsk"
    assert out[: len(payload)] == payload


def test_turbo_bpsk_blind_end_to_end() -> None:
    from core.coding import CODING_CATALOG
    from validation.pipeline import single_carrier_region_to_bytes
    from validation.synth.modulators import modulate
    from validation.types import ModScheme

    payload = bytes(range(24))
    iq = modulate(
        payload, ModScheme.BPSK, sps=512, coding=CODING_CATALOG["turbo_r13"]
    ).astype(np.complex128)
    rng = np.random.default_rng(1)
    noise = (rng.standard_normal(iq.size) + 1j * rng.standard_normal(iq.size)) * 0.1
    rx = (iq + noise).astype(np.complex64)
    out, name = single_carrier_region_to_bytes(rx, 1e6)
    assert name == "turbo_bpsk"
    assert out[: len(payload)] == payload


def test_pipeline_records_resolved_ofdm_profile() -> None:
    from validation import create_synth_dataset
    from validation.pipeline import DetectClassifyPipeline

    ds = create_synth_dataset(
        protocols=["wide"],
        snr_grid_db=[30.0],
        n_per_cell=1,
        profile_by_protocol={"wide": "wifi_40"},
        seed=5,
    )

    class _Clf:
        def classify(self, packet_bytes: bytes, signal_metrics=None) -> str:
            return "wide"

    pipe = DetectClassifyPipeline(_Clf())
    dets = pipe.run(next(iter(ds)))
    assert any(d.resolved_profile == "wifi_40" for d in dets)
