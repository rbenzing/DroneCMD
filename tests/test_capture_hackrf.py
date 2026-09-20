"""Tests for the SoapySDR HackRF (RX-only) capture backend (``core.capture``).

The backend is receive-only and never transmits. These unit tests inject a
fake ``SoapySDR`` module so the RX wiring is verified without any hardware; the
``@pytest.mark.hardware`` smoke test at the end runs a real RX capture only when
SoapySDR and a HackRF are actually present (skipped otherwise).
"""
from __future__ import annotations

import types
from unittest.mock import MagicMock

import numpy as np
import pytest


def _fake_soapysdr() -> types.SimpleNamespace:
    """Stand-in ``SoapySDR`` module whose ``Device`` yields a controllable mock.

    ``readStream`` fills the caller's buffer and returns a status object with
    ``ret`` = the number of samples, mirroring the real SoapySDR contract.
    """
    fake = types.SimpleNamespace()
    fake.SOAPY_SDR_RX = 1
    fake.SOAPY_SDR_CF32 = "CF32"
    dev = MagicMock(name="SoapyDevice")

    def read_stream(stream, buffs, num_elems, timeoutUs=0):  # noqa: N803, ANN001
        buffs[0][:num_elems] = np.complex64(0.1 + 0.2j)
        return types.SimpleNamespace(ret=int(num_elems), flags=0, timeNs=0)

    dev.readStream.side_effect = read_stream
    dev.setupStream.return_value = object()
    fake.Device = MagicMock(name="DeviceFactory", return_value=dev)
    fake._dev = dev  # exposed for assertions
    return fake


def test_hackrf_requires_soapysdr(monkeypatch) -> None:
    import core.capture as cap

    monkeypatch.setattr(cap, "SOAPY_SDR_AVAILABLE", False)
    cfg = cap.SDRConfig(platform=cap.SDRPlatform.HACKRF)
    with pytest.raises(RuntimeError, match="SoapySDR not available"):
        cap.SoapyHackRFHardware(cfg)


def test_hackrf_rx_open_configure_read(monkeypatch) -> None:
    import core.capture as cap

    fake = _fake_soapysdr()
    monkeypatch.setattr(cap, "SoapySDR", fake)
    monkeypatch.setattr(cap, "SOAPY_SDR_AVAILABLE", True)
    cfg = cap.SDRConfig(
        platform=cap.SDRPlatform.HACKRF,
        frequency_hz=2.44e9,
        sample_rate_hz=8e6,
        gain_mode=cap.GainMode.MANUAL,
        gain_db=32.0,
        bandwidth_hz=6e6,
    )
    hw = cap.SoapyHackRFHardware(cfg)
    hw.open()
    assert hw.is_connected
    hw.configure(cfg)

    dev = fake._dev
    dev.setSampleRate.assert_called_once_with(fake.SOAPY_SDR_RX, 0, 8e6)
    dev.setFrequency.assert_called_once_with(fake.SOAPY_SDR_RX, 0, 2.44e9)
    dev.setBandwidth.assert_called_once_with(fake.SOAPY_SDR_RX, 0, 6e6)
    dev.setGain.assert_called_once_with(fake.SOAPY_SDR_RX, 0, 32.0)
    dev.setupStream.assert_called_once_with(fake.SOAPY_SDR_RX, fake.SOAPY_SDR_CF32)
    dev.activateStream.assert_called_once()

    x = hw.read_samples(4096)
    assert x.dtype == np.complex64 and x.shape == (4096,)
    assert np.allclose(x, np.complex64(0.1 + 0.2j))

    # RX-only: the backend must never call any transmit API.
    assert not dev.setupStreamTx.called if hasattr(dev, "setupStreamTx") else True
    for attr in dir(dev):
        assert "writeStream" != attr or not getattr(dev, attr).called

    hw.close()
    dev.deactivateStream.assert_called_once()
    dev.closeStream.assert_called_once()
    assert not hw.is_connected


def test_hackrf_auto_gain_falls_back_when_agc_unsupported(monkeypatch) -> None:
    import core.capture as cap

    fake = _fake_soapysdr()
    fake._dev.setGainMode.side_effect = RuntimeError("AGC unsupported")
    monkeypatch.setattr(cap, "SoapySDR", fake)
    monkeypatch.setattr(cap, "SOAPY_SDR_AVAILABLE", True)
    cfg = cap.SDRConfig(platform=cap.SDRPlatform.HACKRF, gain_mode=cap.GainMode.AUTO)
    hw = cap.SoapyHackRFHardware(cfg)
    hw.open()
    hw.configure(cfg)
    # AGC attempt failed -> a manual default gain was set instead (no crash).
    fake._dev.setGain.assert_called_once()


def test_hackrf_read_stream_error_is_loud(monkeypatch) -> None:
    import core.capture as cap

    fake = _fake_soapysdr()
    fake._dev.readStream.side_effect = lambda *a, **k: types.SimpleNamespace(ret=-1)
    monkeypatch.setattr(cap, "SoapySDR", fake)
    monkeypatch.setattr(cap, "SOAPY_SDR_AVAILABLE", True)
    cfg = cap.SDRConfig(platform=cap.SDRPlatform.HACKRF)
    hw = cap.SoapyHackRFHardware(cfg)
    hw.open()
    hw.configure(cfg)
    with pytest.raises(RuntimeError):
        hw.read_samples(1024)


def test_create_hardware_interface_routes_hackrf(monkeypatch) -> None:
    import core.capture as cap

    fake = _fake_soapysdr()
    monkeypatch.setattr(cap, "SoapySDR", fake)
    monkeypatch.setattr(cap, "SOAPY_SDR_AVAILABLE", True)
    cfg = cap.SDRConfig(platform=cap.SDRPlatform.HACKRF)
    live = cap.EnhancedLiveCapture(cfg)
    assert isinstance(live.hardware, cap.SoapyHackRFHardware)


@pytest.mark.hardware
def test_hackrf_rx_on_air_smoke() -> None:
    """RX-only on-air smoke test — SKIPPED unless SoapySDR + a HackRF are present.

    Captures a short buffer at 2.44 GHz and asserts basic sanity (dtype, length,
    finite, nonzero power). Receive-only: never transmits.
    """
    import core.capture as cap

    if not cap.SOAPY_SDR_AVAILABLE:
        pytest.skip("SoapySDR runtime not installed (see README: PothosSDR)")
    cfg = cap.SDRConfig(
        platform=cap.SDRPlatform.HACKRF,
        frequency_hz=2.44e9,
        sample_rate_hz=8e6,
        gain_mode=cap.GainMode.MANUAL,
        gain_db=32.0,
    )
    hw = cap.SoapyHackRFHardware(cfg)
    try:
        hw.open()
    except RuntimeError as e:
        pytest.skip(f"no HackRF device reachable: {e}")
    try:
        hw.configure(cfg)
        x = hw.read_samples(65536)
        assert x.dtype == np.complex64 and x.size == 65536
        assert np.isfinite(x).all()
        assert float(np.mean(np.abs(x) ** 2)) > 0.0
    finally:
        hw.close()
