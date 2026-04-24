"""
Shared fixtures for DroneCMD test suite.

Hardware fixtures are skipped automatically when SDR devices are absent.
All fixtures that return IQ data use numpy.complex64 (the project standard).
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

IQSamples = npt.NDArray[np.complex64]


# ---------------------------------------------------------------------------
# Signal generation helpers
# ---------------------------------------------------------------------------

def make_tone(
    freq_hz: float = 1_000.0,
    sample_rate: float = 2_048_000.0,
    duration_s: float = 0.01,
    amplitude: float = 0.5,
) -> IQSamples:
    """Return a single-tone complex64 IQ signal."""
    n = int(sample_rate * duration_s)
    t = np.arange(n) / sample_rate
    return (amplitude * np.exp(1j * 2 * np.pi * freq_hz * t)).astype(np.complex64)


def make_noise(n: int = 2048, power: float = 0.01) -> IQSamples:
    """Return AWGN complex64 noise samples."""
    rng = np.random.default_rng(seed=42)
    return (
        np.sqrt(power / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    ).astype(np.complex64)


def make_bpsk_packet(payload: bytes = b"\xDE\xAD\xBE\xEF", sample_rate: int = 2_000_000) -> IQSamples:
    """Return a minimal BPSK-modulated IQ frame for the payload bytes."""
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    symbols = (2 * bits.astype(np.float32) - 1)  # map 0→-1, 1→+1
    sps = sample_rate // 10_000  # 10 kbaud
    samples = np.repeat(symbols, sps).astype(np.complex64)
    return samples


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tone_signal() -> IQSamples:
    return make_tone()


@pytest.fixture
def noise_signal() -> IQSamples:
    return make_noise(4096)


@pytest.fixture
def bpsk_packet() -> IQSamples:
    return make_bpsk_packet()


@pytest.fixture
def dji_raw_packet() -> bytes:
    """Minimal syntactically valid DJI-like packet (header + zeroed payload + dummy CRC)."""
    header = bytes([
        0x55, 0xAA,   # DJI sync bytes
        0x00, 0x10,   # length = 16
        0x01,         # sequence number
        0x00,         # message type: telemetry
    ])
    payload = bytes(8)
    crc = (sum(header + payload) & 0xFFFF).to_bytes(2, "little")
    return header + payload + crc


@pytest.fixture
def mavlink_v1_heartbeat() -> bytes:
    """Minimal MAVLink v1 HEARTBEAT packet (msgid=0)."""
    payload = bytes([
        0x00, 0x00, 0x00, 0x00,  # custom_mode
        0x00,                     # type
        0x00,                     # autopilot
        0x00,                     # base_mode
        0x00,                     # system_status
        0x03,                     # mavlink_version
    ])
    msg_len = len(payload)
    seq = 0
    sys_id = 1
    comp_id = 1
    msg_id = 0
    header = bytes([0xFE, msg_len, seq, sys_id, comp_id, msg_id])
    # CRC-16/MCRF4XX (simplified — just need bytes that look valid for parsing tests)
    crc = bytes([0x00, 0x00])
    return header + payload + crc
