"""
Unit tests for core/fhss.py

Tests cover FCC compliance validation, hop sequence generation, BPSK modulation,
and the prepare_transmit_frames pipeline.  No hardware required.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.fhss import (
    FHSSBand,
    FHSSConfig,
    FHSSCore,
    SimpleFHSS,
    EnhancedFHSSEngine,
    HopFrame,
    create_fcc_compliant_fhss,
)


# ---------------------------------------------------------------------------
# FHSSConfig validation
# ---------------------------------------------------------------------------

class TestFHSSConfig:
    def test_valid_2_4ghz_config(self):
        cfg = FHSSConfig(
            center_freq_hz=2_440_000_000.0,
            channel_spacing_hz=1_000_000.0,
            hop_count=50,
            band=FHSSBand.ISM_2_4_GHz,
        )
        assert cfg.center_freq_hz == 2_440_000_000.0

    def test_auto_band_detection_2_4ghz(self):
        cfg = FHSSConfig(
            center_freq_hz=2_440_000_000.0,
            channel_spacing_hz=1_000_000.0,
            hop_count=50,
            validate_fcc_compliance=True,  # triggers auto-detection
        )
        assert cfg.band == FHSSBand.ISM_2_4_GHz

    def test_invalid_hop_count_raises(self):
        with pytest.raises(ValueError):
            FHSSConfig(center_freq_hz=2_440_000_000.0, channel_spacing_hz=1_000_000.0, hop_count=0)

    def test_invalid_spacing_raises(self):
        with pytest.raises(ValueError):
            FHSSConfig(center_freq_hz=2_440_000_000.0, channel_spacing_hz=0, hop_count=25)


# ---------------------------------------------------------------------------
# FHSSCore — hop sequence
# ---------------------------------------------------------------------------

class TestFHSSCore:
    def setup_method(self):
        cfg = FHSSConfig(
            center_freq_hz=2_440_000_000.0,
            channel_spacing_hz=1_000_000.0,
            hop_count=25,
            band=FHSSBand.ISM_2_4_GHz,
        )
        self.core = FHSSCore(cfg)

    def test_hop_sequence_length(self):
        seq = self.core.generate_hop_sequence(50)
        assert len(seq) == 50

    def test_hop_sequence_deterministic_with_seed(self):
        seq1 = self.core.generate_hop_sequence(20, seed=42)
        seq2 = self.core.generate_hop_sequence(20, seed=42)
        assert seq1 == seq2

    def test_hop_sequence_different_seeds(self):
        seq1 = self.core.generate_hop_sequence(20, seed=1)
        seq2 = self.core.generate_hop_sequence(20, seed=2)
        assert seq1 != seq2

    def test_channel_frequencies_within_band(self):
        band_min, band_max = FHSSBand.ISM_2_4_GHz.frequency_range
        for freq in self.core.channel_frequencies:
            assert band_min <= freq <= band_max, f"Frequency {freq/1e6:.1f} MHz out of band"

    def test_split_packet_into_chunks(self):
        data = bytes(range(64))
        chunks = self.core.split_packet_into_chunks(data, hop_count=4)
        assert len(chunks) == 4
        assert b"".join(chunks) == data


# ---------------------------------------------------------------------------
# SimpleFHSS — backward-compatible interface
# ---------------------------------------------------------------------------

class TestSimpleFHSS:
    def setup_method(self):
        self.fhss = SimpleFHSS(
            center_freq_hz=2_440_000_000.0,
            channel_spacing_hz=1_000_000.0,
            hops=25,
        )

    def test_prepare_transmit_frames_returns_list(self):
        packet = b"\x00\x01\x02\x03" * 8
        frames = self.fhss.prepare_transmit_frames(packet, sample_rate=2_000_000, bitrate=10_000)
        assert isinstance(frames, list)
        assert len(frames) > 0

    def test_prepare_transmit_frames_tuple_structure(self):
        frames = self.fhss.prepare_transmit_frames(b"hello world", sample_rate=2_000_000, bitrate=10_000)
        for freq, samples, duration in frames:
            assert isinstance(freq, float)
            assert isinstance(samples, np.ndarray)
            assert duration > 0.0

    def test_empty_packet_raises(self):
        with pytest.raises((ValueError, Exception)):
            self.fhss.prepare_transmit_frames(b"")


# ---------------------------------------------------------------------------
# EnhancedFHSSEngine — HopFrame interface
# ---------------------------------------------------------------------------

class TestEnhancedFHSSEngine:
    def setup_method(self):
        cfg = FHSSConfig(
            center_freq_hz=2_440_000_000.0,
            channel_spacing_hz=1_000_000.0,
            hop_count=25,
            band=FHSSBand.ISM_2_4_GHz,
        )
        self.engine = EnhancedFHSSEngine(cfg)

    def test_prepare_transmit_frames_returns_hop_frames(self):
        packet = bytes(range(32))
        frames = self.engine.prepare_transmit_frames(packet, sample_rate=2_000_000, bitrate=10_000)
        assert all(isinstance(f, HopFrame) for f in frames)

    def test_hop_frames_have_iq_samples(self):
        frames = self.engine.prepare_transmit_frames(b"test payload", sample_rate=2_000_000, bitrate=10_000)
        for frame in frames:
            assert frame.iq_samples.dtype == np.complex64
            assert len(frame.iq_samples) > 0

    def test_hop_frames_chunk_data_reassembles(self):
        payload = b"reconstruct me"
        frames = self.engine.prepare_transmit_frames(payload, sample_rate=2_000_000, bitrate=10_000)
        reassembled = b"".join(f.chunk_data for f in frames)
        assert reassembled == payload

    def test_frequency_span_property(self):
        span = self.engine.frequency_span_hz
        assert span > 0

    def test_fcc_band_validation_rejects_out_of_band(self):
        cfg = FHSSConfig(
            center_freq_hz=2_440_000_000.0,
            channel_spacing_hz=100_000_000.0,  # huge spacing — will exceed band
            hop_count=25,
            band=FHSSBand.ISM_2_4_GHz,
            validate_fcc_compliance=True,  # must be True to trigger validation
        )
        with pytest.raises(ValueError):
            EnhancedFHSSEngine(cfg)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

class TestCreateFccCompliantFhss:
    def test_returns_enhanced_engine(self):
        engine = create_fcc_compliant_fhss(
            center_freq_hz=2_440_000_000.0,
            band=FHSSBand.ISM_2_4_GHz,
        )
        assert engine is not None

    def test_rejects_out_of_band_center_freq(self):
        with pytest.raises((ValueError, Exception)):
            create_fcc_compliant_fhss(
                center_freq_hz=100_000_000.0,  # FM radio — not ISM
                band=FHSSBand.ISM_2_4_GHz,
            )
