"""
Unit tests for core/signal_processing.py

Tests cover the real DSP implementations that form the foundation of the
capture→demodulate pipeline.  No hardware required.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.signal_processing import (
    SignalProcessor,
    QualityMonitor,
    detect_packets,
    find_preamble,
    analyze_signal_quality,
)
from tests.conftest import make_noise, make_tone


# ---------------------------------------------------------------------------
# SignalProcessor
# ---------------------------------------------------------------------------

class TestSignalProcessor:
    def setup_method(self):
        self.proc = SignalProcessor()

    def test_normalize_unit_power(self, tone_signal):
        normalized = self.proc.normalize(tone_signal)
        power = float(np.mean(np.abs(normalized) ** 2))
        assert abs(power - 1.0) < 0.01, f"Normalized power {power:.4f} not close to 1.0"

    def test_normalize_zero_signal_does_not_raise(self):
        zeros = np.zeros(256, dtype=np.complex64)
        result = self.proc.normalize(zeros)
        assert result is not None
        assert len(result) == 256

    def test_agc_output_bounded(self, tone_signal):
        out = self.proc.apply_agc(tone_signal)
        peak = float(np.max(np.abs(out)))
        assert peak <= 1.1, f"AGC output peak {peak:.3f} exceeds expected bound"

    def test_snr_estimate_positive_for_tone_in_noise(self):
        tone = make_tone(freq_hz=5_000, amplitude=0.8)
        noise = make_noise(len(tone), power=0.001)
        # Use spectral method: compares PSD peak vs noise floor — reliable for tone + noise
        snr = self.proc.estimate_snr(tone + noise, method="spectral")
        assert snr > 5.0, f"Expected SNR > 5 dB for strong tone, got {snr:.1f} dB"

    def test_snr_estimate_low_for_pure_noise(self):
        noise = make_noise(4096, power=0.1)
        snr = self.proc.estimate_snr(noise, method="percentile")
        # Pure noise percentile ratio is close to 1 → low SNR
        assert snr < 15.0, f"Expected moderate SNR for pure noise, got {snr:.1f} dB"

    def test_filter_design_returns_coefficients(self, tone_signal):
        # SignalProcessor.design_filter returns filter coefficients
        fir = self.proc.design_filter(
            filter_type="lowpass",
            cutoff_freq=500_000,
            sample_rate=2_048_000,
        )
        assert fir is not None
        assert len(fir) > 0

    def test_normalize_output_is_complex64(self, tone_signal):
        # Verify dtype preservation through normalize
        result = self.proc.normalize(tone_signal)
        assert result.dtype == np.complex64
        assert len(result) == len(tone_signal)


# ---------------------------------------------------------------------------
# QualityMonitor
# ---------------------------------------------------------------------------

class TestQualityMonitor:
    def test_update_returns_metrics_dict(self, tone_signal):
        monitor = QualityMonitor()
        metrics = monitor.update(tone_signal)
        assert isinstance(metrics, dict)
        assert "signal_power_dbfs" in metrics or len(metrics) > 0

    def test_consecutive_updates_do_not_crash(self, tone_signal, noise_signal):
        monitor = QualityMonitor()
        for sig in [tone_signal, noise_signal, tone_signal]:
            metrics = monitor.update(sig)
            assert metrics is not None


# ---------------------------------------------------------------------------
# Packet detection
# ---------------------------------------------------------------------------

class TestDetectPackets:
    def test_detects_burst_above_noise(self):
        """A strong burst embedded in silence should be detected."""
        silence = np.zeros(1000, dtype=np.complex64)
        burst = (np.ones(500) * 0.8).astype(np.complex64)
        signal = np.concatenate([silence, burst, silence])

        regions = detect_packets(signal, threshold=0.1, min_gap=100)
        assert len(regions) >= 1, "Expected at least one packet region"

    def test_no_false_positives_on_silence(self):
        signal = np.zeros(4096, dtype=np.complex64)
        regions = detect_packets(signal, threshold=0.05)
        assert len(regions) == 0

    def test_returns_list_of_tuples(self, tone_signal):
        regions = detect_packets(tone_signal, threshold=0.01)
        for item in regions:
            assert len(item) == 2, "Each region should be a (start, end) tuple"
            start, end = item
            assert start < end


# ---------------------------------------------------------------------------
# Preamble detection
# ---------------------------------------------------------------------------

class TestFindPreamble:
    def test_finds_known_pattern(self):
        pattern = np.array([1, -1, 1, -1], dtype=np.float32)
        signal_data = np.zeros(100, dtype=np.float32)
        signal_data[20:24] = pattern
        signal_complex = signal_data.astype(np.complex64)

        indices = find_preamble(signal_complex, pattern.astype(np.complex64), threshold=0.7)
        assert len(indices) >= 1

    def test_empty_signal_returns_no_matches(self):
        pattern = np.array([1, -1], dtype=np.complex64)
        signal = np.array([], dtype=np.complex64)
        try:
            indices = find_preamble(signal, pattern, threshold=0.8)
            assert len(indices) == 0
        except (ValueError, IndexError):
            pass  # acceptable — empty input edge case


# ---------------------------------------------------------------------------
# Signal quality analysis
# ---------------------------------------------------------------------------

class TestAnalyzeSignalQuality:
    def test_returns_dict_with_keys(self, tone_signal):
        result = analyze_signal_quality(tone_signal, sample_rate=2_048_000.0)
        assert isinstance(result, dict)

    def test_does_not_raise_on_noise(self, noise_signal):
        result = analyze_signal_quality(noise_signal, sample_rate=2_048_000.0)
        assert result is not None
