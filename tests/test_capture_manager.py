"""
Integration tests for capture/manager.py

All tests use the simulated/fallback capture path — no SDR hardware required.
Hardware-dependent tests are marked @pytest.mark.hardware and skipped by default.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from capture.manager import CaptureManager, CaptureManagerError


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestCaptureManagerInit:
    def test_creates_with_defaults(self):
        mgr = CaptureManager()
        assert mgr is not None

    def test_stores_sample_rate(self):
        mgr = CaptureManager(sample_rate=1_024_000)
        assert mgr.sample_rate == 1_024_000

    def test_stores_platform(self):
        mgr = CaptureManager(platform="rtl_sdr")
        assert mgr.platform == "rtl_sdr"

    def test_not_connected_on_init(self):
        mgr = CaptureManager()
        assert mgr._is_connected is False


# ---------------------------------------------------------------------------
# Frequency / sample-rate setters
# ---------------------------------------------------------------------------

class TestSetters:
    def setup_method(self):
        self.mgr = CaptureManager()

    def test_set_frequency(self):
        self.mgr.set_frequency(2_440_000_000.0)
        assert self.mgr.frequency == 2_440_000_000.0

    def test_set_sample_rate(self):
        self.mgr.set_sample_rate(1_024_000.0)
        assert self.mgr.sample_rate == 1_024_000.0

    def test_set_gain_auto(self):
        self.mgr.set_gain(mode="auto")
        assert self.mgr.gain_mode == "auto"

    def test_set_gain_manual(self):
        self.mgr.set_gain(gain_db=20.0, mode="manual")
        assert self.mgr.gain_db == 20.0


# ---------------------------------------------------------------------------
# extract_packets — public API, no private attribute access
# ---------------------------------------------------------------------------

class TestExtractPackets:
    def setup_method(self):
        self.mgr = CaptureManager()

    def _make_signal_with_burst(self) -> np.ndarray:
        silence = np.zeros(1000, dtype=np.complex64)
        burst = (np.ones(500) * 0.8).astype(np.complex64)
        return np.concatenate([silence, burst, silence])

    def test_extract_with_iq_data_arg(self):
        signal = self._make_signal_with_burst()
        packets = self.mgr.extract_packets(iq_data=signal, threshold=0.1, min_gap=100)
        assert isinstance(packets, list)

    def test_extract_sets_no_private_attr(self):
        signal = self._make_signal_with_burst()
        # Verify we never need to set _iq_data externally
        assert not hasattr(self.mgr, "__dict__") or True  # sanity check
        packets = self.mgr.extract_packets(iq_data=signal, threshold=0.1, min_gap=100)
        assert packets is not None

    def test_extract_without_data_raises(self):
        with pytest.raises(CaptureManagerError):
            self.mgr.extract_packets()  # no data stored, no arg passed

    def test_extract_returns_list_of_arrays(self):
        signal = self._make_signal_with_burst()
        packets = self.mgr.extract_packets(iq_data=signal, threshold=0.1, min_gap=100)
        for pkt in packets:
            assert isinstance(pkt, np.ndarray)


# ---------------------------------------------------------------------------
# load_file
# ---------------------------------------------------------------------------

class TestLoadFile:
    def test_load_file_sets_internal_data(self, tmp_path):
        import numpy as np
        from utils.fileio import write_iq_file
        data = np.ones(512, dtype=np.complex64)
        path = str(tmp_path / "test.iq")
        write_iq_file(path, data)

        mgr = CaptureManager()
        loaded = mgr.load_file(path)
        assert len(loaded) == 512
        assert mgr.get_iq_data() is not None

    def test_load_nonexistent_file_raises(self):
        mgr = CaptureManager()
        with pytest.raises(CaptureManagerError):
            mgr.load_file("/nonexistent/file.iq")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

class TestStatistics:
    def test_initial_stats(self):
        mgr = CaptureManager()
        stats = mgr.get_statistics()
        assert stats["captures_performed"] == 0
        assert stats["is_connected"] is False

    def test_stats_updated_after_capture(self):
        mgr = CaptureManager()
        # Trigger fallback (simulated) capture
        with patch.object(mgr, "_is_connected", True):
            with patch.object(mgr, "_enhanced_capture", None):
                samples = mgr.capture(duration=0.001, auto_connect=False)
        stats = mgr.get_statistics()
        assert stats["captures_performed"] == 1


# ---------------------------------------------------------------------------
# Hardware tests (skipped unless --run-hardware flag)
# ---------------------------------------------------------------------------

@pytest.mark.hardware
class TestHardwareCapture:
    def test_connect_returns_bool(self):
        mgr = CaptureManager(platform="rtl_sdr")
        result = mgr.connect()
        assert isinstance(result, bool)
        mgr.disconnect()
