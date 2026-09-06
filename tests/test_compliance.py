"""
Unit tests for compliance enforcement in injector/suringe.py and core/replay.py

Verifies that power limits, dwell-time limits, and frequency range checks now
raise typed exceptions instead of silently logging and continuing.
"""
from __future__ import annotations

from unittest.mock import MagicMock, AsyncMock

import pytest

from exceptions import PowerLimitError, DwellTimeError, FrequencyViolationError
from injector.suringe import SafetyMonitor, InjectionConfig


# ---------------------------------------------------------------------------
# SafetyMonitor — power compliance
# ---------------------------------------------------------------------------

class TestSafetyMonitorPowerCompliance:
    def setup_method(self):
        self.config = InjectionConfig(max_transmission_power_dbm=10.0, max_injection_duration_s=60.0)
        self.monitor = SafetyMonitor(self.config)

    def test_power_within_limit_does_not_raise(self):
        self.monitor.check_power_compliance(9.9)  # should not raise

    def test_power_at_limit_does_not_raise(self):
        self.monitor.check_power_compliance(10.0)

    def test_power_exceeds_limit_raises(self):
        with pytest.raises(PowerLimitError):
            self.monitor.check_power_compliance(10.1)

    def test_power_far_exceeds_limit_raises(self):
        with pytest.raises(PowerLimitError):
            self.monitor.check_power_compliance(30.0)

    def test_violation_recorded(self):
        try:
            self.monitor.check_power_compliance(50.0)
        except PowerLimitError:
            pass
        assert len(self.monitor.violations) == 1

    def test_multiple_within_limit_no_violations(self):
        for power in [0.0, 5.0, 9.9, 10.0]:
            self.monitor.check_power_compliance(power)
        assert len(self.monitor.violations) == 0


# ---------------------------------------------------------------------------
# SafetyMonitor — dwell time compliance
# ---------------------------------------------------------------------------

class TestSafetyMonitorDurationCompliance:
    def setup_method(self):
        self.config = InjectionConfig(max_transmission_power_dbm=10.0, max_injection_duration_s=0.05)
        self.monitor = SafetyMonitor(self.config)

    def test_within_duration_does_not_raise(self):
        import time
        self.monitor.start_injection_session()
        # Check immediately — well within 50 ms
        self.monitor.check_duration_compliance()

    def test_exceeded_duration_raises(self):
        import time
        self.config = InjectionConfig(max_transmission_power_dbm=10.0, max_injection_duration_s=0.001)
        self.monitor = SafetyMonitor(self.config)
        self.monitor.start_injection_session()
        time.sleep(0.005)  # exceed 1 ms limit
        with pytest.raises(DwellTimeError):
            self.monitor.check_duration_compliance()

    def test_no_session_started_does_not_raise(self):
        self.monitor.check_duration_compliance()  # injection_start_time is None


# ---------------------------------------------------------------------------
# SafetyMonitor — frequency band compliance
# ---------------------------------------------------------------------------

class TestSafetyMonitorFrequencyCompliance:
    def setup_method(self):
        self.config = InjectionConfig()
        self.monitor = SafetyMonitor(self.config)

    def test_24ghz_band_does_not_raise(self):
        self.monitor.check_frequency_compliance(2.44e9)

    def test_433mhz_band_does_not_raise(self):
        self.monitor.check_frequency_compliance(433.5e6)

    def test_915mhz_band_does_not_raise(self):
        self.monitor.check_frequency_compliance(915e6)

    def test_band_edges_do_not_raise(self):
        self.monitor.check_frequency_compliance(2.4e9)      # lower edge
        self.monitor.check_frequency_compliance(2.4835e9)   # upper edge

    def test_out_of_band_raises(self):
        with pytest.raises(FrequencyViolationError):
            self.monitor.check_frequency_compliance(1.5e9)

    def test_just_above_band_raises(self):
        with pytest.raises(FrequencyViolationError):
            self.monitor.check_frequency_compliance(2.5e9)

    def test_violation_recorded(self):
        try:
            self.monitor.check_frequency_compliance(1.5e9)
        except FrequencyViolationError:
            pass
        assert len(self.monitor.violations) == 1

    def test_custom_allowed_range(self):
        cfg = InjectionConfig(allowed_frequency_ranges_hz=((100e6, 200e6),))
        monitor = SafetyMonitor(cfg)
        monitor.check_frequency_compliance(150e6)  # in custom band
        with pytest.raises(FrequencyViolationError):
            monitor.check_frequency_compliance(2.44e9)  # outside custom band


# ---------------------------------------------------------------------------
# InjectionConfig.validate() — hard limits
# ---------------------------------------------------------------------------

class TestInjectionConfigValidation:
    def test_valid_config_no_errors(self):
        config = InjectionConfig(max_transmission_power_dbm=10.0, max_injection_duration_s=30.0)
        errors = config.validate()
        assert errors == []

    def test_power_over_30dbm_is_error(self):
        config = InjectionConfig(max_transmission_power_dbm=31.0)
        errors = config.validate()
        assert any("30" in e for e in errors), f"Expected power-limit error, got: {errors}"

    def test_negative_power_is_error(self):
        config = InjectionConfig(max_transmission_power_dbm=-1.0)
        errors = config.validate()
        assert len(errors) > 0

    def test_zero_duration_is_error(self):
        config = InjectionConfig(max_injection_duration_s=0.0)
        errors = config.validate()
        assert len(errors) > 0

    def test_invalid_jitter_range_is_error(self):
        config = InjectionConfig(jitter_range_s=(0.5, 0.1))
        errors = config.validate()
        assert len(errors) > 0


# ---------------------------------------------------------------------------
# ComplianceMonitor in replay engine
# ---------------------------------------------------------------------------

class TestReplayComplianceMonitor:
    def setup_method(self):
        from core.replay import ReplayConfig, ComplianceMonitor
        self.config = ReplayConfig(safety_timeout_s=0.05)
        self.monitor = ComplianceMonitor(self.config)

    def test_within_timeout_does_not_raise(self):
        self.monitor.start_transmission_timer()
        self.monitor.check_timing_compliance(0.01)

    def test_exceeded_timeout_raises_dwell_time_error(self):
        self.monitor.start_transmission_timer()
        with pytest.raises(DwellTimeError):
            self.monitor.check_timing_compliance(1.0)

    def test_violation_recorded_on_raise(self):
        self.monitor.start_transmission_timer()
        try:
            self.monitor.check_timing_compliance(999.0)
        except DwellTimeError:
            pass
        assert len(self.monitor.violations) == 1
