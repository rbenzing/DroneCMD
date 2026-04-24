"""
Unit tests for exceptions.py

Verifies the exception hierarchy, ModelNotTrainedError, and the compliance
exceptions (PowerLimitError, FrequencyViolationError, DwellTimeError) that
replace the former compliance theater.
"""
from __future__ import annotations

import pytest

from exceptions import (
    DroneCmdError,
    ClassificationError,
    ModelNotTrainedError,
    ComplianceError,
    FCCComplianceError,
    PowerLimitError,
    FrequencyViolationError,
    DwellTimeError,
    ProcessingError,
)


class TestExceptionHierarchy:
    def test_model_not_trained_is_classification_error(self):
        assert issubclass(ModelNotTrainedError, ClassificationError)

    def test_model_not_trained_is_processing_error(self):
        assert issubclass(ModelNotTrainedError, ProcessingError)

    def test_model_not_trained_is_dronecmd_error(self):
        assert issubclass(ModelNotTrainedError, DroneCmdError)

    def test_power_limit_is_fcc_compliance_error(self):
        assert issubclass(PowerLimitError, FCCComplianceError)

    def test_frequency_violation_is_fcc_compliance_error(self):
        assert issubclass(FrequencyViolationError, FCCComplianceError)

    def test_dwell_time_is_fcc_compliance_error(self):
        assert issubclass(DwellTimeError, FCCComplianceError)

    def test_fcc_compliance_is_compliance_error(self):
        assert issubclass(FCCComplianceError, ComplianceError)

    def test_compliance_error_is_dronecmd_error(self):
        assert issubclass(ComplianceError, DroneCmdError)


class TestExceptionMessages:
    def test_model_not_trained_message(self):
        msg = "Run dronecmd train first"
        exc = ModelNotTrainedError(msg)
        assert msg in str(exc)

    def test_power_limit_error_message(self):
        exc = PowerLimitError("35.0 dBm exceeds 30 dBm limit")
        assert "35.0" in str(exc)

    def test_dwell_time_error_message(self):
        exc = DwellTimeError("120s exceeds 60s limit")
        assert "120" in str(exc)


class TestExceptionsCatchable:
    def test_model_not_trained_catchable_as_dronecmd_error(self):
        with pytest.raises(DroneCmdError):
            raise ModelNotTrainedError("no models")

    def test_power_limit_catchable_as_compliance_error(self):
        with pytest.raises(ComplianceError):
            raise PowerLimitError("over limit")

    def test_dwell_time_catchable_as_fcc_compliance_error(self):
        with pytest.raises(FCCComplianceError):
            raise DwellTimeError("too long")
