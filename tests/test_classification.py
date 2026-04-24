"""
Unit tests for core/classification.py

Tests verify that the classifier raises ModelNotTrainedError when no models
are loaded (preventing silent random predictions), that feature extraction
produces well-formed vectors, and that the model loading path works correctly.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from core.classification import (
    AdvancedFeatureExtractor,
    ClassifierConfig,
    ClassificationMethod,
    EnhancedProtocolClassifier,
    FeatureType,
    ModelManager,
)
from exceptions import ModelNotTrainedError


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

class TestAdvancedFeatureExtractor:
    def setup_method(self):
        self.extractor = AdvancedFeatureExtractor(
            feature_types=[
                FeatureType.STATISTICAL,
                FeatureType.HISTOGRAM,
                FeatureType.ENTROPY,
                FeatureType.PROTOCOL_SPECIFIC,
            ]
        )

    def test_extract_returns_nonempty_vector(self, mavlink_v1_heartbeat):
        features, names = self.extractor.extract_features(mavlink_v1_heartbeat)
        assert len(features) > 0
        assert len(names) == len(features)

    def test_feature_names_are_strings(self, mavlink_v1_heartbeat):
        _, names = self.extractor.extract_features(mavlink_v1_heartbeat)
        for name in names:
            assert isinstance(name, str)

    def test_features_are_finite(self, mavlink_v1_heartbeat):
        features, _ = self.extractor.extract_features(mavlink_v1_heartbeat)
        assert np.all(np.isfinite(features)), "Feature vector contains NaN or Inf"

    def test_features_consistent_length(self, mavlink_v1_heartbeat, dji_raw_packet):
        f1, _ = self.extractor.extract_features(mavlink_v1_heartbeat)
        f2, _ = self.extractor.extract_features(dji_raw_packet)
        assert len(f1) == len(f2), "Feature vector length must be consistent across protocols"

    def test_empty_packet_returns_vector_or_empty(self):
        features, names = self.extractor.extract_features(b"")
        assert len(features) == len(names)

    def test_different_protocols_produce_different_features(self, mavlink_v1_heartbeat, dji_raw_packet):
        f_mav, _ = self.extractor.extract_features(mavlink_v1_heartbeat)
        f_dji, _ = self.extractor.extract_features(dji_raw_packet)
        assert not np.allclose(f_mav, f_dji, atol=1e-3), (
            "MAVLink and DJI packets should produce distinct feature vectors"
        )


# ---------------------------------------------------------------------------
# ModelManager — no models loaded
# ---------------------------------------------------------------------------

class TestModelManagerNoModels:
    def test_load_models_returns_false_without_path(self):
        config = ClassifierConfig(model_path=None)
        manager = ModelManager(config)
        result = manager.load_models()
        assert result is False

    def test_load_models_returns_false_for_missing_path(self):
        config = ClassifierConfig(model_path=Path("/nonexistent/path/to/models"))
        manager = ModelManager(config)
        result = manager.load_models()
        assert result is False

    def test_create_fallback_model_returns_false(self):
        config = ClassifierConfig()
        manager = ModelManager(config)
        result = manager.create_fallback_model()
        assert result is False, "create_fallback_model must return False — no dummy models allowed"

    def test_no_models_after_fallback(self):
        config = ClassifierConfig()
        manager = ModelManager(config)
        manager.create_fallback_model()
        assert len(manager.models) == 0


# ---------------------------------------------------------------------------
# EnhancedProtocolClassifier — raises when untrained
# ---------------------------------------------------------------------------

class TestClassifierRaisesWhenUntrained:
    def test_classify_raises_model_not_trained_error(self, mavlink_v1_heartbeat):
        classifier = EnhancedProtocolClassifier(config=ClassifierConfig(model_path=None))
        with pytest.raises(ModelNotTrainedError):
            classifier.classify(mavlink_v1_heartbeat)

    def test_models_ready_flag_false_without_models(self):
        classifier = EnhancedProtocolClassifier(config=ClassifierConfig(model_path=None))
        assert classifier._models_ready is False


# ---------------------------------------------------------------------------
# EnhancedProtocolClassifier — with trained models (sklearn required)
# ---------------------------------------------------------------------------

pytest.importorskip("sklearn", reason="scikit-learn not installed")
pytest.importorskip("joblib", reason="joblib not installed")

import joblib  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402


@pytest.fixture
def trained_model_dir(mavlink_v1_heartbeat, dji_raw_packet):
    """Write a minimal trained RandomForest to a temp directory."""
    tmpdir = Path(tempfile.mkdtemp(prefix="dronecmd_test_"))
    try:
        extractor = AdvancedFeatureExtractor(
            feature_types=[
                FeatureType.STATISTICAL,
                FeatureType.HISTOGRAM,
                FeatureType.SPECTRAL,
                FeatureType.ENTROPY,
                FeatureType.PROTOCOL_SPECIFIC,
            ]
        )

        packets = [mavlink_v1_heartbeat] * 30 + [dji_raw_packet] * 30
        labels = ["mavlink"] * 30 + ["dji"] * 30

        X = np.array([extractor.extract_features(p)[0] for p in packets], dtype=np.float32)
        y = np.array(labels)

        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X, y)

        joblib.dump(clf, tmpdir / "random_forest.pkl")
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


class TestClassifierWithTrainedModels:
    def test_classify_returns_known_label(self, trained_model_dir, mavlink_v1_heartbeat, dji_raw_packet):
        config = ClassifierConfig(
            model_path=trained_model_dir,
            primary_method=ClassificationMethod.RANDOM_FOREST,
            enable_ensemble=False,
            performance_monitoring=True,
        )
        classifier = EnhancedProtocolClassifier(config=config)
        assert classifier._models_ready is True

        result = classifier.classify(mavlink_v1_heartbeat)
        proto = getattr(result, "predicted_protocol", result)
        assert proto in ("mavlink", "dji"), f"Unexpected label: {proto}"

    def test_confidence_between_zero_and_one(self, trained_model_dir, mavlink_v1_heartbeat):
        config = ClassifierConfig(
            model_path=trained_model_dir,
            primary_method=ClassificationMethod.RANDOM_FOREST,
            enable_ensemble=False,
            performance_monitoring=True,
        )
        classifier = EnhancedProtocolClassifier(config=config)
        result = classifier.classify(mavlink_v1_heartbeat)
        confidence = getattr(result, "confidence", None)
        if confidence is not None:
            assert 0.0 <= confidence <= 1.0

    def test_too_short_packet_returns_invalid_result(self, trained_model_dir):
        config = ClassifierConfig(
            model_path=trained_model_dir,
            primary_method=ClassificationMethod.RANDOM_FOREST,
            enable_ensemble=False,
            performance_monitoring=True,
        )
        classifier = EnhancedProtocolClassifier(config=config)
        result = classifier.classify(b"\x00")  # 1 byte — below min_packet_length
        is_valid = getattr(result, "is_valid", True)
        assert is_valid is False or getattr(result, "predicted_protocol", "") == "unknown"
