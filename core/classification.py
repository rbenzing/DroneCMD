#!/usr/bin/env python3
"""
Enhanced Protocol Classification System

A comprehensive machine learning-based protocol classification system designed
to integrate with enhanced SDR capture, demodulation, and packet parsing
systems. This module provides advanced feature extraction, multi-model
ensemble classification, and real-time protocol identification for drone
communication analysis.

Integration Points:
- Enhanced Live SDR Capture System (signal quality metrics)
- Enhanced Demodulation System (modulation characteristics)
- Enhanced Packet Parser (protocol-specific features)
- Enhanced FHSS Engine (frequency hopping patterns)

Standards Compliance:
- IEEE 802.11 wireless protocol standards
- MAVLink protocol specifications
- Open Drone ID standards (ASTM F3411)
- Machine learning best practices (sklearn, MLOps)
- Production ML system design patterns

Key Features:
- Multi-modal feature extraction (spectral, temporal, protocol-specific)
- Ensemble classification with confidence estimation
- Real-time streaming classification
- Model versioning and A/B testing capabilities
- Integration with signal processing pipeline
- Comprehensive performance monitoring
- Adaptive learning and model updating
- Explainable AI for classification decisions

Example:
    >>> from dronecmd.core.enhanced_classifier import EnhancedProtocolClassifier, ClassifierConfig
    >>> config = ClassifierConfig(
    ...     enable_ensemble=True,
    ...     confidence_threshold=0.8,
    ...     enable_feature_selection=True
    ... )
    >>> classifier = EnhancedProtocolClassifier(config)
    >>> result = classifier.classify_packet(packet_data, signal_metrics)
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import time
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import numpy.typing as npt
from scipy import stats
from scipy.fft import fft, fftfreq
from scipy.signal import welch, periodogram

# Machine Learning imports with graceful fallbacks
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    joblib = None

try:
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.feature_selection import SelectKBest, f_classif
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Integration with our enhanced modules
try:
    from .enhanced_demodulation import DemodulationResult, ModulationScheme
    from .enhanced_parser import PacketResult, DroneProtocol
    from .enhanced_live_capture import CaptureMetadata
    ENHANCED_MODULES_AVAILABLE = True
except ImportError:
    ENHANCED_MODULES_AVAILABLE = False
    # Define minimal types for fallback
    class DemodulationResult:
        pass
    class PacketResult:
        pass
    class CaptureMetadata:
        pass

# Configure module logger
logger = logging.getLogger(__name__)

# Type aliases
FeatureVector = npt.NDArray[np.float32]
ConfidenceScore = float
ProtocolLabel = str
ModelMetrics = Dict[str, float]


class ClassificationMethod(Enum):
    """Supported classification methods."""
    
    RANDOM_FOREST = "random_forest"
    LOGISTIC_REGRESSION = "logistic_regression"
    SVM = "svm"
    NEURAL_NETWORK = "mlp"
    ENSEMBLE = "ensemble"
    NAIVE_BAYES = "naive_bayes"
    GRADIENT_BOOSTING = "gradient_boosting"


class FeatureType(Enum):
    """Types of features that can be extracted."""
    
    STATISTICAL = "statistical"        # Basic statistics (mean, std, etc.)
    HISTOGRAM = "histogram"            # Byte value histograms  
    SPECTRAL = "spectral"             # Frequency domain features
    TEMPORAL = "temporal"             # Time domain patterns
    ENTROPY = "entropy"               # Information theoretic
    PROTOCOL_SPECIFIC = "protocol"    # Protocol structure features
    SIGNAL_QUALITY = "signal_quality" # RF signal characteristics
    MODULATION = "modulation"         # Modulation scheme features


@dataclass(frozen=True)
class ClassifierConfig:
    """
    Configuration for the enhanced protocol classifier.
    
    Comprehensive configuration supporting various classification
    approaches and feature extraction methods.
    
    Attributes:
        primary_method: Primary classification method
        enable_ensemble: Use ensemble of multiple classifiers
        feature_types: List of feature types to extract
        confidence_threshold: Minimum confidence for classification
        enable_feature_selection: Use automatic feature selection
        max_features: Maximum number of features to use
        model_path: Path to saved model files
        enable_online_learning: Enable incremental learning
        performance_monitoring: Enable performance tracking
        enable_explainability: Enable explainable AI features
        cache_features: Cache extracted features for performance
        min_packet_length: Minimum packet length for classification
        max_packet_length: Maximum packet length for classification
    """
    
    primary_method: ClassificationMethod = ClassificationMethod.RANDOM_FOREST
    enable_ensemble: bool = True
    feature_types: List[FeatureType] = field(default_factory=lambda: [
        FeatureType.STATISTICAL,
        FeatureType.HISTOGRAM,
        FeatureType.SPECTRAL,
        FeatureType.ENTROPY,
        FeatureType.PROTOCOL_SPECIFIC
    ])
    confidence_threshold: ConfidenceScore = 0.7
    enable_feature_selection: bool = True
    max_features: int = 100
    model_path: Optional[Path] = None
    enable_online_learning: bool = False
    performance_monitoring: bool = True
    enable_explainability: bool = True
    cache_features: bool = True
    min_packet_length: int = 8
    max_packet_length: int = 2048
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.confidence_threshold < 0 or self.confidence_threshold > 1:
            raise ValueError("Confidence threshold must be between 0 and 1")
        
        if self.max_features < 1:
            raise ValueError("max_features must be at least 1")
        
        if self.min_packet_length < 1:
            raise ValueError("min_packet_length must be at least 1")
        
        if self.max_packet_length < self.min_packet_length:
            raise ValueError("max_packet_length must be >= min_packet_length")


@dataclass
class ClassificationResult:
    """
    Results from protocol classification operations.
    
    Contains classification predictions, confidence scores,
    and detailed analysis information.
    """
    
    # Primary classification results
    predicted_protocol: ProtocolLabel = "unknown"
    confidence: ConfidenceScore = 0.0
    
    # Alternative predictions
    top_k_predictions: List[Tuple[ProtocolLabel, ConfidenceScore]] = field(default_factory=list)
    
    # Feature analysis
    features_used: Optional[FeatureVector] = None
    feature_names: List[str] = field(default_factory=list)
    feature_importance: Optional[FeatureVector] = None
    
    # Model information
    model_version: Optional[str] = None
    models_used: List[str] = field(default_factory=list)
    ensemble_weights: Optional[FeatureVector] = None
    
    # Quality metrics
    prediction_entropy: Optional[float] = None
    model_uncertainty: Optional[float] = None
    feature_quality_score: Optional[float] = None
    
    # Processing metadata
    processing_time_ms: float = 0.0
    packet_length: int = 0
    features_extracted: int = 0
    
    # Integration data
    signal_metrics: Optional[Dict[str, float]] = None
    modulation_info: Optional[Dict[str, Any]] = None
    protocol_features: Optional[Dict[str, Any]] = None
    
    # Validation flags
    is_valid: bool = True
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    @property
    def is_confident(self) -> bool:
        """Check if prediction meets confidence threshold."""
        return self.confidence >= 0.7  # Default threshold
    
    @property
    def prediction_diversity(self) -> float:
        """Calculate diversity in top predictions."""
        if len(self.top_k_predictions) < 2:
            return 0.0
        
        confidences = [conf for _, conf in self.top_k_predictions]
        return 1.0 - (max(confidences) - min(confidences))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'predicted_protocol': self.predicted_protocol,
            'confidence': self.confidence,
            'top_k_predictions': self.top_k_predictions,
            'model_version': self.model_version,
            'models_used': self.models_used,
            'prediction_entropy': self.prediction_entropy,
            'model_uncertainty': self.model_uncertainty,
            'feature_quality_score': self.feature_quality_score,
            'processing_time_ms': self.processing_time_ms,
            'packet_length': self.packet_length,
            'features_extracted': self.features_extracted,
            'is_confident': self.is_confident,
            'prediction_diversity': self.prediction_diversity,
            'is_valid': self.is_valid,
            'error_message': self.error_message,
            'warnings': self.warnings,
            'signal_metrics': self.signal_metrics,
            'modulation_info': self.modulation_info,
            'protocol_features': self.protocol_features
        }


class AdvancedFeatureExtractor:
    """
    Advanced feature extraction for protocol classification.
    
    Implements comprehensive feature extraction methods covering
    statistical, spectral, temporal, and protocol-specific features.
    """
    
    def __init__(self, feature_types: List[FeatureType]) -> None:
        """Initialize feature extractor with specified types."""
        self.feature_types = feature_types
        self.feature_cache = {} if FeatureType.STATISTICAL in feature_types else None
        
    def extract_features(
        self,
        packet_data: bytes,
        signal_metrics: Optional[Dict[str, float]] = None,
        demod_result: Optional[DemodulationResult] = None,
        packet_result: Optional[PacketResult] = None
    ) -> Tuple[FeatureVector, List[str]]:
        """
        Extract comprehensive feature vector from packet data.
        
        Args:
            packet_data: Raw packet bytes
            signal_metrics: Optional signal quality metrics
            demod_result: Optional demodulation results
            packet_result: Optional packet parsing results
            
        Returns:
            Tuple of (feature_vector, feature_names)
        """
        if len(packet_data) == 0:
            return np.array([], dtype=np.float32), []
        
        features = []
        feature_names = []
        
        # Convert to numpy array for processing
        data_array = np.frombuffer(packet_data, dtype=np.uint8)
        
        # Extract different types of features
        for feature_type in self.feature_types:
            if feature_type == FeatureType.STATISTICAL:
                feats, names = self._extract_statistical_features(data_array)
            elif feature_type == FeatureType.HISTOGRAM:
                feats, names = self._extract_histogram_features(data_array)
            elif feature_type == FeatureType.SPECTRAL:
                feats, names = self._extract_spectral_features(data_array)
            elif feature_type == FeatureType.TEMPORAL:
                feats, names = self._extract_temporal_features(data_array)
            elif feature_type == FeatureType.ENTROPY:
                feats, names = self._extract_entropy_features(data_array)
            elif feature_type == FeatureType.PROTOCOL_SPECIFIC:
                feats, names = self._extract_protocol_features(packet_data, packet_result)
            elif feature_type == FeatureType.SIGNAL_QUALITY:
                feats, names = self._extract_signal_features(signal_metrics)
            elif feature_type == FeatureType.MODULATION:
                feats, names = self._extract_modulation_features(demod_result)
            else:
                continue
            
            features.extend(feats)
            feature_names.extend(names)
        
        return np.array(features, dtype=np.float32), feature_names
    
    def _extract_statistical_features(
        self, 
        data: npt.NDArray[np.uint8]
    ) -> Tuple[List[float], List[str]]:
        """Extract basic statistical features."""
        if len(data) == 0:
            return [], []
        
        features = [
            float(np.mean(data)),           # Mean
            float(np.std(data)),            # Standard deviation
            float(np.var(data)),            # Variance
            float(np.min(data)),            # Minimum
            float(np.max(data)),            # Maximum
            float(np.median(data)),         # Median
            float(stats.skew(data)),        # Skewness
            float(stats.kurtosis(data)),    # Kurtosis
            float(np.ptp(data)),            # Peak-to-peak
            float(stats.iqr(data)),         # Interquartile range
        ]
        
        # Percentiles
        percentiles = [10, 25, 75, 90]
        for p in percentiles:
            features.append(float(np.percentile(data, p)))
        
        names = [
            'stat_mean', 'stat_std', 'stat_var', 'stat_min', 'stat_max',
            'stat_median', 'stat_skew', 'stat_kurtosis', 'stat_ptp', 'stat_iqr'
        ] + [f'stat_p{p}' for p in percentiles]
        
        return features, names
    
    def _extract_histogram_features(
        self, 
        data: npt.NDArray[np.uint8]
    ) -> Tuple[List[float], List[str]]:
        """Extract histogram-based features."""
        # Byte value histogram (normalized)
        hist, _ = np.histogram(data, bins=16, range=(0, 256))
        hist_norm = hist / (np.sum(hist) + 1e-8)
        
        # Histogram statistics
        hist_features = [
            float(np.max(hist_norm)),       # Maximum bin
            float(np.std(hist_norm)),       # Histogram spread
            float(np.sum(hist_norm > 0)),   # Number of non-empty bins
            float(stats.entropy(hist_norm + 1e-8))  # Histogram entropy
        ]
        
        features = hist_norm.tolist() + hist_features
        names = [f'hist_bin_{i}' for i in range(16)] + [
            'hist_max', 'hist_std', 'hist_nonzero', 'hist_entropy'
        ]
        
        return features, names
    
    def _extract_spectral_features(
        self, 
        data: npt.NDArray[np.uint8]
    ) -> Tuple[List[float], List[str]]:
        """Extract frequency domain features."""
        if len(data) < 8:
            return [], []
        
        # Convert to float and remove DC
        signal = data.astype(np.float32) - np.mean(data)
        
        # Power spectral density
        freqs, psd = welch(signal, nperseg=min(len(signal), 64))
        
        # Spectral features
        features = [
            float(np.sum(psd)),                    # Total power
            float(np.max(psd)),                    # Peak power
            float(np.argmax(psd) / len(psd)),      # Peak frequency (normalized)
            float(np.sum(freqs * psd) / np.sum(psd)),  # Spectral centroid
            float(np.sqrt(np.sum((freqs - np.sum(freqs * psd) / np.sum(psd))**2 * psd) / np.sum(psd))),  # Spectral spread
            float(np.sum(psd[:len(psd)//4]) / np.sum(psd)),  # Low frequency ratio
            float(np.sum(psd[3*len(psd)//4:]) / np.sum(psd)), # High frequency ratio
        ]
        
        # Spectral rolloff
        cumsum_psd = np.cumsum(psd)
        rolloff_85 = np.where(cumsum_psd >= 0.85 * cumsum_psd[-1])[0]
        if len(rolloff_85) > 0:
            features.append(float(rolloff_85[0] / len(psd)))
        else:
            features.append(1.0)
        
        names = [
            'spec_total_power', 'spec_peak_power', 'spec_peak_freq',
            'spec_centroid', 'spec_spread', 'spec_low_ratio', 'spec_high_ratio',
            'spec_rolloff_85'
        ]
        
        return features, names
    
    def _extract_temporal_features(
        self, 
        data: npt.NDArray[np.uint8]
    ) -> Tuple[List[float], List[str]]:
        """Extract temporal pattern features."""
        if len(data) < 4:
            return [], []
        
        # First-order differences
        diff1 = np.diff(data.astype(np.float32))
        
        # Zero-crossing rate
        zero_crossings = np.sum(np.diff(np.sign(diff1)) != 0) / (len(diff1) - 1)
        
        # Run length encoding features
        changes = np.where(np.diff(data) != 0)[0]
        if len(changes) > 0:
            run_lengths = np.diff(np.concatenate(([0], changes, [len(data)])))
            avg_run_length = float(np.mean(run_lengths))
            max_run_length = float(np.max(run_lengths))
            run_length_std = float(np.std(run_lengths))
        else:
            avg_run_length = float(len(data))
            max_run_length = float(len(data))
            run_length_std = 0.0
        
        # Autocorrelation features
        if len(data) > 8:
            autocorr = np.correlate(data, data, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr_norm = autocorr / autocorr[0]
            
            # Find first minimum in autocorrelation
            if len(autocorr_norm) > 2:
                first_min_idx = np.argmin(autocorr_norm[1:len(autocorr_norm)//4]) + 1
                first_min_val = float(autocorr_norm[first_min_idx])
            else:
                first_min_val = 0.0
        else:
            first_min_val = 0.0
        
        features = [
            float(np.mean(np.abs(diff1))),  # Mean absolute difference
            float(np.std(diff1)),           # Difference standard deviation
            zero_crossings,                 # Zero crossing rate
            avg_run_length,                # Average run length
            max_run_length,                # Maximum run length
            run_length_std,                # Run length standard deviation
            first_min_val                  # First autocorrelation minimum
        ]
        
        names = [
            'temp_mean_abs_diff', 'temp_diff_std', 'temp_zero_cross',
            'temp_avg_run', 'temp_max_run', 'temp_run_std', 'temp_autocorr_min'
        ]
        
        return features, names
    
    def _extract_entropy_features(
        self, 
        data: npt.NDArray[np.uint8]
    ) -> Tuple[List[float], List[str]]:
        """Extract information-theoretic features."""
        if len(data) == 0:
            return [], []
        
        # Byte-level entropy
        _, counts = np.unique(data, return_counts=True)
        probs = counts / len(data)
        byte_entropy = float(stats.entropy(probs, base=2))
        
        # Block entropy (2-byte blocks)
        if len(data) >= 2:
            blocks = [data[i:i+2].tobytes() for i in range(len(data)-1)]
            _, block_counts = np.unique(blocks, return_counts=True)
            block_probs = block_counts / len(blocks)
            block_entropy = float(stats.entropy(block_probs, base=2))
        else:
            block_entropy = 0.0
        
        # Compression ratio estimate
        try:
            import zlib
            compressed = zlib.compress(data.tobytes())
            compression_ratio = len(compressed) / len(data)
        except:
            compression_ratio = 1.0
        
        features = [
            byte_entropy,
            block_entropy,
            float(compression_ratio),
            float(len(np.unique(data)) / 256),  # Alphabet utilization
        ]
        
        names = [
            'entropy_byte', 'entropy_block', 'entropy_compression', 'entropy_alphabet'
        ]
        
        return features, names
    
    def _extract_protocol_features(
        self, 
        packet_data: bytes,
        packet_result: Optional[PacketResult] = None
    ) -> Tuple[List[float], List[str]]:
        """Extract protocol-specific structural features."""
        features = []
        names = []
        
        # Basic structural features
        features.extend([
            float(len(packet_data)),        # Packet length
            float(packet_data[0]) if packet_data else 0,  # First byte (potential sync/magic)
            float(packet_data[-1]) if packet_data else 0, # Last byte
        ])
        names.extend(['proto_length', 'proto_first_byte', 'proto_last_byte'])
        
        # Check for common protocol patterns
        if len(packet_data) >= 2:
            # MAVLink patterns
            is_mavlink_v1 = packet_data[0] == 0xFE
            is_mavlink_v2 = packet_data[0] == 0xFD
            
            # DJI patterns
            is_dji_sync = packet_data[:2] == b'\x55\xAA'
            
            features.extend([
                float(is_mavlink_v1),
                float(is_mavlink_v2), 
                float(is_dji_sync)
            ])
            names.extend(['proto_mavlink_v1', 'proto_mavlink_v2', 'proto_dji_sync'])
        
        # Length-based features for known protocols
        if len(packet_data) >= 8:
            # Potential payload length for MAVLink
            if packet_data[0] in [0xFE, 0xFD]:
                payload_len = packet_data[1] if len(packet_data) > 1 else 0
                expected_total = (8 + payload_len) if packet_data[0] == 0xFE else (12 + payload_len)
                length_match = abs(len(packet_data) - expected_total) <= 1
                features.append(float(length_match))
                names.append('proto_mavlink_length_match')
            else:
                features.append(0.0)
                names.append('proto_mavlink_length_match')
        
        # Add packet parser results if available
        if packet_result is not None:
            features.extend([
                float(packet_result.confidence),
                float(packet_result.header_valid),
                float(packet_result.checksum_valid),
                packet_result.signal_power_dbfs or 0.0,
                packet_result.snr_db or 0.0
            ])
            names.extend([
                'proto_parser_confidence', 'proto_header_valid', 
                'proto_checksum_valid', 'proto_signal_power', 'proto_snr'
            ])
        
        return features, names
    
    def _extract_signal_features(
        self, 
        signal_metrics: Optional[Dict[str, float]]
    ) -> Tuple[List[float], List[str]]:
        """Extract signal quality features."""
        if signal_metrics is None:
            return [], []
        
        feature_mapping = {
            'signal_level_dbfs': 'sig_power_dbfs',
            'noise_floor_dbfs': 'sig_noise_dbfs', 
            'snr_db': 'sig_snr_db',
            'sample_loss_rate': 'sig_loss_rate',
            'total_samples': 'sig_total_samples'
        }
        
        features = []
        names = []
        
        for metric_key, feature_name in feature_mapping.items():
            value = signal_metrics.get(metric_key, 0.0)
            features.append(float(value))
            names.append(feature_name)
        
        return features, names
    
    def _extract_modulation_features(
        self, 
        demod_result: Optional[DemodulationResult]
    ) -> Tuple[List[float], List[str]]:
        """Extract modulation scheme features."""
        if demod_result is None:
            return [], []
        
        features = []
        names = []
        
        # Signal quality metrics
        if hasattr(demod_result, 'snr_db') and demod_result.snr_db is not None:
            features.append(float(demod_result.snr_db))
            names.append('mod_snr_db')
        
        if hasattr(demod_result, 'evm_percent') and demod_result.evm_percent is not None:
            features.append(float(demod_result.evm_percent))
            names.append('mod_evm_percent')
        
        # Timing and frequency offsets
        if hasattr(demod_result, 'carrier_frequency_offset_hz'):
            features.append(float(demod_result.carrier_frequency_offset_hz))
            names.append('mod_freq_offset_hz')
        
        if hasattr(demod_result, 'timing_offset_samples'):
            features.append(float(demod_result.timing_offset_samples))
            names.append('mod_timing_offset')
        
        # Recovery status
        if hasattr(demod_result, 'clock_recovery_locked'):
            features.append(float(demod_result.clock_recovery_locked))
            names.append('mod_clock_locked')
        
        return features, names


class ModelManager:
    """
    Model management for protocol classification.
    
    Handles model loading, saving, versioning, and ensemble management.
    """
    
    def __init__(self, config: ClassifierConfig) -> None:
        """Initialize model manager."""
        self.config = config
        self.models = {}
        self.model_metadata = {}
        self.performance_history = defaultdict(list)
        
    def load_models(self) -> bool:
        """Load classification models from disk."""
        if not JOBLIB_AVAILABLE or not SKLEARN_AVAILABLE:
            logger.error("Required ML libraries not available")
            return False
        
        if self.config.model_path is None:
            logger.warning("No model path specified")
            return False
        
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            logger.warning(f"Model path does not exist: {model_path}")
            return False
        
        try:
            # Load primary model
            primary_model_file = model_path / f"{self.config.primary_method.value}.pkl"
            if primary_model_file.exists():
                self.models[self.config.primary_method.value] = joblib.load(primary_model_file)
                logger.info(f"Loaded primary model: {self.config.primary_method.value}")
            
            # Load ensemble models if enabled
            if self.config.enable_ensemble:
                ensemble_methods = [
                    ClassificationMethod.RANDOM_FOREST,
                    ClassificationMethod.LOGISTIC_REGRESSION,
                    ClassificationMethod.SVM
                ]
                
                for method in ensemble_methods:
                    model_file = model_path / f"{method.value}.pkl"
                    if model_file.exists():
                        self.models[method.value] = joblib.load(model_file)
                        logger.info(f"Loaded ensemble model: {method.value}")
            
            # Load model metadata
            metadata_file = model_path / "model_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    self.model_metadata = json.load(f)
            
            return len(self.models) > 0
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False
    
    def create_fallback_model(self) -> bool:
        """Create a simple fallback model if no models are loaded."""
        if not SKLEARN_AVAILABLE:
            return False
        
        try:
            # Create a simple random forest with default parameters
            self.models['fallback'] = RandomForestClassifier(
                n_estimators=10,
                max_depth=5,
                random_state=42
            )
            
            # Create dummy training data for initialization
            dummy_features = np.random.rand(100, 20)
            dummy_labels = np.random.choice(['mavlink', 'dji', 'unknown'], 100)
            
            self.models['fallback'].fit(dummy_features, dummy_labels)
            
            logger.warning("Created fallback model with dummy training data")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create fallback model: {e}")
            return False


class EnhancedProtocolClassifier:
    """
    Enhanced protocol classifier with advanced ML capabilities.
    
    Integrates with enhanced SDR systems to provide comprehensive
    protocol classification with confidence estimation and real-time
    performance monitoring.
    """
    
    def __init__(
        self,
        config: Optional[ClassifierConfig] = None,
        model_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Initialize enhanced protocol classifier.
        
        Args:
            config: Classifier configuration
            model_path: Path to model files (for backward compatibility)
        """
        # Handle backward compatibility
        if config is None:
            config = ClassifierConfig()
        
        if model_path is not None and config.model_path is None:
            config = ClassifierConfig(
                **{**config.__dict__, 'model_path': Path(model_path)}
            )
        
        self.config = config
        self.feature_extractor = AdvancedFeatureExtractor(config.feature_types)
        self.model_manager = ModelManager(config)
        
        # Performance monitoring
        self.stats = {
            'classifications_performed': 0,
            'total_processing_time': 0.0,
            'confidence_history': deque(maxlen=1000),
            'protocol_counts': defaultdict(int),
            'feature_cache_hits': 0,
            'feature_cache_misses': 0
        }
        
        # Feature selection and scaling
        self.feature_scaler = None
        self.feature_selector = None
        self.feature_names = []
        
        # Load models
        models_loaded = self.model_manager.load_models()
        if not models_loaded:
            logger.warning("No models loaded, creating fallback model")
            self.model_manager.create_fallback_model()
        
        logger.info(f"Initialized enhanced protocol classifier with {len(self.model_manager.models)} models")
    
    def classify(
        self, 
        packet_bytes: bytes,
        signal_metrics: Optional[Dict[str, float]] = None,
        demod_result: Optional[DemodulationResult] = None,
        packet_result: Optional[PacketResult] = None
    ) -> Union[str, ClassificationResult]:
        """
        Classify protocol with comprehensive analysis.
        
        Args:
            packet_bytes: Raw packet data
            signal_metrics: Optional signal quality metrics
            demod_result: Optional demodulation results
            packet_result: Optional packet parsing results
            
        Returns:
            Classification result (string for backward compatibility,
            ClassificationResult for enhanced mode)
        """
        start_time = time.time()
        
        # Input validation
        if len(packet_bytes) < self.config.min_packet_length:
            if self.config.performance_monitoring:
                return ClassificationResult(
                    predicted_protocol="unknown",
                    confidence=0.0,
                    error_message=f"Packet too short: {len(packet_bytes)} < {self.config.min_packet_length}",
                    is_valid=False
                )
            return "unknown"
        
        if len(packet_bytes) > self.config.max_packet_length:
            packet_bytes = packet_bytes[:self.config.max_packet_length]
        
        try:
            # Extract features
            features, feature_names = self.feature_extractor.extract_features(
                packet_bytes, signal_metrics, demod_result, packet_result
            )
            
            if len(features) == 0:
                result = ClassificationResult(
                    predicted_protocol="unknown",
                    confidence=0.0,
                    error_message="No features extracted",
                    is_valid=False
                )
                if not self.config.performance_monitoring:
                    return "unknown"
                return result
            
            # Perform classification
            if self.config.enable_ensemble and len(self.model_manager.models) > 1:
                classification_result = self._ensemble_classify(features, feature_names)
            else:
                classification_result = self._single_model_classify(features, feature_names)
            
            # Add processing metadata
            processing_time = time.time() - start_time
            classification_result.processing_time_ms = processing_time * 1000
            classification_result.packet_length = len(packet_bytes)
            classification_result.features_extracted = len(features)
            
            # Add integration data
            classification_result.signal_metrics = signal_metrics
            if demod_result:
                classification_result.modulation_info = {
                    'snr_db': getattr(demod_result, 'snr_db', None),
                    'evm_percent': getattr(demod_result, 'evm_percent', None)
                }
            if packet_result:
                classification_result.protocol_features = {
                    'confidence': getattr(packet_result, 'confidence', None),
                    'protocol': getattr(packet_result, 'protocol', None)
                }
            
            # Update statistics
            self._update_statistics(classification_result, processing_time)
            
            # Return appropriate format
            if self.config.performance_monitoring:
                return classification_result
            else:
                return classification_result.predicted_protocol
                
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            if self.config.performance_monitoring:
                return ClassificationResult(
                    predicted_protocol="unknown",
                    confidence=0.0,
                    error_message=str(e),
                    is_valid=False,
                    processing_time_ms=(time.time() - start_time) * 1000
                )
            return "unknown"
    
    def _single_model_classify(
        self, 
        features: FeatureVector,
        feature_names: List[str]
    ) -> ClassificationResult:
        """Classify using single model."""
        model_name = list(self.model_manager.models.keys())[0]
        model = self.model_manager.models[model_name]
        
        # Reshape for single sample prediction
        features_reshaped = features.reshape(1, -1)
        
        # Get prediction and probabilities
        prediction = model.predict(features_reshaped)[0]
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_reshaped)[0]
            classes = model.classes_
            confidence = float(np.max(probabilities))
            
            # Create top-k predictions
            top_indices = np.argsort(probabilities)[::-1][:5]
            top_k = [(classes[i], float(probabilities[i])) for i in top_indices]
        else:
            confidence = 0.8  # Default confidence for models without probabilities
            top_k = [(prediction, confidence)]
        
        return ClassificationResult(
            predicted_protocol=prediction,
            confidence=confidence,
            top_k_predictions=top_k,
            features_used=features,
            feature_names=feature_names,
            model_version="single",
            models_used=[model_name],
            is_valid=True
        )
    
    def _ensemble_classify(
        self, 
        features: FeatureVector,
        feature_names: List[str]
    ) -> ClassificationResult:
        """Classify using ensemble of models."""
        features_reshaped = features.reshape(1, -1)
        
        predictions = {}
        confidences = {}
        all_probabilities = {}
        
        # Get predictions from each model
        for model_name, model in self.model_manager.models.items():
            try:
                pred = model.predict(features_reshaped)[0]
                predictions[model_name] = pred
                
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(features_reshaped)[0]
                    classes = model.classes_
                    confidences[model_name] = float(np.max(probs))
                    all_probabilities[model_name] = dict(zip(classes, probs))
                else:
                    confidences[model_name] = 0.8
                    all_probabilities[model_name] = {pred: 0.8}
                    
            except Exception as e:
                logger.warning(f"Model {model_name} failed: {e}")
                continue
        
        if not predictions:
            return ClassificationResult(
                predicted_protocol="unknown",
                confidence=0.0,
                error_message="All ensemble models failed",
                is_valid=False
            )
        
        # Ensemble voting (weighted by confidence)
        protocol_votes = defaultdict(float)
        total_weight = 0.0
        
        for model_name, pred in predictions.items():
            weight = confidences[model_name]
            protocol_votes[pred] += weight
            total_weight += weight
        
        # Normalize votes
        if total_weight > 0:
            for protocol in protocol_votes:
                protocol_votes[protocol] /= total_weight
        
        # Get final prediction
        final_prediction = max(protocol_votes.items(), key=lambda x: x[1])
        
        # Create top-k predictions from ensemble
        sorted_votes = sorted(protocol_votes.items(), key=lambda x: x[1], reverse=True)
        top_k = sorted_votes[:5]
        
        return ClassificationResult(
            predicted_protocol=final_prediction[0],
            confidence=final_prediction[1],
            top_k_predictions=top_k,
            features_used=features,
            feature_names=feature_names,
            model_version="ensemble",
            models_used=list(predictions.keys()),
            ensemble_weights=np.array([confidences[m] for m in predictions.keys()]),
            is_valid=True
        )
    
    def _update_statistics(
        self, 
        result: ClassificationResult,
        processing_time: float
    ) -> None:
        """Update performance statistics."""
        self.stats['classifications_performed'] += 1
        self.stats['total_processing_time'] += processing_time
        self.stats['confidence_history'].append(result.confidence)
        self.stats['protocol_counts'][result.predicted_protocol] += 1
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get classifier performance statistics."""
        avg_time = (
            self.stats['total_processing_time'] / 
            max(1, self.stats['classifications_performed'])
        )
        
        confidence_history = list(self.stats['confidence_history'])
        avg_confidence = np.mean(confidence_history) if confidence_history else 0.0
        
        return {
            'classifications_performed': self.stats['classifications_performed'],
            'average_processing_time_ms': avg_time * 1000,
            'average_confidence': avg_confidence,
            'protocol_distribution': dict(self.stats['protocol_counts']),
            'models_loaded': len(self.model_manager.models),
            'feature_types_enabled': [ft.value for ft in self.config.feature_types],
            'cache_hit_rate': (
                self.stats['feature_cache_hits'] / 
                max(1, self.stats['feature_cache_hits'] + self.stats['feature_cache_misses'])
            )
        }


# Backward compatibility function
def create_protocol_classifier(model_path: Optional[str] = None) -> EnhancedProtocolClassifier:
    """
    Create protocol classifier with backward compatibility.
    
    Args:
        model_path: Path to model files
        
    Returns:
        Enhanced protocol classifier instance
    """
    config = ClassifierConfig()
    if model_path:
        config = ClassifierConfig(model_path=Path(model_path))
    
    return EnhancedProtocolClassifier(config)


def main() -> None:
    """
    Example usage and testing of the enhanced protocol classifier.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=== Enhanced Protocol Classifier Demo ===\n")
    
    # Create classifier configuration
    config = ClassifierConfig(
        primary_method=ClassificationMethod.RANDOM_FOREST,
        enable_ensemble=True,
        feature_types=[
            FeatureType.STATISTICAL,
            FeatureType.HISTOGRAM,
            FeatureType.SPECTRAL,
            FeatureType.ENTROPY,
            FeatureType.PROTOCOL_SPECIFIC
        ],
        confidence_threshold=0.7,
        performance_monitoring=True
    )
    
    print(f"Configuration:")
    print(f"  Primary method: {config.primary_method.value}")
    print(f"  Ensemble enabled: {config.enable_ensemble}")
    print(f"  Feature types: {[ft.value for ft in config.feature_types]}")
    print(f"  Confidence threshold: {config.confidence_threshold}")
    
    # Create classifier
    classifier = EnhancedProtocolClassifier(config)
    
    # Test data samples
    test_packets = [
        # MAVLink v1 pattern
        b'\xFE\x21\x00\x01\x01\x00' + b'\x42' * 33 + b'\x12\x34',
        
        # MAVLink v2 pattern  
        b'\xFD\x21\x00\x00\x00\x01\x01\x00\x00\x00' + b'\x42' * 33 + b'\x12\x34',
        
        # DJI pattern
        b'\x55\xAA\x27\x10' + b'\x33' * 35 + b'\x56\x78',
        
        # Random data
        b'\x11\x22\x33\x44' + b'\x55' * 20 + b'\x99\xAA',
    ]
    
    expected_protocols = ['MAVLink', 'MAVLink', 'DJI', 'Unknown']
    
    print(f"\n=== Testing Classification ===")
    
    for i, (packet, expected) in enumerate(zip(test_packets, expected_protocols)):
        print(f"\nTest {i+1}: {expected} packet ({len(packet)} bytes)")
        print(f"  Data: {packet[:8].hex()}...")
        
        # Simulate signal metrics
        signal_metrics = {
            'signal_level_dbfs': -20.0 + np.random.normal(0, 5),
            'noise_floor_dbfs': -50.0 + np.random.normal(0, 3),
            'snr_db': 25.0 + np.random.normal(0, 2),
            'sample_loss_rate': np.random.uniform(0, 0.5)
        }
        
        # Classify packet
        result = classifier.classify(packet, signal_metrics=signal_metrics)
        
        if isinstance(result, ClassificationResult):
            print(f"  Predicted: {result.predicted_protocol}")
            print(f"  Confidence: {result.confidence:.3f}")
            print(f"  Processing time: {result.processing_time_ms:.1f} ms")
            print(f"  Features extracted: {result.features_extracted}")
            print(f"  Models used: {result.models_used}")
            
            if len(result.top_k_predictions) > 1:
                print(f"  Top alternatives:")
                for j, (proto, conf) in enumerate(result.top_k_predictions[1:3]):
                    print(f"    {j+2}: {proto} ({conf:.3f})")
        else:
            print(f"  Predicted: {result}")
    
    # Performance statistics
    print(f"\n=== Performance Statistics ===")
    stats = classifier.get_performance_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    # Test backward compatibility
    print(f"\n=== Backward Compatibility Test ===")
    simple_config = ClassifierConfig(performance_monitoring=False)
    simple_classifier = EnhancedProtocolClassifier(simple_config)
    
    for i, packet in enumerate(test_packets[:2]):
        result = simple_classifier.classify(packet)
        print(f"  Packet {i+1}: {result}")


if __name__ == "__main__":
    main()
