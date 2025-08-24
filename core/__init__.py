#!/usr/bin/env python3
"""
DroneCmd Core Module

This module contains the production-ready, feature-complete implementations
of all major DroneCmd systems. These are the enhanced versions designed for
professional use, research applications, and production deployments.

Core Systems:
- Enhanced Live Capture: Production SDR capture with quality monitoring
- Advanced Demodulation: Multi-scheme demodulation with carrier recovery
- Protocol Classification: ML-based protocol identification
- FHSS Engine: FCC-compliant frequency hopping systems
- Signal Processing: Comprehensive signal analysis and conditioning
- Replay Engine: Advanced signal replay with timing control

All core modules are designed for:
- High performance and reliability
- Comprehensive error handling
- Advanced configuration options
- Integration with monitoring systems
- Standards compliance (FCC, IEEE, etc.)

Usage:
    >>> from dronecmd.core import EnhancedLiveCapture, SDRConfig
    >>> config = SDRConfig(frequency_hz=2.44e9, sample_rate_hz=2.048e6)
    >>> capture = EnhancedLiveCapture(config)
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Union

# Configure module logger
logger = logging.getLogger(__name__)

# =============================================================================
# ENHANCED CAPTURE SYSTEM
# =============================================================================

try:
    from .capture import (
        EnhancedLiveCapture,
        SDRConfig, 
        CaptureMetadata,
        SDRPlatform,
        GainMode,
        CaptureFormat,
        SignalQualityMonitor
    )
    CAPTURE_AVAILABLE = True
    logger.debug("Enhanced capture system loaded successfully")
    
except ImportError as e:
    logger.warning(f"Enhanced capture system not available: {e}")
    CAPTURE_AVAILABLE = False
    # Set to None for graceful degradation
    EnhancedLiveCapture = None
    SDRConfig = None
    CaptureMetadata = None
    SDRPlatform = None
    GainMode = None
    CaptureFormat = None
    SignalQualityMonitor = None

# =============================================================================
# DEMODULATION SYSTEM
# =============================================================================

try:
    from .demodulation import (
        DemodulationEngine,
        DemodConfig,
        DemodulationResult,
        ModulationScheme,
        FilterType,
        SignalProcessor as DemodSignalProcessor
    )
    DEMODULATION_AVAILABLE = True
    logger.debug("Demodulation system loaded successfully")
    
except ImportError as e:
    logger.warning(f"Demodulation system not available: {e}")
    DEMODULATION_AVAILABLE = False
    DemodulationEngine = None
    DemodConfig = None
    DemodulationResult = None
    ModulationScheme = None
    FilterType = None
    DemodSignalProcessor = None

# =============================================================================
# PROTOCOL CLASSIFICATION SYSTEM
# =============================================================================

try:
    from .classification import (
        EnhancedProtocolClassifier,
        ClassifierConfig,
        ClassificationResult,
        ClassificationMethod,
        FeatureType
    )
    CLASSIFICATION_AVAILABLE = True
    logger.debug("Protocol classification system loaded successfully")
    
except ImportError as e:
    logger.warning(f"Protocol classification system not available: {e}")
    CLASSIFICATION_AVAILABLE = False
    EnhancedProtocolClassifier = None
    ClassifierConfig = None
    ClassificationResult = None
    ClassificationMethod = None
    FeatureType = None

# =============================================================================
# FHSS SYSTEM
# =============================================================================

try:
    from .fhss import (
        EnhancedFHSSEngine,
        SimpleFHSS,
        FHSSConfig,
        FHSSBand,
        HopFrame,
        create_fhss_engine,
        create_fcc_compliant_fhss
    )
    FHSS_AVAILABLE = True
    logger.debug("FHSS system loaded successfully")
    
except ImportError as e:
    logger.warning(f"FHSS system not available: {e}")
    FHSS_AVAILABLE = False
    EnhancedFHSSEngine = None
    SimpleFHSS = None
    FHSSConfig = None
    FHSSBand = None
    HopFrame = None
    create_fhss_engine = None
    create_fcc_compliant_fhss = None

# =============================================================================
# SIGNAL PROCESSING SYSTEM
# =============================================================================

try:
    from .signal_processing import (
        SignalProcessor,
        QualityMonitor,
        detect_packets,
        find_preamble,
        normalize_signal as advanced_normalize,
        analyze_signal_quality,
        correct_iq_imbalance,
        estimate_modulation_scheme,
        create_test_signal,
        convert_power_units
    )
    SIGNAL_PROCESSING_AVAILABLE = True
    logger.debug("Signal processing system loaded successfully")
    
except ImportError as e:
    logger.warning(f"Signal processing system not available: {e}")
    SIGNAL_PROCESSING_AVAILABLE = False
    SignalProcessor = None
    QualityMonitor = None
    detect_packets = None
    find_preamble = None
    advanced_normalize = None
    analyze_signal_quality = None
    correct_iq_imbalance = None
    estimate_modulation_scheme = None
    create_test_signal = None
    convert_power_units = None

# =============================================================================
# PACKET PARSING SYSTEM
# =============================================================================

try:
    from .parsing import (
        EnhancedPacketParser,
        ParserConfig,
        PacketResult,
        DroneProtocol
    )
    PARSING_AVAILABLE = True
    logger.debug("Packet parsing system loaded successfully")
    
except ImportError as e:
    logger.warning(f"Packet parsing system not available: {e}")
    PARSING_AVAILABLE = False
    EnhancedPacketParser = None
    ParserConfig = None
    PacketResult = None
    DroneProtocol = None

# =============================================================================
# REPLAY SYSTEM
# =============================================================================

try:
    from .replay import (
        EnhancedReplayEngine,
        ReplayConfig,
        ReplayResult,
        ReplayStrategy,
        TransmissionMode,
        ComplianceLevel
    )
    REPLAY_AVAILABLE = True
    logger.debug("Replay system loaded successfully")
    
except ImportError as e:
    logger.warning(f"Replay system not available: {e}")
    REPLAY_AVAILABLE = False
    EnhancedReplayEngine = None
    ReplayConfig = None
    ReplayResult = None
    ReplayStrategy = None
    TransmissionMode = None
    ComplianceLevel = None

# =============================================================================
# CORE FACTORY FUNCTIONS
# =============================================================================

def create_capture_system(
    platform: str = "rtl_sdr",
    **config_kwargs: Any
) -> Optional[Any]:
    """
    Create enhanced capture system with automatic configuration.
    
    Args:
        platform: SDR platform name
        **config_kwargs: SDR configuration parameters
        
    Returns:
        Enhanced capture system instance or None if not available
    """
    if not CAPTURE_AVAILABLE:
        logger.error("Enhanced capture system not available")
        return None
    
    try:
        # Map platform string to enum
        platform_map = {
            'rtl_sdr': SDRPlatform.RTL_SDR,
            'rtl_sdr_tcp': SDRPlatform.RTL_SDR_TCP,
            'hackrf': SDRPlatform.HACKRF,
            'airspy': SDRPlatform.AIRSPY,
            'sdrplay': SDRPlatform.SDRPLAY,
            'pluto_sdr': SDRPlatform.PLUTO_SDR,
            'usrp': SDRPlatform.USRP,
            'lime_sdr': SDRPlatform.LIME_SDR
        }
        
        platform_enum = platform_map.get(platform.lower(), SDRPlatform.RTL_SDR)
        
        # Create configuration
        config = SDRConfig(platform=platform_enum, **config_kwargs)
        
        # Create capture system
        return EnhancedLiveCapture(config)
        
    except Exception as e:
        logger.error(f"Failed to create capture system: {e}")
        return None


def create_demodulation_system(
    scheme: Union[str, ModulationScheme],
    **config_kwargs: Any
) -> Optional[Any]:
    """
    Create demodulation system for specified scheme.
    
    Args:
        scheme: Modulation scheme name or enum
        **config_kwargs: Demodulation configuration parameters
        
    Returns:
        Demodulation engine instance or None if not available
    """
    if not DEMODULATION_AVAILABLE:
        logger.error("Demodulation system not available")
        return None
    
    try:
        # Convert string to enum if needed
        if isinstance(scheme, str):
            scheme_map = {
                'ook': ModulationScheme.OOK,
                'ask': ModulationScheme.ASK,
                'fsk': ModulationScheme.FSK,
                'gfsk': ModulationScheme.GFSK,
                'msk': ModulationScheme.MSK,
                'psk': ModulationScheme.PSK,
                'bpsk': ModulationScheme.BPSK,
                'qpsk': ModulationScheme.QPSK,
                'dpsk': ModulationScheme.DPSK
            }
            scheme = scheme_map.get(scheme.lower(), ModulationScheme.OOK)
        
        # Create configuration
        config = DemodConfig(scheme=scheme, **config_kwargs)
        
        # Create demodulation engine
        return DemodulationEngine(config)
        
    except Exception as e:
        logger.error(f"Failed to create demodulation system: {e}")
        return None


def create_classification_system(**config_kwargs: Any) -> Optional[Any]:
    """
    Create protocol classification system.
    
    Args:
        **config_kwargs: Classifier configuration parameters
        
    Returns:
        Protocol classifier instance or None if not available
    """
    if not CLASSIFICATION_AVAILABLE:
        logger.error("Classification system not available")
        return None
    
    try:
        config = ClassifierConfig(**config_kwargs)
        return EnhancedProtocolClassifier(config)
        
    except Exception as e:
        logger.error(f"Failed to create classification system: {e}")
        return None


def create_integrated_analysis_pipeline(
    capture_config: Optional[Dict[str, Any]] = None,
    demod_config: Optional[Dict[str, Any]] = None,
    classifier_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create integrated analysis pipeline with all core systems.
    
    Args:
        capture_config: Capture system configuration
        demod_config: Demodulation system configuration  
        classifier_config: Classification system configuration
        
    Returns:
        Dictionary with initialized systems
    """
    pipeline = {}
    
    # Create capture system
    if CAPTURE_AVAILABLE and capture_config:
        pipeline['capture'] = create_capture_system(**capture_config)
    
    # Create demodulation system
    if DEMODULATION_AVAILABLE and demod_config:
        scheme = demod_config.pop('scheme', 'ook')
        pipeline['demodulation'] = create_demodulation_system(scheme, **demod_config)
    
    # Create classification system
    if CLASSIFICATION_AVAILABLE and classifier_config:
        pipeline['classification'] = create_classification_system(**classifier_config)
    
    # Create signal processor
    if SIGNAL_PROCESSING_AVAILABLE:
        pipeline['signal_processor'] = SignalProcessor()
        pipeline['quality_monitor'] = QualityMonitor()
    
    # Create FHSS engine if available
    if FHSS_AVAILABLE:
        pipeline['fhss'] = create_fhss_engine
    
    logger.info(f"Created integrated pipeline with {len(pipeline)} systems")
    return pipeline

# =============================================================================
# SYSTEM STATUS AND DIAGNOSTICS
# =============================================================================

def get_core_system_status() -> Dict[str, bool]:
    """
    Get status of all core systems.
    
    Returns:
        Dictionary mapping system names to availability status
    """
    return {
        'capture': CAPTURE_AVAILABLE,
        'demodulation': DEMODULATION_AVAILABLE,
        'classification': CLASSIFICATION_AVAILABLE,
        'fhss': FHSS_AVAILABLE,
        'signal_processing': SIGNAL_PROCESSING_AVAILABLE,
        'parsing': PARSING_AVAILABLE,
        'replay': REPLAY_AVAILABLE
    }


def check_core_dependencies() -> List[str]:
    """
    Check core system dependencies and return missing items.
    
    Returns:
        List of missing dependencies
    """
    missing = []
    
    # Check required packages
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    
    try:
        import scipy
    except ImportError:
        missing.append("scipy")
    
    try:
        import sklearn
    except ImportError:
        missing.append("scikit-learn (for classification)")
    
    # Check optional SDR packages
    try:
        import rtlsdr
    except ImportError:
        missing.append("pyrtlsdr (for RTL-SDR support)")
    
    try:
        import SoapySDR
    except ImportError:
        missing.append("SoapySDR (for additional SDR support)")
    
    return missing


def validate_core_integration() -> Dict[str, Any]:
    """
    Validate integration between core systems.
    
    Returns:
        Dictionary with validation results
    """
    results = {
        'status': 'ok',
        'issues': [],
        'warnings': [],
        'recommendations': []
    }
    
    # Check system availability
    status = get_core_system_status()
    unavailable = [name for name, available in status.items() if not available]
    
    if unavailable:
        results['warnings'].append(f"Unavailable systems: {', '.join(unavailable)}")
    
    # Check dependencies
    missing_deps = check_core_dependencies()
    if missing_deps:
        results['issues'].append(f"Missing dependencies: {', '.join(missing_deps)}")
        results['status'] = 'degraded'
    
    # Integration checks
    if CAPTURE_AVAILABLE and SIGNAL_PROCESSING_AVAILABLE:
        results['recommendations'].append("Capture + Signal Processing: Full pipeline available")
    
    if DEMODULATION_AVAILABLE and CLASSIFICATION_AVAILABLE:
        results['recommendations'].append("Demod + Classification: Complete protocol analysis available")
    
    if FHSS_AVAILABLE and REPLAY_AVAILABLE:
        results['recommendations'].append("FHSS + Replay: Advanced transmission testing available")
    
    return results

# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Core system classes
    'EnhancedLiveCapture',
    'DemodulationEngine', 
    'EnhancedProtocolClassifier',
    'EnhancedFHSSEngine',
    'SignalProcessor',
    'EnhancedReplayEngine',
    'EnhancedPacketParser',
    
    # Configuration classes
    'SDRConfig',
    'DemodConfig',
    'ClassifierConfig', 
    'FHSSConfig',
    'ReplayConfig',
    'ParserConfig',
    
    # Result classes
    'CaptureMetadata',
    'DemodulationResult',
    'ClassificationResult',
    'ReplayResult',
    'PacketResult',
    'HopFrame',
    
    # Enums
    'SDRPlatform',
    'GainMode',
    'CaptureFormat',
    'ModulationScheme',
    'FilterType',
    'ClassificationMethod',
    'FeatureType',
    'FHSSBand',
    'ReplayStrategy',
    'TransmissionMode',
    'ComplianceLevel',
    'DroneProtocol',
    
    # Simple interfaces
    'SimpleFHSS',
    'QualityMonitor',
    'SignalQualityMonitor',
    
    # Factory functions
    'create_capture_system',
    'create_demodulation_system',
    'create_classification_system',
    'create_integrated_analysis_pipeline',
    'create_fhss_engine',
    'create_fcc_compliant_fhss',
    
    # Signal processing functions
    'detect_packets',
    'find_preamble',
    'advanced_normalize',
    'analyze_signal_quality',
    'correct_iq_imbalance',
    'estimate_modulation_scheme',
    'create_test_signal',
    'convert_power_units',
    
    # System utilities
    'get_core_system_status',
    'check_core_dependencies',
    'validate_core_integration'
]

# Filter __all__ based on availability
available_exports = []
for name in __all__:
    if globals().get(name) is not None:
        available_exports.append(name)

__all__ = available_exports

# Log core module initialization
logger.info(f"DroneCmd core module initialized with {len(__all__)} exports")
logger.debug(f"Available systems: {get_core_system_status()}")

# Issue warnings for missing systems
missing_systems = [name for name, status in get_core_system_status().items() if not status]
if missing_systems:
    warnings.warn(
        f"Some core systems not available: {', '.join(missing_systems)}. "
        f"Check dependencies and installation.",
        UserWarning,
        stacklevel=2
    )