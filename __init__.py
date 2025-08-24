#!/usr/bin/env python3
"""
DroneCmd: Comprehensive Drone Command and Interference Toolkit

A full-spectrum SDR-based toolkit for drone communication analysis, providing
both simple interfaces for common tasks and advanced systems for production
use. The library follows progressive enhancement principles, allowing users
to start with simple APIs and graduate to advanced features as needed.

Key Features:
- Multi-platform SDR support (RTL-SDR, HackRF, Airspy, etc.)
- Advanced signal processing and demodulation
- Protocol classification and packet parsing
- FHSS (Frequency Hopping) support with FCC compliance
- Real-time signal quality monitoring
- Comprehensive plugin system for protocol extensions

Quick Start:
    >>> import dronecmd
    >>> # Simple capture
    >>> manager = dronecmd.CaptureManager()
    >>> samples = manager.capture(frequency=2.44e9, duration=10)
    >>> 
    >>> # Protocol analysis
    >>> classifier = dronecmd.ProtocolClassifier()
    >>> result = classifier.classify(packet_data)

For Advanced Use:
    >>> from dronecmd.core import EnhancedLiveCapture, DemodulationEngine
    >>> # Production-ready capture with quality monitoring
    >>> capture = EnhancedLiveCapture(config)
    >>> # Advanced demodulation with multiple schemes
    >>> demod = DemodulationEngine(config)
"""

# Package metadata
__version__ = "1.0.0"
__author__ = "DroneCmd Development Team"
__email__ = "contact@dronecmd.dev"
__license__ = "Educational/Research Use Only"
__description__ = "Comprehensive Drone Command and Interference Toolkit"

# Python version check
import sys
if sys.version_info < (3, 8):
    raise RuntimeError("DroneCmd requires Python 3.8 or higher")

import logging
import warnings
from typing import Any, Dict, List, Optional

# Configure package logger
logger = logging.getLogger(__name__)

# =============================================================================
# CORE EXPORTS - Simple Interfaces (Primary API)
# =============================================================================

try:
    # Simple capture interface
    from .capture import CaptureManager, PacketSniffer
    
    # Simple signal processing
    from .utils import (
        normalize_signal, 
        iq_to_complex, 
        calculate_power,
        frequency_shift
    )
    
    # Configuration and logging
    from .utils import configure_logging, get_logger
    
    SIMPLE_INTERFACES_AVAILABLE = True
    
except ImportError as e:
    logger.warning(f"Simple interfaces not fully available: {e}")
    SIMPLE_INTERFACES_AVAILABLE = False
    # Provide minimal fallbacks
    CaptureManager = None
    PacketSniffer = None

# =============================================================================
# ENHANCED CORE EXPORTS - Advanced Features
# =============================================================================

try:
    # Advanced core systems
    from .core.capture import EnhancedLiveCapture, SDRConfig, SDRPlatform
    from .core.demodulation import DemodulationEngine, DemodConfig, ModulationScheme
    from .core.classification import EnhancedProtocolClassifier, ClassifierConfig
    from .core.fhss import EnhancedFHSSEngine, SimpleFHSS, FHSSBand, FHSSConfig
    from .core.signal_processing import SignalProcessor, QualityMonitor
    from .core.replay import EnhancedReplayEngine, ReplayConfig
    
    ENHANCED_CORE_AVAILABLE = True
    
except ImportError as e:
    logger.warning(f"Enhanced core systems not fully available: {e}")
    ENHANCED_CORE_AVAILABLE = False
    # Set to None for graceful degradation
    EnhancedLiveCapture = None
    DemodulationEngine = None
    EnhancedProtocolClassifier = None
    EnhancedFHSSEngine = None
    SignalProcessor = None
    EnhancedReplayEngine = None

# =============================================================================
# PLUGIN SYSTEM EXPORTS
# =============================================================================

try:
    from .plugins import PluginRegistry, BasePlugin, load_plugins
    from .plugins.registry import register_plugin, get_plugin, list_plugins
    
    PLUGINS_AVAILABLE = True
    
except ImportError as e:
    logger.warning(f"Plugin system not available: {e}")
    PLUGINS_AVAILABLE = False
    PluginRegistry = None
    BasePlugin = None

# =============================================================================
# FACTORY FUNCTIONS - Easy Creation
# =============================================================================

def create_capture_manager(
    platform: str = "rtl_sdr",
    simple: bool = True,
    **kwargs: Any
) -> Any:
    """
    Factory function to create appropriate capture manager.
    
    Args:
        platform: SDR platform ("rtl_sdr", "hackrf", etc.)
        simple: If True, return simple interface; if False, return enhanced
        **kwargs: Additional configuration parameters
        
    Returns:
        Capture manager instance
        
    Example:
        >>> # Simple usage
        >>> manager = create_capture_manager(platform="rtl_sdr")
        >>> 
        >>> # Advanced usage
        >>> manager = create_capture_manager(simple=False, 
        ...                                 frequency_hz=2.44e9,
        ...                                 sample_rate_hz=2.048e6)
    """
    if simple and SIMPLE_INTERFACES_AVAILABLE and CaptureManager:
        return CaptureManager(platform=platform, **kwargs)
    elif ENHANCED_CORE_AVAILABLE and EnhancedLiveCapture:
        # Create enhanced capture with configuration
        from .core.capture import SDRConfig, SDRPlatform
        
        # Map platform string to enum
        platform_map = {
            'rtl_sdr': SDRPlatform.RTL_SDR,
            'hackrf': SDRPlatform.HACKRF,
            'airspy': SDRPlatform.AIRSPY,
            'sdrplay': SDRPlatform.SDRPLAY
        }
        platform_enum = platform_map.get(platform.lower(), SDRPlatform.RTL_SDR)
        
        config = SDRConfig(platform=platform_enum, **kwargs)
        return EnhancedLiveCapture(config)
    else:
        raise RuntimeError("No capture interfaces available")


def create_protocol_classifier(simple: bool = True, **kwargs: Any) -> Any:
    """
    Factory function to create protocol classifier.
    
    Args:
        simple: If True, return simple interface; if False, return enhanced
        **kwargs: Configuration parameters
        
    Returns:
        Protocol classifier instance
    """
    if simple:
        # Try to import simple classifier or create wrapper
        try:
            from .capture import SimpleProtocolClassifier
            return SimpleProtocolClassifier(**kwargs)
        except ImportError:
            pass
    
    if ENHANCED_CORE_AVAILABLE and EnhancedProtocolClassifier:
        from .core.classification import ClassifierConfig
        config = ClassifierConfig(**kwargs)
        return EnhancedProtocolClassifier(config)
    else:
        raise RuntimeError("No protocol classifier available")


def create_fhss_engine(
    center_freq_hz: float,
    simple: bool = True,
    **kwargs: Any
) -> Any:
    """
    Factory function to create FHSS engine.
    
    Args:
        center_freq_hz: Center frequency in Hz
        simple: If True, return simple interface; if False, return enhanced
        **kwargs: Configuration parameters
        
    Returns:
        FHSS engine instance
    """
    if ENHANCED_CORE_AVAILABLE:
        from .core.fhss import create_fhss_engine
        return create_fhss_engine(center_freq_hz, simple=simple, **kwargs)
    else:
        raise RuntimeError("FHSS engine not available")


def create_signal_processor() -> Any:
    """Create signal processor instance."""
    if ENHANCED_CORE_AVAILABLE and SignalProcessor:
        return SignalProcessor()
    else:
        raise RuntimeError("Signal processor not available")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_version_info() -> Dict[str, Any]:
    """
    Get comprehensive version and feature information.
    
    Returns:
        Dictionary with version and capability information
    """
    info = {
        'version': __version__,
        'python_version': sys.version,
        'simple_interfaces': SIMPLE_INTERFACES_AVAILABLE,
        'enhanced_core': ENHANCED_CORE_AVAILABLE,
        'plugins': PLUGINS_AVAILABLE,
    }
    
    # Check optional dependencies
    optional_deps = {}
    
    try:
        import numpy
        optional_deps['numpy'] = numpy.__version__
    except ImportError:
        optional_deps['numpy'] = None
    
    try:
        import scipy
        optional_deps['scipy'] = scipy.__version__
    except ImportError:
        optional_deps['scipy'] = None
    
    try:
        import sklearn
        optional_deps['sklearn'] = sklearn.__version__
    except ImportError:
        optional_deps['sklearn'] = None
    
    try:
        from rtlsdr import RtlSdr
        optional_deps['rtlsdr'] = "available"
    except ImportError:
        optional_deps['rtlsdr'] = None
    
    info['dependencies'] = optional_deps
    
    return info


def check_system_requirements() -> List[str]:
    """
    Check system requirements and return list of issues.
    
    Returns:
        List of requirement issues (empty if all OK)
    """
    issues = []
    
    # Python version
    if sys.version_info < (3, 8):
        issues.append("Python 3.8+ required")
    
    # Required dependencies
    try:
        import numpy
        if tuple(map(int, numpy.__version__.split('.')[:2])) < (1, 20):
            issues.append("NumPy 1.20+ recommended")
    except ImportError:
        issues.append("NumPy is required")
    
    try:
        import scipy
    except ImportError:
        issues.append("SciPy is required")
    
    # Optional but recommended
    try:
        import sklearn
    except ImportError:
        issues.append("scikit-learn recommended for classification")
    
    return issues


def configure_package_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> None:
    """
    Configure logging for the entire package.
    
    Args:
        level: Logging level
        format_string: Custom format string
    """
    if format_string is None:
        format_string = '[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=[logging.StreamHandler()]
    )
    
    # Set package logger level
    package_logger = logging.getLogger(__name__)
    package_logger.setLevel(level)


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================

# Legacy imports for existing code
try:
    from .core.fhss import SimpleFHSS as FHSSEngine  # Original name
    from .utils import normalize_signal as normalize_iq_signal
    from .capture import CaptureManager as SDRCaptureManager
except ImportError:
    pass

# =============================================================================
# PACKAGE INITIALIZATION
# =============================================================================

def _check_and_warn():
    """Check system and issue warnings if needed."""
    issues = check_system_requirements()
    if issues:
        warnings.warn(
            f"System requirement issues detected: {', '.join(issues)}. "
            f"Some features may not work correctly.",
            UserWarning,
            stacklevel=2
        )
    
    # Feature availability warnings
    if not SIMPLE_INTERFACES_AVAILABLE:
        warnings.warn(
            "Simple interfaces not available. Some import errors occurred.",
            UserWarning,
            stacklevel=2
        )
    
    if not ENHANCED_CORE_AVAILABLE:
        warnings.warn(
            "Enhanced core systems not available. Advanced features limited.",
            UserWarning,
            stacklevel=2
        )

# Run checks on import
_check_and_warn()

# =============================================================================
# PUBLIC API DEFINITION
# =============================================================================

__all__ = [
    # Metadata
    '__version__',
    '__author__',
    '__license__',
    
    # Factory functions (primary interface)
    'create_capture_manager',
    'create_protocol_classifier', 
    'create_fhss_engine',
    'create_signal_processor',
    
    # Simple interfaces (if available)
    'CaptureManager',
    'PacketSniffer',
    
    # Enhanced core (if available)
    'EnhancedLiveCapture',
    'DemodulationEngine', 
    'EnhancedProtocolClassifier',
    'EnhancedFHSSEngine',
    'SignalProcessor',
    'EnhancedReplayEngine',
    
    # Configuration classes
    'SDRConfig',
    'DemodConfig',
    'ClassifierConfig',
    'FHSSConfig',
    'ReplayConfig',
    
    # Enums
    'SDRPlatform',
    'ModulationScheme',
    'FHSSBand',
    
    # Utility functions
    'normalize_signal',
    'iq_to_complex',
    'calculate_power',
    'frequency_shift',
    
    # Plugin system
    'PluginRegistry',
    'BasePlugin',
    'load_plugins',
    
    # Package utilities
    'get_version_info',
    'check_system_requirements',
    'configure_package_logging',
    
    # Legacy aliases
    'FHSSEngine',  # Legacy name for SimpleFHSS
]

# Add conditional exports based on availability
if SIMPLE_INTERFACES_AVAILABLE:
    __all__.extend(['CaptureManager', 'PacketSniffer'])

if ENHANCED_CORE_AVAILABLE:
    __all__.extend([
        'EnhancedLiveCapture', 'DemodulationEngine', 'EnhancedProtocolClassifier',
        'EnhancedFHSSEngine', 'SignalProcessor', 'EnhancedReplayEngine'
    ])

if PLUGINS_AVAILABLE:
    __all__.extend(['PluginRegistry', 'BasePlugin', 'load_plugins'])

# Package initialization message
logger.info(f"DroneCmd v{__version__} initialized successfully")
logger.debug(f"Features: Simple={SIMPLE_INTERFACES_AVAILABLE}, "
            f"Enhanced={ENHANCED_CORE_AVAILABLE}, "
            f"Plugins={PLUGINS_AVAILABLE}")