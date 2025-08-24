#!/usr/bin/env python3
"""
DroneCmd Capture Module - Simplified Interfaces

This module provides easy-to-use interfaces for SDR capture operations,
wrapping the enhanced core systems with simplified APIs that are perfect
for getting started, scripting, and common use cases.

Key Features:
- Simple capture operations with automatic configuration
- Packet detection and extraction with minimal setup
- Backward compatibility with existing code
- Progressive enhancement to core systems
- Built-in error handling and validation

The capture module follows the adapter pattern, providing simple interfaces
that delegate to the enhanced core systems while hiding complexity.

Quick Start:
    >>> from dronecmd.capture import CaptureManager, PacketSniffer
    >>> 
    >>> # Simple capture
    >>> manager = CaptureManager()
    >>> manager.set_frequency(2.44e9)
    >>> samples = manager.capture(duration=10)
    >>> 
    >>> # Packet sniffing
    >>> sniffer = PacketSniffer("captured_data.iq")
    >>> packets = sniffer.sniff()
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

# Configure module logger
logger = logging.getLogger(__name__)

# =============================================================================
# SIMPLE CAPTURE MANAGER
# =============================================================================

try:
    from .manager import CaptureManager, CaptureManagerError
    MANAGER_AVAILABLE = True
    logger.debug("Capture manager loaded successfully")
    
except ImportError as e:
    logger.warning(f"Capture manager not available: {e}")
    MANAGER_AVAILABLE = False
    CaptureManager = None
    CaptureManagerError = None

# =============================================================================
# PACKET SNIFFER
# =============================================================================

try:
    from .sniffer import PacketSniffer, SnifferError
    SNIFFER_AVAILABLE = True
    logger.debug("Packet sniffer loaded successfully")
    
except ImportError as e:
    logger.warning(f"Packet sniffer not available: {e}")
    SNIFFER_AVAILABLE = False
    PacketSniffer = None
    SnifferError = None

# =============================================================================
# SIGNAL DETECTOR
# =============================================================================

try:
    from .detector import SignalDetector, DetectorConfig
    DETECTOR_AVAILABLE = True
    logger.debug("Signal detector loaded successfully")
    
except ImportError as e:
    logger.warning(f"Signal detector not available: {e}")
    DETECTOR_AVAILABLE = False
    SignalDetector = None
    DetectorConfig = None

# =============================================================================
# SIMPLE PROTOCOL CLASSIFIER
# =============================================================================

class SimpleProtocolClassifier:
    """
    Simple protocol classifier wrapper for easy use.
    
    Provides a simplified interface to the enhanced protocol classification
    system with automatic configuration and basic result formatting.
    """
    
    def __init__(self, model_path: Optional[str] = None) -> None:
        """
        Initialize simple protocol classifier.
        
        Args:
            model_path: Path to trained model files (optional)
        """
        self._classifier = None
        self._model_path = model_path
        
        # Try to initialize enhanced classifier
        try:
            from ..core.classification import EnhancedProtocolClassifier, ClassifierConfig
            
            config = ClassifierConfig()
            if model_path:
                config = ClassifierConfig(model_path=Path(model_path))
            
            self._classifier = EnhancedProtocolClassifier(config)
            self._enhanced_available = True
            
        except ImportError:
            logger.warning("Enhanced classifier not available, using fallback")
            self._enhanced_available = False
    
    def classify(self, packet_data: bytes) -> str:
        """
        Classify a packet and return protocol name.
        
        Args:
            packet_data: Raw packet bytes
            
        Returns:
            Predicted protocol name
        """
        if self._classifier and self._enhanced_available:
            try:
                result = self._classifier.classify(packet_data)
                if hasattr(result, 'predicted_protocol'):
                    return result.predicted_protocol
                else:
                    return str(result)
            except Exception as e:
                logger.error(f"Classification failed: {e}")
                return "unknown"
        else:
            # Simple fallback classification
            return self._simple_classify(packet_data)
    
    def classify_detailed(self, packet_data: bytes) -> Dict[str, Any]:
        """
        Classify packet and return detailed results.
        
        Args:
            packet_data: Raw packet bytes
            
        Returns:
            Dictionary with classification details
        """
        if self._classifier and self._enhanced_available:
            try:
                result = self._classifier.classify(packet_data)
                if hasattr(result, 'to_dict'):
                    return result.to_dict()
                else:
                    return {'predicted_protocol': str(result), 'confidence': 0.5}
            except Exception as e:
                logger.error(f"Detailed classification failed: {e}")
                return {'predicted_protocol': 'unknown', 'confidence': 0.0, 'error': str(e)}
        else:
            # Simple fallback
            protocol = self._simple_classify(packet_data)
            return {'predicted_protocol': protocol, 'confidence': 0.5, 'method': 'fallback'}
    
    def _simple_classify(self, packet_data: bytes) -> str:
        """Simple fallback classification based on packet structure."""
        if len(packet_data) < 4:
            return "unknown"
        
        # Check for common patterns
        first_bytes = packet_data[:4]
        
        # MAVLink patterns
        if packet_data[0] == 0xFE:
            return "mavlink_v1"
        elif packet_data[0] == 0xFD:
            return "mavlink_v2"
        
        # DJI patterns
        elif first_bytes[:2] == b'\x55\xAA':
            return "dji"
        
        # Generic patterns based on structure
        elif len(packet_data) > 20 and packet_data[0] in [0x7E, 0x7F, 0xFF]:
            return "generic_framed"
        
        else:
            return "unknown"

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def quick_capture(
    frequency_hz: float,
    duration_s: float,
    sample_rate_hz: float = 2.048e6,
    platform: str = "rtl_sdr",
    filename: Optional[str] = None
) -> Optional[Any]:
    """
    Quick capture function for one-shot operations.
    
    Args:
        frequency_hz: Center frequency in Hz
        duration_s: Capture duration in seconds
        sample_rate_hz: Sample rate in Hz
        platform: SDR platform name
        filename: Optional filename to save to
        
    Returns:
        IQ samples or filename if saved to file
    """
    if not MANAGER_AVAILABLE:
        logger.error("Capture manager not available")
        return None
    
    try:
        # Create manager
        manager = CaptureManager(platform=platform, sample_rate=sample_rate_hz)
        manager.set_frequency(frequency_hz)
        
        # Capture
        if filename:
            return manager.capture_to_file(filename, duration_s)
        else:
            return manager.capture(duration_s)
            
    except Exception as e:
        logger.error(f"Quick capture failed: {e}")
        return None


def quick_packet_extraction(
    iq_file: Union[str, Path],
    threshold: float = 0.05,
    min_packet_size: int = 100
) -> List[bytes]:
    """
    Quick packet extraction from IQ file.
    
    Args:
        iq_file: Path to IQ file
        threshold: Detection threshold
        min_packet_size: Minimum packet size in samples
        
    Returns:
        List of packet byte sequences
    """
    if not SNIFFER_AVAILABLE:
        logger.error("Packet sniffer not available")
        return []
    
    try:
        sniffer = PacketSniffer(str(iq_file))
        return sniffer.sniff(threshold=threshold)
        
    except Exception as e:
        logger.error(f"Quick packet extraction failed: {e}")
        return []


def analyze_capture_file(
    iq_file: Union[str, Path],
    generate_report: bool = True
) -> Dict[str, Any]:
    """
    Analyze IQ capture file and generate summary.
    
    Args:
        iq_file: Path to IQ file  
        generate_report: Generate detailed analysis report
        
    Returns:
        Analysis results dictionary
    """
    results = {
        'file': str(iq_file),
        'valid': False,
        'error': None
    }
    
    try:
        if not MANAGER_AVAILABLE:
            results['error'] = "Capture manager not available"
            return results
        
        # Load file
        manager = CaptureManager()
        samples = manager.load_file(iq_file)
        
        if samples is None:
            results['error'] = "Failed to load IQ file"
            return results
        
        # Basic analysis
        results.update({
            'valid': True,
            'total_samples': len(samples),
            'duration_estimate_s': len(samples) / 2.048e6,  # Assume 2.048 MHz
            'file_size_mb': Path(iq_file).stat().st_size / (1024*1024)
        })
        
        # Signal quality analysis
        quality = manager.analyze_signal_quality(samples)
        results['signal_quality'] = quality
        
        # Packet detection
        packets = manager.extract_packets(samples)
        results['packets_detected'] = len(packets)
        
        if packets:
            packet_sizes = [len(p) for p in packets]
            results['packet_stats'] = {
                'count': len(packets),
                'avg_size': sum(packet_sizes) / len(packet_sizes),
                'min_size': min(packet_sizes),
                'max_size': max(packet_sizes)
            }
        
        # Protocol analysis if available
        if packets and hasattr(SimpleProtocolClassifier, '__init__'):
            classifier = SimpleProtocolClassifier()
            protocols = {}
            
            for packet in packets[:10]:  # Analyze first 10 packets
                # Convert IQ to bytes (simplified)
                packet_bytes = packet.real.astype('uint8').tobytes()[:64]  # First 64 bytes
                protocol = classifier.classify(packet_bytes)
                protocols[protocol] = protocols.get(protocol, 0) + 1
            
            results['protocols_detected'] = protocols
        
        logger.info(f"Analyzed {iq_file}: {len(samples):,} samples, {len(packets)} packets")
        
    except Exception as e:
        results['error'] = str(e)
        logger.error(f"Analysis failed: {e}")
    
    return results


def get_supported_platforms() -> List[str]:
    """
    Get list of supported SDR platforms.
    
    Returns:
        List of platform names
    """
    platforms = ["rtl_sdr"]  # Always available in fallback
    
    # Check for additional platform support
    try:
        from ..core.capture import SDRPlatform
        platforms = [platform.value for platform in SDRPlatform]
    except ImportError:
        pass
    
    # Check actual hardware availability
    available_platforms = []
    
    for platform in platforms:
        if platform == "rtl_sdr":
            try:
                import rtlsdr
                available_platforms.append(platform)
            except ImportError:
                pass
        elif platform == "hackrf":
            try:
                import SoapySDR
                available_platforms.append(platform)
            except ImportError:
                pass
        # Add other platform checks as needed
    
    return available_platforms if available_platforms else ["rtl_sdr"]  # Fallback


def create_capture_session(
    platform: str = "rtl_sdr",
    **kwargs: Any
) -> Optional[Any]:
    """
    Create a configured capture session.
    
    Args:
        platform: SDR platform name
        **kwargs: Additional configuration
        
    Returns:
        Capture manager instance or None
    """
    if not MANAGER_AVAILABLE:
        logger.error("Capture manager not available")
        return None
    
    try:
        return CaptureManager(platform=platform, **kwargs)
    except Exception as e:
        logger.error(f"Failed to create capture session: {e}")
        return None

# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================

# Legacy class names and functions for existing code
try:
    from .manager import CaptureManager as SDRCapture  # Legacy name
    from ..utils.fileio import read_iq_file, write_iq_file
    
    # Legacy function aliases
    def load_iq_data(filename: str):
        """Legacy function for loading IQ data."""
        return read_iq_file(filename)
    
    def save_iq_data(filename: str, data):
        """Legacy function for saving IQ data."""
        return write_iq_file(filename, data)
        
except ImportError:
    SDRCapture = None
    load_iq_data = None
    save_iq_data = None

# =============================================================================
# MODULE VALIDATION
# =============================================================================

def get_capture_capabilities() -> Dict[str, bool]:
    """
    Get available capture capabilities.
    
    Returns:
        Dictionary of capability flags
    """
    return {
        'manager': MANAGER_AVAILABLE,
        'sniffer': SNIFFER_AVAILABLE,
        'detector': DETECTOR_AVAILABLE,
        'enhanced_core': _check_enhanced_core(),
        'rtl_sdr_hardware': _check_rtl_sdr(),
        'file_io': _check_file_io()
    }


def _check_enhanced_core() -> bool:
    """Check if enhanced core systems are available."""
    try:
        from ..core.capture import EnhancedLiveCapture
        return True
    except ImportError:
        return False


def _check_rtl_sdr() -> bool:
    """Check if RTL-SDR hardware support is available."""
    try:
        import rtlsdr
        return True
    except ImportError:
        return False


def _check_file_io() -> bool:
    """Check if file I/O utilities are available."""
    try:
        from ..utils.fileio import read_iq_file, write_iq_file
        return True
    except ImportError:
        return False

# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Main classes
    'CaptureManager',
    'PacketSniffer', 
    'SignalDetector',
    'SimpleProtocolClassifier',
    
    # Configuration classes
    'DetectorConfig',
    
    # Error classes
    'CaptureManagerError',
    'SnifferError',
    
    # Utility functions
    'quick_capture',
    'quick_packet_extraction',
    'analyze_capture_file',
    'get_supported_platforms',
    'create_capture_session',
    
    # File I/O functions
    'load_iq_data',
    'save_iq_data',
    
    # Legacy aliases
    'SDRCapture',
    
    # System functions
    'get_capture_capabilities'
]

# Filter exports based on availability
available_exports = []
for name in __all__:
    if globals().get(name) is not None:
        available_exports.append(name)

__all__ = available_exports

# Module initialization
logger.info(f"DroneCmd capture module initialized with {len(__all__)} exports")

# Check capabilities and warn if limited
capabilities = get_capture_capabilities()
if not any(capabilities.values()):
    warnings.warn(
        "No capture capabilities available. Check installation and dependencies.",
        UserWarning,
        stacklevel=2
    )
elif not capabilities.get('enhanced_core', False):
    logger.info("Using simplified capture interfaces (enhanced core not available)")

# Log available platforms
platforms = get_supported_platforms()
logger.debug(f"Supported SDR platforms: {platforms}")