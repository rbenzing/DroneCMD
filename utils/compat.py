#!/usr/bin/env python3
"""
Backward Compatibility Layer

This module provides backward compatibility for the refactored dronecmd library,
ensuring existing code continues to work while providing migration paths to
enhanced functionality.

Key Features:
- Import aliases for moved modules
- Wrapper functions for changed APIs
- Deprecation warnings with upgrade paths
- Automatic fallbacks for missing dependencies
- Version compatibility checks

Usage:
    # Old code continues to work
    from dronecmd.utils.signal_tools import FHSSEngine
    
    # New enhanced features available
    from dronecmd.core.fhss import EnhancedFHSSEngine
    
    # Compatibility helpers
    from dronecmd.utils.compat import check_compatibility, migrate_config
"""

from __future__ import annotations

import logging
import warnings
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

# Version information
__version__ = "2.0.0"
__compatibility_version__ = "1.0.0"

# Configure module logger
logger = logging.getLogger(__name__)

# Compatibility flags
_compatibility_mode = False
_deprecation_warnings_enabled = True


class CompatibilityError(Exception):
    """Exception raised when compatibility cannot be maintained."""
    pass


class DeprecationHelper:
    """Helper class for managing deprecation warnings and migrations."""
    
    def __init__(self):
        self._warned_functions = set()
        self._warned_modules = set()
    
    def warn_deprecated_function(
        self,
        old_name: str,
        new_name: str,
        version: str = "2.0.0",
        removal_version: str = "3.0.0"
    ) -> None:
        """Issue deprecation warning for function."""
        if not _deprecation_warnings_enabled or old_name in self._warned_functions:
            return
        
        warnings.warn(
            f"{old_name} is deprecated since version {version} and will be "
            f"removed in version {removal_version}. Use {new_name} instead.",
            DeprecationWarning,
            stacklevel=3
        )
        self._warned_functions.add(old_name)
    
    def warn_deprecated_module(
        self,
        old_module: str,
        new_module: str,
        version: str = "2.0.0"
    ) -> None:
        """Issue deprecation warning for module."""
        if not _deprecation_warnings_enabled or old_module in self._warned_modules:
            return
        
        warnings.warn(
            f"Module {old_module} is deprecated since version {version}. "
            f"Use {new_module} instead.",
            DeprecationWarning,
            stacklevel=3
        )
        self._warned_modules.add(old_module)


# Global deprecation helper
_deprecation_helper = DeprecationHelper()


# =============================================================================
# MODULE IMPORT COMPATIBILITY
# =============================================================================

def _safe_import(module_name: str, fallback_name: Optional[str] = None) -> Any:
    """Safely import module with fallback."""
    try:
        parts = module_name.split('.')
        module = __import__(module_name, fromlist=[parts[-1]])
        return module
    except ImportError as e:
        if fallback_name:
            logger.debug(f"Failed to import {module_name}, trying fallback {fallback_name}")
            try:
                parts = fallback_name.split('.')
                module = __import__(fallback_name, fromlist=[parts[-1]])
                return module
            except ImportError:
                pass
        
        logger.warning(f"Could not import {module_name}: {e}")
        return None


# Import enhanced modules with fallbacks
try:
    from ..core import fhss as _core_fhss
    from ..core import signal_processing as _core_signal_processing
    from ..core import capture as _core_capture
    from ..core import demodulation as _core_demodulation
    from ..core import classification as _core_classification
    from ..capture import manager as _capture_manager
    ENHANCED_MODULES_AVAILABLE = True
except ImportError:
    _core_fhss = None
    _core_signal_processing = None
    _core_capture = None
    _core_demodulation = None
    _core_classification = None
    _capture_manager = None
    ENHANCED_MODULES_AVAILABLE = False
    logger.warning("Enhanced modules not available, running in compatibility mode")


# =============================================================================
# SIGNAL_TOOLS COMPATIBILITY (signal_tools.py -> core/fhss.py)
# =============================================================================

class FHSSEngine:
    """
    Backward compatible FHSS engine.
    
    This class provides the exact same interface as the original signal_tools.FHSSEngine
    while delegating to the enhanced implementation.
    """
    
    def __init__(
        self,
        center_freq_hz: float,
        channel_spacing_hz: float,
        hops: int = 8,
        seed: Optional[int] = None,
        hop_sequence: Optional[List[int]] = None
    ):
        """Initialize FHSS engine (backward compatible interface)."""
        _deprecation_helper.warn_deprecated_function(
            "dronecmd.utils.signal_tools.FHSSEngine",
            "dronecmd.core.fhss.SimpleFHSS"
        )
        
        if ENHANCED_MODULES_AVAILABLE and _core_fhss:
            # Use enhanced implementation
            self._engine = _core_fhss.SimpleFHSS(
                center_freq_hz, channel_spacing_hz, hops, seed, hop_sequence
            )
        else:
            # Fallback implementation
            self._init_fallback(center_freq_hz, channel_spacing_hz, hops, seed, hop_sequence)
    
    def _init_fallback(self, center_freq_hz, channel_spacing_hz, hops, seed, hop_sequence):
        """Fallback initialization for when enhanced modules aren't available."""
        import numpy as np
        
        self.center = float(center_freq_hz)
        self.spacing = float(channel_spacing_hz)
        self.hops = int(hops)
        self.rng = np.random.default_rng(seed)
        
        # Generate channel frequencies
        half_span = (hops - 1) / 2.0
        self.channels = np.array([
            center_freq_hz + (i - half_span) * channel_spacing_hz
            for i in range(hops)
        ])
        
        # Set hop sequence
        if hop_sequence is not None:
            self.hop_sequence = list(hop_sequence)
        else:
            seq = np.arange(hops)
            self.rng.shuffle(seq)
            self.hop_sequence = seq.tolist()
    
    def generate_hop_map(self, length: int, seed: Optional[int] = None) -> List[int]:
        """Generate hop map (backward compatible)."""
        if hasattr(self._engine, 'generate_hop_map'):
            return self._engine.generate_hop_map(length, seed)
        else:
            # Fallback implementation
            rng = np.random.default_rng(seed) if seed is not None else self.rng
            sequence = []
            while len(sequence) < length:
                block = self.hop_sequence.copy()
                rng.shuffle(block)
                sequence.extend(block)
            return sequence[:length]
    
    def split_packet_into_hops(self, packet: bytes, hop_count: int) -> List[bytes]:
        """Split packet into hops (backward compatible)."""
        if hasattr(self._engine, 'split_packet_into_hops'):
            return self._engine.split_packet_into_hops(packet, hop_count)
        else:
            # Fallback implementation
            if hop_count <= 0:
                return []
            if not packet:
                return [b""] * hop_count
            
            n = len(packet)
            q, r = divmod(n, hop_count)
            chunks = []
            idx = 0
            for i in range(hop_count):
                take = q + (1 if i < r else 0)
                chunks.append(packet[idx:idx + take])
                idx += take
            return chunks
    
    def prepare_transmit_frames(
        self,
        packet: bytes,
        sample_rate: int = 2_000_000,
        bitrate: int = 1000,
        **kwargs
    ) -> List[Tuple[float, np.ndarray, float]]:
        """Prepare transmit frames (backward compatible)."""
        if hasattr(self._engine, 'prepare_transmit_frames'):
            return self._engine.prepare_transmit_frames(packet, sample_rate, bitrate, **kwargs)
        else:
            # Basic fallback - return empty frames
            logger.warning("Using basic fallback for prepare_transmit_frames")
            return [(self.channels[0], np.array([]), 0.1)]


# Helper functions from signal_tools.py
def _bytes_to_bits(data: bytes) -> np.ndarray:
    """Convert bytes to bits (backward compatible)."""
    if ENHANCED_MODULES_AVAILABLE and _core_fhss:
        return _core_fhss.FHSSCore.bytes_to_bits(data)
    else:
        import numpy as np
        if not data:
            return np.array([], dtype=np.int8)
        arr = np.frombuffer(data, dtype=np.uint8)
        bits = np.unpackbits(arr)
        return bits.astype(np.int8)


def _bits_to_bytes(bits: np.ndarray) -> bytes:
    """Convert bits to bytes (backward compatible)."""
    if ENHANCED_MODULES_AVAILABLE and _core_fhss:
        return _core_fhss.FHSSCore.bits_to_bytes(bits)
    else:
        import numpy as np
        if bits.size == 0:
            return b""
        if bits.size % 8 != 0:
            bits = bits[:-(bits.size % 8)]
        return np.packbits(bits).tobytes()


# =============================================================================
# CAPTURE COMPATIBILITY
# =============================================================================

class CaptureManager:
    """
    Backward compatible capture manager.
    
    Delegates to the new simplified capture manager while maintaining
    the original interface.
    """
    
    def __init__(self, freq_hz: float = 0, sample_rate: float = 2.048e6, duration_sec: float = 0):
        """Initialize capture manager (backward compatible)."""
        _deprecation_helper.warn_deprecated_function(
            "dronecmd.capture.capture_manager.CaptureManager",
            "dronecmd.capture.manager.CaptureManager"
        )
        
        if ENHANCED_MODULES_AVAILABLE and _capture_manager:
            self._manager = _capture_manager.CaptureManager(
                sample_rate=sample_rate
            )
            if freq_hz > 0:
                self._manager.set_frequency(freq_hz)
        else:
            # Basic fallback
            self.freq_hz = freq_hz
            self.sample_rate = sample_rate
            self.duration_sec = duration_sec
            self.captured_iq = []
    
    def capture(self, frequency: float, duration: float) -> str:
        """Capture to file (backward compatible)."""
        if hasattr(self._manager, 'capture_to_file'):
            self._manager.set_frequency(frequency)
            timestamp = int(time.time())
            filename = f"iq_capture_{timestamp}.iq"
            return self._manager.capture_to_file(filename, duration)
        else:
            # Fallback
            import time
            timestamp = int(time.time())
            filename = f"iq_capture_{timestamp}.iq"
            logger.warning(f"Fallback capture simulation: {filename}")
            return filename
    
    def load_file(self, path: str) -> None:
        """Load file (backward compatible)."""
        if hasattr(self._manager, 'load_file'):
            data = self._manager.load_file(path)
            self.captured_iq = [data]
        else:
            # Fallback
            try:
                from ..utils.fileio import read_iq_file
                data = read_iq_file(path)
                self.captured_iq = [data]
            except:
                logger.warning(f"Could not load file: {path}")
                self.captured_iq = []
    
    def segment_packets(self) -> List[np.ndarray]:
        """Segment packets (backward compatible)."""
        if hasattr(self._manager, 'extract_packets') and self.captured_iq:
            return self._manager.extract_packets(self.captured_iq[0])
        else:
            # Basic fallback
            if self.captured_iq:
                # Return the whole capture as one "packet"
                return self.captured_iq
            return []


# =============================================================================
# UTILITY COMPATIBILITY
# =============================================================================

def normalize_signal(iq_data: np.ndarray) -> np.ndarray:
    """Normalize signal (backward compatible)."""
    if ENHANCED_MODULES_AVAILABLE and _core_signal_processing:
        return _core_signal_processing.normalize_signal(iq_data)
    else:
        # Basic fallback
        import numpy as np
        if len(iq_data) == 0:
            return iq_data
        max_val = np.max(np.abs(iq_data))
        if max_val == 0:
            return iq_data
        return iq_data / max_val


def iq_to_complex(i_samples, q_samples) -> np.ndarray:
    """Convert I/Q to complex (backward compatible)."""
    if ENHANCED_MODULES_AVAILABLE and _core_signal_processing:
        return _core_signal_processing.iq_to_complex(i_samples, q_samples)
    else:
        import numpy as np
        return np.array(i_samples) + 1j * np.array(q_samples)


# =============================================================================
# CONFIGURATION MIGRATION
# =============================================================================

def migrate_config(old_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Migrate old configuration to new format.
    
    Args:
        old_config: Dictionary with old configuration format
        
    Returns:
        Dictionary with new configuration format
    """
    new_config = {}
    
    # Map old keys to new keys
    key_mapping = {
        'freq_hz': 'frequency_hz',
        'sample_rate': 'sample_rate_hz',
        'duration_sec': 'duration_s',
        'gain': 'gain_db',
        'device': 'device_index'
    }
    
    for old_key, value in old_config.items():
        new_key = key_mapping.get(old_key, old_key)
        new_config[new_key] = value
    
    return new_config


def check_compatibility() -> Dict[str, Any]:
    """
    Check compatibility status of the library.
    
    Returns:
        Dictionary with compatibility information
    """
    import sys
    
    compatibility_info = {
        'library_version': __version__,
        'compatibility_version': __compatibility_version__,
        'python_version': sys.version,
        'enhanced_modules_available': ENHANCED_MODULES_AVAILABLE,
        'compatibility_mode': _compatibility_mode,
        'deprecation_warnings': _deprecation_warnings_enabled
    }
    
    # Check for required dependencies
    dependencies = {
        'numpy': False,
        'scipy': False,
        'sklearn': False,
        'pyrtlsdr': False
    }
    
    for dep in dependencies:
        try:
            __import__(dep)
            dependencies[dep] = True
        except ImportError:
            dependencies[dep] = False
    
    compatibility_info['dependencies'] = dependencies
    
    # Check for enhanced modules
    enhanced_modules = {
        'core.fhss': _core_fhss is not None,
        'core.signal_processing': _core_signal_processing is not None,
        'core.capture': _core_capture is not None,
        'capture.manager': _capture_manager is not None
    }
    
    compatibility_info['enhanced_modules'] = enhanced_modules
    
    return compatibility_info


def enable_compatibility_mode(enable: bool = True) -> None:
    """
    Enable or disable compatibility mode.
    
    Args:
        enable: Whether to enable compatibility mode
    """
    global _compatibility_mode
    _compatibility_mode = enable
    
    if enable:
        logger.info("Enabled compatibility mode for legacy code")
    else:
        logger.info("Disabled compatibility mode")


def enable_deprecation_warnings(enable: bool = True) -> None:
    """
    Enable or disable deprecation warnings.
    
    Args:
        enable: Whether to enable deprecation warnings
    """
    global _deprecation_warnings_enabled
    _deprecation_warnings_enabled = enable
    
    if enable:
        warnings.filterwarnings('default', category=DeprecationWarning)
        logger.info("Enabled deprecation warnings")
    else:
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        logger.info("Disabled deprecation warnings")


def get_migration_guide() -> List[str]:
    """
    Get migration guide for upgrading to enhanced modules.
    
    Returns:
        List of migration steps
    """
    return [
        "1. Update imports:",
        "   Old: from dronecmd.utils.signal_tools import FHSSEngine",
        "   New: from dronecmd.core.fhss import SimpleFHSS",
        "",
        "2. Update capture manager:",
        "   Old: from dronecmd.capture.capture_manager import CaptureManager", 
        "   New: from dronecmd.capture.manager import CaptureManager",
        "",
        "3. Use enhanced signal processing:",
        "   Old: from dronecmd.utils import utils",
        "   New: from dronecmd.core.signal_processing import SignalProcessor",
        "",
        "4. Configuration changes:",
        "   Old: config = {'freq_hz': 2.4e9, 'sample_rate': 2e6}",
        "   New: config = {'frequency_hz': 2.4e9, 'sample_rate_hz': 2e6}",
        "",
        "5. Enable enhanced features:",
        "   Use EnhancedFHSSEngine for FCC compliance",
        "   Use SignalProcessor for advanced signal processing",
        "   Use QualityMonitor for real-time monitoring"
    ]


# =============================================================================
# PLUGIN COMPATIBILITY
# =============================================================================

def wrap_legacy_plugin(plugin_class):
    """
    Decorator to wrap legacy plugin classes for compatibility.
    
    Args:
        plugin_class: Legacy plugin class to wrap
        
    Returns:
        Wrapped plugin class
    """
    class WrappedPlugin(plugin_class):
        def __init__(self, *args, **kwargs):
            _deprecation_helper.warn_deprecated_function(
                f"{plugin_class.__name__}",
                "dronecmd.plugins.base.BasePlugin"
            )
            super().__init__(*args, **kwargs)
    
    return WrappedPlugin


# =============================================================================
# VERSION COMPATIBILITY
# =============================================================================

def require_version(min_version: str) -> None:
    """
    Require minimum library version.
    
    Args:
        min_version: Minimum required version string
        
    Raises:
        CompatibilityError: If version requirement not met
    """
    from packaging import version
    
    if version.parse(__version__) < version.parse(min_version):
        raise CompatibilityError(
            f"Library version {__version__} is below required {min_version}"
        )


def check_python_version(min_version: str = "3.8") -> bool:
    """
    Check if Python version meets requirements.
    
    Args:
        min_version: Minimum Python version required
        
    Returns:
        True if version is compatible
    """
    import sys
    from packaging import version
    
    current_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    return version.parse(current_version) >= version.parse(min_version)


# =============================================================================
# INITIALIZATION
# =============================================================================

def _initialize_compatibility():
    """Initialize compatibility layer."""
    # Check Python version
    if not check_python_version("3.8"):
        warnings.warn(
            "Python version < 3.8 may not be fully supported",
            RuntimeWarning
        )
    
    # Check for enhanced modules
    if not ENHANCED_MODULES_AVAILABLE:
        logger.warning(
            "Enhanced modules not available. Some features may be limited. "
            "Consider upgrading to access advanced functionality."
        )
        enable_compatibility_mode(True)
    
    # Set up deprecation warnings
    warnings.filterwarnings('default', category=DeprecationWarning)
    
    logger.info(f"Compatibility layer initialized (version {__version__})")


# Initialize on import
_initialize_compatibility()


# Example usage
if __name__ == "__main__":
    print("=== DroneCmd Compatibility Layer ===")
    
    # Check compatibility
    compat_info = check_compatibility()
    print(f"Library version: {compat_info['library_version']}")
    print(f"Enhanced modules: {compat_info['enhanced_modules_available']}")
    print(f"Python version: {compat_info['python_version']}")
    
    # Show migration guide
    print("\n=== Migration Guide ===")
    for step in get_migration_guide():
        print(step)
    
    # Test backward compatibility
    print("\n=== Testing Backward Compatibility ===")
    
    try:
        # Test FHSS engine
        fhss = FHSSEngine(center_freq_hz=2.44e9, hops=8)
        print(f"✓ FHSS engine created with {len(fhss.channels)} channels")
        
        # Test signal normalization
        import numpy as np
        test_signal = np.random.complex64(np.random.randn(1000) + 1j * np.random.randn(1000))
        normalized = normalize_signal(test_signal)
        print(f"✓ Signal normalized: {np.max(np.abs(normalized)):.3f}")
        
        print("Backward compatibility tests passed!")
        
    except Exception as e:
        print(f"✗ Compatibility test failed: {e}")