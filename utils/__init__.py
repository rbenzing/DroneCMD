#!/usr/bin/env python3
"""
Utils Package Interface

This module provides a clean API for all utility functions and classes,
organizing them by functionality and providing both simple and advanced
interfaces. It serves as the main entry point for utility functionality.

Organized by functionality:
- File I/O operations (fileio)
- Cryptographic utilities (crypto)
- Logging configuration (logging)
- Configuration management (config)
- Backward compatibility (compat)

Example Usage:
    # File operations
    >>> from dronecmd.utils import read_iq_file, write_iq_file
    >>> data = read_iq_file("capture.iq")
    
    # Crypto operations
    >>> from dronecmd.utils import sha256_digest, hmac_sha256
    >>> digest = sha256_digest("test data")
    
    # Logging setup
    >>> from dronecmd.utils import configure_logging, get_logger
    >>> configure_logging(level='INFO')
    >>> logger = get_logger(__name__)
    
    # Configuration
    >>> from dronecmd.utils import load_config, get_config_value
    >>> config = load_config("settings.json")
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, Optional, Union
from pathlib import Path

# =============================================================================
# FILE I/O UTILITIES
# =============================================================================

try:
    from .fileio import read_iq_file, write_iq_file
    FILEIO_AVAILABLE = True
except ImportError:
    FILEIO_AVAILABLE = False
    
    def read_iq_file(filepath):
        """Fallback IQ file reader."""
        import numpy as np
        return np.fromfile(filepath, dtype=np.complex64)
    
    def write_iq_file(filepath, data):
        """Fallback IQ file writer."""
        with open(filepath, "wb") as f:
            f.write(data.astype(np.complex64).tobytes())

# =============================================================================
# CRYPTOGRAPHIC UTILITIES
# =============================================================================

try:
    from .crypto import sha256_digest, hmac_sha256
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    
    def sha256_digest(data):
        """Fallback SHA256 digest."""
        import hashlib
        if isinstance(data, str):
            data = data.encode()
        return hashlib.sha256(data).hexdigest()
    
    def hmac_sha256(key, data):
        """Fallback HMAC-SHA256."""
        import hashlib
        import hmac
        if isinstance(key, str):
            key = key.encode()
        if isinstance(data, str):
            data = data.encode()
        return hmac.new(key, data, hashlib.sha256).hexdigest()

# =============================================================================
# LOGGING UTILITIES
# =============================================================================

try:
    from .logger import configure_logging, get_logger
    LOGGING_AVAILABLE = True
except ImportError:
    LOGGING_AVAILABLE = False
    
    def configure_logging(level=logging.INFO, logfile=None, verbose=False):
        """Fallback logging configuration."""
        if verbose:
            level = logging.DEBUG
        
        logging.basicConfig(
            level=level,
            format='[%(asctime)s] %(levelname)s %(name)s: %(message)s',
            handlers=[
                logging.StreamHandler()
            ] + ([logging.FileHandler(logfile)] if logfile else [])
        )
    
    def get_logger(name):
        """Fallback logger getter."""
        return logging.getLogger(name)

# =============================================================================
# CONFIGURATION MANAGEMENT
# =============================================================================

try:
    from .config import load_config, save_config, get_config_value, set_config_value
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    
    # Simple fallback configuration management
    _config_store = {}
    
    def load_config(filepath=None):
        """Fallback config loader."""
        global _config_store
        if filepath and Path(filepath).exists():
            import json
            with open(filepath, 'r') as f:
                _config_store.update(json.load(f))
        return _config_store.copy()
    
    def save_config(config_dict, filepath):
        """Fallback config saver."""
        import json
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def get_config_value(key, default=None):
        """Fallback config getter."""
        return _config_store.get(key, default)
    
    def set_config_value(key, value):
        """Fallback config setter."""
        _config_store[key] = value

# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================

try:
    from .compat import (
        deprecated, 
        check_enhanced_availability,
        get_upgrade_suggestions,
        migrate_old_config
    )
    COMPAT_AVAILABLE = True
except ImportError:
    COMPAT_AVAILABLE = False
    
    def deprecated(func_name, replacement=None):
        """Simple deprecation decorator."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                warnings.warn(
                    f"{func_name} is deprecated" + 
                    (f", use {replacement} instead" if replacement else ""),
                    DeprecationWarning,
                    stacklevel=2
                )
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def check_enhanced_availability():
        """Check if enhanced modules are available."""
        try:
            from ..core import capture, demodulation, fhss
            return True
        except ImportError:
            return False
    
    def get_upgrade_suggestions(operation):
        """Get suggestions for enhanced alternatives."""
        suggestions = {
            'capture': ['Use enhanced live capture for better performance'],
            'demodulation': ['Use enhanced demodulation for more schemes'],
            'fhss': ['Use enhanced FHSS for FCC compliance'],
            'classification': ['Use enhanced classifier for better accuracy']
        }
        return suggestions.get(operation, [])
    
    def migrate_old_config(old_config):
        """Migrate old configuration format."""
        return old_config  # Simple passthrough

# =============================================================================
# PACKAGE INFORMATION AND INTEGRATION STATUS
# =============================================================================

def get_utils_info() -> Dict[str, Any]:
    """
    Get information about utils package and component availability.
    
    Returns:
        Dictionary with component availability and versions
    """
    return {
        'fileio_available': FILEIO_AVAILABLE,
        'crypto_available': CRYPTO_AVAILABLE,
        'logging_available': LOGGING_AVAILABLE,
        'config_available': CONFIG_AVAILABLE,
        'compat_available': COMPAT_AVAILABLE,
        'enhanced_core_available': check_enhanced_availability(),
        'package_version': '1.0.0',  # Should be dynamic in production
    }


def setup_utils(
    log_level: str = 'INFO',
    config_file: Optional[str] = None,
    enable_enhanced: bool = True
) -> Dict[str, Any]:
    """
    One-stop setup function for common utils configuration.
    
    Args:
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        config_file: Optional configuration file to load
        enable_enhanced: Whether to try using enhanced features
        
    Returns:
        Setup status and configuration
    """
    setup_status = {}
    
    # Setup logging
    try:
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR
        }
        configure_logging(level=level_map.get(log_level.upper(), logging.INFO))
        setup_status['logging'] = 'configured'
    except Exception as e:
        setup_status['logging'] = f'failed: {e}'
    
    # Load configuration
    if config_file:
        try:
            config = load_config(config_file)
            setup_status['config'] = f'loaded from {config_file}'
            setup_status['config_values'] = len(config)
        except Exception as e:
            setup_status['config'] = f'failed: {e}'
    else:
        setup_status['config'] = 'not specified'
    
    # Check enhanced features
    if enable_enhanced:
        enhanced_available = check_enhanced_availability()
        setup_status['enhanced_features'] = 'available' if enhanced_available else 'not available'
        
        if not enhanced_available:
            setup_status['suggestions'] = [
                'Install enhanced modules for better performance',
                'Check core module imports',
                'Verify SDR dependencies'
            ]
    
    return setup_status


# =============================================================================
# CONVENIENCE ALIASES AND SHORTCUTS
# =============================================================================

# File operations
read_iq = read_iq_file
write_iq = write_iq_file

# Crypto shortcuts
hash_data = sha256_digest
sign_data = hmac_sha256

# Logging shortcuts
setup_logging = configure_logging
logger = get_logger

# Config shortcuts
load_settings = load_config
save_settings = save_config
get_setting = get_config_value
set_setting = set_config_value

# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # File I/O
    'read_iq_file',
    'write_iq_file',
    'read_iq',
    'write_iq',
    
    # Cryptography
    'sha256_digest',
    'hmac_sha256',
    'hash_data',
    'sign_data',
    
    # Logging
    'configure_logging',
    'get_logger',
    'setup_logging',
    'logger',
    
    # Configuration
    'load_config',
    'save_config',
    'get_config_value',
    'set_config_value',
    'load_settings',
    'save_settings',
    'get_setting',
    'set_setting',
    
    # Compatibility
    'deprecated',
    'check_enhanced_availability',
    'get_upgrade_suggestions',
    'migrate_old_config',
    
    # Package utilities
    'get_utils_info',
    'setup_utils',
]

# =============================================================================
# INITIALIZATION
# =============================================================================

# Configure module logger
_logger = get_logger(__name__)

# Log component availability
if not all([FILEIO_AVAILABLE, CRYPTO_AVAILABLE, LOGGING_AVAILABLE]):
    _logger.debug("Some utils components using fallback implementations")

# Check for enhanced features
if check_enhanced_availability():
    _logger.debug("Enhanced core modules available")
else:
    _logger.debug("Enhanced core modules not available, using basic implementations")

# Version compatibility check
try:
    import sys
    if sys.version_info < (3, 8):
        _logger.warning("Python version < 3.8 detected, some features may not work correctly")
except Exception:
    pass