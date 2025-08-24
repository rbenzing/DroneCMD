#!/usr/bin/env python3
"""
Protocol Plugin System

This module provides the protocol plugin infrastructure for drone command
analysis and generation. It includes base classes, registration mechanisms,
and automatic discovery of protocol implementations.

Key Features:
- Automatic plugin discovery and loading
- Protocol detection and validation
- Packet encoding/decoding interfaces
- Plugin capability registration
- Performance monitoring and caching
- Extensible architecture for new protocols

Supported Protocol Types:
- MAVLink (ArduPilot, PX4)
- DJI proprietary protocols
- Parrot protocols
- Generic OOK/FSK protocols
- Custom protocol implementations

Example Usage:
    >>> from dronecmd.plugins.protocols import get_protocol_plugins, detect_protocol
    >>> plugins = get_protocol_plugins()
    >>> result = detect_protocol(packet_data)
    >>> if result.confidence > 0.8:
    ...     decoded = result.plugin.decode_packet(packet_data)
"""

from __future__ import annotations

import logging
import importlib
import pkgutil
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from pathlib import Path

import numpy as np
import numpy.typing as npt

# Configure module logger
logger = logging.getLogger(__name__)

# Type aliases
PacketData = Union[bytes, npt.NDArray[np.uint8]]
IQSamples = npt.NDArray[np.complex64]
ConfidenceScore = float


class ProtocolType(Enum):
    """Standard protocol types for drone communications."""
    
    MAVLINK = "mavlink"
    DJI = "dji"
    PARROT = "parrot"
    SKYDIO = "skydio"
    AUTEL = "autel"
    YUNEEC = "yuneec"
    GENERIC_OOK = "generic_ook"
    GENERIC_FSK = "generic_fsk"
    CUSTOM = "custom"
    UNKNOWN = "unknown"


class ProtocolCapability(Enum):
    """Capabilities that protocol plugins can support."""
    
    DETECTION = "detection"           # Can detect if packet matches protocol
    DECODING = "decoding"            # Can decode packet contents  
    ENCODING = "encoding"            # Can encode packets
    VALIDATION = "validation"        # Can validate packet integrity
    COMMAND_INJECTION = "injection"  # Can create command packets
    TELEMETRY_PARSING = "telemetry"  # Can parse telemetry data
    REAL_TIME = "real_time"          # Supports real-time processing


@dataclass
class ProtocolDetectionResult:
    """Result from protocol detection operations."""
    
    protocol_type: ProtocolType
    confidence: ConfidenceScore
    plugin: Optional['BaseProtocolPlugin'] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    detection_time_ms: float = 0.0
    
    @property
    def is_confident(self) -> bool:
        """Check if detection confidence is high enough."""
        return self.confidence >= 0.7
    
    @property
    def is_valid(self) -> bool:
        """Check if detection result is valid."""
        return self.plugin is not None and self.confidence > 0.0


@dataclass  
class PacketDecodeResult:
    """Result from packet decoding operations."""
    
    success: bool
    protocol_type: ProtocolType
    message_type: Optional[str] = None
    message_id: Optional[int] = None
    payload: Optional[Dict[str, Any]] = None
    raw_data: Optional[bytes] = None
    validation_errors: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0
    
    @property
    def is_valid(self) -> bool:
        """Check if decode was successful and valid."""
        return self.success and len(self.validation_errors) == 0


class BaseProtocolPlugin(ABC):
    """
    Abstract base class for all protocol plugins.
    
    All protocol plugins must inherit from this class and implement
    the required methods. This ensures a consistent interface across
    all protocol implementations.
    """
    
    def __init__(self) -> None:
        """Initialize base protocol plugin."""
        self._capabilities = set()
        self._performance_stats = {
            'detections': 0,
            'successful_detections': 0,
            'decodings': 0,
            'successful_decodings': 0,
            'total_processing_time': 0.0
        }
    
    @property
    @abstractmethod
    def protocol_type(self) -> ProtocolType:
        """Get the protocol type this plugin handles."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get human-readable name of the protocol."""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Get version of the protocol implementation."""
        pass
    
    @property
    def capabilities(self) -> List[ProtocolCapability]:
        """Get list of capabilities supported by this plugin."""
        return list(self._capabilities)
    
    def has_capability(self, capability: ProtocolCapability) -> bool:
        """Check if plugin supports specific capability."""
        return capability in self._capabilities
    
    @abstractmethod
    def detect(
        self,
        packet_data: PacketData,
        iq_samples: Optional[IQSamples] = None,
        sample_rate: Optional[float] = None
    ) -> ProtocolDetectionResult:
        """
        Detect if packet data matches this protocol.
        
        Args:
            packet_data: Raw packet bytes or bit array
            iq_samples: Optional IQ samples for signal analysis
            sample_rate: Sample rate of IQ data
            
        Returns:
            Detection result with confidence score
        """
        pass
    
    @abstractmethod
    def decode_packet(
        self,
        packet_data: PacketData,
        validate: bool = True
    ) -> PacketDecodeResult:
        """
        Decode packet data according to protocol specification.
        
        Args:
            packet_data: Raw packet bytes
            validate: Whether to perform validation checks
            
        Returns:
            Decoded packet information
        """
        pass
    
    def encode_packet(
        self,
        message_type: str,
        payload: Dict[str, Any],
        **kwargs: Any
    ) -> bytes:
        """
        Encode packet data according to protocol specification.
        
        Args:
            message_type: Type of message to encode
            payload: Message payload data
            **kwargs: Additional encoding parameters
            
        Returns:
            Encoded packet bytes
            
        Raises:
            NotImplementedError: If encoding not supported
        """
        if ProtocolCapability.ENCODING not in self._capabilities:
            raise NotImplementedError(f"{self.name} does not support packet encoding")
        
        raise NotImplementedError("encode_packet must be implemented by subclass")
    
    def validate_packet(
        self,
        packet_data: PacketData
    ) -> Tuple[bool, List[str]]:
        """
        Validate packet integrity and structure.
        
        Args:
            packet_data: Raw packet bytes
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        if ProtocolCapability.VALIDATION not in self._capabilities:
            return True, []  # Default to valid if validation not supported
        
        # Basic validation - subclasses should override
        if len(packet_data) == 0:
            return False, ["Empty packet"]
        
        return True, []
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get plugin performance statistics."""
        stats = self._performance_stats.copy()
        
        # Calculate derived metrics
        if stats['detections'] > 0:
            stats['detection_success_rate'] = stats['successful_detections'] / stats['detections']
            stats['avg_processing_time_ms'] = (stats['total_processing_time'] / stats['detections']) * 1000
        else:
            stats['detection_success_rate'] = 0.0
            stats['avg_processing_time_ms'] = 0.0
        
        if stats['decodings'] > 0:
            stats['decoding_success_rate'] = stats['successful_decodings'] / stats['decodings']
        else:
            stats['decoding_success_rate'] = 0.0
        
        return stats
    
    def _update_stats(self, operation: str, success: bool, processing_time: float) -> None:
        """Update internal performance statistics."""
        if operation == 'detection':
            self._performance_stats['detections'] += 1
            if success:
                self._performance_stats['successful_detections'] += 1
        elif operation == 'decoding':
            self._performance_stats['decodings'] += 1
            if success:
                self._performance_stats['successful_decodings'] += 1
        
        self._performance_stats['total_processing_time'] += processing_time
    
    def __str__(self) -> str:
        """String representation of plugin."""
        return f"{self.name} v{self.version} ({self.protocol_type.value})"
    
    def __repr__(self) -> str:
        """Detailed representation of plugin."""
        caps = [cap.value for cap in self.capabilities]
        return f"{self.__class__.__name__}(name='{self.name}', capabilities={caps})"


class ProtocolRegistry:
    """
    Registry for protocol plugins with automatic discovery and loading.
    
    Manages all available protocol plugins and provides methods for
    plugin discovery, registration, and selection.
    """
    
    def __init__(self) -> None:
        """Initialize protocol registry."""
        self._plugins: Dict[ProtocolType, BaseProtocolPlugin] = {}
        self._plugin_classes: Dict[str, Type[BaseProtocolPlugin]] = {}
        self._loaded = False
        logger.debug("Initialized protocol registry")
    
    def register_plugin(
        self,
        plugin: BaseProtocolPlugin,
        override: bool = False
    ) -> bool:
        """
        Register a protocol plugin.
        
        Args:
            plugin: Plugin instance to register
            override: Whether to override existing plugin
            
        Returns:
            True if registration successful
        """
        protocol_type = plugin.protocol_type
        
        if protocol_type in self._plugins and not override:
            logger.warning(f"Plugin for {protocol_type.value} already registered")
            return False
        
        self._plugins[protocol_type] = plugin
        self._plugin_classes[plugin.__class__.__name__] = plugin.__class__
        
        logger.info(f"Registered plugin: {plugin}")
        return True
    
    def get_plugin(self, protocol_type: ProtocolType) -> Optional[BaseProtocolPlugin]:
        """Get plugin for specific protocol type."""
        if not self._loaded:
            self.discover_plugins()
        
        return self._plugins.get(protocol_type)
    
    def get_all_plugins(self) -> List[BaseProtocolPlugin]:
        """Get all registered plugins."""
        if not self._loaded:
            self.discover_plugins()
        
        return list(self._plugins.values())
    
    def discover_plugins(self, package_path: Optional[str] = None) -> int:
        """
        Discover and load protocol plugins automatically.
        
        Args:
            package_path: Optional package path to search
            
        Returns:
            Number of plugins loaded
        """
        if package_path is None:
            # Use current package path
            package_path = __name__
        
        plugins_loaded = 0
        
        try:
            # Import current package to get path
            package = importlib.import_module(package_path)
            package_dir = Path(package.__file__).parent
            
            # Iterate through all Python files in protocols directory
            for module_info in pkgutil.iter_modules([str(package_dir)]):
                if module_info.name.startswith('_'):
                    continue  # Skip private modules
                
                try:
                    module_name = f"{package_path}.{module_info.name}"
                    module = importlib.import_module(module_name)
                    
                    # Look for plugin classes
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        
                        # Check if it's a plugin class
                        if (isinstance(attr, type) and 
                            issubclass(attr, BaseProtocolPlugin) and 
                            attr != BaseProtocolPlugin):
                            
                            # Instantiate and register plugin
                            try:
                                plugin_instance = attr()
                                if self.register_plugin(plugin_instance):
                                    plugins_loaded += 1
                                    logger.debug(f"Loaded plugin from {module_name}: {attr_name}")
                            except Exception as e:
                                logger.warning(f"Failed to instantiate plugin {attr_name}: {e}")
                
                except Exception as e:
                    logger.warning(f"Failed to load module {module_info.name}: {e}")
        
        except Exception as e:
            logger.error(f"Plugin discovery failed: {e}")
        
        self._loaded = True
        logger.info(f"Plugin discovery complete: {plugins_loaded} plugins loaded")
        return plugins_loaded
    
    def detect_protocol(
        self,
        packet_data: PacketData,
        iq_samples: Optional[IQSamples] = None,
        sample_rate: Optional[float] = None,
        min_confidence: float = 0.5
    ) -> ProtocolDetectionResult:
        """
        Detect protocol using all available plugins.
        
        Args:
            packet_data: Raw packet data
            iq_samples: Optional IQ samples
            sample_rate: Sample rate of IQ data
            min_confidence: Minimum confidence threshold
            
        Returns:
            Best detection result across all plugins
        """
        if not self._loaded:
            self.discover_plugins()
        
        best_result = ProtocolDetectionResult(
            protocol_type=ProtocolType.UNKNOWN,
            confidence=0.0
        )
        
        for plugin in self._plugins.values():
            try:
                result = plugin.detect(packet_data, iq_samples, sample_rate)
                
                if result.confidence > best_result.confidence:
                    best_result = result
                    
            except Exception as e:
                logger.warning(f"Detection failed for {plugin.name}: {e}")
        
        # Only return result if it meets minimum confidence
        if best_result.confidence < min_confidence:
            best_result.protocol_type = ProtocolType.UNKNOWN
            best_result.plugin = None
        
        return best_result
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics and plugin information."""
        stats = {
            'total_plugins': len(self._plugins),
            'plugins_loaded': self._loaded,
            'protocol_types': [pt.value for pt in self._plugins.keys()],
            'plugin_details': {}
        }
        
        for protocol_type, plugin in self._plugins.items():
            stats['plugin_details'][protocol_type.value] = {
                'name': plugin.name,
                'version': plugin.version,
                'capabilities': [cap.value for cap in plugin.capabilities],
                'performance': plugin.get_performance_stats()
            }
        
        return stats


# Global protocol registry instance
_protocol_registry = ProtocolRegistry()


# =============================================================================
# PUBLIC API FUNCTIONS
# =============================================================================

def get_protocol_plugins() -> List[BaseProtocolPlugin]:
    """
    Get all available protocol plugins.
    
    Returns:
        List of registered protocol plugins
    """
    return _protocol_registry.get_all_plugins()


def get_protocol_plugin(protocol_type: ProtocolType) -> Optional[BaseProtocolPlugin]:
    """
    Get specific protocol plugin by type.
    
    Args:
        protocol_type: Protocol type to get plugin for
        
    Returns:
        Plugin instance or None if not found
    """
    return _protocol_registry.get_plugin(protocol_type)


def register_protocol_plugin(
    plugin: BaseProtocolPlugin,
    override: bool = False
) -> bool:
    """
    Register a new protocol plugin.
    
    Args:
        plugin: Plugin instance to register
        override: Whether to override existing plugin
        
    Returns:
        True if registration successful
    """
    return _protocol_registry.register_plugin(plugin, override)


def detect_protocol(
    packet_data: PacketData,
    iq_samples: Optional[IQSamples] = None,
    sample_rate: Optional[float] = None,
    min_confidence: float = 0.5
) -> ProtocolDetectionResult:
    """
    Detect protocol type from packet data.
    
    Args:
        packet_data: Raw packet bytes or bit array
        iq_samples: Optional IQ samples for signal analysis
        sample_rate: Sample rate of IQ data
        min_confidence: Minimum confidence threshold
        
    Returns:
        Detection result with best matching protocol
    """
    return _protocol_registry.detect_protocol(
        packet_data, iq_samples, sample_rate, min_confidence
    )


def discover_protocol_plugins(package_path: Optional[str] = None) -> int:
    """
    Discover and load protocol plugins.
    
    Args:
        package_path: Optional package path to search
        
    Returns:
        Number of plugins loaded
    """
    return _protocol_registry.discover_plugins(package_path)


def get_protocol_registry_stats() -> Dict[str, Any]:
    """
    Get protocol registry statistics.
    
    Returns:
        Dictionary with registry and plugin statistics
    """
    return _protocol_registry.get_registry_stats()


def decode_packet_with_auto_detection(
    packet_data: PacketData,
    iq_samples: Optional[IQSamples] = None,
    sample_rate: Optional[float] = None,
    min_confidence: float = 0.7
) -> Tuple[ProtocolDetectionResult, Optional[PacketDecodeResult]]:
    """
    Auto-detect protocol and decode packet in one operation.
    
    Args:
        packet_data: Raw packet data
        iq_samples: Optional IQ samples
        sample_rate: Sample rate of IQ data  
        min_confidence: Minimum detection confidence
        
    Returns:
        Tuple of (detection_result, decode_result)
    """
    # First detect the protocol
    detection_result = detect_protocol(packet_data, iq_samples, sample_rate, min_confidence)
    
    decode_result = None
    if detection_result.is_confident and detection_result.plugin:
        try:
            # Attempt to decode with detected plugin
            decode_result = detection_result.plugin.decode_packet(packet_data)
        except Exception as e:
            logger.warning(f"Decoding failed after detection: {e}")
    
    return detection_result, decode_result


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_generic_plugin(
    protocol_type: ProtocolType,
    name: str,
    detection_patterns: List[bytes],
    **kwargs: Any
) -> BaseProtocolPlugin:
    """
    Create a generic protocol plugin with basic pattern matching.
    
    Args:
        protocol_type: Protocol type
        name: Plugin name
        detection_patterns: List of byte patterns for detection
        **kwargs: Additional plugin parameters
        
    Returns:
        Generic protocol plugin instance
    """
    # Import and create generic plugin
    from .generic import GenericProtocolPlugin
    
    return GenericProtocolPlugin(
        protocol_type=protocol_type,
        name=name,
        detection_patterns=detection_patterns,
        **kwargs
    )


def validate_plugin_implementation(plugin: BaseProtocolPlugin) -> Tuple[bool, List[str]]:
    """
    Validate that a plugin properly implements the required interface.
    
    Args:
        plugin: Plugin to validate
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check required properties
    try:
        _ = plugin.protocol_type
        _ = plugin.name
        _ = plugin.version
        _ = plugin.capabilities
    except Exception as e:
        errors.append(f"Missing required property: {e}")
    
    # Check required methods
    required_methods = ['detect', 'decode_packet']
    for method_name in required_methods:
        if not hasattr(plugin, method_name):
            errors.append(f"Missing required method: {method_name}")
        elif not callable(getattr(plugin, method_name)):
            errors.append(f"Method {method_name} is not callable")
    
    # Test basic functionality
    try:
        test_data = b'\x00\x01\x02\x03'
        result = plugin.detect(test_data)
        if not isinstance(result, ProtocolDetectionResult):
            errors.append("detect() does not return ProtocolDetectionResult")
    except Exception as e:
        errors.append(f"detect() method failed: {e}")
    
    return len(errors) == 0, errors


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums
    'ProtocolType',
    'ProtocolCapability',
    
    # Data classes
    'ProtocolDetectionResult',
    'PacketDecodeResult',
    
    # Base classes
    'BaseProtocolPlugin',
    
    # Registry
    'ProtocolRegistry',
    
    # Main API functions
    'get_protocol_plugins',
    'get_protocol_plugin',
    'register_protocol_plugin',
    'detect_protocol',
    'discover_protocol_plugins',
    'decode_packet_with_auto_detection',
    
    # Utility functions
    'create_generic_plugin',
    'validate_plugin_implementation',
    'get_protocol_registry_stats',
]

# =============================================================================
# INITIALIZATION
# =============================================================================

# Automatically discover plugins on import
logger.debug("Initializing protocol plugin system")

try:
    plugins_found = discover_protocol_plugins()
    logger.info(f"Protocol plugin system initialized with {plugins_found} plugins")
except Exception as e:
    logger.warning(f"Plugin auto-discovery failed: {e}")