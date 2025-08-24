#!/usr/bin/env python3
"""
Base Plugin Architecture

This module defines the base classes and interfaces for the DroneCmd plugin system.
Provides a standardized way to create protocol-specific plugins with proper
detection, decoding, and command injection capabilities.

Key Features:
- Abstract base classes for different plugin types
- Standardized plugin interface
- Built-in validation and error handling
- Plugin metadata and capabilities
- Integration with enhanced core systems

Plugin Types:
- ProtocolPlugin: Base for protocol detection/decoding
- InjectionPlugin: Base for command injection
- AnalysisPlugin: Base for signal analysis
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

# Configure module logger
logger = logging.getLogger(__name__)

# Type aliases
IQSamples = npt.NDArray[np.complex64]
PacketBytes = bytes
ConfidenceScore = float
ProtocolName = str


class PluginType(Enum):
    """Types of plugins supported by the system."""
    
    PROTOCOL = "protocol"        # Protocol detection and decoding
    INJECTION = "injection"      # Command injection
    ANALYSIS = "analysis"        # Signal analysis
    DECODER = "decoder"          # Packet decoding
    ENCODER = "encoder"          # Packet encoding


class PluginCapability(Enum):
    """Capabilities that plugins can advertise."""
    
    DETECT = "detect"                    # Can detect protocol in signals
    DECODE = "decode"                    # Can decode packets
    ENCODE = "encode"                    # Can encode commands
    INJECT = "inject"                    # Can inject commands
    ANALYZE = "analyze"                  # Can analyze signals
    CLASSIFY = "classify"                # Can classify protocols
    VALIDATE = "validate"                # Can validate packets
    REAL_TIME = "real_time"             # Supports real-time processing
    BATCH = "batch"                     # Supports batch processing
    STREAMING = "streaming"              # Supports streaming data


@dataclass
class PluginMetadata:
    """
    Metadata describing a plugin's capabilities and requirements.
    
    This standardized metadata allows the plugin system to properly
    load, configure, and utilize plugins.
    """
    
    # Basic identification
    name: str
    version: str
    description: str
    author: str
    
    # Plugin characteristics
    plugin_type: PluginType
    capabilities: List[PluginCapability] = field(default_factory=list)
    supported_protocols: List[str] = field(default_factory=list)
    
    # Technical requirements
    min_packet_length: int = 8
    max_packet_length: int = 2048
    requires_iq_data: bool = False
    requires_demodulation: bool = False
    
    # Performance characteristics
    typical_confidence_threshold: float = 0.7
    processing_complexity: str = "medium"  # low, medium, high
    memory_usage: str = "low"              # low, medium, high
    
    # Dependencies and compatibility
    required_modules: List[str] = field(default_factory=list)
    python_version_min: str = "3.8"
    compatible_platforms: List[str] = field(default_factory=lambda: ["all"])
    
    # Additional metadata
    tags: List[str] = field(default_factory=list)
    documentation_url: Optional[str] = None
    license: str = "Educational/Research Use Only"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for serialization."""
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'author': self.author,
            'plugin_type': self.plugin_type.value,
            'capabilities': [c.value for c in self.capabilities],
            'supported_protocols': self.supported_protocols,
            'min_packet_length': self.min_packet_length,
            'max_packet_length': self.max_packet_length,
            'requires_iq_data': self.requires_iq_data,
            'requires_demodulation': self.requires_demodulation,
            'typical_confidence_threshold': self.typical_confidence_threshold,
            'processing_complexity': self.processing_complexity,
            'memory_usage': self.memory_usage,
            'required_modules': self.required_modules,
            'python_version_min': self.python_version_min,
            'compatible_platforms': self.compatible_platforms,
            'tags': self.tags,
            'documentation_url': self.documentation_url,
            'license': self.license
        }


@dataclass
class DetectionResult:
    """Result from protocol detection operations."""
    
    detected: bool = False
    confidence: ConfidenceScore = 0.0
    protocol_name: str = "unknown"
    packet_start: Optional[int] = None
    packet_length: Optional[int] = None
    
    # Additional detection info
    signal_quality: Optional[Dict[str, float]] = None
    detection_method: Optional[str] = None
    processing_time_ms: float = 0.0
    
    # Error handling
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class DecodingResult:
    """Result from packet decoding operations."""
    
    success: bool = False
    decoded_data: Optional[Dict[str, Any]] = None
    confidence: ConfidenceScore = 0.0
    
    # Protocol-specific fields
    message_type: Optional[str] = None
    sequence_number: Optional[int] = None
    payload: Optional[bytes] = None
    checksum_valid: bool = False
    
    # Quality metrics
    bit_errors: int = 0
    signal_quality: Optional[Dict[str, float]] = None
    processing_time_ms: float = 0.0
    
    # Error handling
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class EncodingResult:
    """Result from command encoding operations."""
    
    success: bool = False
    encoded_packet: Optional[PacketBytes] = None
    packet_length: int = 0
    
    # Encoding metadata
    encoding_method: Optional[str] = None
    checksum_added: bool = False
    encryption_applied: bool = False
    
    # Quality assurance
    validation_passed: bool = False
    estimated_transmission_time: Optional[float] = None
    
    # Error handling
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


class PluginError(Exception):
    """Base exception for plugin-related errors."""
    
    def __init__(self, message: str, plugin_name: str = "unknown"):
        self.plugin_name = plugin_name
        super().__init__(f"Plugin '{plugin_name}': {message}")


class PluginValidationError(PluginError):
    """Exception raised when plugin validation fails."""
    pass


class PluginNotFoundError(PluginError):
    """Exception raised when a requested plugin is not found."""
    pass


class BasePlugin(ABC):
    """
    Abstract base class for all DroneCmd plugins.
    
    This class defines the common interface and functionality that all
    plugins must implement, regardless of their specific type.
    """
    
    def __init__(self) -> None:
        """Initialize base plugin."""
        self._metadata: Optional[PluginMetadata] = None
        self._is_initialized = False
        self._configuration = {}
        self._statistics = {
            'operations_performed': 0,
            'successful_operations': 0,
            'total_processing_time': 0.0,
            'average_confidence': 0.0
        }
    
    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        pass
    
    @abstractmethod
    def initialize(self, **kwargs: Any) -> bool:
        """
        Initialize plugin with configuration.
        
        Args:
            **kwargs: Plugin-specific configuration parameters
            
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def validate(self) -> bool:
        """
        Validate plugin configuration and dependencies.
        
        Returns:
            True if validation passes, False otherwise
        """
        pass
    
    def configure(self, **kwargs: Any) -> None:
        """
        Configure plugin parameters.
        
        Args:
            **kwargs: Configuration parameters
        """
        self._configuration.update(kwargs)
        logger.debug(f"Configured plugin {self.metadata.name}: {kwargs}")
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get current plugin configuration."""
        return self._configuration.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get plugin performance statistics."""
        stats = self._statistics.copy()
        stats.update({
            'plugin_name': self.metadata.name,
            'plugin_version': self.metadata.version,
            'is_initialized': self._is_initialized,
            'configuration_keys': list(self._configuration.keys())
        })
        return stats
    
    def reset_statistics(self) -> None:
        """Reset plugin statistics."""
        self._statistics = {
            'operations_performed': 0,
            'successful_operations': 0,
            'total_processing_time': 0.0,
            'average_confidence': 0.0
        }
    
    def _update_statistics(self, success: bool, confidence: float, processing_time: float) -> None:
        """Update plugin statistics."""
        self._statistics['operations_performed'] += 1
        if success:
            self._statistics['successful_operations'] += 1
        
        self._statistics['total_processing_time'] += processing_time
        
        # Update average confidence (running average)
        ops = self._statistics['operations_performed']
        current_avg = self._statistics['average_confidence']
        self._statistics['average_confidence'] = ((current_avg * (ops - 1)) + confidence) / ops
    
    def cleanup(self) -> None:
        """Cleanup plugin resources."""
        self._is_initialized = False
        logger.debug(f"Cleaned up plugin {self.metadata.name}")


class ProtocolPlugin(BasePlugin):
    """
    Base class for protocol detection and decoding plugins.
    
    Protocol plugins are responsible for:
    - Detecting their specific protocol in signal data
    - Decoding packets according to protocol specifications
    - Validating packet integrity
    - Extracting meaningful information from packets
    """
    
    @abstractmethod
    def detect(
        self,
        data: Union[IQSamples, PacketBytes],
        **kwargs: Any
    ) -> DetectionResult:
        """
        Detect protocol in signal or packet data.
        
        Args:
            data: IQ samples or packet bytes to analyze
            **kwargs: Additional detection parameters
            
        Returns:
            Detection result with confidence and metadata
        """
        pass
    
    @abstractmethod
    def decode_packet(
        self,
        packet_data: PacketBytes,
        **kwargs: Any
    ) -> DecodingResult:
        """
        Decode packet according to protocol specifications.
        
        Args:
            packet_data: Raw packet bytes
            **kwargs: Additional decoding parameters
            
        Returns:
            Decoding result with extracted information
        """
        pass
    
    def validate_packet(self, packet_data: PacketBytes) -> bool:
        """
        Validate packet integrity and format.
        
        Args:
            packet_data: Raw packet bytes
            
        Returns:
            True if packet is valid, False otherwise
        """
        # Basic validation - subclasses should override
        if len(packet_data) < self.metadata.min_packet_length:
            return False
        if len(packet_data) > self.metadata.max_packet_length:
            return False
        return True
    
    def extract_features(self, packet_data: PacketBytes) -> Dict[str, Any]:
        """
        Extract protocol-specific features for classification.
        
        Args:
            packet_data: Raw packet bytes
            
        Returns:
            Dictionary of extracted features
        """
        # Basic features - subclasses should override
        return {
            'packet_length': len(packet_data),
            'first_byte': packet_data[0] if packet_data else 0,
            'last_byte': packet_data[-1] if packet_data else 0
        }


class InjectionPlugin(BasePlugin):
    """
    Base class for command injection plugins.
    
    Injection plugins are responsible for:
    - Encoding commands into protocol-specific packets
    - Validating command parameters
    - Generating proper packet structure
    - Applying necessary encryption/authentication
    """
    
    @abstractmethod
    def encode_command(
        self,
        command: Dict[str, Any],
        **kwargs: Any
    ) -> EncodingResult:
        """
        Encode command into protocol-specific packet.
        
        Args:
            command: Command parameters to encode
            **kwargs: Additional encoding parameters
            
        Returns:
            Encoding result with packet bytes
        """
        pass
    
    @abstractmethod
    def get_supported_commands(self) -> List[str]:
        """
        Get list of supported command types.
        
        Returns:
            List of supported command names
        """
        pass
    
    def validate_command(self, command: Dict[str, Any]) -> bool:
        """
        Validate command parameters.
        
        Args:
            command: Command parameters to validate
            
        Returns:
            True if command is valid, False otherwise
        """
        # Basic validation - subclasses should override
        return isinstance(command, dict) and len(command) > 0
    
    def get_command_schema(self, command_type: str) -> Dict[str, Any]:
        """
        Get schema for specific command type.
        
        Args:
            command_type: Type of command
            
        Returns:
            JSON schema describing command parameters
        """
        # Default schema - subclasses should override
        return {
            "type": "object",
            "properties": {
                "command_type": {"type": "string"},
                "parameters": {"type": "object"}
            },
            "required": ["command_type"]
        }


class AnalysisPlugin(BasePlugin):
    """
    Base class for signal analysis plugins.
    
    Analysis plugins are responsible for:
    - Analyzing signal characteristics
    - Extracting signal features
    - Performing specialized signal processing
    - Providing signal quality metrics
    """
    
    @abstractmethod
    def analyze_signal(
        self,
        iq_data: IQSamples,
        sample_rate: float,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Analyze signal characteristics.
        
        Args:
            iq_data: IQ signal samples
            sample_rate: Sample rate in Hz
            **kwargs: Additional analysis parameters
            
        Returns:
            Dictionary of analysis results
        """
        pass
    
    def extract_signal_features(
        self,
        iq_data: IQSamples,
        sample_rate: float
    ) -> Dict[str, float]:
        """
        Extract numerical features from signal.
        
        Args:
            iq_data: IQ signal samples
            sample_rate: Sample rate in Hz
            
        Returns:
            Dictionary of numerical features
        """
        # Basic features - subclasses should override
        if len(iq_data) == 0:
            return {}
        
        power = np.mean(np.abs(iq_data) ** 2)
        return {
            'signal_power_dbfs': 10 * np.log10(power + 1e-12),
            'peak_power_dbfs': 10 * np.log10(np.max(np.abs(iq_data) ** 2) + 1e-12),
            'sample_count': len(iq_data)
        }


class CompositePlugin(BasePlugin):
    """
    Base class for plugins that combine multiple capabilities.
    
    Composite plugins can implement multiple plugin interfaces,
    allowing them to provide comprehensive protocol support.
    """
    
    def __init__(self) -> None:
        """Initialize composite plugin."""
        super().__init__()
        self._sub_plugins = {}
        self._primary_capability = None
    
    def register_sub_plugin(
        self,
        capability: PluginCapability,
        plugin: BasePlugin
    ) -> None:
        """
        Register a sub-plugin for specific capability.
        
        Args:
            capability: Capability provided by sub-plugin
            plugin: Plugin instance
        """
        self._sub_plugins[capability] = plugin
        if self._primary_capability is None:
            self._primary_capability = capability
        
        logger.debug(f"Registered sub-plugin for {capability.value}")
    
    def get_sub_plugin(self, capability: PluginCapability) -> Optional[BasePlugin]:
        """Get sub-plugin for specific capability."""
        return self._sub_plugins.get(capability)
    
    def has_capability(self, capability: PluginCapability) -> bool:
        """Check if plugin has specific capability."""
        return capability in self._sub_plugins or capability in self.metadata.capabilities


# Utility functions for plugin validation

def validate_plugin_metadata(metadata: PluginMetadata) -> List[str]:
    """
    Validate plugin metadata for completeness and correctness.
    
    Args:
        metadata: Plugin metadata to validate
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    # Required fields
    if not metadata.name:
        errors.append("Plugin name is required")
    if not metadata.version:
        errors.append("Plugin version is required")
    if not metadata.description:
        errors.append("Plugin description is required")
    
    # Validate numeric constraints
    if metadata.min_packet_length < 1:
        errors.append("Minimum packet length must be at least 1")
    if metadata.max_packet_length < metadata.min_packet_length:
        errors.append("Maximum packet length must be >= minimum packet length")
    if not (0.0 <= metadata.typical_confidence_threshold <= 1.0):
        errors.append("Confidence threshold must be between 0.0 and 1.0")
    
    # Validate enums
    try:
        PluginType(metadata.plugin_type)
    except ValueError:
        errors.append(f"Invalid plugin type: {metadata.plugin_type}")
    
    for cap in metadata.capabilities:
        try:
            PluginCapability(cap)
        except (ValueError, AttributeError):
            errors.append(f"Invalid capability: {cap}")
    
    return errors


def check_plugin_compatibility(
    plugin: BasePlugin,
    required_capabilities: List[PluginCapability]
) -> bool:
    """
    Check if plugin supports required capabilities.
    
    Args:
        plugin: Plugin to check
        required_capabilities: List of required capabilities
        
    Returns:
        True if plugin supports all required capabilities
    """
    plugin_caps = set(plugin.metadata.capabilities)
    required_caps = set(required_capabilities)
    
    return required_caps.issubset(plugin_caps)


# Example plugin for testing/reference
class ExampleProtocolPlugin(ProtocolPlugin):
    """Example protocol plugin for testing and reference."""
    
    @property
    def metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        return PluginMetadata(
            name="example_protocol",
            version="1.0.0",
            description="Example protocol plugin for testing",
            author="DroneCmd Framework",
            plugin_type=PluginType.PROTOCOL,
            capabilities=[
                PluginCapability.DETECT,
                PluginCapability.DECODE,
                PluginCapability.VALIDATE
            ],
            supported_protocols=["example"],
            min_packet_length=8,
            max_packet_length=256
        )
    
    def initialize(self, **kwargs: Any) -> bool:
        """Initialize plugin."""
        self._is_initialized = True
        return True
    
    def validate(self) -> bool:
        """Validate plugin."""
        return True
    
    def detect(self, data: Union[IQSamples, PacketBytes], **kwargs: Any) -> DetectionResult:
        """Detect example protocol."""
        # Simple detection based on magic bytes
        if isinstance(data, bytes) and len(data) >= 4:
            if data[:4] == b'\xDE\xAD\xBE\xEF':
                return DetectionResult(
                    detected=True,
                    confidence=0.95,
                    protocol_name="example",
                    packet_start=0,
                    packet_length=len(data)
                )
        
        return DetectionResult(detected=False, confidence=0.0)
    
    def decode_packet(self, packet_data: PacketBytes, **kwargs: Any) -> DecodingResult:
        """Decode example protocol packet."""
        if not self.validate_packet(packet_data):
            return DecodingResult(
                success=False,
                error_message="Invalid packet format"
            )
        
        # Simple decoding
        decoded_data = {
            'magic': packet_data[:4].hex(),
            'payload_length': len(packet_data) - 8,
            'payload': packet_data[4:-4].hex(),
            'checksum': packet_data[-4:].hex()
        }
        
        return DecodingResult(
            success=True,
            decoded_data=decoded_data,
            confidence=0.9,
            checksum_valid=True
        )


if __name__ == "__main__":
    # Test example plugin
    plugin = ExampleProtocolPlugin()
    
    print("=== Plugin System Demo ===")
    print(f"Plugin: {plugin.metadata.name} v{plugin.metadata.version}")
    print(f"Capabilities: {[c.value for c in plugin.metadata.capabilities]}")
    
    # Test initialization
    success = plugin.initialize()
    print(f"Initialization: {'Success' if success else 'Failed'}")
    
    # Test detection
    test_packet = b'\xDE\xAD\xBE\xEF' + b'Hello World' + b'\x12\x34\x56\x78'
    detection = plugin.detect(test_packet)
    print(f"Detection: {detection.detected} (confidence: {detection.confidence:.2f})")
    
    # Test decoding
    if detection.detected:
        decoding = plugin.decode_packet(test_packet)
        print(f"Decoding: {'Success' if decoding.success else 'Failed'}")
        if decoding.success:
            print(f"Decoded data: {decoding.decoded_data}")
    
    # Show statistics
    stats = plugin.get_statistics()
    print(f"Statistics: {stats}")