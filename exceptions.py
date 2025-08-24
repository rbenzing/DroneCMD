#!/usr/bin/env python3
"""
Centralized Exception Handling

This module defines all custom exceptions used throughout the DroneCmd framework.
Provides a hierarchical exception structure that makes error handling more
precise and enables better debugging and user feedback.

Exception Hierarchy:
- DroneCmdError (base)
  ├── ConfigurationError
  ├── HardwareError
  │   ├── SDRError
  │   ├── CaptureError
  │   └── TransmissionError
  ├── ProcessingError
  │   ├── SignalProcessingError
  │   ├── DemodulationError
  │   ├── ClassificationError
  │   └── ParsingError
  ├── PluginError
  │   ├── PluginLoadError
  │   ├── PluginValidationError
  │   └── PluginNotFoundError
  ├── ProtocolError
  │   ├── ProtocolNotSupportedError
  │   ├── PacketFormatError
  │   └── ChecksumError
  └── ComplianceError
      ├── FCCComplianceError
      └── SafetyError
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union


class DroneCmdError(Exception):
    """
    Base exception for all DroneCmd framework errors.
    
    All custom exceptions in the framework inherit from this base class,
    making it easy to catch any framework-specific error.
    
    Attributes:
        message: Human-readable error message
        error_code: Optional error code for programmatic handling
        context: Additional context information
        suggestions: Optional suggestions for resolving the error
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        suggestions: Optional[List[str]] = None
    ) -> None:
        """
        Initialize base DroneCmd error.
        
        Args:
            message: Human-readable error message
            error_code: Optional error code for programmatic handling
            context: Additional context information
            suggestions: Optional suggestions for resolving the error
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.suggestions = suggestions or []
        
    def __str__(self) -> str:
        """Return formatted error message."""
        result = self.message
        
        if self.error_code:
            result = f"[{self.error_code}] {result}"
        
        if self.suggestions:
            result += f"\n\nSuggestions:\n"
            for suggestion in self.suggestions:
                result += f"  - {suggestion}\n"
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'error_code': self.error_code,
            'context': self.context,
            'suggestions': self.suggestions
        }


# =============================================================================
# CONFIGURATION ERRORS
# =============================================================================

class ConfigurationError(DroneCmdError):
    """Raised when there are configuration-related errors."""
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_value: Optional[Any] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize configuration error.
        
        Args:
            message: Error message
            config_key: Configuration key that caused the error
            config_value: Configuration value that caused the error
            **kwargs: Additional arguments for base class
        """
        context = kwargs.get('context', {})
        if config_key:
            context['config_key'] = config_key
        if config_value is not None:
            context['config_value'] = config_value
        
        kwargs['context'] = context
        super().__init__(message, **kwargs)


class InvalidParameterError(ConfigurationError):
    """Raised when invalid parameters are provided."""
    pass


class MissingParameterError(ConfigurationError):
    """Raised when required parameters are missing."""
    pass


class ParameterRangeError(ConfigurationError):
    """Raised when parameters are outside valid ranges."""
    pass


# =============================================================================
# HARDWARE ERRORS
# =============================================================================

class HardwareError(DroneCmdError):
    """Base class for hardware-related errors."""
    pass


class SDRError(HardwareError):
    """Raised when SDR hardware errors occur."""
    
    def __init__(
        self,
        message: str,
        sdr_type: Optional[str] = None,
        device_index: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize SDR error.
        
        Args:
            message: Error message
            sdr_type: Type of SDR device
            device_index: Device index
            **kwargs: Additional arguments for base class
        """
        context = kwargs.get('context', {})
        if sdr_type:
            context['sdr_type'] = sdr_type
        if device_index is not None:
            context['device_index'] = device_index
        
        kwargs['context'] = context
        
        # Add common suggestions
        suggestions = kwargs.get('suggestions', [])
        suggestions.extend([
            "Check if SDR device is properly connected",
            "Verify device drivers are installed",
            "Try different USB port or cable",
            "Check device permissions"
        ])
        kwargs['suggestions'] = suggestions
        
        super().__init__(message, **kwargs)


class CaptureError(HardwareError):
    """Raised when signal capture errors occur."""
    
    def __init__(
        self,
        message: str,
        frequency_hz: Optional[float] = None,
        sample_rate_hz: Optional[float] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize capture error.
        
        Args:
            message: Error message
            frequency_hz: Capture frequency
            sample_rate_hz: Sample rate
            **kwargs: Additional arguments for base class
        """
        context = kwargs.get('context', {})
        if frequency_hz:
            context['frequency_hz'] = frequency_hz
            context['frequency_mhz'] = frequency_hz / 1e6
        if sample_rate_hz:
            context['sample_rate_hz'] = sample_rate_hz
            context['sample_rate_mhz'] = sample_rate_hz / 1e6
        
        kwargs['context'] = context
        super().__init__(message, **kwargs)


class TransmissionError(HardwareError):
    """Raised when signal transmission errors occur."""
    pass


class DeviceNotFoundError(SDRError):
    """Raised when SDR device is not found."""
    
    def __init__(self, device_type: str = "SDR", device_index: int = 0, **kwargs: Any) -> None:
        message = f"{device_type} device {device_index} not found"
        super().__init__(message, sdr_type=device_type, device_index=device_index, **kwargs)


class DeviceInitializationError(SDRError):
    """Raised when SDR device initialization fails."""
    pass


class SampleRateError(CaptureError):
    """Raised when sample rate is invalid or unsupported."""
    pass


class FrequencyRangeError(CaptureError):
    """Raised when frequency is outside supported range."""
    pass


# =============================================================================
# PROCESSING ERRORS
# =============================================================================

class ProcessingError(DroneCmdError):
    """Base class for signal processing errors."""
    pass


class SignalProcessingError(ProcessingError):
    """Raised when signal processing operations fail."""
    pass


class DemodulationError(ProcessingError):
    """Raised when demodulation operations fail."""
    
    def __init__(
        self,
        message: str,
        modulation_scheme: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize demodulation error.
        
        Args:
            message: Error message
            modulation_scheme: Modulation scheme being processed
            **kwargs: Additional arguments for base class
        """
        context = kwargs.get('context', {})
        if modulation_scheme:
            context['modulation_scheme'] = modulation_scheme
        
        kwargs['context'] = context
        super().__init__(message, **kwargs)


class ClassificationError(ProcessingError):
    """Raised when protocol classification fails."""
    pass


class ParsingError(ProcessingError):
    """Raised when packet parsing fails."""
    
    def __init__(
        self,
        message: str,
        packet_length: Optional[int] = None,
        protocol: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize parsing error.
        
        Args:
            message: Error message
            packet_length: Length of packet being parsed
            protocol: Protocol being parsed
            **kwargs: Additional arguments for base class
        """
        context = kwargs.get('context', {})
        if packet_length is not None:
            context['packet_length'] = packet_length
        if protocol:
            context['protocol'] = protocol
        
        kwargs['context'] = context
        super().__init__(message, **kwargs)


class InsufficientDataError(ProcessingError):
    """Raised when insufficient data is available for processing."""
    pass


class InvalidDataFormatError(ProcessingError):
    """Raised when data format is invalid for the operation."""
    pass


# =============================================================================
# PLUGIN ERRORS
# =============================================================================

class PluginError(DroneCmdError):
    """Base class for plugin-related errors."""
    
    def __init__(
        self,
        message: str,
        plugin_name: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize plugin error.
        
        Args:
            message: Error message
            plugin_name: Name of plugin that caused the error
            **kwargs: Additional arguments for base class
        """
        context = kwargs.get('context', {})
        if plugin_name:
            context['plugin_name'] = plugin_name
        
        kwargs['context'] = context
        super().__init__(message, **kwargs)


class PluginLoadError(PluginError):
    """Raised when plugin loading fails."""
    pass


class PluginValidationError(PluginError):
    """Raised when plugin validation fails."""
    pass


class PluginNotFoundError(PluginError):
    """Raised when a requested plugin is not found."""
    pass


class PluginInitializationError(PluginError):
    """Raised when plugin initialization fails."""
    pass


class PluginCompatibilityError(PluginError):
    """Raised when plugin compatibility check fails."""
    pass


# =============================================================================
# PROTOCOL ERRORS
# =============================================================================

class ProtocolError(DroneCmdError):
    """Base class for protocol-related errors."""
    
    def __init__(
        self,
        message: str,
        protocol: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize protocol error.
        
        Args:
            message: Error message
            protocol: Protocol name
            **kwargs: Additional arguments for base class
        """
        context = kwargs.get('context', {})
        if protocol:
            context['protocol'] = protocol
        
        kwargs['context'] = context
        super().__init__(message, **kwargs)


class ProtocolNotSupportedError(ProtocolError):
    """Raised when a protocol is not supported."""
    pass


class PacketFormatError(ProtocolError):
    """Raised when packet format is invalid."""
    
    def __init__(
        self,
        message: str,
        expected_format: Optional[str] = None,
        actual_format: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize packet format error.
        
        Args:
            message: Error message
            expected_format: Expected packet format
            actual_format: Actual packet format detected
            **kwargs: Additional arguments for base class
        """
        context = kwargs.get('context', {})
        if expected_format:
            context['expected_format'] = expected_format
        if actual_format:
            context['actual_format'] = actual_format
        
        kwargs['context'] = context
        super().__init__(message, **kwargs)


class ChecksumError(ProtocolError):
    """Raised when packet checksum validation fails."""
    
    def __init__(
        self,
        message: str = "Packet checksum validation failed",
        expected_checksum: Optional[Union[int, str]] = None,
        actual_checksum: Optional[Union[int, str]] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize checksum error.
        
        Args:
            message: Error message
            expected_checksum: Expected checksum value
            actual_checksum: Actual checksum value
            **kwargs: Additional arguments for base class
        """
        context = kwargs.get('context', {})
        if expected_checksum is not None:
            context['expected_checksum'] = expected_checksum
        if actual_checksum is not None:
            context['actual_checksum'] = actual_checksum
        
        kwargs['context'] = context
        super().__init__(message, **kwargs)


class ProtocolVersionError(ProtocolError):
    """Raised when protocol version is unsupported."""
    pass


class SequenceError(ProtocolError):
    """Raised when packet sequence is invalid."""
    pass


# =============================================================================
# COMPLIANCE ERRORS
# =============================================================================

class ComplianceError(DroneCmdError):
    """Base class for regulatory compliance errors."""
    pass


class FCCComplianceError(ComplianceError):
    """Raised when FCC compliance violations are detected."""
    
    def __init__(
        self,
        message: str,
        regulation: Optional[str] = None,
        frequency_hz: Optional[float] = None,
        power_dbm: Optional[float] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize FCC compliance error.
        
        Args:
            message: Error message
            regulation: Specific FCC regulation violated
            frequency_hz: Frequency involved in violation
            power_dbm: Power level involved in violation
            **kwargs: Additional arguments for base class
        """
        context = kwargs.get('context', {})
        if regulation:
            context['regulation'] = regulation
        if frequency_hz:
            context['frequency_hz'] = frequency_hz
            context['frequency_mhz'] = frequency_hz / 1e6
        if power_dbm is not None:
            context['power_dbm'] = power_dbm
        
        kwargs['context'] = context
        
        # Add compliance-specific suggestions
        suggestions = kwargs.get('suggestions', [])
        suggestions.extend([
            "Review FCC Part 15 regulations",
            "Ensure operation is within authorized frequency bands",
            "Verify power levels are within legal limits",
            "Consider using laboratory environment for testing"
        ])
        kwargs['suggestions'] = suggestions
        
        super().__init__(message, **kwargs)


class SafetyError(ComplianceError):
    """Raised when safety violations are detected."""
    
    def __init__(
        self,
        message: str,
        safety_type: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize safety error.
        
        Args:
            message: Error message
            safety_type: Type of safety concern
            **kwargs: Additional arguments for base class
        """
        context = kwargs.get('context', {})
        if safety_type:
            context['safety_type'] = safety_type
        
        kwargs['context'] = context
        
        # Add safety-specific suggestions
        suggestions = kwargs.get('suggestions', [])
        suggestions.extend([
            "Review safety protocols",
            "Ensure proper protective equipment is used",
            "Verify operation is in authorized environment",
            "Check exposure limits and safety guidelines"
        ])
        kwargs['suggestions'] = suggestions
        
        super().__init__(message, **kwargs)


class PowerLimitError(FCCComplianceError):
    """Raised when transmission power exceeds legal limits."""
    pass


class FrequencyViolationError(FCCComplianceError):
    """Raised when frequency usage violates regulations."""
    pass


class DwellTimeError(FCCComplianceError):
    """Raised when FHSS dwell time exceeds limits."""
    pass


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_error_context(
    operation: str,
    parameters: Dict[str, Any],
    timestamp: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create standardized error context information.
    
    Args:
        operation: Operation being performed when error occurred
        parameters: Parameters passed to the operation
        timestamp: Optional timestamp of error
        
    Returns:
        Standardized error context dictionary
    """
    context = {
        'operation': operation,
        'parameters': parameters
    }
    
    if timestamp:
        context['timestamp'] = timestamp
    else:
        import datetime
        context['timestamp'] = datetime.datetime.now().isoformat()
    
    return context


def format_error_for_user(error: Exception) -> str:
    """
    Format error for user-friendly display.
    
    Args:
        error: Exception to format
        
    Returns:
        User-friendly error message
    """
    if isinstance(error, DroneCmdError):
        return str(error)
    else:
        # Handle non-DroneCmd errors
        error_type = error.__class__.__name__
        return f"{error_type}: {str(error)}"


def is_recoverable_error(error: Exception) -> bool:
    """
    Determine if an error is potentially recoverable.
    
    Args:
        error: Exception to check
        
    Returns:
        True if error might be recoverable
    """
    # Configuration errors are often recoverable
    if isinstance(error, ConfigurationError):
        return True
    
    # Some hardware errors might be recoverable
    if isinstance(error, (CaptureError, TransmissionError)):
        return True
    
    # Processing errors with insufficient data might be recoverable
    if isinstance(error, InsufficientDataError):
        return True
    
    # Plugin errors might be recoverable with different plugins
    if isinstance(error, (PluginNotFoundError, PluginValidationError)):
        return True
    
    # Compliance errors are usually not recoverable without changes
    if isinstance(error, ComplianceError):
        return False
    
    # Hardware initialization errors are usually not recoverable
    if isinstance(error, (DeviceNotFoundError, DeviceInitializationError)):
        return False
    
    # Default to potentially recoverable
    return True


def suggest_error_resolution(error: Exception) -> List[str]:
    """
    Suggest potential resolutions for an error.
    
    Args:
        error: Exception to analyze
        
    Returns:
        List of suggested resolution steps
    """
    if isinstance(error, DroneCmdError):
        return error.suggestions
    
    # Generic suggestions for non-DroneCmd errors
    suggestions = [
        "Check input parameters and configuration",
        "Verify hardware connections and drivers",
        "Review system logs for additional details",
        "Try the operation again after a short delay"
    ]
    
    # Add specific suggestions based on error type
    if "permission" in str(error).lower():
        suggestions.append("Check file/device permissions")
    
    if "memory" in str(error).lower():
        suggestions.append("Free up system memory and try again")
    
    if "timeout" in str(error).lower():
        suggestions.append("Increase timeout values or check network connectivity")
    
    return suggestions


# =============================================================================
# ERROR REPORTING
# =============================================================================

class ErrorReporter:
    """
    Centralized error reporting system.
    
    Provides consistent error logging and reporting across the framework.
    """
    
    def __init__(self, logger_name: str = "dronecmd.errors") -> None:
        """Initialize error reporter."""
        import logging
        self.logger = logging.getLogger(logger_name)
        self._error_counts = {}
    
    def report_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        severity: str = "ERROR"
    ) -> None:
        """
        Report an error with full context.
        
        Args:
            error: Exception that occurred
            context: Additional context information
            severity: Error severity level
        """
        error_type = error.__class__.__name__
        
        # Track error frequency
        self._error_counts[error_type] = self._error_counts.get(error_type, 0) + 1
        
        # Format error message
        message = f"{error_type}: {str(error)}"
        
        if context:
            message += f" | Context: {context}"
        
        if isinstance(error, DroneCmdError) and error.context:
            message += f" | Error Context: {error.context}"
        
        # Log based on severity
        if severity == "CRITICAL":
            self.logger.critical(message)
        elif severity == "ERROR":
            self.logger.error(message)
        elif severity == "WARNING":
            self.logger.warning(message)
        else:
            self.logger.info(message)
    
    def get_error_statistics(self) -> Dict[str, int]:
        """Get error occurrence statistics."""
        return self._error_counts.copy()


# Global error reporter instance
_global_error_reporter = ErrorReporter()


def report_error(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    severity: str = "ERROR"
) -> None:
    """Report error using global error reporter."""
    _global_error_reporter.report_error(error, context, severity)


def get_error_statistics() -> Dict[str, int]:
    """Get global error statistics."""
    return _global_error_reporter.get_error_statistics()


# Exception handling decorator
def handle_errors(
    error_types: Union[type, Tuple[type, ...]] = Exception,
    default_return: Any = None,
    reraise: bool = False
):
    """
    Decorator for consistent error handling.
    
    Args:
        error_types: Exception types to handle
        default_return: Default return value on error
        reraise: Whether to reraise the exception after handling
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except error_types as e:
                report_error(e, {'function': func.__name__, 'args': args, 'kwargs': kwargs})
                
                if reraise:
                    raise
                
                return default_return
        
        return wrapper
    return decorator


if __name__ == "__main__":
    # Test exception system
    print("=== DroneCmd Exception System Demo ===")
    
    # Test basic exception
    try:
        raise ConfigurationError(
            "Invalid sample rate",
            error_code="CFG001",
            config_key="sample_rate_hz",
            config_value=-1000,
            suggestions=["Use a positive sample rate value", "Check configuration file"]
        )
    except DroneCmdError as e:
        print("Configuration Error:")
        print(str(e))
        print(f"Context: {e.context}")
        print()
    
    # Test SDR error
    try:
        raise DeviceNotFoundError("RTL-SDR", 0)
    except SDRError as e:
        print("SDR Error:")
        print(str(e))
        print()
    
    # Test error reporting
    reporter = ErrorReporter()
    test_error = ParsingError("Failed to parse packet", packet_length=0, protocol="mavlink")
    reporter.report_error(test_error, {"source": "test"})
    
    print("Error Statistics:")
    print(reporter.get_error_statistics())