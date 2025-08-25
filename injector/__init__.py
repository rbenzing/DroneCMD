#!/usr/bin/env python3
"""
DroneCmd Injection Module

This module provides signal injection and transmission capabilities for
authorized testing and research purposes. All functionality is designed
for laboratory use only and includes compliance monitoring to ensure
responsible usage.

IMPORTANT LEGAL NOTICE:
This module is intended for authorized testing, research, and educational
purposes only. Users are responsible for compliance with all applicable
laws and regulations including FCC Part 15, local spectrum regulations,
and drone operation guidelines. Unauthorized transmission is illegal.
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Union, Callable
from pathlib import Path

# Configure module logger
logger = logging.getLogger(__name__)

# Issue legal compliance warning
warnings.warn(
    "DroneCmd Injection Module: This module is for authorized testing and "
    "research only. Users are responsible for compliance with all applicable "
    "laws and regulations. Unauthorized transmission is illegal.",
    UserWarning,
    stacklevel=2
)

# =============================================================================
# SIGNAL INJECTOR
# =============================================================================

try:
    from .injector import (
        SignalInjector,
        InjectionConfig,
        InjectionResult,
        InjectionMode,
        ComplianceMonitor,
        InjectorError
    )
    INJECTOR_AVAILABLE = True
    logger.debug("Signal injector loaded successfully")
    
except ImportError as e:
    logger.warning(f"Signal injector not available: {e}")
    INJECTOR_AVAILABLE = False
    SignalInjector = None
    InjectionConfig = None
    InjectionResult = None
    InjectionMode = None
    ComplianceMonitor = None
    InjectorError = None

# =============================================================================
# OBFUSCATION ENGINE
# =============================================================================

try:
    from .obfuscation import (
        ObfuscationEngine,
        ObfuscationMethod,
        TimingObfuscator,
        PacketObfuscator,
        FrequencyObfuscator
    )
    OBFUSCATION_AVAILABLE = True
    logger.debug("Obfuscation engine loaded successfully")
    
except ImportError as e:
    logger.warning(f"Obfuscation engine not available: {e}")
    OBFUSCATION_AVAILABLE = False
    ObfuscationEngine = None
    ObfuscationMethod = None
    TimingObfuscator = None
    PacketObfuscator = None
    FrequencyObfuscator = None

# =============================================================================
# COMPLIANCE AND SAFETY SYSTEMS
# =============================================================================

class ComplianceChecker:
    """
    Compliance checking system for injection operations.
    
    Provides comprehensive compliance validation for laboratory testing
    including frequency limits, power restrictions, and safety protocols.
    """
    
    def __init__(self, strict_mode: bool = True) -> None:
        """
        Initialize compliance checker.
        
        Args:
            strict_mode: Enable strict compliance checking
        """
        self.strict_mode = strict_mode
        self.violations = []
        self.session_start_time = None
        
        # Compliance limits (conservative defaults)
        self.frequency_limits = {
            'min_hz': 433.05e6,     # ISM band start
            'max_hz': 434.79e6,     # ISM band end (conservative)
            'ism_2_4_min': 2400e6,  # 2.4 GHz ISM start
            'ism_2_4_max': 2483.5e6 # 2.4 GHz ISM end
        }
        
        self.power_limits = {
            'max_power_dbm': 10.0,   # Conservative limit
            'max_duty_cycle': 0.1,   # 10% duty cycle max
            'max_session_time_s': 300 # 5 minute session limit
        }
        
        logger.info(f"Compliance checker initialized (strict={strict_mode})")
    
    def check_frequency_compliance(self, frequency_hz: float) -> bool:
        """
        Check if frequency is within allowed ranges.
        
        Args:
            frequency_hz: Frequency to check
            
        Returns:
            True if compliant, False otherwise
        """
        # Check ISM bands (conservative)
        ism_433 = (self.frequency_limits['min_hz'] <= frequency_hz <= 
                  self.frequency_limits['max_hz'])
        
        ism_2_4 = (self.frequency_limits['ism_2_4_min'] <= frequency_hz <= 
                  self.frequency_limits['ism_2_4_max'])
        
        if not (ism_433 or ism_2_4):
            violation = f"Frequency {frequency_hz/1e6:.1f} MHz outside allowed ISM bands"
            self.violations.append(violation)
            logger.error(f"COMPLIANCE VIOLATION: {violation}")
            return False
        
        return True
    
    def check_power_compliance(self, power_dbm: float) -> bool:
        """
        Check if power level is within limits.
        
        Args:
            power_dbm: Power level in dBm
            
        Returns:
            True if compliant, False otherwise
        """
        if power_dbm > self.power_limits['max_power_dbm']:
            violation = f"Power {power_dbm} dBm exceeds limit {self.power_limits['max_power_dbm']} dBm"
            self.violations.append(violation)
            logger.error(f"COMPLIANCE VIOLATION: {violation}")
            return False
        
        return True
    
    def check_session_compliance(self) -> bool:
        """Check session duration compliance."""
        import time
        
        if self.session_start_time is None:
            self.session_start_time = time.time()
            return True
        
        session_duration = time.time() - self.session_start_time
        
        if session_duration > self.power_limits['max_session_time_s']:
            violation = f"Session duration {session_duration:.1f}s exceeds limit"
            self.violations.append(violation)
            logger.error(f"COMPLIANCE VIOLATION: {violation}")
            return False
        
        return True
    
    def validate_injection_request(
        self,
        frequency_hz: float,
        power_dbm: float,
        duration_s: float
    ) -> Dict[str, Any]:
        """
        Validate injection request for full compliance.
        
        Args:
            frequency_hz: Requested frequency
            power_dbm: Requested power level
            duration_s: Injection duration
            
        Returns:
            Validation result dictionary
        """
        result = {
            'compliant': True,
            'violations': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Check individual parameters
        if not self.check_frequency_compliance(frequency_hz):
            result['compliant'] = False
            result['violations'].append("Frequency compliance violation")
        
        if not self.check_power_compliance(power_dbm):
            result['compliant'] = False
            result['violations'].append("Power compliance violation")
        
        if not self.check_session_compliance():
            result['compliant'] = False
            result['violations'].append("Session compliance violation")
        
        # Duration checks
        if duration_s > 30.0:  # Conservative limit
            result['warnings'].append("Long injection duration - ensure compliance")
        
        # Recommendations
        if frequency_hz < 1e9:  # Below 1 GHz
            result['recommendations'].append("Consider using 2.4 GHz ISM band")
        
        if power_dbm > 0:  # Above 0 dBm
            result['recommendations'].append("Consider reducing power for testing")
        
        return result


class SafetyMonitor:
    """
    Safety monitoring system for injection operations.
    
    Provides real-time safety monitoring including emergency shutdown,
    power monitoring, and session limits.
    """
    
    def __init__(self) -> None:
        """Initialize safety monitor."""
        self.emergency_stop = False
        self.active_injections = 0
        self.total_transmission_time = 0.0
        self.max_concurrent_injections = 1
        self.emergency_callbacks = []
        
        logger.info("Safety monitor initialized")
    
    def register_emergency_callback(self, callback: Callable[[], None]) -> None:
        """Register emergency shutdown callback."""
        self.emergency_callbacks.append(callback)
    
    def trigger_emergency_stop(self, reason: str) -> None:
        """Trigger emergency stop of all injections."""
        self.emergency_stop = True
        logger.critical(f"EMERGENCY STOP TRIGGERED: {reason}")
        
        # Call all emergency callbacks
        for callback in self.emergency_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Emergency callback failed: {e}")
    
    def check_safety_limits(self) -> bool:
        """Check current safety status."""
        if self.emergency_stop:
            return False
        
        if self.active_injections >= self.max_concurrent_injections:
            return False
        
        # Check total transmission time
        if self.total_transmission_time > 600:  # 10 minutes total
            self.trigger_emergency_stop("Total transmission time limit exceeded")
            return False
        
        return True
    
    def start_injection(self) -> bool:
        """Record start of injection."""
        if not self.check_safety_limits():
            return False
        
        self.active_injections += 1
        return True
    
    def end_injection(self, duration_s: float) -> None:
        """Record end of injection."""
        self.active_injections = max(0, self.active_injections - 1)
        self.total_transmission_time += duration_s


# =============================================================================
# SAFE INJECTION WRAPPER
# =============================================================================

class SafeInjector:
    """
    Safe injection wrapper with comprehensive compliance and safety checking.
    
    This class wraps the signal injector with mandatory compliance checking,
    safety monitoring, and emergency stop capabilities.
    """
    
    def __init__(
        self,
        injector_config: Optional[Dict[str, Any]] = None,
        enable_compliance: bool = True,
        enable_safety_monitor: bool = True
    ) -> None:
        """
        Initialize safe injector.
        
        Args:
            injector_config: Configuration for underlying injector
            enable_compliance: Enable compliance checking
            enable_safety_monitor: Enable safety monitoring
        """
        self.compliance_checker = ComplianceChecker() if enable_compliance else None
        self.safety_monitor = SafetyMonitor() if enable_safety_monitor else None
        
        # Initialize underlying injector if available
        if INJECTOR_AVAILABLE and injector_config:
            try:
                config = InjectionConfig(**injector_config)
                self.injector = SignalInjector(config)
            except Exception as e:
                logger.error(f"Failed to initialize injector: {e}")
                self.injector = None
        else:
            self.injector = None
        
        # Register safety callbacks
        if self.safety_monitor and self.injector:
            self.safety_monitor.register_emergency_callback(self._emergency_shutdown)
        
        logger.info("Safe injector initialized")
    
    def _emergency_shutdown(self) -> None:
        """Emergency shutdown callback."""
        if self.injector:
            try:
                self.injector.stop_all_injections()
            except Exception as e:
                logger.error(f"Emergency shutdown failed: {e}")
    
    def validate_injection(
        self,
        frequency_hz: float,
        power_dbm: float = 0.0,
        duration_s: float = 1.0
    ) -> Dict[str, Any]:
        """
        Validate injection parameters before execution.
        
        Args:
            frequency_hz: Injection frequency
            power_dbm: Power level
            duration_s: Injection duration
            
        Returns:
            Validation result
        """
        if not self.compliance_checker:
            return {'compliant': False, 'error': 'Compliance checker not available'}
        
        return self.compliance_checker.validate_injection_request(
            frequency_hz, power_dbm, duration_s
        )
    
    def safe_inject(
        self,
        packet_data: bytes,
        frequency_hz: float,
        power_dbm: float = 0.0,
        duration_s: float = 1.0,
        force_inject: bool = False
    ) -> Dict[str, Any]:
        """
        Perform safe injection with full compliance checking.
        
        Args:
            packet_data: Data to inject
            frequency_hz: Injection frequency
            power_dbm: Power level
            duration_s: Injection duration
            force_inject: Force injection despite warnings (not violations)
            
        Returns:
            Injection result
        """
        result = {
            'success': False,
            'injected': False,
            'compliance_checked': True,
            'safety_checked': True
        }
        
        # Compliance validation
        if self.compliance_checker:
            validation = self.validate_injection(frequency_hz, power_dbm, duration_s)
            result['compliance_result'] = validation
            
            if not validation['compliant']:
                result['error'] = f"Compliance violation: {validation['violations']}"
                logger.error(result['error'])
                return result
            
            if validation['warnings'] and not force_inject:
                result['error'] = f"Compliance warnings: {validation['warnings']}"
                result['requires_force'] = True
                return result
        
        # Safety checks
        if self.safety_monitor:
            if not self.safety_monitor.start_injection():
                result['error'] = "Safety limits exceeded"
                return result
        
        # Perform injection if available
        if self.injector:
            try:
                injection_result = self.injector.inject_packet(
                    packet_data, frequency_hz, power_dbm, duration_s
                )
                result.update({
                    'success': True,
                    'injected': True,
                    'injection_result': injection_result
                })
                
            except Exception as e:
                result['error'] = f"Injection failed: {e}"
                
            finally:
                if self.safety_monitor:
                    self.safety_monitor.end_injection(duration_s)
        else:
            result['error'] = "Injector not available"
        
        return result

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_safe_injector(
    platform: str = "hackrf",
    **config_kwargs: Any
) -> Optional[SafeInjector]:
    """
    Create safe injector with automatic configuration.
    
    Args:
        platform: Transmission platform
        **config_kwargs: Additional configuration
        
    Returns:
        Safe injector instance or None
    """
    try:
        injector_config = {
            'platform': platform,
            'enable_compliance_monitoring': True,
            'enable_safety_features': True,
            **config_kwargs
        }
        
        return SafeInjector(
            injector_config=injector_config,
            enable_compliance=True,
            enable_safety_monitor=True
        )
        
    except Exception as e:
        logger.error(f"Failed to create safe injector: {e}")
        return None


def check_injection_legality(
    frequency_hz: float,
    power_dbm: float,
    location: Optional[str] = None
) -> Dict[str, Any]:
    """
    Check injection legality for given parameters.
    
    Args:
        frequency_hz: Proposed frequency
        power_dbm: Proposed power level
        location: Location (for regional compliance)
        
    Returns:
        Legality assessment
    """
    checker = ComplianceChecker()
    
    # Basic compliance check
    freq_ok = checker.check_frequency_compliance(frequency_hz)
    power_ok = checker.check_power_compliance(power_dbm)
    
    return {
        'frequency_compliant': freq_ok,
        'power_compliant': power_ok,
        'overall_compliant': freq_ok and power_ok,
        'violations': checker.violations,
        'warning': "This is a basic check only. Consult local regulations.",
        'disclaimer': "Users are responsible for full legal compliance."
    }


def get_injection_capabilities() -> Dict[str, bool]:
    """
    Get available injection capabilities.
    
    Returns:
        Dictionary of capability flags
    """
    return {
        'injector': INJECTOR_AVAILABLE,
        'obfuscation': OBFUSCATION_AVAILABLE,
        'compliance_checking': True,  # Always available
        'safety_monitoring': True,    # Always available
        'hackrf_support': _check_hackrf_support(),
        'soapy_sdr_support': _check_soapy_support()
    }


def _check_hackrf_support() -> bool:
    """Check if HackRF support is available."""
    try:
        import pyhackrf
        return True
    except ImportError:
        return False


def _check_soapy_support() -> bool:
    """Check if SoapySDR support is available."""
    try:
        import SoapySDR
        return True
    except ImportError:
        return False

# =============================================================================
# LEGAL AND SAFETY WARNINGS
# =============================================================================

def display_legal_warning() -> None:
    """Display comprehensive legal warning."""
    warning_text = """
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                          LEGAL WARNING                               ║
    ║                                                                      ║
    ║  This injection module is for AUTHORIZED TESTING ONLY in controlled ║
    ║  laboratory environments. Users are LEGALLY RESPONSIBLE for:        ║
    ║                                                                      ║
    ║  • FCC Part 15 compliance and licensing requirements                 ║
    ║  • Local and international spectrum regulations                      ║
    ║  • Drone operation laws and safety protocols                        ║
    ║  • Privacy and security regulations                                  ║
    ║                                                                      ║
    ║  UNAUTHORIZED TRANSMISSION IS ILLEGAL and may result in:             ║
    ║  • Substantial fines and legal penalties                             ║
    ║  • Criminal prosecution                                              ║
    ║  • Equipment confiscation                                            ║
    ║                                                                      ║
    ║  USE ONLY IN AUTHORIZED LABORATORY ENVIRONMENTS                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """
    print(warning_text)
    logger.warning("Legal warning displayed to user")

# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Main classes
    'SignalInjector',
    'SafeInjector',
    'ObfuscationEngine',
    'ComplianceChecker',
    'SafetyMonitor',
    
    # Configuration classes
    'InjectionConfig',
    
    # Result classes
    'InjectionResult',
    
    # Enums
    'InjectionMode',
    'ObfuscationMethod',
    
    # Obfuscation components
    'TimingObfuscator',
    'PacketObfuscator',
    'FrequencyObfuscator',
    
    # Error classes
    'InjectorError',
    
    # Utility functions
    'create_safe_injector',
    'check_injection_legality',
    'get_injection_capabilities',
    'display_legal_warning',
    
    # Monitoring classes
    'ComplianceMonitor'
]

# Filter exports based on availability
available_exports = []
for name in __all__:
    if globals().get(name) is not None:
        available_exports.append(name)

__all__ = available_exports

# Module initialization with legal warning
logger.info(f"DroneCmd injection module initialized with {len(__all__)} exports")
logger.warning("INJECTION MODULE: For authorized laboratory testing only")

# Display legal warning on import
display_legal_warning()

# Check capabilities
capabilities = get_injection_capabilities()
if not any(capabilities.values()):
    warnings.warn(
        "Limited injection capabilities. Check hardware and dependencies.",
        UserWarning,
        stacklevel=2
    )

logger.info(f"Injection capabilities: {capabilities}")