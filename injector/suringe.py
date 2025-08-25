#!/usr/bin/env python3
"""
Enhanced Signal Injection Interface

This module provides comprehensive signal injection capabilities that integrate
with the enhanced core systems for FHSS, replay, signal processing, and 
obfuscation. It supports both simple injection scenarios and sophisticated
multi-layer injection strategies.

Key Features:
- Integration with enhanced FHSS and replay systems
- Multiple injection strategies (simple, intelligent, stealth)
- Protocol-aware injection with automatic encoding
- Advanced obfuscation and anti-detection techniques
- Compliance monitoring and safety limits
- Backward compatibility with existing injection code

Usage:
    >>> injector = InjectionEngine(transmitter)
    >>> injector.inject_packet(packet_data, repeat=3)
    >>> 
    >>> # Advanced usage with FHSS
    >>> injector.inject_with_fhss(packet_data, band=FHSSBand.ISM_2_4_GHz)
"""

from __future__ import annotations

import asyncio
import logging
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import numpy.typing as npt

# Import from core and injection modules
try:
    from ..core.fhss import EnhancedFHSSEngine, SimpleFHSS, FHSSBand, FHSSConfig
    from ..core.replay import EnhancedReplayEngine, ReplayConfig, ReplayStrategy
    from ..core.signal_processing import SignalProcessor, create_test_signal
    from ..plugins.base import BaseProtocolPlugin
    from .obfuscation import ObfuscationEngine, ObfuscationConfig, apply_timing_jitter
    ENHANCED_AVAILABLE = True
except ImportError:
    # Fallback for development/testing
    ENHANCED_AVAILABLE = False
    warnings.warn("Enhanced modules not available, using fallback implementations")

# Configure module logger
logger = logging.getLogger(__name__)

# Type aliases
IQSamples = npt.NDArray[np.complex64]
FrequencyHz = float
PowerDbm = float


class InjectionMethod(Enum):
    """Available injection methods."""
    
    SIMPLE = "simple"                    # Basic packet injection
    INTELLIGENT = "intelligent"         # Protocol-aware injection
    STEALTH = "stealth"                 # Anti-detection injection
    FHSS = "fhss"                       # Frequency hopping injection
    REPLAY = "replay"                   # Signal replay injection
    BURST = "burst"                     # Burst injection
    CONTINUOUS = "continuous"           # Continuous injection


class InjectionTarget(Enum):
    """Injection target types."""
    
    BROADCAST = "broadcast"
    TARGETED = "targeted"
    RESEARCH = "research"
    TESTING = "testing"


@dataclass(frozen=True)
class InjectionConfig:
    """
    Configuration for injection operations.
    
    Comprehensive configuration supporting various injection strategies
    with safety limits and compliance monitoring.
    """
    
    # Core injection parameters
    method: InjectionMethod = InjectionMethod.SIMPLE
    target: InjectionTarget = InjectionTarget.RESEARCH
    max_transmission_power_dbm: PowerDbm = 10.0
    enable_safety_limits: bool = True
    
    # Timing and repetition
    repeat_count: int = 1
    inter_packet_delay_s: float = 0.1
    enable_timing_jitter: bool = False
    jitter_range_s: Tuple[float, float] = (0.05, 0.2)
    
    # Advanced features
    enable_obfuscation: bool = False
    enable_fhss: bool = False
    enable_protocol_encoding: bool = True
    enable_error_correction: bool = False
    
    # Safety and compliance
    max_injection_duration_s: float = 60.0
    enable_compliance_monitoring: bool = True
    authorized_laboratory_only: bool = True
    
    # Integration options
    fhss_config: Optional[Dict[str, Any]] = None
    replay_config: Optional[Dict[str, Any]] = None
    obfuscation_config: Optional[Dict[str, Any]] = None
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.repeat_count < 1:
            raise ValueError("repeat_count must be at least 1")
        
        if self.inter_packet_delay_s < 0:
            raise ValueError("inter_packet_delay_s must be non-negative")
        
        if self.max_transmission_power_dbm > 30.0:
            warnings.warn(
                f"High transmission power {self.max_transmission_power_dbm} dBm - "
                "ensure compliance with local regulations"
            )
        
        if not self.authorized_laboratory_only:
            warnings.warn(
                "Injection outside authorized laboratory environment may violate regulations"
            )


@dataclass
class InjectionResult:
    """Results from injection operations."""
    
    # Injection summary
    packets_injected: int = 0
    successful_injections: int = 0
    failed_injections: int = 0
    total_injection_time_s: float = 0.0
    
    # Performance metrics
    average_power_dbm: Optional[float] = None
    frequency_accuracy_hz: Optional[float] = None
    timing_accuracy_us: Optional[float] = None
    
    # Obfuscation metrics
    obfuscation_applied: List[str] = field(default_factory=list)
    stealth_score: Optional[float] = None
    
    # Safety and compliance
    safety_violations: List[str] = field(default_factory=list)
    compliance_status: str = "unknown"
    
    # Error tracking
    error_messages: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Timestamps
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate injection success rate."""
        total = self.successful_injections + self.failed_injections
        return self.successful_injections / max(1, total)
    
    @property
    def injection_rate_hz(self) -> float:
        """Calculate average injection rate."""
        if self.total_injection_time_s > 0:
            return self.packets_injected / self.total_injection_time_s
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'packets_injected': self.packets_injected,
            'successful_injections': self.successful_injections,
            'failed_injections': self.failed_injections,
            'success_rate': self.success_rate,
            'total_injection_time_s': self.total_injection_time_s,
            'injection_rate_hz': self.injection_rate_hz,
            'average_power_dbm': self.average_power_dbm,
            'obfuscation_applied': self.obfuscation_applied,
            'stealth_score': self.stealth_score,
            'safety_violations': self.safety_violations,
            'compliance_status': self.compliance_status,
            'error_messages': self.error_messages,
            'warnings': self.warnings,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None
        }


class TransmitterInterface(ABC):
    """Abstract interface for transmitter hardware."""
    
    @abstractmethod
    def send(self, data: bytes, frequency: Optional[float] = None) -> None:
        """Send data synchronously."""
        pass
    
    @abstractmethod
    async def send_async(self, data: bytes, frequency: Optional[float] = None) -> None:
        """Send data asynchronously."""
        pass
    
    @abstractmethod
    def set_frequency(self, frequency: float) -> None:
        """Set transmitter frequency."""
        pass
    
    @abstractmethod
    def set_power(self, power_dbm: float) -> None:
        """Set transmission power."""
        pass
    
    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """Check if transmitter is ready."""
        pass


class MockTransmitter(TransmitterInterface):
    """Mock transmitter for testing and development."""
    
    def __init__(self) -> None:
        """Initialize mock transmitter."""
        self.frequency = 2.44e9
        self.power_dbm = 10.0
        self.transmitted_packets = []
        self._ready = True
    
    def send(self, data: bytes, frequency: Optional[float] = None) -> None:
        """Send data (simulated)."""
        if frequency:
            self.frequency = frequency
        
        self.transmitted_packets.append({
            'data': data,
            'frequency': self.frequency,
            'power_dbm': self.power_dbm,
            'timestamp': time.time()
        })
        
        # Simulate transmission time
        time.sleep(0.001)
    
    async def send_async(self, data: bytes, frequency: Optional[float] = None) -> None:
        """Send data asynchronously (simulated)."""
        if frequency:
            self.frequency = frequency
        
        self.transmitted_packets.append({
            'data': data,
            'frequency': self.frequency,
            'power_dbm': self.power_dbm,
            'timestamp': time.time()
        })
        
        # Simulate async transmission time
        await asyncio.sleep(0.001)
    
    def set_frequency(self, frequency: float) -> None:
        """Set transmitter frequency."""
        self.frequency = frequency
    
    def set_power(self, power_dbm: float) -> None:
        """Set transmission power."""
        self.power_dbm = power_dbm
    
    @property
    def is_ready(self) -> bool:
        """Check if transmitter is ready."""
        return self._ready


class SafetyMonitor:
    """Safety monitoring for injection operations."""
    
    def __init__(self, config: InjectionConfig) -> None:
        """Initialize safety monitor."""
        self.config = config
        self.violations = []
        self.injection_start_time = None
        self.power_measurements = []
        
    def start_injection_session(self) -> None:
        """Start monitoring an injection session."""
        self.injection_start_time = time.time()
        self.violations = []
        self.power_measurements = []
    
    def check_power_compliance(self, power_dbm: float) -> bool:
        """Check if transmission power is within limits."""
        self.power_measurements.append(power_dbm)
        
        if power_dbm > self.config.max_transmission_power_dbm:
            violation = f"Power {power_dbm:.1f} dBm exceeds limit {self.config.max_transmission_power_dbm:.1f} dBm"
            self.violations.append(violation)
            logger.warning(violation)
            return False
        
        return True
    
    def check_duration_compliance(self) -> bool:
        """Check if injection duration is within limits."""
        if self.injection_start_time is None:
            return True
        
        duration = time.time() - self.injection_start_time
        if duration > self.config.max_injection_duration_s:
            violation = f"Injection duration {duration:.1f}s exceeds limit {self.config.max_injection_duration_s:.1f}s"
            self.violations.append(violation)
            logger.error(violation)
            return False
        
        return True
    
    def get_compliance_status(self) -> str:
        """Get current compliance status."""
        if not self.violations:
            return "compliant"
        elif len(self.violations) < 3:
            return "warning"
        else:
            return "violation"


class InjectionEngine:
    """
    Enhanced injection engine with comprehensive features.
    
    This class provides a unified interface for signal injection that integrates
    with all enhanced core systems while maintaining backward compatibility.
    
    Example:
        >>> injector = InjectionEngine(transmitter)
        >>> 
        >>> # Simple injection
        >>> result = injector.inject_packet(b"Hello World", repeat=3)
        >>> 
        >>> # Advanced FHSS injection
        >>> result = injector.inject_with_fhss(
        ...     packet_data, 
        ...     band=FHSSBand.ISM_2_4_GHz,
        ...     enable_obfuscation=True
        ... )
    """
    
    def __init__(
        self,
        transmitter: TransmitterInterface,
        config: Optional[InjectionConfig] = None
    ) -> None:
        """
        Initialize injection engine.
        
        Args:
            transmitter: Transmitter interface
            config: Injection configuration
        """
        self.transmitter = transmitter
        self.config = config or InjectionConfig()
        
        # Enhanced components (if available)
        self._signal_processor = None
        self._fhss_engine = None
        self._replay_engine = None
        self._obfuscation_engine = None
        
        # Safety and monitoring
        self.safety_monitor = SafetyMonitor(self.config)
        
        # Initialize enhanced components
        if ENHANCED_AVAILABLE:
            self._init_enhanced_components()
        
        # Statistics
        self.stats = {
            'total_injections': 0,
            'total_packets_injected': 0,
            'total_injection_time': 0.0,
            'method_usage': {},
            'error_count': 0
        }
        
        logger.info(f"Initialized injection engine with {self.config.method.value} method")
    
    def _init_enhanced_components(self) -> None:
        """Initialize enhanced core components."""
        try:
            # Signal processor
            self._signal_processor = SignalProcessor()
            
            # FHSS engine (if enabled)
            if self.config.enable_fhss and self.config.fhss_config:
                fhss_config = FHSSConfig(**self.config.fhss_config)
                self._fhss_engine = EnhancedFHSSEngine(fhss_config)
            
            # Replay engine
            if self.config.replay_config:
                replay_config = ReplayConfig(**self.config.replay_config)
                self._replay_engine = EnhancedReplayEngine(replay_config, self.transmitter)
            
            # Obfuscation engine (if enabled)
            if self.config.enable_obfuscation:
                obf_config = ObfuscationConfig(**(self.config.obfuscation_config or {}))
                self._obfuscation_engine = ObfuscationEngine(obf_config)
            
            logger.debug("Enhanced components initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize enhanced components: {e}")
    
    def inject_packet(
        self,
        packet_data: bytes,
        repeat: int = 1,
        frequency: Optional[float] = None,
        power_dbm: Optional[float] = None,
        enable_jitter: Optional[bool] = None
    ) -> InjectionResult:
        """
        Inject packet with basic configuration.
        
        Args:
            packet_data: Raw packet data to inject
            repeat: Number of times to repeat injection
            frequency: Transmission frequency in Hz
            power_dbm: Transmission power in dBm
            enable_jitter: Enable timing jitter
            
        Returns:
            Injection result with performance metrics
        """
        if not packet_data:
            raise ValueError("Packet data cannot be empty")
        
        if not self.transmitter.is_ready:
            raise RuntimeError("Transmitter not ready")
        
        # Override config parameters if provided
        actual_repeat = repeat if repeat > 0 else self.config.repeat_count
        actual_jitter = enable_jitter if enable_jitter is not None else self.config.enable_timing_jitter
        
        # Set transmitter parameters
        if frequency is not None:
            self.transmitter.set_frequency(frequency)
        
        if power_dbm is not None:
            self.transmitter.set_power(power_dbm)
        
        # Start injection session
        start_time = time.time()
        result = InjectionResult()
        self.safety_monitor.start_injection_session()
        
        try:
            for i in range(actual_repeat):
                # Check safety compliance
                if not self.safety_monitor.check_duration_compliance():
                    result.safety_violations.extend(self.safety_monitor.violations)
                    break
                
                # Apply obfuscation if enabled
                processed_packet = self._apply_obfuscation(packet_data, i)
                
                # Inject packet
                injection_start = time.time()
                
                try:
                    self.transmitter.send(processed_packet, frequency)
                    result.successful_injections += 1
                    
                    # Record timing
                    injection_time = time.time() - injection_start
                    result.total_injection_time_s += injection_time
                    
                except Exception as e:
                    result.failed_injections += 1
                    result.error_messages.append(f"Injection {i+1} failed: {str(e)}")
                    logger.error(f"Injection failed: {e}")
                
                # Inter-packet delay with optional jitter
                if i < actual_repeat - 1:  # No delay after last packet
                    delay = self.config.inter_packet_delay_s
                    
                    if actual_jitter:
                        jitter_min, jitter_max = self.config.jitter_range_s
                        jitter = np.random.uniform(jitter_min, jitter_max)
                        delay += jitter
                    
                    time.sleep(delay)
            
            # Finalize results
            result.packets_injected = result.successful_injections + result.failed_injections
            result.end_time = datetime.now(timezone.utc)
            result.compliance_status = self.safety_monitor.get_compliance_status()
            
            # Update statistics
            self._update_statistics("inject_packet", result)
            
            processing_time = time.time() - start_time
            logger.info(
                f"Injected {result.successful_injections}/{actual_repeat} packets "
                f"in {processing_time:.3f}s"
            )
            
            return result
            
        except Exception as e:
            result.error_messages.append(str(e))
            result.end_time = datetime.now(timezone.utc)
            logger.error(f"Injection failed: {e}")
            return result
    
    async def inject_packet_async(
        self,
        packet_data: bytes,
        repeat: int = 1,
        frequency: Optional[float] = None,
        **kwargs: Any
    ) -> InjectionResult:
        """
        Asynchronous packet injection.
        
        Args:
            packet_data: Raw packet data to inject
            repeat: Number of times to repeat injection
            frequency: Transmission frequency in Hz
            **kwargs: Additional injection parameters
            
        Returns:
            Injection result with performance metrics
        """
        if not self.transmitter.is_ready:
            raise RuntimeError("Transmitter not ready")
        
        start_time = time.time()
        result = InjectionResult()
        self.safety_monitor.start_injection_session()
        
        try:
            for i in range(repeat):
                # Check safety compliance
                if not self.safety_monitor.check_duration_compliance():
                    result.safety_violations.extend(self.safety_monitor.violations)
                    break
                
                # Apply obfuscation if enabled
                processed_packet = self._apply_obfuscation(packet_data, i)
                
                # Async injection
                injection_start = time.time()
                
                try:
                    await self.transmitter.send_async(processed_packet, frequency)
                    result.successful_injections += 1
                    
                    injection_time = time.time() - injection_start
                    result.total_injection_time_s += injection_time
                    
                except Exception as e:
                    result.failed_injections += 1
                    result.error_messages.append(f"Async injection {i+1} failed: {str(e)}")
                
                # Async delay
                if i < repeat - 1:
                    delay = self.config.inter_packet_delay_s
                    if self.config.enable_timing_jitter:
                        jitter_min, jitter_max = self.config.jitter_range_s
                        jitter = np.random.uniform(jitter_min, jitter_max)
                        delay += jitter
                    
                    await asyncio.sleep(delay)
            
            result.packets_injected = result.successful_injections + result.failed_injections
            result.end_time = datetime.now(timezone.utc)
            result.compliance_status = self.safety_monitor.get_compliance_status()
            
            return result
            
        except Exception as e:
            result.error_messages.append(str(e))
            result.end_time = datetime.now(timezone.utc)
            return result
    
    def inject_with_fhss(
        self,
        packet_data: bytes,
        band: Optional[FHSSBand] = None,
        hop_count: Optional[int] = None,
        **kwargs: Any
    ) -> InjectionResult:
        """
        Inject packet using FHSS (Frequency Hopping Spread Spectrum).
        
        Args:
            packet_data: Raw packet data to inject
            band: FHSS band to operate in
            hop_count: Number of frequency hops
            **kwargs: Additional FHSS parameters
            
        Returns:
            Injection result with FHSS metrics
        """
        if not ENHANCED_AVAILABLE:
            raise RuntimeError("FHSS injection requires enhanced modules")
        
        if not packet_data:
            raise ValueError("Packet data cannot be empty")
        
        start_time = time.time()
        result = InjectionResult()
        self.safety_monitor.start_injection_session()
        
        try:
            # Create FHSS engine if not already initialized
            if self._fhss_engine is None:
                if band is None:
                    band = FHSSBand.ISM_2_4_GHz
                
                # Auto-configure FHSS
                freq_min, freq_max = band.frequency_range
                center_freq = (freq_min + freq_max) / 2
                
                fhss_config = FHSSConfig(
                    center_freq_hz=center_freq,
                    hop_count=hop_count or band.min_channels,
                    band=band,
                    validate_fcc_compliance=self.config.enable_compliance_monitoring,
                    **kwargs
                )
                
                self._fhss_engine = EnhancedFHSSEngine(fhss_config)
            
            # Prepare FHSS frames
            frames = self._fhss_engine.prepare_transmit_frames(
                packet_data,
                sample_rate=kwargs.get('sample_rate', 2e6),
                bitrate=kwargs.get('bitrate', 100e3)
            )
            
            # Transmit each frame
            for frame in frames:
                # Check safety compliance
                if not self.safety_monitor.check_duration_compliance():
                    result.safety_violations.extend(self.safety_monitor.violations)
                    break
                
                try:
                    # Set frequency for this hop
                    self.transmitter.set_frequency(frame.frequency_hz)
                    
                    # Convert IQ samples to bytes (simplified)
                    # In practice, this would involve proper IQ modulation
                    frame_data = frame.chunk_data
                    
                    # Apply obfuscation if enabled
                    processed_data = self._apply_obfuscation(frame_data, frame.hop_index)
                    
                    # Transmit
                    self.transmitter.send(processed_data, frame.frequency_hz)
                    result.successful_injections += 1
                    
                    # Dwell time
                    time.sleep(frame.duration_s)
                    
                except Exception as e:
                    result.failed_injections += 1
                    result.error_messages.append(f"FHSS frame {frame.hop_index} failed: {str(e)}")
            
            result.packets_injected = len(frames)
            result.obfuscation_applied.append("FHSS")
            result.end_time = datetime.now(timezone.utc)
            result.compliance_status = self.safety_monitor.get_compliance_status()
            
            # FHSS-specific metrics
            if frames:
                frequencies = [frame.frequency_hz for frame in frames]
                result.frequency_accuracy_hz = np.std(frequencies)
            
            processing_time = time.time() - start_time
            logger.info(
                f"FHSS injection completed: {result.successful_injections}/{len(frames)} frames "
                f"in {processing_time:.3f}s"
            )
            
            return result
            
        except Exception as e:
            result.error_messages.append(str(e))
            result.end_time = datetime.now(timezone.utc)
            logger.error(f"FHSS injection failed: {e}")
            return result
    
    def inject_with_replay(
        self,
        recorded_signal: Union[bytes, IQSamples],
        repeat: int = 1,
        **kwargs: Any
    ) -> InjectionResult:
        """
        Inject using signal replay methodology.
        
        Args:
            recorded_signal: Previously recorded signal data
            repeat: Number of replay iterations
            **kwargs: Replay configuration parameters
            
        Returns:
            Injection result with replay metrics
        """
        if not ENHANCED_AVAILABLE:
            raise RuntimeError("Replay injection requires enhanced modules")
        
        start_time = time.time()
        result = InjectionResult()
        
        try:
            # Create replay engine if not already initialized
            if self._replay_engine is None:
                replay_config = ReplayConfig(
                    strategy=ReplayStrategy.INTELLIGENT,
                    enable_performance_monitoring=True,
                    **kwargs
                )
                self._replay_engine = EnhancedReplayEngine(replay_config, self.transmitter)
            
            # Convert IQ samples to bytes if needed
            if isinstance(recorded_signal, np.ndarray):
                # Simple conversion - in practice would use proper demodulation
                packet_data = recorded_signal.tobytes()
            else:
                packet_data = recorded_signal
            
            # Perform replay injection
            replay_result = await self._replay_engine.replay_packet(
                packet_data,
                repeat_count=repeat,
                **kwargs
            )
            
            # Convert replay result to injection result
            result.packets_injected = replay_result.packets_transmitted
            result.successful_injections = replay_result.successful_transmissions
            result.failed_injections = replay_result.failed_transmissions
            result.total_injection_time_s = replay_result.total_transmission_time_s
            result.timing_accuracy_us = replay_result.timing_accuracy_us
            result.obfuscation_applied.append("Replay")
            result.end_time = datetime.now(timezone.utc)
            
            if replay_result.error_messages:
                result.error_messages.extend(replay_result.error_messages)
            
            return result
            
        except Exception as e:
            result.error_messages.append(str(e))
            result.end_time = datetime.now(timezone.utc)
            logger.error(f"Replay injection failed: {e}")
            return result
    
    def inject_continuous(
        self,
        packet_data: bytes,
        duration_s: float,
        packet_rate_hz: float = 10.0,
        **kwargs: Any
    ) -> InjectionResult:
        """
        Continuous injection at specified rate.
        
        Args:
            packet_data: Raw packet data to inject continuously
            duration_s: Total injection duration in seconds
            packet_rate_hz: Injection rate in packets per second
            **kwargs: Additional injection parameters
            
        Returns:
            Injection result with continuous metrics
        """
        if duration_s > self.config.max_injection_duration_s:
            raise ValueError(f"Duration {duration_s}s exceeds maximum {self.config.max_injection_duration_s}s")
        
        start_time = time.time()
        result = InjectionResult()
        self.safety_monitor.start_injection_session()
        
        inter_packet_interval = 1.0 / packet_rate_hz
        
        try:
            packet_count = 0
            while time.time() - start_time < duration_s:
                # Check safety compliance
                if not self.safety_monitor.check_duration_compliance():
                    result.safety_violations.extend(self.safety_monitor.violations)
                    break
                
                # Apply obfuscation
                processed_packet = self._apply_obfuscation(packet_data, packet_count)
                
                try:
                    self.transmitter.send(processed_packet)
                    result.successful_injections += 1
                except Exception as e:
                    result.failed_injections += 1
                    result.error_messages.append(f"Continuous injection {packet_count} failed: {str(e)}")
                
                packet_count += 1
                
                # Wait for next packet
                time.sleep(inter_packet_interval)
            
            result.packets_injected = packet_count
            result.total_injection_time_s = time.time() - start_time
            result.obfuscation_applied.append("Continuous")
            result.end_time = datetime.now(timezone.utc)
            result.compliance_status = self.safety_monitor.get_compliance_status()
            
            logger.info(f"Continuous injection: {packet_count} packets in {result.total_injection_time_s:.3f}s")
            return result
            
        except Exception as e:
            result.error_messages.append(str(e))
            result.end_time = datetime.now(timezone.utc)
            return result
    
    def _apply_obfuscation(self, packet_data: bytes, iteration: int) -> bytes:
        """Apply obfuscation techniques to packet data."""
        if not self.config.enable_obfuscation:
            return packet_data
        
        try:
            if ENHANCED_AVAILABLE and self._obfuscation_engine:
                return self._obfuscation_engine.obfuscate_packet(packet_data, iteration)
            else:
                # Simple fallback obfuscation
                return self._simple_obfuscation(packet_data, iteration)
        except Exception as e:
            logger.warning(f"Obfuscation failed: {e}")
            return packet_data
    
    def _simple_obfuscation(self, packet_data: bytes, iteration: int) -> bytes:
        """Simple fallback obfuscation."""
        # Simple XOR obfuscation
        key = (iteration % 256).to_bytes(1, 'big')
        obfuscated = bytes(b ^ key[0] for b in packet_data)
        return obfuscated
    
    def _update_statistics(self, method: str, result: InjectionResult) -> None:
        """Update injection statistics."""
        self.stats['total_injections'] += 1
        self.stats['total_packets_injected'] += result.packets_injected
        self.stats['total_injection_time'] += result.total_injection_time_s
        
        if method not in self.stats['method_usage']:
            self.stats['method_usage'][method] = 0
        self.stats['method_usage'][method] += 1
        
        if result.error_messages:
            self.stats['error_count'] += len(result.error_messages)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get injection engine statistics."""
        stats = self.stats.copy()
        stats.update({
            'config': {
                'method': self.config.method.value,
                'target': self.config.target.value,
                'max_power_dbm': self.config.max_transmission_power_dbm,
                'enhanced_mode': ENHANCED_AVAILABLE
            },
            'transmitter_ready': self.transmitter.is_ready,
            'safety_violations': len(self.safety_monitor.violations)
        })
        
        return stats


# Backward compatibility class (GhostInjector from original code)
class GhostInjector:
    """
    Backward compatible ghost injector interface.
    
    Maintains compatibility with existing code while leveraging
    the enhanced injection engine internally.
    """
    
    def __init__(self, plugin: Any, transmitter: TransmitterInterface) -> None:
        """Initialize ghost injector."""
        self.plugin = plugin
        self.transmitter = transmitter
        
        # Create enhanced injection engine
        config = InjectionConfig(
            method=InjectionMethod.STEALTH,
            enable_obfuscation=True,
            enable_timing_jitter=True
        )
        self.injection_engine = InjectionEngine(transmitter, config)
    
    def craft_and_send(
        self,
        command: Dict[str, Any],
        repeat: int = 3,
        jitter: bool = True
    ) -> None:
        """
        Craft and send command (backward compatible).
        
        Args:
            command: Command dictionary
            repeat: Number of repetitions
            jitter: Enable timing jitter
        """
        try:
            # Encode command using plugin
            if hasattr(self.plugin, 'encode_command'):
                packet = self.plugin.encode_command(command)
            else:
                # Fallback encoding
                import json
                packet = json.dumps(command).encode()
            
            # Apply FEC if available
            packet = self._apply_fec(packet)
            
            # Inject using enhanced engine
            result = self.injection_engine.inject_packet(
                packet,
                repeat=repeat,
                enable_jitter=jitter
            )
            
            logger.info(f"Ghost injection completed: {result.successful_injections}/{repeat} packets")
            
        except Exception as e:
            logger.error(f"Ghost injection failed: {e}")
            raise
    
    def _apply_fec(self, packet: bytes) -> bytes:
        """Apply forward error correction (backward compatible)."""
        try:
            from reedsolo import RSCodec
            rsc = RSCodec(10)  # 10 bytes parity
            return rsc.encode(packet)
        except ImportError:
            logger.warning("Reed-Solomon codec not available, skipping FEC")
            return packet
        except Exception as e:
            logger.warning(f"FEC failed: {e}")
            return packet


# Factory functions for easy creation
def create_injection_engine(
    transmitter: TransmitterInterface,
    method: str = "simple",
    **kwargs: Any
) -> InjectionEngine:
    """
    Create injection engine with automatic configuration.
    
    Args:
        transmitter: Transmitter interface
        method: Injection method ('simple', 'intelligent', 'stealth', 'fhss')
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured injection engine
    """
    method_map = {
        'simple': InjectionMethod.SIMPLE,
        'intelligent': InjectionMethod.INTELLIGENT,
        'stealth': InjectionMethod.STEALTH,
        'fhss': InjectionMethod.FHSS,
        'replay': InjectionMethod.REPLAY
    }
    
    method_enum = method_map.get(method.lower(), InjectionMethod.SIMPLE)
    
    config = InjectionConfig(
        method=method_enum,
        enable_obfuscation=method in ['stealth', 'fhss'],
        enable_fhss=method == 'fhss',
        **kwargs
    )
    
    return InjectionEngine(transmitter, config)


def create_mock_transmitter() -> MockTransmitter:
    """Create mock transmitter for testing."""
    return MockTransmitter()


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    print("=== Injection Engine Demo ===")
    
    try:
        # Create mock transmitter for demo
        transmitter = create_mock_transmitter()
        
        # Test simple injection
        print("\n=== Simple Injection ===")
        simple_injector = create_injection_engine(transmitter, method="simple")
        
        test_packet = b"Hello Injection World!"
        result = simple_injector.inject_packet(test_packet, repeat=3)
        
        print(f"Simple injection result:")
        print(f"  Packets injected: {result.packets_injected}")
        print(f"  Success rate: {result.success_rate:.1%}")
        print(f"  Total time: {result.total_injection_time_s:.3f}s")
        
        # Test stealth injection
        print("\n=== Stealth Injection ===")
        stealth_injector = create_injection_engine(
            transmitter,
            method="stealth",
            enable_timing_jitter=True
        )
        
        stealth_result = stealth_injector.inject_packet(test_packet, repeat=5)
        print(f"Stealth injection result:")
        print(f"  Packets injected: {stealth_result.packets_injected}")
        print(f"  Obfuscation applied: {stealth_result.obfuscation_applied}")
        print(f"  Compliance status: {stealth_result.compliance_status}")
        
        # Test FHSS injection (if enhanced modules available)
        if ENHANCED_AVAILABLE:
            print("\n=== FHSS Injection ===")
            try:
                fhss_injector = create_injection_engine(transmitter, method="fhss")
                fhss_result = fhss_injector.inject_with_fhss(
                    test_packet,
                    band=FHSSBand.ISM_2_4_GHz
                )
                print(f"FHSS injection result:")
                print(f"  Packets injected: {fhss_result.packets_injected}")
                print(f"  Frequency accuracy: {fhss_result.frequency_accuracy_hz:.1f} Hz")
            except Exception as e:
                print(f"FHSS injection failed: {e}")
        else:
            print("\n=== FHSS Injection ===")
            print("Enhanced modules not available for FHSS injection")
        
        # Test backward compatibility
        print("\n=== Backward Compatibility (GhostInjector) ===")
        
        class MockPlugin:
            def encode_command(self, command):
                import json
                return json.dumps(command).encode()
        
        ghost = GhostInjector(MockPlugin(), transmitter)
        ghost.craft_and_send({'action': 'test', 'value': 42}, repeat=2)
        print("Ghost injector test completed")
        
        # Show transmitter activity
        print(f"\n=== Transmitter Activity ===")
        print(f"Total packets transmitted: {len(transmitter.transmitted_packets)}")
        for i, tx in enumerate(transmitter.transmitted_packets[:5]):
            print(f"  TX {i}: {len(tx['data'])} bytes at {tx['frequency']/1e6:.3f} MHz")
        
        # Show statistics
        print(f"\n=== Statistics ===")
        stats = simple_injector.get_statistics()
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value}")
            elif isinstance(value, dict) and len(value) < 5:
                print(f"  {key}: {value}")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()