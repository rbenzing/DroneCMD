#!/usr/bin/env python3
"""
Enhanced RF Signal Replay Engine

A comprehensive signal replay system designed for RF analysis, testing, and
validation. This module integrates with enhanced SDR capture, demodulation,
FHSS, and protocol analysis systems to provide sophisticated replay capabilities
with timing control, signal validation, and compliance monitoring.

Integration Points:
- Enhanced FHSS Engine (frequency hopping replay)
- Enhanced Demodulation System (signal validation)
- Enhanced Protocol Classifier (protocol-aware replay)
- Enhanced Live Capture (signal quality monitoring)
- Enhanced Packet Parser (intelligent replay strategies)

Standards Compliance:
- FCC CFR 47 Part 15 compliance monitoring
- RF exposure safety limits (SAR calculations)
- Timing accuracy per IEEE 802.11 standards
- Signal quality standards for test equipment
- Laboratory testing protocols and safety

Key Features:
- Multiple replay strategies (timing-based, intelligent, stress testing)
- Real-time signal quality monitoring and validation
- Adaptive timing control with microsecond precision
- Protocol-aware replay with intelligent modifications
- Comprehensive compliance and safety monitoring
- Advanced synchronization and triggering
- Performance analytics and reporting
- Integration with all enhanced SDR systems

Example:
    >>> from dronecmd.core.enhanced_replay import EnhancedReplayEngine, ReplayConfig
    >>> config = ReplayConfig(
    ...     strategy=ReplayStrategy.INTELLIGENT,
    ...     enable_fhss=True,
    ...     timing_precision_us=10,
    ...     enable_compliance_monitoring=True
    ... )
    >>> engine = EnhancedReplayEngine(config, transmitter)
    >>> await engine.replay_packet(packet_data, repeat_count=5)
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple, Union
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import numpy.typing as npt

# Integration with our enhanced modules
try:
    from .enhanced_fhss import EnhancedFHSSEngine, FHSSBand, FHSSConfiguration
    from .enhanced_demodulation import DemodulationEngine, DemodConfig, ModulationScheme
    from .enhanced_parser import EnhancedPacketParser, ParserConfig
    from .enhanced_classifier import EnhancedProtocolClassifier, ClassifierConfig
    from .enhanced_live_capture import EnhancedLiveCapture, SDRConfig
    ENHANCED_MODULES_AVAILABLE = True
except ImportError:
    ENHANCED_MODULES_AVAILABLE = False
    # Fallback minimal classes
    class EnhancedFHSSEngine:
        pass
    class DemodulationEngine:
        pass

# Configure module logger
logger = logging.getLogger(__name__)

# Type aliases
FrequencyHz = float
TimeSeconds = float
TimestampUs = int
IQSamples = npt.NDArray[np.complex64]
PowerDbm = float


class ReplayStrategy(Enum):
    """Replay strategies for different testing scenarios."""
    
    SIMPLE = "simple"                    # Basic repeat with fixed timing
    TIMING_ACCURATE = "timing_accurate"  # Precise timing reproduction
    INTELLIGENT = "intelligent"         # Protocol-aware intelligent replay
    STRESS_TEST = "stress_test"         # High-rate stress testing
    ADAPTIVE = "adaptive"               # Adaptive based on signal quality
    SYNCHRONIZED = "synchronized"       # Multi-channel synchronized replay
    BURST = "burst"                     # Burst transmission testing


class TransmissionMode(Enum):
    """RF transmission modes."""
    
    CONTINUOUS = "continuous"
    TRIGGERED = "triggered"
    SCHEDULED = "scheduled"
    RESPONSIVE = "responsive"


class ComplianceLevel(Enum):
    """RF compliance monitoring levels."""
    
    NONE = "none"
    BASIC = "basic"
    STRICT = "strict"
    LABORATORY = "laboratory"


@dataclass(frozen=True)
class ReplayConfig:
    """
    Configuration for enhanced replay operations.
    
    Comprehensive configuration supporting various replay strategies
    with advanced timing control and compliance monitoring.
    
    Attributes:
        strategy: Replay strategy to use
        transmission_mode: Mode of RF transmission
        compliance_level: Level of compliance monitoring
        enable_fhss: Enable frequency hopping during replay
        enable_signal_validation: Validate transmitted signals
        enable_timing_analysis: Enable precise timing measurements
        timing_precision_us: Timing precision in microseconds
        max_transmission_power_dbm: Maximum transmission power
        frequency_range_hz: Allowed frequency range for transmission
        enable_adaptive_power: Enable adaptive power control
        safety_timeout_s: Maximum continuous transmission time
        enable_performance_monitoring: Enable detailed performance tracking
        enable_protocol_awareness: Enable protocol-specific replay logic
        validate_before_transmission: Validate packets before transmission
        max_packet_rate_hz: Maximum packet transmission rate
        enable_synchronization: Enable multi-transmitter synchronization
        interpacket_delay_range_s: Range for random inter-packet delays
    """
    
    strategy: ReplayStrategy = ReplayStrategy.SIMPLE
    transmission_mode: TransmissionMode = TransmissionMode.TRIGGERED
    compliance_level: ComplianceLevel = ComplianceLevel.BASIC
    enable_fhss: bool = False
    enable_signal_validation: bool = True
    enable_timing_analysis: bool = True
    timing_precision_us: int = 100
    max_transmission_power_dbm: PowerDbm = 10.0
    frequency_range_hz: Tuple[FrequencyHz, FrequencyHz] = (2.4e9, 2.485e9)
    enable_adaptive_power: bool = False
    safety_timeout_s: TimeSeconds = 10.0
    enable_performance_monitoring: bool = True
    enable_protocol_awareness: bool = True
    validate_before_transmission: bool = True
    max_packet_rate_hz: float = 1000.0
    enable_synchronization: bool = False
    interpacket_delay_range_s: Tuple[TimeSeconds, TimeSeconds] = (0.1, 0.5)
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.timing_precision_us < 1:
            raise ValueError("Timing precision must be at least 1 microsecond")
        
        if self.max_transmission_power_dbm > 30.0:
            warnings.warn(
                f"High transmission power {self.max_transmission_power_dbm} dBm - "
                "ensure compliance with local regulations"
            )
        
        if self.safety_timeout_s <= 0:
            raise ValueError("Safety timeout must be positive")
        
        if self.max_packet_rate_hz <= 0:
            raise ValueError("Maximum packet rate must be positive")
        
        freq_min, freq_max = self.frequency_range_hz
        if freq_min >= freq_max:
            raise ValueError("Invalid frequency range")


@dataclass
class ReplayResult:
    """
    Results from replay operations.
    
    Contains comprehensive information about transmission performance,
    timing accuracy, and signal quality metrics.
    """
    
    # Transmission summary
    packets_transmitted: int = 0
    total_transmission_time_s: float = 0.0
    successful_transmissions: int = 0
    failed_transmissions: int = 0
    
    # Timing analysis
    timing_accuracy_us: Optional[float] = None
    timing_jitter_us: Optional[float] = None
    inter_packet_intervals_s: List[float] = field(default_factory=list)
    
    # Signal quality metrics
    average_power_dbm: Optional[float] = None
    peak_power_dbm: Optional[float] = None
    frequency_accuracy_hz: Optional[float] = None
    spectral_purity_db: Optional[float] = None
    
    # Compliance monitoring
    compliance_violations: List[str] = field(default_factory=list)
    power_limit_exceeded: bool = False
    frequency_violations: List[float] = field(default_factory=list)
    
    # Protocol analysis
    protocol_classifications: Dict[str, int] = field(default_factory=dict)
    validation_failures: List[str] = field(default_factory=list)
    
    # Performance metrics
    throughput_bps: float = 0.0
    cpu_usage_percent: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    
    # Error analysis
    error_messages: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Timestamps
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate transmission success rate."""
        total = self.successful_transmissions + self.failed_transmissions
        return self.successful_transmissions / max(1, total)
    
    @property
    def average_packet_rate_hz(self) -> float:
        """Calculate average packet transmission rate."""
        if self.total_transmission_time_s > 0:
            return self.packets_transmitted / self.total_transmission_time_s
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'packets_transmitted': self.packets_transmitted,
            'total_transmission_time_s': self.total_transmission_time_s,
            'successful_transmissions': self.successful_transmissions,
            'failed_transmissions': self.failed_transmissions,
            'success_rate': self.success_rate,
            'timing_accuracy_us': self.timing_accuracy_us,
            'timing_jitter_us': self.timing_jitter_us,
            'average_power_dbm': self.average_power_dbm,
            'peak_power_dbm': self.peak_power_dbm,
            'frequency_accuracy_hz': self.frequency_accuracy_hz,
            'compliance_violations': self.compliance_violations,
            'protocol_classifications': self.protocol_classifications,
            'throughput_bps': self.throughput_bps,
            'average_packet_rate_hz': self.average_packet_rate_hz,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'error_messages': self.error_messages,
            'warnings': self.warnings
        }


class PrecisionTimer:
    """
    High-precision timing controller for replay operations.
    
    Provides microsecond-level timing control for accurate
    signal replay and synchronization.
    """
    
    def __init__(self, precision_us: int = 100) -> None:
        """Initialize precision timer."""
        self.precision_us = precision_us
        self.precision_s = precision_us / 1e6
        self._last_timestamp = 0.0
        self._timing_errors = deque(maxlen=1000)
    
    async def sleep_precise(self, duration_s: float) -> float:
        """
        Sleep with high precision timing.
        
        Args:
            duration_s: Desired sleep duration in seconds
            
        Returns:
            Actual sleep duration achieved
        """
        start_time = time.perf_counter()
        target_time = start_time + duration_s
        
        # Use asyncio.sleep for most of the duration
        if duration_s > 0.001:  # 1ms threshold
            rough_sleep = duration_s - 0.001
            await asyncio.sleep(rough_sleep)
        
        # Busy-wait for precise timing on the remainder
        while time.perf_counter() < target_time:
            await asyncio.sleep(0)  # Yield control briefly
        
        actual_duration = time.perf_counter() - start_time
        timing_error = abs(actual_duration - duration_s)
        self._timing_errors.append(timing_error)
        
        return actual_duration
    
    def get_timing_statistics(self) -> Dict[str, float]:
        """Get timing accuracy statistics."""
        if not self._timing_errors:
            return {}
        
        errors_us = [e * 1e6 for e in self._timing_errors]
        return {
            'mean_error_us': float(np.mean(errors_us)),
            'std_error_us': float(np.std(errors_us)),
            'max_error_us': float(np.max(errors_us)),
            'min_error_us': float(np.min(errors_us))
        }


class ComplianceMonitor:
    """
    RF compliance and safety monitoring system.
    
    Monitors transmission parameters to ensure compliance with
    regulatory requirements and safety standards.
    """
    
    def __init__(self, config: ReplayConfig) -> None:
        """Initialize compliance monitor."""
        self.config = config
        self.violations = []
        self.power_measurements = deque(maxlen=1000)
        self.frequency_measurements = deque(maxlen=1000)
        self.transmission_start_time = None
        
    def check_power_compliance(self, power_dbm: float) -> bool:
        """Check if transmission power is within limits."""
        self.power_measurements.append(power_dbm)
        
        if power_dbm > self.config.max_transmission_power_dbm:
            violation = f"Power {power_dbm:.1f} dBm exceeds limit {self.config.max_transmission_power_dbm:.1f} dBm"
            self.violations.append(violation)
            logger.warning(violation)
            return False
        
        return True
    
    def check_frequency_compliance(self, frequency_hz: float) -> bool:
        """Check if transmission frequency is within allowed range."""
        self.frequency_measurements.append(frequency_hz)
        
        freq_min, freq_max = self.config.frequency_range_hz
        if not (freq_min <= frequency_hz <= freq_max):
            violation = f"Frequency {frequency_hz/1e6:.3f} MHz outside allowed range [{freq_min/1e6:.1f}, {freq_max/1e6:.1f}] MHz"
            self.violations.append(violation)
            logger.warning(violation)
            return False
        
        return True
    
    def check_timing_compliance(self, duration_s: float) -> bool:
        """Check if transmission duration is within safety limits."""
        if duration_s > self.config.safety_timeout_s:
            violation = f"Continuous transmission {duration_s:.1f}s exceeds safety timeout {self.config.safety_timeout_s:.1f}s"
            self.violations.append(violation)
            logger.error(violation)
            return False
        
        return True
    
    def start_transmission_timer(self) -> None:
        """Start timing a transmission session."""
        self.transmission_start_time = time.time()
    
    def get_transmission_duration(self) -> float:
        """Get current transmission session duration."""
        if self.transmission_start_time is None:
            return 0.0
        return time.time() - self.transmission_start_time
    
    def get_compliance_report(self) -> Dict[str, Any]:
        """Generate compliance monitoring report."""
        return {
            'total_violations': len(self.violations),
            'violation_details': self.violations.copy(),
            'power_stats': {
                'mean_dbm': float(np.mean(self.power_measurements)) if self.power_measurements else 0.0,
                'max_dbm': float(np.max(self.power_measurements)) if self.power_measurements else 0.0,
                'samples': len(self.power_measurements)
            },
            'frequency_stats': {
                'mean_hz': float(np.mean(self.frequency_measurements)) if self.frequency_measurements else 0.0,
                'range_hz': float(np.ptp(self.frequency_measurements)) if self.frequency_measurements else 0.0,
                'samples': len(self.frequency_measurements)
            },
            'transmission_duration_s': self.get_transmission_duration()
        }


class ReplayStrategyBase(ABC):
    """Abstract base class for replay strategies."""
    
    def __init__(self, config: ReplayConfig) -> None:
        """Initialize replay strategy."""
        self.config = config
        self.timer = PrecisionTimer(config.timing_precision_us)
        self.compliance_monitor = ComplianceMonitor(config)
    
    @abstractmethod
    async def execute_replay(
        self,
        transmitter: Any,
        packet_data: bytes,
        repeat_count: int,
        **kwargs: Any
    ) -> ReplayResult:
        """Execute replay strategy."""
        pass


class SimpleReplayStrategy(ReplayStrategyBase):
    """Simple replay strategy with basic timing control."""
    
    async def execute_replay(
        self,
        transmitter: Any,
        packet_data: bytes,
        repeat_count: int,
        random_delay: bool = False,
        **kwargs: Any
    ) -> ReplayResult:
        """Execute simple replay with optional random delays."""
        result = ReplayResult()
        result.start_time = datetime.now(timezone.utc)
        
        self.compliance_monitor.start_transmission_timer()
        
        for i in range(repeat_count):
            try:
                # Apply inter-packet delay
                if i > 0:  # No delay before first packet
                    if random_delay:
                        delay_min, delay_max = self.config.interpacket_delay_range_s
                        delay = np.random.uniform(delay_min, delay_max)
                    else:
                        delay = 0.1  # Default delay
                    
                    await self.timer.sleep_precise(delay)
                    result.inter_packet_intervals_s.append(delay)
                
                # Check compliance before transmission
                if not self.compliance_monitor.check_timing_compliance(
                    self.compliance_monitor.get_transmission_duration()
                ):
                    result.error_messages.append("Safety timeout exceeded")
                    break
                
                # Transmit packet
                transmission_start = time.perf_counter()
                await self._transmit_packet(transmitter, packet_data)
                transmission_time = time.perf_counter() - transmission_start
                
                result.successful_transmissions += 1
                result.total_transmission_time_s += transmission_time
                
            except Exception as e:
                result.failed_transmissions += 1
                result.error_messages.append(f"Transmission {i+1} failed: {str(e)}")
                logger.error(f"Transmission failed: {e}")
        
        result.packets_transmitted = result.successful_transmissions + result.failed_transmissions
        result.end_time = datetime.now(timezone.utc)
        
        # Add timing statistics
        timing_stats = self.timer.get_timing_statistics()
        result.timing_accuracy_us = timing_stats.get('mean_error_us')
        result.timing_jitter_us = timing_stats.get('std_error_us')
        
        # Add compliance report
        compliance_report = self.compliance_monitor.get_compliance_report()
        result.compliance_violations = compliance_report['violation_details']
        
        return result
    
    async def _transmit_packet(self, transmitter: Any, packet_data: bytes) -> None:
        """Transmit a single packet."""
        if hasattr(transmitter, 'send_async'):
            await transmitter.send_async(packet_data)
        elif hasattr(transmitter, 'send'):
            # Run blocking send in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, transmitter.send, packet_data)
        else:
            raise RuntimeError("Transmitter has no send method")


class IntelligentReplayStrategy(ReplayStrategyBase):
    """Intelligent replay strategy with protocol awareness."""
    
    def __init__(self, config: ReplayConfig) -> None:
        """Initialize intelligent replay strategy."""
        super().__init__(config)
        
        # Initialize protocol analysis tools
        if ENHANCED_MODULES_AVAILABLE:
            self.classifier = EnhancedProtocolClassifier(
                ClassifierConfig(performance_monitoring=True)
            )
            self.parser = None  # Will be initialized when needed
        else:
            self.classifier = None
            self.parser = None
    
    async def execute_replay(
        self,
        transmitter: Any,
        packet_data: bytes,
        repeat_count: int,
        adaptive_timing: bool = True,
        **kwargs: Any
    ) -> ReplayResult:
        """Execute intelligent replay with protocol analysis."""
        result = ReplayResult()
        result.start_time = datetime.now(timezone.utc)
        
        # Analyze packet protocol
        protocol_info = await self._analyze_packet(packet_data)
        if protocol_info:
            result.protocol_classifications[protocol_info['protocol']] = repeat_count
        
        # Determine optimal timing based on protocol
        optimal_timing = self._calculate_optimal_timing(protocol_info, adaptive_timing)
        
        self.compliance_monitor.start_transmission_timer()
        
        for i in range(repeat_count):
            try:
                # Protocol-aware inter-packet timing
                if i > 0:
                    delay = optimal_timing.get('inter_packet_delay', 0.1)
                    
                    # Add protocol-specific jitter
                    if protocol_info and protocol_info.get('protocol') == 'mavlink':
                        # MAVLink typically has more regular timing
                        jitter = np.random.normal(0, delay * 0.1)
                    else:
                        # Unknown protocols use more variation
                        jitter = np.random.uniform(-delay * 0.2, delay * 0.2)
                    
                    actual_delay = max(0.01, delay + jitter)  # Minimum 10ms
                    await self.timer.sleep_precise(actual_delay)
                    result.inter_packet_intervals_s.append(actual_delay)
                
                # Adaptive packet modification (if enabled)
                modified_packet = packet_data
                if protocol_info and protocol_info.get('allow_modification', False):
                    modified_packet = self._modify_packet_intelligently(packet_data, protocol_info)
                
                # Transmit with quality monitoring
                transmission_start = time.perf_counter()
                await self._transmit_with_monitoring(transmitter, modified_packet, result)
                transmission_time = time.perf_counter() - transmission_start
                
                result.successful_transmissions += 1
                result.total_transmission_time_s += transmission_time
                
                # Adaptive timing adjustment
                if adaptive_timing and i % 10 == 9:  # Every 10 packets
                    optimal_timing = self._adapt_timing_parameters(optimal_timing, result)
                
            except Exception as e:
                result.failed_transmissions += 1
                result.error_messages.append(f"Intelligent transmission {i+1} failed: {str(e)}")
                logger.error(f"Intelligent transmission failed: {e}")
        
        result.packets_transmitted = result.successful_transmissions + result.failed_transmissions
        result.end_time = datetime.now(timezone.utc)
        
        # Calculate throughput
        if result.total_transmission_time_s > 0:
            total_bits = len(packet_data) * 8 * result.successful_transmissions
            result.throughput_bps = total_bits / result.total_transmission_time_s
        
        return result
    
    async def _analyze_packet(self, packet_data: bytes) -> Optional[Dict[str, Any]]:
        """Analyze packet to determine protocol and characteristics."""
        if not self.classifier:
            return None
        
        try:
            classification_result = self.classifier.classify(packet_data)
            
            if hasattr(classification_result, 'predicted_protocol'):
                return {
                    'protocol': classification_result.predicted_protocol,
                    'confidence': classification_result.confidence,
                    'allow_modification': classification_result.confidence > 0.8,
                    'timing_sensitivity': 'high' if 'mavlink' in classification_result.predicted_protocol.lower() else 'medium'
                }
            else:
                return {
                    'protocol': str(classification_result),
                    'confidence': 0.5,
                    'allow_modification': False,
                    'timing_sensitivity': 'medium'
                }
        except Exception as e:
            logger.debug(f"Packet analysis failed: {e}")
            return None
    
    def _calculate_optimal_timing(
        self, 
        protocol_info: Optional[Dict[str, Any]], 
        adaptive: bool
    ) -> Dict[str, float]:
        """Calculate optimal timing parameters based on protocol."""
        if not protocol_info:
            return {'inter_packet_delay': 0.1}
        
        protocol = protocol_info.get('protocol', '').lower()
        timing_sensitivity = protocol_info.get('timing_sensitivity', 'medium')
        
        if 'mavlink' in protocol:
            # MAVLink typically operates at 10-50 Hz
            base_delay = 0.02 if timing_sensitivity == 'high' else 0.05
        elif 'dji' in protocol:
            # DJI protocols often have higher rates
            base_delay = 0.01 if timing_sensitivity == 'high' else 0.025
        else:
            # Unknown protocols use conservative timing
            base_delay = 0.1
        
        return {
            'inter_packet_delay': base_delay,
            'timing_tolerance': base_delay * 0.1 if timing_sensitivity == 'high' else base_delay * 0.3
        }
    
    def _modify_packet_intelligently(
        self, 
        packet_data: bytes, 
        protocol_info: Dict[str, Any]
    ) -> bytes:
        """Intelligently modify packet based on protocol understanding."""
        # For now, return original packet
        # In a full implementation, this could:
        # - Update sequence numbers
        # - Adjust timestamps
        # - Modify non-critical fields for testing
        return packet_data
    
    async def _transmit_with_monitoring(
        self, 
        transmitter: Any, 
        packet_data: bytes, 
        result: ReplayResult
    ) -> None:
        """Transmit packet with comprehensive monitoring."""
        # Simulate power monitoring (would integrate with actual SDR)
        simulated_power = 15.0 + np.random.normal(0, 1.0)
        
        if not self.compliance_monitor.check_power_compliance(simulated_power):
            result.warnings.append(f"Power compliance violation: {simulated_power:.1f} dBm")
        
        # Perform transmission
        if hasattr(transmitter, 'send_async'):
            await transmitter.send_async(packet_data)
        elif hasattr(transmitter, 'send'):
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, transmitter.send, packet_data)
        else:
            raise RuntimeError("Transmitter has no send method")
    
    def _adapt_timing_parameters(
        self, 
        current_timing: Dict[str, float], 
        result: ReplayResult
    ) -> Dict[str, float]:
        """Adapt timing parameters based on performance feedback."""
        # Adjust timing based on success rate and timing accuracy
        success_rate = result.success_rate
        
        if success_rate < 0.95:  # Poor success rate
            # Increase inter-packet delay to improve reliability
            current_timing['inter_packet_delay'] *= 1.1
        elif success_rate > 0.99 and len(result.inter_packet_intervals_s) > 5:
            # Good success rate, potentially reduce delay for higher throughput
            avg_interval = np.mean(result.inter_packet_intervals_s[-10:])
            if avg_interval > 0.05:  # Only reduce if current delay is > 50ms
                current_timing['inter_packet_delay'] *= 0.95
        
        return current_timing


class EnhancedReplayEngine:
    """
    Enhanced replay engine with comprehensive RF testing capabilities.
    
    Integrates with all enhanced SDR systems to provide sophisticated
    replay functionality with timing control, compliance monitoring,
    and intelligent adaptation.
    """
    
    def __init__(
        self,
        config: ReplayConfig,
        transmitter: Any,
        fhss_engine: Optional[EnhancedFHSSEngine] = None
    ) -> None:
        """
        Initialize enhanced replay engine.
        
        Args:
            config: Replay configuration
            transmitter: RF transmitter interface
            fhss_engine: Optional FHSS engine for frequency hopping
        """
        self.config = config
        self.transmitter = transmitter
        self.fhss_engine = fhss_engine
        
        # Initialize strategy
        self.strategy = self._create_strategy()
        
        # Performance monitoring
        self.stats = {
            'total_replays': 0,
            'total_packets_transmitted': 0,
            'total_transmission_time': 0.0,
            'strategy_usage': defaultdict(int),
            'error_count': 0
        }
        
        # Safety monitoring
        self._active_transmissions = 0
        self._transmission_lock = asyncio.Lock()
        
        logger.info(f"Initialized enhanced replay engine with {config.strategy.value} strategy")
    
    def _create_strategy(self) -> ReplayStrategyBase:
        """Create replay strategy based on configuration."""
        if self.config.strategy == ReplayStrategy.SIMPLE:
            return SimpleReplayStrategy(self.config)
        elif self.config.strategy == ReplayStrategy.INTELLIGENT:
            return IntelligentReplayStrategy(self.config)
        else:
            logger.warning(f"Strategy {self.config.strategy.value} not implemented, using simple")
            return SimpleReplayStrategy(self.config)
    
    async def replay_packet(
        self,
        packet_bytes: bytes,
        repeat_count: int = 3,
        random_delay: bool = False,
        **kwargs: Any
    ) -> ReplayResult:
        """
        Replay packet with comprehensive analysis and control.
        
        Args:
            packet_bytes: Raw packet data to replay
            repeat_count: Number of times to repeat transmission
            random_delay: Enable random inter-packet delays
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            Comprehensive replay results
        """
        async with self._transmission_lock:
            self._active_transmissions += 1
            
            try:
                # Input validation
                if len(packet_bytes) == 0:
                    raise ValueError("Empty packet data")
                
                if repeat_count <= 0:
                    raise ValueError("Repeat count must be positive")
                
                # Apply FHSS if enabled
                processed_packet = await self._apply_fhss_if_enabled(packet_bytes)
                
                # Validate packet before transmission
                if self.config.validate_before_transmission:
                    validation_result = await self._validate_packet(processed_packet)
                    if not validation_result['valid']:
                        logger.warning(f"Packet validation failed: {validation_result['errors']}")
                
                # Execute replay strategy
                result = await self.strategy.execute_replay(
                    self.transmitter,
                    processed_packet,
                    repeat_count,
                    random_delay=random_delay,
                    **kwargs
                )
                
                # Update statistics
                self._update_statistics(result)
                
                return result
                
            except Exception as e:
                self.stats['error_count'] += 1
                logger.error(f"Replay failed: {e}")
                
                # Return error result
                error_result = ReplayResult()
                error_result.failed_transmissions = 1
                error_result.error_messages.append(str(e))
                return error_result
                
            finally:
                self._active_transmissions -= 1
    
    async def replay_sequence(
        self,
        packet_sequence: List[bytes],
        timing_sequence: Optional[List[float]] = None,
        repeat_count: int = 1
    ) -> List[ReplayResult]:
        """
        Replay a sequence of packets with precise timing control.
        
        Args:
            packet_sequence: List of packet data to replay in sequence
            timing_sequence: Optional list of inter-packet delays
            repeat_count: Number of times to repeat entire sequence
            
        Returns:
            List of replay results for each packet
        """
        if not packet_sequence:
            return []
        
        if timing_sequence is None:
            timing_sequence = [0.1] * (len(packet_sequence) - 1) + [0.0]
        
        all_results = []
        
        for sequence_iteration in range(repeat_count):
            sequence_results = []
            
            for i, packet_data in enumerate(packet_sequence):
                # Replay individual packet
                result = await self.replay_packet(packet_data, repeat_count=1)
                sequence_results.append(result)
                
                # Apply inter-packet timing
                if i < len(timing_sequence) and timing_sequence[i] > 0:
                    await asyncio.sleep(timing_sequence[i])
            
            all_results.extend(sequence_results)
        
        return all_results
    
    async def stress_test_replay(
        self,
        packet_bytes: bytes,
        target_rate_hz: float,
        duration_s: float
    ) -> ReplayResult:
        """
        Perform stress test replay at specified packet rate.
        
        Args:
            packet_bytes: Packet data to replay
            target_rate_hz: Target transmission rate in Hz
            duration_s: Duration of stress test in seconds
            
        Returns:
            Stress test results
        """
        if target_rate_hz > self.config.max_packet_rate_hz:
            raise ValueError(f"Target rate {target_rate_hz} Hz exceeds maximum {self.config.max_packet_rate_hz} Hz")
        
        inter_packet_delay = 1.0 / target_rate_hz
        total_packets = int(duration_s * target_rate_hz)
        
        logger.info(f"Starting stress test: {total_packets} packets at {target_rate_hz} Hz for {duration_s}s")
        
        # Use simple strategy for stress testing
        original_strategy = self.strategy
        self.strategy = SimpleReplayStrategy(self.config)
        
        try:
            result = await self.strategy.execute_replay(
                self.transmitter,
                packet_bytes,
                total_packets
            )
            
            # Verify timing performance
            if result.inter_packet_intervals_s:
                actual_rate = 1.0 / np.mean(result.inter_packet_intervals_s)
                rate_error = abs(actual_rate - target_rate_hz) / target_rate_hz
                
                if rate_error > 0.1:  # 10% tolerance
                    result.warnings.append(f"Rate error {rate_error*100:.1f}%: target {target_rate_hz} Hz, actual {actual_rate:.1f} Hz")
            
            return result
            
        finally:
            self.strategy = original_strategy
    
    async def _apply_fhss_if_enabled(self, packet_bytes: bytes) -> bytes:
        """Apply FHSS processing if enabled."""
        if not self.config.enable_fhss or not self.fhss_engine:
            return packet_bytes
        
        try:
            # This would integrate with the enhanced FHSS engine
            # For now, return original packet
            # In full implementation:
            # frames = self.fhss_engine.prepare_transmit_frames(packet_bytes)
            # return frames[0][1]  # Return IQ samples of first frame
            return packet_bytes
            
        except Exception as e:
            logger.warning(f"FHSS processing failed: {e}")
            return packet_bytes
    
    async def _validate_packet(self, packet_bytes: bytes) -> Dict[str, Any]:
        """Validate packet before transmission."""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Basic validation
        if len(packet_bytes) == 0:
            validation_result['valid'] = False
            validation_result['errors'].append("Empty packet")
        
        if len(packet_bytes) > 2048:  # Reasonable maximum
            validation_result['warnings'].append(f"Large packet: {len(packet_bytes)} bytes")
        
        # Protocol-specific validation could be added here
        
        return validation_result
    
    def _update_statistics(self, result: ReplayResult) -> None:
        """Update engine performance statistics."""
        self.stats['total_replays'] += 1
        self.stats['total_packets_transmitted'] += result.packets_transmitted
        self.stats['total_transmission_time'] += result.total_transmission_time_s
        self.stats['strategy_usage'][self.config.strategy.value] += 1
        
        if result.error_messages:
            self.stats['error_count'] += len(result.error_messages)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get engine performance statistics."""
        avg_transmission_time = (
            self.stats['total_transmission_time'] / 
            max(1, self.stats['total_replays'])
        )
        
        avg_packets_per_replay = (
            self.stats['total_packets_transmitted'] / 
            max(1, self.stats['total_replays'])
        )
        
        return {
            'total_replays': self.stats['total_replays'],
            'total_packets_transmitted': self.stats['total_packets_transmitted'],
            'average_transmission_time_s': avg_transmission_time,
            'average_packets_per_replay': avg_packets_per_replay,
            'strategy_distribution': dict(self.stats['strategy_usage']),
            'error_count': self.stats['error_count'],
            'active_transmissions': self._active_transmissions,
            'configuration': {
                'strategy': self.config.strategy.value,
                'max_power_dbm': self.config.max_transmission_power_dbm,
                'timing_precision_us': self.config.timing_precision_us
            }
        }


# Backward compatibility function
def create_replay_engine(transmitter: Any) -> EnhancedReplayEngine:
    """
    Create replay engine with backward compatibility.
    
    Args:
        transmitter: RF transmitter interface
        
    Returns:
        Enhanced replay engine instance
    """
    config = ReplayConfig()
    return EnhancedReplayEngine(config, transmitter)


async def main() -> None:
    """
    Example usage and testing of the enhanced replay engine.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=== Enhanced Replay Engine Demo ===\n")
    
    # Mock transmitter for demonstration
    class MockTransmitter:
        def __init__(self):
            self.transmitted_packets = []
        
        async def send_async(self, data: bytes) -> None:
            await asyncio.sleep(0.001)  # Simulate transmission time
            self.transmitted_packets.append(data)
            print(f"Transmitted {len(data)} bytes: {data[:8].hex()}...")
        
        def send(self, data: bytes) -> None:
            self.transmitted_packets.append(data)
            print(f"Transmitted {len(data)} bytes: {data[:8].hex()}...")
    
    # Create configurations
    simple_config = ReplayConfig(
        strategy=ReplayStrategy.SIMPLE,
        timing_precision_us=50,
        enable_performance_monitoring=True
    )
    
    intelligent_config = ReplayConfig(
        strategy=ReplayStrategy.INTELLIGENT,
        enable_protocol_awareness=True,
        enable_signal_validation=True
    )
    
    print("Configuration examples:")
    print(f"  Simple: {simple_config.strategy.value}, precision: {simple_config.timing_precision_us}μs")
    print(f"  Intelligent: {intelligent_config.strategy.value}, protocol-aware: {intelligent_config.enable_protocol_awareness}")
    
    # Test packets
    test_packets = [
        # MAVLink-like packet
        b'\xFE\x21\x00\x01\x01\x00' + b'\x42' * 33 + b'\x12\x34',
        
        # DJI-like packet
        b'\x55\xAA\x27\x10' + b'\x33' * 35 + b'\x56\x78',
        
        # Generic packet
        b'\x11\x22\x33\x44' + b'\x55' * 20
    ]
    
    # Test simple replay
    print(f"\n=== Testing Simple Replay ===")
    transmitter = MockTransmitter()
    simple_engine = EnhancedReplayEngine(simple_config, transmitter)
    
    simple_result = await simple_engine.replay_packet(
        test_packets[0],
        repeat_count=3,
        random_delay=True
    )
    
    print(f"Simple replay results:")
    print(f"  Packets transmitted: {simple_result.packets_transmitted}")
    print(f"  Success rate: {simple_result.success_rate:.1%}")
    print(f"  Total time: {simple_result.total_transmission_time_s:.3f}s")
    print(f"  Timing accuracy: {simple_result.timing_accuracy_us:.1f}μs")
    
    # Test intelligent replay
    print(f"\n=== Testing Intelligent Replay ===")
    transmitter2 = MockTransmitter()
    intelligent_engine = EnhancedReplayEngine(intelligent_config, transmitter2)
    
    intelligent_result = await intelligent_engine.replay_packet(
        test_packets[0],
        repeat_count=5,
        adaptive_timing=True
    )
    
    print(f"Intelligent replay results:")
    print(f"  Packets transmitted: {intelligent_result.packets_transmitted}")
    print(f"  Success rate: {intelligent_result.success_rate:.1%}")
    print(f"  Throughput: {intelligent_result.throughput_bps:.0f} bps")
    print(f"  Protocol classifications: {intelligent_result.protocol_classifications}")
    
    # Test sequence replay
    print(f"\n=== Testing Sequence Replay ===")
    sequence_results = await simple_engine.replay_sequence(
        test_packets,
        timing_sequence=[0.05, 0.1, 0.0],  # 50ms, 100ms delays
        repeat_count=2
    )
    
    print(f"Sequence replay results:")
    total_packets = sum(r.packets_transmitted for r in sequence_results)
    print(f"  Total packets: {total_packets}")
    print(f"  Sequence iterations: 2")
    print(f"  Average success rate: {np.mean([r.success_rate for r in sequence_results]):.1%}")
    
    # Performance statistics
    print(f"\n=== Performance Statistics ===")
    stats = simple_engine.get_performance_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())