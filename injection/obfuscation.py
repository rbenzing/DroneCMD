#!/usr/bin/env python3
"""
Advanced Obfuscation Techniques

This module provides comprehensive obfuscation capabilities for signal injection
operations, including timing, frequency, protocol, and data obfuscation techniques
designed to evade detection and analysis.

Key Features:
- Multiple obfuscation layers (timing, frequency, protocol, data)
- Anti-detection and anti-analysis techniques
- Configurable obfuscation strength and methods
- Integration with injection and FHSS systems
- Backward compatibility with simple obfuscation functions
- Traffic analysis resistance

Usage:
    >>> obfuscator = ObfuscationEngine()
    >>> obfuscated_packet = obfuscator.obfuscate_packet(packet_data)
    >>> 
    >>> # Simple timing jitter (backward compatible)
    >>> apply_timing_jitter(delay_range=(0.1, 0.5))
"""

from __future__ import annotations

import hashlib
import logging
import random
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

# Configure module logger
logger = logging.getLogger(__name__)

# Type aliases
ObfuscationMethod = str
ByteArray = npt.NDArray[np.uint8]


class ObfuscationType(Enum):
    """Types of obfuscation techniques."""
    
    TIMING = "timing"               # Timing-based obfuscation
    FREQUENCY = "frequency"         # Frequency domain obfuscation
    PROTOCOL = "protocol"           # Protocol-level obfuscation
    DATA = "data"                   # Data content obfuscation
    TRAFFIC = "traffic"             # Traffic analysis resistance
    STEALTH = "stealth"             # Advanced stealth techniques
    ENCRYPTION = "encryption"       # Cryptographic obfuscation


class StealthLevel(Enum):
    """Stealth operation levels."""
    
    NONE = "none"                   # No stealth
    LOW = "low"                     # Basic obfuscation
    MEDIUM = "medium"               # Moderate stealth
    HIGH = "high"                   # Advanced stealth
    MAXIMUM = "maximum"             # Maximum stealth (may impact performance)


@dataclass(frozen=True)
class ObfuscationConfig:
    """
    Configuration for obfuscation operations.
    
    Comprehensive configuration supporting various obfuscation techniques
    with adjustable stealth levels and detection resistance.
    """
    
    # Core obfuscation settings
    stealth_level: StealthLevel = StealthLevel.MEDIUM
    enabled_types: List[ObfuscationType] = field(default_factory=lambda: [
        ObfuscationType.TIMING,
        ObfuscationType.DATA
    ])
    
    # Timing obfuscation
    timing_jitter_enabled: bool = True
    timing_jitter_range_s: Tuple[float, float] = (0.05, 0.3)
    timing_randomization: bool = True
    burst_interval_variation: float = 0.2
    
    # Frequency obfuscation
    frequency_hopping: bool = False
    frequency_jitter_hz: float = 1000.0
    carrier_offset_variation: bool = True
    
    # Protocol obfuscation
    protocol_masking: bool = True
    header_randomization: bool = True
    payload_padding: bool = True
    padding_range: Tuple[int, int] = (1, 16)
    
    # Data obfuscation
    data_encoding: bool = True
    xor_obfuscation: bool = True
    substitution_cipher: bool = False
    data_compression: bool = False
    
    # Traffic analysis resistance
    traffic_shaping: bool = True
    dummy_packets: bool = False
    packet_size_variation: bool = True
    
    # Advanced stealth
    mimicry_enabled: bool = False
    covert_channels: bool = False
    anti_forensics: bool = False
    
    # Encryption settings
    encryption_enabled: bool = False
    encryption_key: Optional[bytes] = None
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.timing_jitter_range_s[0] >= self.timing_jitter_range_s[1]:
            raise ValueError("Invalid timing jitter range")
        
        if self.padding_range[0] >= self.padding_range[1]:
            raise ValueError("Invalid padding range")
        
        if self.frequency_jitter_hz < 0:
            raise ValueError("Frequency jitter must be non-negative")


class TimingObfuscator:
    """Timing-based obfuscation techniques."""
    
    def __init__(self, config: ObfuscationConfig) -> None:
        """Initialize timing obfuscator."""
        self.config = config
        self._last_transmission_time = 0.0
        self._interval_history = []
    
    def apply_jitter(self, base_delay: float = 0.1) -> float:
        """
        Apply timing jitter to transmission intervals.
        
        Args:
            base_delay: Base delay in seconds
            
        Returns:
            Jittered delay value
        """
        if not self.config.timing_jitter_enabled:
            return base_delay
        
        jitter_min, jitter_max = self.config.timing_jitter_range_s
        jitter = random.uniform(jitter_min, jitter_max)
        
        if self.config.timing_randomization:
            # Add random component
            random_factor = random.uniform(0.5, 1.5)
            jitter *= random_factor
        
        return base_delay + jitter
    
    def calculate_adaptive_interval(self, packet_size: int) -> float:
        """
        Calculate adaptive interval based on packet characteristics.
        
        Args:
            packet_size: Size of packet in bytes
            
        Returns:
            Adaptive interval in seconds
        """
        # Base interval proportional to packet size
        base_interval = 0.001 * packet_size  # 1ms per byte
        
        # Add variation based on stealth level
        if self.config.stealth_level == StealthLevel.HIGH:
            variation = random.uniform(0.5, 2.0)
            base_interval *= variation
        elif self.config.stealth_level == StealthLevel.MAXIMUM:
            # More sophisticated adaptive timing
            variation = self._calculate_traffic_mimicry_interval()
            base_interval *= variation
        
        return self.apply_jitter(base_interval)
    
    def _calculate_traffic_mimicry_interval(self) -> float:
        """Calculate interval that mimics legitimate traffic patterns."""
        # Simple pattern mimicry - in practice would analyze real traffic
        patterns = [0.1, 0.2, 0.5, 1.0, 2.0]  # Common intervals
        weights = [0.3, 0.25, 0.2, 0.15, 0.1]  # Weighted selection
        
        return np.random.choice(patterns, p=weights)
    
    def wait_with_jitter(self, base_delay: float = 0.1) -> None:
        """
        Wait with applied timing jitter.
        
        Args:
            base_delay: Base delay in seconds
        """
        actual_delay = self.apply_jitter(base_delay)
        time.sleep(actual_delay)
        
        # Track timing for adaptation
        current_time = time.time()
        if self._last_transmission_time > 0:
            interval = current_time - self._last_transmission_time
            self._interval_history.append(interval)
            
            # Keep limited history
            if len(self._interval_history) > 100:
                self._interval_history.pop(0)
        
        self._last_transmission_time = current_time


class FrequencyObfuscator:
    """Frequency-domain obfuscation techniques."""
    
    def __init__(self, config: ObfuscationConfig) -> None:
        """Initialize frequency obfuscator."""
        self.config = config
        self._frequency_history = []
    
    def apply_frequency_jitter(self, base_frequency: float) -> float:
        """
        Apply frequency jitter to transmission frequency.
        
        Args:
            base_frequency: Base frequency in Hz
            
        Returns:
            Jittered frequency
        """
        if not self.config.carrier_offset_variation:
            return base_frequency
        
        max_jitter = self.config.frequency_jitter_hz
        jitter = random.uniform(-max_jitter, max_jitter)
        
        jittered_frequency = base_frequency + jitter
        
        # Track frequency usage
        self._frequency_history.append(jittered_frequency)
        if len(self._frequency_history) > 50:
            self._frequency_history.pop(0)
        
        return jittered_frequency
    
    def select_random_frequency(
        self,
        frequency_range: Tuple[float, float],
        avoid_recent: bool = True
    ) -> float:
        """
        Select random frequency from range.
        
        Args:
            frequency_range: (min_freq, max_freq) in Hz
            avoid_recent: Avoid recently used frequencies
            
        Returns:
            Selected frequency
        """
        min_freq, max_freq = frequency_range
        
        if avoid_recent and len(self._frequency_history) > 5:
            # Avoid frequencies used in last 5 transmissions
            recent_freqs = set(self._frequency_history[-5:])
            
            # Try to find unused frequency
            for _ in range(10):  # Max 10 attempts
                freq = random.uniform(min_freq, max_freq)
                if not any(abs(freq - rf) < 10000 for rf in recent_freqs):  # 10kHz separation
                    return freq
        
        # Fallback to any frequency in range
        return random.uniform(min_freq, max_freq)
    
    def generate_frequency_pattern(
        self,
        base_frequencies: List[float],
        pattern_length: int = 10
    ) -> List[float]:
        """
        Generate pseudo-random frequency hopping pattern.
        
        Args:
            base_frequencies: Available frequencies
            pattern_length: Length of pattern
            
        Returns:
            Frequency hopping pattern
        """
        if not base_frequencies:
            return []
        
        # Ensure equal usage of all frequencies
        pattern = []
        frequencies = base_frequencies.copy()
        
        for _ in range(pattern_length):
            if not frequencies:
                frequencies = base_frequencies.copy()
            
            freq = random.choice(frequencies)
            frequencies.remove(freq)
            pattern.append(freq)
        
        return pattern


class ProtocolObfuscator:
    """Protocol-level obfuscation techniques."""
    
    def __init__(self, config: ObfuscationConfig) -> None:
        """Initialize protocol obfuscator."""
        self.config = config
    
    def obfuscate_header(self, packet: bytes) -> bytes:
        """
        Obfuscate packet header fields.
        
        Args:
            packet: Original packet bytes
            
        Returns:
            Packet with obfuscated header
        """
        if not self.config.header_randomization or len(packet) < 8:
            return packet
        
        # Simple header obfuscation - randomize first few bytes
        packet_array = bytearray(packet)
        
        # Randomize first 2-4 bytes (avoiding critical sync patterns)
        header_length = min(4, len(packet_array) // 4)
        
        for i in range(header_length):
            if random.random() < 0.3:  # 30% chance to modify each byte
                packet_array[i] = random.randint(0, 255)
        
        return bytes(packet_array)
    
    def add_padding(self, packet: bytes) -> bytes:
        """
        Add random padding to packet.
        
        Args:
            packet: Original packet bytes
            
        Returns:
            Padded packet
        """
        if not self.config.payload_padding:
            return packet
        
        padding_min, padding_max = self.config.padding_range
        padding_length = random.randint(padding_min, padding_max)
        
        # Generate random padding
        padding = bytes(random.randint(0, 255) for _ in range(padding_length))
        
        # Add padding at random location
        if random.random() < 0.5:
            # Prepend padding
            return padding + packet
        else:
            # Append padding
            return packet + padding
    
    def fragment_packet(self, packet: bytes, max_fragment_size: int = 64) -> List[bytes]:
        """
        Fragment packet into smaller pieces.
        
        Args:
            packet: Original packet bytes
            max_fragment_size: Maximum fragment size
            
        Returns:
            List of packet fragments
        """
        if len(packet) <= max_fragment_size:
            return [packet]
        
        fragments = []
        offset = 0
        
        while offset < len(packet):
            # Random fragment size (within limits)
            remaining = len(packet) - offset
            fragment_size = random.randint(
                min(8, remaining),
                min(max_fragment_size, remaining)
            )
            
            fragment = packet[offset:offset + fragment_size]
            fragments.append(fragment)
            offset += fragment_size
        
        return fragments
    
    def apply_protocol_mimicry(self, packet: bytes, target_protocol: str = "generic") -> bytes:
        """
        Apply protocol mimicry techniques.
        
        Args:
            packet: Original packet bytes
            target_protocol: Protocol to mimic
            
        Returns:
            Modified packet with protocol mimicry
        """
        if not self.config.protocol_masking:
            return packet
        
        # Simple protocol mimicry - add common protocol headers
        if target_protocol.lower() == "http":
            # Add HTTP-like header
            fake_header = b"GET / HTTP/1.1\r\n\r\n"
            return fake_header + packet
        
        elif target_protocol.lower() == "dns":
            # Add DNS-like header
            fake_header = b"\x12\x34\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00"
            return fake_header + packet
        
        else:
            # Generic obfuscation
            fake_header = bytes(random.randint(0, 255) for _ in range(4))
            return fake_header + packet


class DataObfuscator:
    """Data content obfuscation techniques."""
    
    def __init__(self, config: ObfuscationConfig) -> None:
        """Initialize data obfuscator."""
        self.config = config
        self._xor_keys = {}
    
    def apply_xor_obfuscation(self, data: bytes, key_id: Optional[str] = None) -> bytes:
        """
        Apply XOR obfuscation to data.
        
        Args:
            data: Input data bytes
            key_id: Optional key identifier for key reuse
            
        Returns:
            XOR obfuscated data
        """
        if not self.config.xor_obfuscation or not data:
            return data
        
        # Generate or retrieve XOR key
        if key_id and key_id in self._xor_keys:
            key = self._xor_keys[key_id]
        else:
            key = secrets.randbits(8)  # 8-bit key
            if key_id:
                self._xor_keys[key_id] = key
        
        # Apply XOR
        obfuscated = bytes(b ^ key for b in data)
        return obfuscated
    
    def apply_substitution_cipher(self, data: bytes) -> bytes:
        """
        Apply simple substitution cipher.
        
        Args:
            data: Input data bytes
            
        Returns:
            Substitution encrypted data
        """
        if not self.config.substitution_cipher or not data:
            return data
        
        # Create substitution table
        substitution_table = list(range(256))
        random.shuffle(substitution_table)
        
        # Apply substitution
        substituted = bytes(substitution_table[b] for b in data)
        return substituted
    
    def apply_data_encoding(self, data: bytes, encoding: str = "base64") -> bytes:
        """
        Apply data encoding obfuscation.
        
        Args:
            data: Input data bytes
            encoding: Encoding method
            
        Returns:
            Encoded data
        """
        if not self.config.data_encoding or not data:
            return data
        
        try:
            if encoding == "base64":
                import base64
                return base64.b64encode(data)
            
            elif encoding == "hex":
                return data.hex().encode()
            
            elif encoding == "url":
                import urllib.parse
                return urllib.parse.quote(data).encode()
            
            else:
                return data
                
        except Exception as e:
            logger.warning(f"Data encoding failed: {e}")
            return data
    
    def scramble_bytes(self, data: bytes, seed: Optional[int] = None) -> bytes:
        """
        Scramble byte order using deterministic algorithm.
        
        Args:
            data: Input data bytes
            seed: Random seed for reproducible scrambling
            
        Returns:
            Scrambled data
        """
        if not data:
            return data
        
        # Convert to list for manipulation
        data_list = list(data)
        
        # Create random state
        rng = random.Random(seed)
        
        # Scramble bytes
        rng.shuffle(data_list)
        
        return bytes(data_list)


class TrafficObfuscator:
    """Traffic analysis resistance techniques."""
    
    def __init__(self, config: ObfuscationConfig) -> None:
        """Initialize traffic obfuscator."""
        self.config = config
        self._dummy_packet_count = 0
    
    def generate_dummy_packet(self, size_range: Tuple[int, int] = (32, 128)) -> bytes:
        """
        Generate dummy packet for traffic padding.
        
        Args:
            size_range: Size range for dummy packet
            
        Returns:
            Dummy packet bytes
        """
        min_size, max_size = size_range
        packet_size = random.randint(min_size, max_size)
        
        # Generate realistic-looking dummy data
        dummy_data = bytearray()
        
        # Add fake header
        dummy_data.extend(bytes(random.randint(0, 255) for _ in range(8)))
        
        # Add payload with patterns
        remaining = packet_size - len(dummy_data)
        if remaining > 0:
            # Mix of zero bytes, random bytes, and patterns
            for i in range(remaining):
                if i % 4 == 0:
                    dummy_data.append(0x00)  # Zero byte
                elif i % 4 == 1:
                    dummy_data.append(0xFF)  # Max byte
                else:
                    dummy_data.append(random.randint(0, 255))  # Random
        
        self._dummy_packet_count += 1
        return bytes(dummy_data)
    
    def vary_packet_size(self, packet: bytes, variation_percent: float = 0.2) -> bytes:
        """
        Vary packet size for traffic shaping.
        
        Args:
            packet: Original packet
            variation_percent: Size variation percentage
            
        Returns:
            Size-varied packet
        """
        if not self.config.packet_size_variation:
            return packet
        
        original_size = len(packet)
        max_variation = int(original_size * variation_percent)
        
        if max_variation == 0:
            return packet
        
        size_change = random.randint(-max_variation, max_variation)
        
        if size_change > 0:
            # Add padding
            padding = bytes(random.randint(0, 255) for _ in range(size_change))
            return packet + padding
        elif size_change < 0 and len(packet) > abs(size_change):
            # Truncate (carefully)
            return packet[:size_change]
        else:
            return packet
    
    def calculate_traffic_timing(self, packet_count: int) -> List[float]:
        """
        Calculate timing pattern for traffic shaping.
        
        Args:
            packet_count: Number of packets
            
        Returns:
            List of inter-packet intervals
        """
        if not self.config.traffic_shaping:
            return [0.1] * packet_count  # Fixed interval
        
        # Generate varied timing pattern
        intervals = []
        
        for i in range(packet_count):
            # Base interval with variation
            base_interval = 0.1
            
            # Add pattern-based variation
            if i % 5 == 0:  # Every 5th packet
                interval = base_interval * 2.0  # Longer pause
            elif i % 3 == 0:  # Every 3rd packet
                interval = base_interval * 0.5  # Shorter pause
            else:
                interval = base_interval
            
            # Add random jitter
            jitter = random.uniform(-0.02, 0.02)
            interval = max(0.01, interval + jitter)
            
            intervals.append(interval)
        
        return intervals


class ObfuscationEngine:
    """
    Comprehensive obfuscation engine integrating all techniques.
    
    This class provides a unified interface for applying multiple
    obfuscation techniques to achieve desired stealth levels.
    
    Example:
        >>> obfuscator = ObfuscationEngine(ObfuscationConfig(
        ...     stealth_level=StealthLevel.HIGH,
        ...     enabled_types=[ObfuscationType.TIMING, ObfuscationType.DATA]
        ... ))
        >>> obfuscated = obfuscator.obfuscate_packet(packet_data)
    """
    
    def __init__(self, config: Optional[ObfuscationConfig] = None) -> None:
        """
        Initialize obfuscation engine.
        
        Args:
            config: Obfuscation configuration
        """
        self.config = config or ObfuscationConfig()
        
        # Initialize obfuscation components
        self.timing_obfuscator = TimingObfuscator(self.config)
        self.frequency_obfuscator = FrequencyObfuscator(self.config)
        self.protocol_obfuscator = ProtocolObfuscator(self.config)
        self.data_obfuscator = DataObfuscator(self.config)
        self.traffic_obfuscator = TrafficObfuscator(self.config)
        
        # Statistics
        self.stats = {
            'packets_obfuscated': 0,
            'techniques_applied': {},
            'stealth_level': self.config.stealth_level.value
        }
        
        logger.info(f"Initialized obfuscation engine (stealth: {self.config.stealth_level.value})")
    
    def obfuscate_packet(
        self,
        packet_data: bytes,
        iteration: int = 0,
        custom_key: Optional[str] = None
    ) -> bytes:
        """
        Apply comprehensive packet obfuscation.
        
        Args:
            packet_data: Original packet data
            iteration: Packet iteration number
            custom_key: Custom obfuscation key
            
        Returns:
            Obfuscated packet data
        """
        if not packet_data:
            return packet_data
        
        obfuscated = packet_data
        techniques_applied = []
        
        try:
            # Data obfuscation
            if ObfuscationType.DATA in self.config.enabled_types:
                if self.config.xor_obfuscation:
                    obfuscated = self.data_obfuscator.apply_xor_obfuscation(
                        obfuscated, custom_key
                    )
                    techniques_applied.append("XOR")
                
                if self.config.data_encoding:
                    # Only apply for high stealth levels to avoid size increase
                    if self.config.stealth_level in [StealthLevel.HIGH, StealthLevel.MAXIMUM]:
                        obfuscated = self.data_obfuscator.apply_data_encoding(obfuscated)
                        techniques_applied.append("Encoding")
                
                # Byte scrambling for maximum stealth
                if self.config.stealth_level == StealthLevel.MAXIMUM:
                    obfuscated = self.data_obfuscator.scramble_bytes(obfuscated, iteration)
                    techniques_applied.append("Scrambling")
            
            # Protocol obfuscation
            if ObfuscationType.PROTOCOL in self.config.enabled_types:
                if self.config.header_randomization:
                    obfuscated = self.protocol_obfuscator.obfuscate_header(obfuscated)
                    techniques_applied.append("HeaderObfuscation")
                
                if self.config.payload_padding:
                    obfuscated = self.protocol_obfuscator.add_padding(obfuscated)
                    techniques_applied.append("Padding")
                
                # Protocol mimicry for high stealth
                if self.config.stealth_level in [StealthLevel.HIGH, StealthLevel.MAXIMUM]:
                    if self.config.protocol_masking:
                        obfuscated = self.protocol_obfuscator.apply_protocol_mimicry(obfuscated)
                        techniques_applied.append("ProtocolMimicry")
            
            # Traffic obfuscation
            if ObfuscationType.TRAFFIC in self.config.enabled_types:
                if self.config.packet_size_variation:
                    obfuscated = self.traffic_obfuscator.vary_packet_size(obfuscated)
                    techniques_applied.append("SizeVariation")
            
            # Update statistics
            self.stats['packets_obfuscated'] += 1
            for technique in techniques_applied:
                self.stats['techniques_applied'][technique] = (
                    self.stats['techniques_applied'].get(technique, 0) + 1
                )
            
            logger.debug(f"Applied obfuscation: {techniques_applied}")
            return obfuscated
            
        except Exception as e:
            logger.error(f"Packet obfuscation failed: {e}")
            return packet_data
    
    def apply_timing_obfuscation(self, base_delay: float = 0.1) -> float:
        """
        Apply timing obfuscation to transmission timing.
        
        Args:
            base_delay: Base delay in seconds
            
        Returns:
            Obfuscated delay
        """
        if ObfuscationType.TIMING in self.config.enabled_types:
            return self.timing_obfuscator.apply_jitter(base_delay)
        return base_delay
    
    def apply_frequency_obfuscation(self, base_frequency: float) -> float:
        """
        Apply frequency obfuscation to transmission frequency.
        
        Args:
            base_frequency: Base frequency in Hz
            
        Returns:
            Obfuscated frequency
        """
        if ObfuscationType.FREQUENCY in self.config.enabled_types:
            return self.frequency_obfuscator.apply_frequency_jitter(base_frequency)
        return base_frequency
    
    def generate_dummy_traffic(
        self,
        packet_count: int,
        size_range: Tuple[int, int] = (32, 128)
    ) -> List[bytes]:
        """
        Generate dummy traffic for traffic analysis resistance.
        
        Args:
            packet_count: Number of dummy packets
            size_range: Size range for packets
            
        Returns:
            List of dummy packets
        """
        if not self.config.dummy_packets:
            return []
        
        dummy_packets = []
        for _ in range(packet_count):
            dummy = self.traffic_obfuscator.generate_dummy_packet(size_range)
            dummy_packets.append(dummy)
        
        return dummy_packets
    
    def calculate_stealth_score(self, original_packet: bytes, obfuscated_packet: bytes) -> float:
        """
        Calculate stealth score based on obfuscation applied.
        
        Args:
            original_packet: Original packet data
            obfuscated_packet: Obfuscated packet data
            
        Returns:
            Stealth score (0-1, higher is more stealthy)
        """
        if not original_packet:
            return 0.0
        
        score = 0.0
        max_score = 0.0
        
        # Size difference component
        size_diff = abs(len(obfuscated_packet) - len(original_packet))
        size_ratio = size_diff / len(original_packet)
        if size_ratio > 0:
            score += min(0.2, size_ratio)  # Max 0.2 for size changes
        max_score += 0.2
        
        # Content difference component
        if len(obfuscated_packet) >= len(original_packet):
            common_bytes = sum(
                1 for i in range(len(original_packet))
                if original_packet[i] == obfuscated_packet[i]
            )
            content_diff = 1 - (common_bytes / len(original_packet))
            score += content_diff * 0.4  # Max 0.4 for content changes
        max_score += 0.4
        
        # Technique diversity component
        techniques_count = len(self.stats['techniques_applied'])
        score += min(0.4, techniques_count * 0.1)  # Max 0.4 for technique diversity
        max_score += 0.4
        
        return score / max_score if max_score > 0 else 0.0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get obfuscation engine statistics."""
        return {
            'config': {
                'stealth_level': self.config.stealth_level.value,
                'enabled_types': [t.value for t in self.config.enabled_types],
                'timing_jitter': self.config.timing_jitter_enabled,
                'protocol_masking': self.config.protocol_masking
            },
            'statistics': self.stats.copy()
        }


# =============================================================================
# BACKWARD COMPATIBLE FUNCTIONS
# =============================================================================

def apply_timing_jitter(delay_range: Tuple[float, float] = (0.1, 0.5)) -> None:
    """
    Apply timing jitter (backward compatible).
    
    Args:
        delay_range: Range of delay values in seconds
    """
    delay = random.uniform(delay_range[0], delay_range[1])
    time.sleep(delay)


def randomize_packet(packet: bytes) -> bytes:
    """
    Randomize packet bytes (backward compatible).
    
    Args:
        packet: Original packet bytes
        
    Returns:
        Randomized packet bytes
    """
    if not packet:
        return packet
    
    arr = bytearray(packet)
    random.shuffle(arr)
    return bytes(arr)


def time_jitter(delay_range: Tuple[float, float] = (0.1, 0.5)) -> None:
    """
    Backward compatible time jitter function.
    
    Args:
        delay_range: Range of delay values
    """
    apply_timing_jitter(delay_range)


# Factory functions for easy creation
def create_obfuscation_engine(stealth_level: str = "medium", **kwargs: Any) -> ObfuscationEngine:
    """
    Create obfuscation engine with automatic configuration.
    
    Args:
        stealth_level: Stealth level ('none', 'low', 'medium', 'high', 'maximum')
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured obfuscation engine
    """
    stealth_map = {
        'none': StealthLevel.NONE,
        'low': StealthLevel.LOW,
        'medium': StealthLevel.MEDIUM,
        'high': StealthLevel.HIGH,
        'maximum': StealthLevel.MAXIMUM
    }
    
    stealth_enum = stealth_map.get(stealth_level.lower(), StealthLevel.MEDIUM)
    
    # Auto-configure based on stealth level
    if stealth_enum == StealthLevel.NONE:
        enabled_types = []
    elif stealth_enum == StealthLevel.LOW:
        enabled_types = [ObfuscationType.TIMING]
    elif stealth_enum == StealthLevel.MEDIUM:
        enabled_types = [ObfuscationType.TIMING, ObfuscationType.DATA]
    elif stealth_enum == StealthLevel.HIGH:
        enabled_types = [
            ObfuscationType.TIMING, ObfuscationType.DATA,
            ObfuscationType.PROTOCOL, ObfuscationType.FREQUENCY
        ]
    else:  # MAXIMUM
        enabled_types = [
            ObfuscationType.TIMING, ObfuscationType.DATA,
            ObfuscationType.PROTOCOL, ObfuscationType.FREQUENCY,
            ObfuscationType.TRAFFIC, ObfuscationType.STEALTH
        ]
    
    config = ObfuscationConfig(
        stealth_level=stealth_enum,
        enabled_types=enabled_types,
        **kwargs
    )
    
    return ObfuscationEngine(config)


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    print("=== Obfuscation Engine Demo ===")
    
    try:
        # Test data
        test_packet = b"Hello Obfuscation World! This is a test packet for demonstration."
        print(f"Original packet: {len(test_packet)} bytes")
        print(f"Original hex: {test_packet[:16].hex()}...")
        
        # Test different stealth levels
        stealth_levels = ['low', 'medium', 'high', 'maximum']
        
        for level in stealth_levels:
            print(f"\n=== Testing {level.upper()} stealth level ===")
            
            obfuscator = create_obfuscation_engine(stealth_level=level)
            obfuscated = obfuscator.obfuscate_packet(test_packet)
            
            print(f"Obfuscated: {len(obfuscated)} bytes")
            print(f"Obfuscated hex: {obfuscated[:16].hex()}...")
            
            # Calculate stealth score
            score = obfuscator.calculate_stealth_score(test_packet, obfuscated)
            print(f"Stealth score: {score:.3f}")
            
            # Show statistics
            stats = obfuscator.get_statistics()
            techniques = stats['statistics']['techniques_applied']
            print(f"Techniques applied: {list(techniques.keys())}")
        
        # Test timing obfuscation
        print(f"\n=== Testing Timing Obfuscation ===")
        
        timing_obfuscator = create_obfuscation_engine(stealth_level='high')
        
        print("Applying timing jitter (watch delays):")
        for i in range(5):
            start_time = time.time()
            delay = timing_obfuscator.apply_timing_obfuscation(0.1)
            time.sleep(delay)
            actual_delay = time.time() - start_time
            print(f"  Iteration {i+1}: {actual_delay:.3f}s (requested: {delay:.3f}s)")
        
        # Test backward compatibility
        print(f"\n=== Testing Backward Compatibility ===")
        
        # Original functions
        randomized = randomize_packet(test_packet)
        print(f"Randomized packet: {len(randomized)} bytes")
        print(f"Different from original: {randomized != test_packet}")
        
        print("Applying timing jitter...")
        start_time = time.time()
        apply_timing_jitter((0.05, 0.15))
        jitter_time = time.time() - start_time
        print(f"Jitter delay: {jitter_time:.3f}s")
        
        # Test dummy traffic generation
        print(f"\n=== Testing Dummy Traffic ===")
        
        traffic_obfuscator = create_obfuscation_engine(
            stealth_level='high',
            dummy_packets=True
        )
        
        dummy_packets = traffic_obfuscator.generate_dummy_traffic(5, (32, 64))
        print(f"Generated {len(dummy_packets)} dummy packets:")
        
        for i, dummy in enumerate(dummy_packets):
            print(f"  Dummy {i+1}: {len(dummy)} bytes, hex: {dummy[:8].hex()}...")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()