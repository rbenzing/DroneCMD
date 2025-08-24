#!/usr/bin/env python3
"""
Enhanced Packet Parsing System

This module provides comprehensive packet parsing capabilities for drone communication
protocols. It consolidates the original packet_parser.py functionality while adding
advanced features for protocol detection, parsing, and analysis.

Key Features:
- Multi-protocol support (MAVLink, DJI, Parrot, etc.)
- Advanced packet detection and segmentation
- Integration with enhanced demodulation and classification
- Plugin-based protocol architecture
- Real-time parsing with quality metrics
- Backward compatibility with simple interfaces

Integration Points:
- Enhanced Demodulation System (for signal conditioning)
- Enhanced Protocol Classifier (for protocol identification)
- Enhanced Signal Processing (for packet detection)
- Plugin Registry (for protocol-specific parsers)
"""

from __future__ import annotations

import logging
import time
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import numpy.typing as npt

# Integration with enhanced modules
try:
    from .signal_processing import detect_packets, find_preamble, SignalProcessor
    from .classification import EnhancedProtocolClassifier, ClassificationResult
    from .demodulation import DemodulationEngine, DemodConfig, ModulationScheme
    ENHANCED_MODULES_AVAILABLE = True
except ImportError:
    ENHANCED_MODULES_AVAILABLE = False
    warnings.warn("Enhanced modules not available, using fallback implementations")

# Configure module logger
logger = logging.getLogger(__name__)

# Type aliases
IQSamples = npt.NDArray[np.complex64]
PacketBytes = bytes
ProtocolName = str
ConfidenceScore = float
ParsedData = Dict[str, Any]


class DroneProtocol(Enum):
    """Supported drone communication protocols."""
    
    MAVLINK_V1 = "mavlink_v1"
    MAVLINK_V2 = "mavlink_v2"
    DJI_LEGACY = "dji_legacy"
    DJI_OCUSYNC = "dji_ocusync"
    PARROT = "parrot"
    SKYDIO = "skydio"
    AUTEL = "autel"
    YUNEEC = "yuneec"
    SKYDRONE = "skydrone"
    GENERIC = "generic"
    UNKNOWN = "unknown"
    
    @property
    def sync_patterns(self) -> List[bytes]:
        """Get known sync/magic patterns for this protocol."""
        patterns = {
            self.MAVLINK_V1: [b'\xFE'],
            self.MAVLINK_V2: [b'\xFD'],
            self.DJI_LEGACY: [b'\x55\xAA'],
            self.DJI_OCUSYNC: [b'\x55\xAA', b'\xAA\x55'],
            self.PARROT: [b'\x42', b'\x43'],
            self.SKYDIO: [b'\x5A\x5A'],
            self.AUTEL: [b'\xAB\xCD'],
            self.YUNEEC: [b'\xEF\x12'],
            self.SKYDRONE: [b'\x12\x34']
        }
        return patterns.get(self, [])
    
    @property
    def expected_packet_sizes(self) -> Tuple[int, int]:
        """Get (min_size, max_size) for this protocol."""
        sizes = {
            self.MAVLINK_V1: (8, 263),      # Header + payload + checksum
            self.MAVLINK_V2: (12, 280),     # Extended header + payload + checksum
            self.DJI_LEGACY: (10, 1024),    # Variable size DJI packets
            self.DJI_OCUSYNC: (16, 2048),   # Larger OcuSync packets
            self.PARROT: (8, 512),          # Parrot ARSDK packets
            self.SKYDIO: (12, 256),         # Skydio protocol packets
            self.AUTEL: (10, 512),          # Autel packets
            self.YUNEEC: (8, 256),          # Yuneec packets
            self.SKYDRONE: (8, 128)         # SkyDrone packets
        }
        return sizes.get(self, (8, 512))


@dataclass
class PacketResult:
    """
    Results from packet parsing operations.
    
    Contains parsed packet data, protocol information, and quality metrics.
    """
    
    # Packet identification
    protocol: DroneProtocol = DroneProtocol.UNKNOWN
    confidence: ConfidenceScore = 0.0
    packet_bytes: PacketBytes = b""
    packet_index: int = 0
    
    # Parsing results
    parsed_data: Optional[ParsedData] = None
    header_valid: bool = False
    checksum_valid: bool = False
    payload_length: int = 0
    
    # Protocol-specific fields
    message_id: Optional[int] = None
    sequence_number: Optional[int] = None
    system_id: Optional[int] = None
    component_id: Optional[int] = None
    
    # Signal quality (if available)
    signal_power_dbfs: Optional[float] = None
    snr_db: Optional[float] = None
    evm_percent: Optional[float] = None
    
    # Processing metadata
    processing_time_ms: float = 0.0
    parse_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Raw data
    raw_header: Optional[bytes] = None
    raw_payload: Optional[bytes] = None
    raw_checksum: Optional[bytes] = None
    
    @property
    def is_valid(self) -> bool:
        """Check if packet parsing was successful."""
        return (self.protocol != DroneProtocol.UNKNOWN and 
                self.confidence > 0.5 and
                len(self.parse_errors) == 0)
    
    @property
    def total_length(self) -> int:
        """Get total packet length in bytes."""
        return len(self.packet_bytes)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'protocol': self.protocol.value,
            'confidence': self.confidence,
            'packet_length': len(self.packet_bytes),
            'packet_index': self.packet_index,
            'parsed_data': self.parsed_data,
            'header_valid': self.header_valid,
            'checksum_valid': self.checksum_valid,
            'payload_length': self.payload_length,
            'message_id': self.message_id,
            'sequence_number': self.sequence_number,
            'system_id': self.system_id,
            'component_id': self.component_id,
            'signal_power_dbfs': self.signal_power_dbfs,
            'snr_db': self.snr_db,
            'evm_percent': self.evm_percent,
            'processing_time_ms': self.processing_time_ms,
            'is_valid': self.is_valid,
            'total_length': self.total_length,
            'parse_errors': self.parse_errors,
            'warnings': self.warnings
        }


class ProtocolParserBase(ABC):
    """
    Abstract base class for protocol-specific parsers.
    
    Defines the interface that all protocol parsers must implement.
    """
    
    @abstractmethod
    def detect(self, packet_bytes: PacketBytes) -> Tuple[bool, ConfidenceScore]:
        """
        Detect if packet matches this protocol.
        
        Args:
            packet_bytes: Raw packet bytes
            
        Returns:
            Tuple of (is_match, confidence_score)
        """
        pass
    
    @abstractmethod
    def parse(self, packet_bytes: PacketBytes) -> PacketResult:
        """
        Parse packet according to protocol specification.
        
        Args:
            packet_bytes: Raw packet bytes
            
        Returns:
            Parsed packet result
        """
        pass
    
    @property
    @abstractmethod
    def protocol_name(self) -> DroneProtocol:
        """Get the protocol this parser handles."""
        pass
    
    @property
    @abstractmethod
    def supported_versions(self) -> List[str]:
        """Get list of supported protocol versions."""
        pass


class MAVLinkParser(ProtocolParserBase):
    """
    MAVLink protocol parser supporting v1.0 and v2.0.
    
    Implements parsing for the MAVLink micro air vehicle communication protocol
    used by ArduPilot, PX4, and other flight control systems.
    """
    
    def __init__(self) -> None:
        """Initialize MAVLink parser."""
        self.crc_table = self._generate_crc_table()
    
    @property
    def protocol_name(self) -> DroneProtocol:
        """Get protocol name."""
        return DroneProtocol.MAVLINK_V1  # Will be updated based on detection
    
    @property
    def supported_versions(self) -> List[str]:
        """Get supported versions."""
        return ["1.0", "2.0"]
    
    def detect(self, packet_bytes: PacketBytes) -> Tuple[bool, ConfidenceScore]:
        """Detect MAVLink packet."""
        if len(packet_bytes) < 8:
            return False, 0.0
        
        # Check magic bytes
        if packet_bytes[0] == 0xFE:  # MAVLink v1
            if len(packet_bytes) >= 8:
                payload_len = packet_bytes[1]
                expected_len = 8 + payload_len
                if len(packet_bytes) >= expected_len:
                    return True, 0.9
        elif packet_bytes[0] == 0xFD:  # MAVLink v2
            if len(packet_bytes) >= 12:
                payload_len = packet_bytes[1]
                expected_len = 12 + payload_len
                if len(packet_bytes) >= expected_len:
                    return True, 0.95
        
        return False, 0.0
    
    def parse(self, packet_bytes: PacketBytes) -> PacketResult:
        """Parse MAVLink packet."""
        start_time = time.time()
        
        if packet_bytes[0] == 0xFE:
            result = self._parse_mavlink_v1(packet_bytes)
        elif packet_bytes[0] == 0xFD:
            result = self._parse_mavlink_v2(packet_bytes)
        else:
            result = PacketResult(
                protocol=DroneProtocol.UNKNOWN,
                packet_bytes=packet_bytes,
                parse_errors=["Invalid MAVLink magic byte"]
            )
        
        result.processing_time_ms = (time.time() - start_time) * 1000
        return result
    
    def _parse_mavlink_v1(self, packet_bytes: PacketBytes) -> PacketResult:
        """Parse MAVLink v1.0 packet."""
        if len(packet_bytes) < 8:
            return PacketResult(
                protocol=DroneProtocol.MAVLINK_V1,
                packet_bytes=packet_bytes,
                parse_errors=["Packet too short for MAVLink v1"]
            )
        
        # Parse header
        magic = packet_bytes[0]          # 0xFE
        payload_len = packet_bytes[1]    # Payload length
        seq = packet_bytes[2]            # Sequence number
        sys_id = packet_bytes[3]         # System ID
        comp_id = packet_bytes[4]        # Component ID
        msg_id = packet_bytes[5]         # Message ID
        
        expected_total_len = 8 + payload_len
        if len(packet_bytes) < expected_total_len:
            return PacketResult(
                protocol=DroneProtocol.MAVLINK_V1,
                packet_bytes=packet_bytes,
                parse_errors=[f"Incomplete packet: {len(packet_bytes)} < {expected_total_len}"]
            )
        
        # Extract payload and checksum
        payload = packet_bytes[6:6+payload_len]
        checksum = packet_bytes[6+payload_len:6+payload_len+2]
        
        # Validate checksum
        calculated_crc = self._calculate_crc(packet_bytes[1:6+payload_len])
        received_crc = int.from_bytes(checksum, byteorder='little')
        checksum_valid = calculated_crc == received_crc
        
        # Create parsed data
        parsed_data = {
            'magic': magic,
            'payload_length': payload_len,
            'sequence': seq,
            'system_id': sys_id,
            'component_id': comp_id,
            'message_id': msg_id,
            'payload': payload.hex(),
            'checksum': checksum.hex(),
            'version': '1.0'
        }
        
        return PacketResult(
            protocol=DroneProtocol.MAVLINK_V1,
            confidence=0.9 if checksum_valid else 0.7,
            packet_bytes=packet_bytes[:expected_total_len],
            parsed_data=parsed_data,
            header_valid=True,
            checksum_valid=checksum_valid,
            payload_length=payload_len,
            message_id=msg_id,
            sequence_number=seq,
            system_id=sys_id,
            component_id=comp_id,
            raw_header=packet_bytes[:6],
            raw_payload=payload,
            raw_checksum=checksum
        )
    
    def _parse_mavlink_v2(self, packet_bytes: PacketBytes) -> PacketResult:
        """Parse MAVLink v2.0 packet."""
        if len(packet_bytes) < 12:
            return PacketResult(
                protocol=DroneProtocol.MAVLINK_V2,
                packet_bytes=packet_bytes,
                parse_errors=["Packet too short for MAVLink v2"]
            )
        
        # Parse header
        magic = packet_bytes[0]          # 0xFD
        payload_len = packet_bytes[1]    # Payload length
        incompat_flags = packet_bytes[2] # Incompatible flags
        compat_flags = packet_bytes[3]   # Compatible flags
        seq = packet_bytes[4]            # Sequence number
        sys_id = packet_bytes[5]         # System ID
        comp_id = packet_bytes[6]        # Component ID
        msg_id = int.from_bytes(packet_bytes[7:10], byteorder='little')  # Message ID (24-bit)
        
        expected_total_len = 12 + payload_len
        if len(packet_bytes) < expected_total_len:
            return PacketResult(
                protocol=DroneProtocol.MAVLINK_V2,
                packet_bytes=packet_bytes,
                parse_errors=[f"Incomplete packet: {len(packet_bytes)} < {expected_total_len}"]
            )
        
        # Extract payload and checksum
        payload = packet_bytes[10:10+payload_len]
        checksum = packet_bytes[10+payload_len:10+payload_len+2]
        
        # Validate checksum (simplified - real implementation would include CRC_EXTRA)
        calculated_crc = self._calculate_crc(packet_bytes[1:10+payload_len])
        received_crc = int.from_bytes(checksum, byteorder='little')
        checksum_valid = calculated_crc == received_crc
        
        # Create parsed data
        parsed_data = {
            'magic': magic,
            'payload_length': payload_len,
            'incompat_flags': incompat_flags,
            'compat_flags': compat_flags,
            'sequence': seq,
            'system_id': sys_id,
            'component_id': comp_id,
            'message_id': msg_id,
            'payload': payload.hex(),
            'checksum': checksum.hex(),
            'version': '2.0'
        }
        
        return PacketResult(
            protocol=DroneProtocol.MAVLINK_V2,
            confidence=0.95 if checksum_valid else 0.8,
            packet_bytes=packet_bytes[:expected_total_len],
            parsed_data=parsed_data,
            header_valid=True,
            checksum_valid=checksum_valid,
            payload_length=payload_len,
            message_id=msg_id,
            sequence_number=seq,
            system_id=sys_id,
            component_id=comp_id,
            raw_header=packet_bytes[:10],
            raw_payload=payload,
            raw_checksum=checksum
        )
    
    def _generate_crc_table(self) -> List[int]:
        """Generate CRC lookup table for MAVLink."""
        crc_table = []
        for i in range(256):
            crc = i
            for _ in range(8):
                if crc & 1:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc >>= 1
            crc_table.append(crc)
        return crc_table
    
    def _calculate_crc(self, data: bytes) -> int:
        """Calculate CRC-16-CCITT for MAVLink."""
        crc = 0xFFFF
        for byte in data:
            crc = ((crc >> 8) ^ self.crc_table[(crc ^ byte) & 0xFF]) & 0xFFFF
        return crc


class DJIParser(ProtocolParserBase):
    """
    DJI protocol parser for legacy and OcuSync protocols.
    
    Handles various DJI drone communication protocols including
    legacy formats and newer OcuSync systems.
    """
    
    @property
    def protocol_name(self) -> DroneProtocol:
        """Get protocol name."""
        return DroneProtocol.DJI_LEGACY
    
    @property
    def supported_versions(self) -> List[str]:
        """Get supported versions."""
        return ["Legacy", "OcuSync 1.0", "OcuSync 2.0"]
    
    def detect(self, packet_bytes: PacketBytes) -> Tuple[bool, ConfidenceScore]:
        """Detect DJI packet."""
        if len(packet_bytes) < 4:
            return False, 0.0
        
        # Check for DJI sync patterns
        if packet_bytes[:2] == b'\x55\xAA':
            if len(packet_bytes) >= 10:  # Minimum DJI packet size
                return True, 0.85
        elif packet_bytes[:2] == b'\xAA\x55':
            if len(packet_bytes) >= 16:  # OcuSync packets are typically larger
                return True, 0.9
        
        return False, 0.0
    
    def parse(self, packet_bytes: PacketBytes) -> PacketResult:
        """Parse DJI packet."""
        start_time = time.time()
        
        if packet_bytes[:2] == b'\x55\xAA':
            result = self._parse_dji_legacy(packet_bytes)
        elif packet_bytes[:2] == b'\xAA\x55':
            result = self._parse_dji_ocusync(packet_bytes)
        else:
            result = PacketResult(
                protocol=DroneProtocol.UNKNOWN,
                packet_bytes=packet_bytes,
                parse_errors=["Invalid DJI sync pattern"]
            )
        
        result.processing_time_ms = (time.time() - start_time) * 1000
        return result
    
    def _parse_dji_legacy(self, packet_bytes: PacketBytes) -> PacketResult:
        """Parse DJI legacy format packet."""
        if len(packet_bytes) < 10:
            return PacketResult(
                protocol=DroneProtocol.DJI_LEGACY,
                packet_bytes=packet_bytes,
                parse_errors=["Packet too short for DJI legacy"]
            )
        
        # Parse basic DJI header
        sync = packet_bytes[:2]          # 0x55AA
        length = int.from_bytes(packet_bytes[2:4], byteorder='little')
        packet_type = packet_bytes[4]
        source = packet_bytes[5]
        target = packet_bytes[6]
        seq = packet_bytes[7]
        
        # Validate length
        if len(packet_bytes) < length:
            return PacketResult(
                protocol=DroneProtocol.DJI_LEGACY,
                packet_bytes=packet_bytes,
                parse_errors=[f"Incomplete packet: {len(packet_bytes)} < {length}"]
            )
        
        # Extract payload (simplified)
        header_size = 8
        payload_size = length - header_size - 2  # Subtract header and checksum
        if payload_size > 0:
            payload = packet_bytes[header_size:header_size+payload_size]
        else:
            payload = b""
        
        # Extract checksum
        checksum = packet_bytes[length-2:length]
        
        # Basic checksum validation (simplified)
        calculated_sum = sum(packet_bytes[:length-2]) & 0xFFFF
        received_sum = int.from_bytes(checksum, byteorder='little')
        checksum_valid = calculated_sum == received_sum
        
        parsed_data = {
            'sync': sync.hex(),
            'length': length,
            'packet_type': packet_type,
            'source': source,
            'target': target,
            'sequence': seq,
            'payload': payload.hex() if payload else "",
            'checksum': checksum.hex(),
            'version': 'Legacy'
        }
        
        return PacketResult(
            protocol=DroneProtocol.DJI_LEGACY,
            confidence=0.85 if checksum_valid else 0.6,
            packet_bytes=packet_bytes[:length],
            parsed_data=parsed_data,
            header_valid=True,
            checksum_valid=checksum_valid,
            payload_length=len(payload),
            sequence_number=seq,
            raw_header=packet_bytes[:header_size],
            raw_payload=payload,
            raw_checksum=checksum
        )
    
    def _parse_dji_ocusync(self, packet_bytes: PacketBytes) -> PacketResult:
        """Parse DJI OcuSync packet (simplified implementation)."""
        # This is a simplified implementation - real OcuSync is more complex
        if len(packet_bytes) < 16:
            return PacketResult(
                protocol=DroneProtocol.DJI_OCUSYNC,
                packet_bytes=packet_bytes,
                parse_errors=["Packet too short for DJI OcuSync"]
            )
        
        parsed_data = {
            'sync': packet_bytes[:2].hex(),
            'header': packet_bytes[2:16].hex(),
            'version': 'OcuSync',
            'note': 'Simplified parsing - full OcuSync protocol is proprietary'
        }
        
        return PacketResult(
            protocol=DroneProtocol.DJI_OCUSYNC,
            confidence=0.7,  # Lower confidence due to simplified parsing
            packet_bytes=packet_bytes,
            parsed_data=parsed_data,
            header_valid=True,
            checksum_valid=False,  # Cannot validate without full protocol
            payload_length=len(packet_bytes) - 16,
            raw_header=packet_bytes[:16]
        )


class GenericParser(ProtocolParserBase):
    """
    Generic protocol parser for unknown or custom protocols.
    
    Provides basic packet structure analysis when specific
    protocol parsers cannot identify the packet.
    """
    
    @property
    def protocol_name(self) -> DroneProtocol:
        """Get protocol name."""
        return DroneProtocol.GENERIC
    
    @property
    def supported_versions(self) -> List[str]:
        """Get supported versions."""
        return ["Generic"]
    
    def detect(self, packet_bytes: PacketBytes) -> Tuple[bool, ConfidenceScore]:
        """Generic detection - always matches with low confidence."""
        if len(packet_bytes) >= 8:
            return True, 0.1
        return False, 0.0
    
    def parse(self, packet_bytes: PacketBytes) -> PacketResult:
        """Parse packet using generic analysis."""
        start_time = time.time()
        
        # Basic analysis
        entropy = self._calculate_entropy(packet_bytes)
        repeating_patterns = self._find_repeating_patterns(packet_bytes)
        
        parsed_data = {
            'length': len(packet_bytes),
            'entropy': entropy,
            'first_bytes': packet_bytes[:min(8, len(packet_bytes))].hex(),
            'last_bytes': packet_bytes[-min(8, len(packet_bytes)):].hex(),
            'repeating_patterns': repeating_patterns,
            'analysis': 'Generic packet analysis'
        }
        
        result = PacketResult(
            protocol=DroneProtocol.GENERIC,
            confidence=0.3,
            packet_bytes=packet_bytes,
            parsed_data=parsed_data,
            header_valid=False,
            checksum_valid=False,
            payload_length=len(packet_bytes)
        )
        
        result.processing_time_ms = (time.time() - start_time) * 1000
        return result
    
    def _calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of data."""
        if len(data) == 0:
            return 0.0
        
        byte_counts = defaultdict(int)
        for byte in data:
            byte_counts[byte] += 1
        
        entropy = 0.0
        data_len = len(data)
        for count in byte_counts.values():
            p = count / data_len
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy
    
    def _find_repeating_patterns(self, data: bytes, min_length: int = 2) -> List[str]:
        """Find repeating byte patterns."""
        patterns = []
        
        for length in range(min_length, min(8, len(data) // 2)):
            for start in range(len(data) - length):
                pattern = data[start:start + length]
                # Look for this pattern elsewhere
                count = 0
                pos = 0
                while pos < len(data) - length:
                    if data[pos:pos + length] == pattern:
                        count += 1
                    pos += 1
                
                if count >= 3:  # Pattern appears at least 3 times
                    patterns.append(f"{pattern.hex()} (appears {count} times)")
                    break  # Don't look for longer patterns starting at same position
        
        return patterns[:5]  # Return up to 5 patterns


class EnhancedPacketParser:
    """
    Enhanced packet parser with multi-protocol support and advanced features.
    
    Provides comprehensive packet parsing capabilities with protocol detection,
    quality analysis, and integration with enhanced classification systems.
    """
    
    def __init__(
        self,
        enable_classification: bool = True,
        enable_demodulation_integration: bool = True,
        confidence_threshold: float = 0.5
    ) -> None:
        """
        Initialize enhanced packet parser.
        
        Args:
            enable_classification: Use enhanced protocol classification
            enable_demodulation_integration: Integrate with demodulation system
            confidence_threshold: Minimum confidence for valid parsing
        """
        self.enable_classification = enable_classification
        self.enable_demodulation_integration = enable_demodulation_integration
        self.confidence_threshold = confidence_threshold
        
        # Initialize parsers
        self.parsers = {
            DroneProtocol.MAVLINK_V1: MAVLinkParser(),
            DroneProtocol.MAVLINK_V2: MAVLinkParser(),
            DroneProtocol.DJI_LEGACY: DJIParser(),
            DroneProtocol.DJI_OCUSYNC: DJIParser(),
            DroneProtocol.GENERIC: GenericParser()
        }
        
        # Enhanced components
        self.classifier = None
        self.signal_processor = None
        
        if ENHANCED_MODULES_AVAILABLE:
            if enable_classification:
                try:
                    self.classifier = EnhancedProtocolClassifier()
                except Exception as e:
                    logger.warning(f"Failed to initialize classifier: {e}")
            
            if enable_demodulation_integration:
                try:
                    self.signal_processor = SignalProcessor()
                except Exception as e:
                    logger.warning(f"Failed to initialize signal processor: {e}")
        
        # Statistics
        self.stats = {
            'packets_parsed': 0,
            'protocols_detected': defaultdict(int),
            'total_processing_time': 0.0,
            'parsing_errors': 0
        }
        
        logger.info(f"Initialized enhanced packet parser with {len(self.parsers)} protocol parsers")
    
    def parse_packet(
        self,
        packet_bytes: PacketBytes,
        signal_metrics: Optional[Dict[str, float]] = None
    ) -> PacketResult:
        """
        Parse a single packet with comprehensive analysis.
        
        Args:
            packet_bytes: Raw packet bytes
            signal_metrics: Optional signal quality metrics
            
        Returns:
            Parsed packet result
        """
        start_time = time.time()
        
        try:
            # Input validation
            if len(packet_bytes) == 0:
                return PacketResult(
                    protocol=DroneProtocol.UNKNOWN,
                    packet_bytes=packet_bytes,
                    parse_errors=["Empty packet"]
                )
            
            # Protocol detection
            best_parser = None
            best_confidence = 0.0
            best_protocol = DroneProtocol.UNKNOWN
            
            # Try classification first if available
            if self.classifier is not None:
                try:
                    classification_result = self.classifier.classify(packet_bytes)
                    if hasattr(classification_result, 'predicted_protocol'):
                        protocol_name = classification_result.predicted_protocol.lower()
                        confidence = classification_result.confidence
                        
                        # Map classification to protocol enum
                        for protocol in DroneProtocol:
                            if protocol_name in protocol.value:
                                best_protocol = protocol
                                best_confidence = confidence
                                break
                        
                        logger.debug(f"Classifier detected: {protocol_name} ({confidence:.3f})")
                        
                except Exception as e:
                    logger.debug(f"Classification failed: {e}")
            
            # Try protocol-specific parsers
            parser_results = []
            for protocol, parser in self.parsers.items():
                try:
                    is_match, confidence = parser.detect(packet_bytes)
                    if is_match and confidence > best_confidence:
                        best_parser = parser
                        best_confidence = confidence
                        best_protocol = protocol
                    
                    parser_results.append((protocol, is_match, confidence))
                    
                except Exception as e:
                    logger.debug(f"Parser {protocol.value} detection failed: {e}")
            
            # Use best parser or fallback to generic
            if best_parser is None or best_confidence < self.confidence_threshold:
                best_parser = self.parsers[DroneProtocol.GENERIC]
                best_protocol = DroneProtocol.GENERIC
            
            # Parse packet
            result = best_parser.parse(packet_bytes)
            
            # Add signal metrics if available
            if signal_metrics:
                result.signal_power_dbfs = signal_metrics.get('signal_level_dbfs')
                result.snr_db = signal_metrics.get('snr_db')
                result.evm_percent = signal_metrics.get('evm_percent')
            
            # Update statistics
            self._update_stats(result, time.time() - start_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Packet parsing failed: {e}")
            self.stats['parsing_errors'] += 1
            
            return PacketResult(
                protocol=DroneProtocol.UNKNOWN,
                packet_bytes=packet_bytes,
                parse_errors=[str(e)],
                processing_time_ms=(time.time() - start_time) * 1000
            )
    
    def parse_iq_samples(
        self,
        iq_samples: IQSamples,
        sample_rate: float,
        bitrate: float = 9600,
        modulation: ModulationScheme = ModulationScheme.OOK
    ) -> List[PacketResult]:
        """
        Parse packets from IQ samples with integrated demodulation.
        
        Args:
            iq_samples: Complex IQ samples
            sample_rate: Sample rate in Hz
            bitrate: Expected bit rate
            modulation: Modulation scheme
            
        Returns:
            List of parsed packet results
        """
        if not ENHANCED_MODULES_AVAILABLE or not self.enable_demodulation_integration:
            raise NotImplementedError("Demodulation integration not available")
        
        try:
            # Detect packets in IQ samples
            if self.signal_processor:
                packet_regions = detect_packets(iq_samples, threshold=0.05, min_gap=1000)
            else:
                packet_regions = [(0, len(iq_samples))]  # Fallback
            
            results = []
            
            for i, (start, end) in enumerate(packet_regions):
                packet_iq = iq_samples[start:end]
                
                # Demodulate packet
                try:
                    demod_config = DemodConfig(
                        scheme=modulation,
                        sample_rate_hz=sample_rate,
                        bitrate_bps=bitrate
                    )
                    demod_engine = DemodulationEngine(demod_config)
                    demod_result = demod_engine.demodulate(packet_iq)
                    
                    if demod_result.is_valid and len(demod_result.bits) > 0:
                        # Convert bits to bytes
                        packet_bytes = self._bits_to_bytes(demod_result.bits)
                        
                        # Parse packet
                        signal_metrics = {
                            'signal_level_dbfs': demod_result.signal_power_dbfs,
                            'snr_db': demod_result.snr_db,
                            'evm_percent': demod_result.evm_percent
                        }
                        
                        parse_result = self.parse_packet(packet_bytes, signal_metrics)
                        parse_result.packet_index = i
                        results.append(parse_result)
                    
                except Exception as e:
                    logger.debug(f"Demodulation failed for packet {i}: {e}")
                    continue
            
            logger.info(f"Parsed {len(results)} packets from {len(packet_regions)} detected regions")
            return results
            
        except Exception as e:
            logger.error(f"IQ sample parsing failed: {e}")
            return []
    
    def parse_multiple_packets(
        self,
        packet_list: List[PacketBytes],
        signal_metrics_list: Optional[List[Dict[str, float]]] = None
    ) -> List[PacketResult]:
        """
        Parse multiple packets efficiently.
        
        Args:
            packet_list: List of packet byte arrays
            signal_metrics_list: Optional list of signal metrics per packet
            
        Returns:
            List of parsed packet results
        """
        results = []
        
        for i, packet_bytes in enumerate(packet_list):
            signal_metrics = None
            if signal_metrics_list and i < len(signal_metrics_list):
                signal_metrics = signal_metrics_list[i]
            
            result = self.parse_packet(packet_bytes, signal_metrics)
            result.packet_index = i
            results.append(result)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get parser statistics."""
        stats = self.stats.copy()
        
        if stats['packets_parsed'] > 0:
            stats['average_processing_time_ms'] = (
                (stats['total_processing_time'] / stats['packets_parsed']) * 1000
            )
        else:
            stats['average_processing_time_ms'] = 0.0
        
        stats['protocols_detected'] = dict(stats['protocols_detected'])
        stats['success_rate'] = (
            (stats['packets_parsed'] - stats['parsing_errors']) / 
            max(1, stats['packets_parsed'])
        )
        
        return stats
    
    def _update_stats(self, result: PacketResult, processing_time: float) -> None:
        """Update parser statistics."""
        self.stats['packets_parsed'] += 1
        self.stats['protocols_detected'][result.protocol.value] += 1
        self.stats['total_processing_time'] += processing_time
        
        if not result.is_valid:
            self.stats['parsing_errors'] += 1
    
    def _bits_to_bytes(self, bits: npt.NDArray) -> bytes:
        """Convert bit array to bytes."""
        if len(bits) == 0:
            return b""
        
        # Pad to byte boundary
        if len(bits) % 8 != 0:
            padding = 8 - (len(bits) % 8)
            bits = np.pad(bits, (0, padding), 'constant')
        
        # Pack bits into bytes
        bits_uint8 = bits.astype(np.uint8)
        return np.packbits(bits_uint8).tobytes()


# =============================================================================
# BACKWARD COMPATIBILITY INTERFACE
# =============================================================================

class PacketParser:
    """
    Simplified packet parser for backward compatibility.
    
    Maintains the original interface while using the enhanced parser backend.
    """
    
    def __init__(self, preamble_patterns: Optional[Dict[str, npt.NDArray]] = None) -> None:
        """
        Initialize packet parser (backward compatible).
        
        Args:
            preamble_patterns: Dictionary of {manufacturer: pattern} (deprecated)
        """
        self.preamble_patterns = preamble_patterns or {}
        self._enhanced_parser = EnhancedPacketParser()
        
        # Backward compatibility warning
        if preamble_patterns:
            warnings.warn(
                "preamble_patterns parameter is deprecated. "
                "Use enhanced protocol detection instead.",
                DeprecationWarning
            )
    
    def parse(self, iq_samples: npt.NDArray, sample_rate: int) -> List[Dict[str, Any]]:
        """
        Parse IQ samples (backward compatible interface).
        
        Args:
            iq_samples: IQ samples to parse
            sample_rate: Sample rate in Hz
            
        Returns:
            List of parsing results
        """
        try:
            # Use enhanced parser
            results = self._enhanced_parser.parse_iq_samples(
                iq_samples, sample_rate
            )
            
            # Convert to backward compatible format
            legacy_results = []
            for result in results:
                legacy_result = {
                    "packet_index": result.packet_index,
                    "plugin": result.protocol.value,
                    "decoded": result.parsed_data
                }
                legacy_results.append(legacy_result)
            
            return legacy_results
            
        except Exception as e:
            logger.error(f"Legacy parsing failed: {e}")
            return []
    
    def detect_packets(self, iq_data: npt.NDArray) -> Dict[str, List[int]]:
        """Detect packets using preamble patterns (backward compatible)."""
        detected = {}
        
        if ENHANCED_MODULES_AVAILABLE:
            for manufacturer, pattern in self.preamble_patterns.items():
                try:
                    hits = find_preamble(iq_data, pattern, threshold=0.8)
                    if len(hits) > 0:
                        detected[manufacturer] = hits.tolist()
                except Exception as e:
                    logger.debug(f"Preamble detection failed for {manufacturer}: {e}")
        
        return detected
    
    def extract_packets(
        self,
        iq_data: npt.NDArray,
        start_indices: List[int],
        length: int
    ) -> List[npt.NDArray]:
        """Extract packets at specified indices (backward compatible)."""
        packets = []
        for idx in start_indices:
            if idx + length <= len(iq_data):
                packets.append(iq_data[idx:idx + length])
        return packets


# Factory functions
def create_packet_parser(
    enhanced: bool = True,
    **kwargs: Any
) -> Union[PacketParser, EnhancedPacketParser]:
    """
    Create packet parser instance.
    
    Args:
        enhanced: Use enhanced parser if True, legacy if False
        **kwargs: Additional configuration parameters
        
    Returns:
        Parser instance
    """
    if enhanced:
        return EnhancedPacketParser(**kwargs)
    else:
        return PacketParser(**kwargs)


def create_protocol_parser(protocol: DroneProtocol) -> ProtocolParserBase:
    """
    Create protocol-specific parser.
    
    Args:
        protocol: Protocol to create parser for
        
    Returns:
        Protocol parser instance
    """
    parser_map = {
        DroneProtocol.MAVLINK_V1: MAVLinkParser,
        DroneProtocol.MAVLINK_V2: MAVLinkParser,
        DroneProtocol.DJI_LEGACY: DJIParser,
        DroneProtocol.DJI_OCUSYNC: DJIParser,
        DroneProtocol.GENERIC: GenericParser
    }
    
    parser_class = parser_map.get(protocol, GenericParser)
    return parser_class()


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    print("=== Enhanced Packet Parser Demo ===")
    
    # Create enhanced parser
    parser = EnhancedPacketParser()
    
    # Test MAVLink packet
    mavlink_packet = b'\xFE\x21\x00\x01\x01\x00' + b'\x42' * 33 + b'\x12\x34'
    result = parser.parse_packet(mavlink_packet)
    
    print(f"MAVLink Result:")
    print(f"  Protocol: {result.protocol.value}")
    print(f"  Confidence: {result.confidence:.3f}")
    print(f"  Valid: {result.is_valid}")
    print(f"  Processing time: {result.processing_time_ms:.1f} ms")
    
    # Test DJI packet
    dji_packet = b'\x55\xAA\x27\x10' + b'\x33' * 35 + b'\x56\x78'
    result = parser.parse_packet(dji_packet)
    
    print(f"\nDJI Result:")
    print(f"  Protocol: {result.protocol.value}")
    print(f"  Confidence: {result.confidence:.3f}")
    print(f"  Valid: {result.is_valid}")
    
    # Show statistics
    stats = parser.get_statistics()
    print(f"\nStatistics:")
    print(f"  Packets parsed: {stats['packets_parsed']}")
    print(f"  Success rate: {stats['success_rate']:.1%}")
    print(f"  Protocol distribution: {stats['protocols_detected']}")