#!/usr/bin/env python3
"""
DJI Protocol Plugin

Comprehensive plugin for detecting, parsing, and encoding DJI drone communication protocols.
Supports multiple DJI protocol variants including OcuSync, Lightbridge, and WiFi-based protocols.

Protocol Support:
- DJI OcuSync (2.4GHz and 5.8GHz variants)
- DJI Lightbridge (legacy systems)
- DJI WiFi protocols (Spark, Tello)
- DJI Enhanced WiFi (newer consumer drones)
- DJI SDK communication protocols

Key Features:
- Multi-variant protocol detection
- Packet structure analysis and parsing
- Command/telemetry differentiation
- Encryption detection and handling
- Flight data extraction
- Real-time protocol analysis
"""

from __future__ import annotations

import logging
import struct
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

# Import from core modules and plugin base
try:
    from ...core.signal_processing import SignalProcessor, analyze_signal_quality
    from ..base import BaseProtocolPlugin, ProtocolDetectionResult, ProtocolParseResult
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False
    warnings.warn("Core modules not available, using fallback implementations")

# Configure module logger
logger = logging.getLogger(__name__)

# Type aliases
IQSamples = npt.NDArray[np.complex64]


class DJIProtocolVariant(Enum):
    """DJI protocol variants and versions."""
    
    OCUSYNC_1 = "ocusync_v1"
    OCUSYNC_2 = "ocusync_v2" 
    OCUSYNC_3 = "ocusync_v3"
    LIGHTBRIDGE = "lightbridge"
    WIFI_LEGACY = "wifi_legacy"
    WIFI_ENHANCED = "wifi_enhanced"
    SDK_PROTOCOL = "sdk_protocol"
    UNKNOWN_DJI = "unknown_dji"


class DJIMessageType(Enum):
    """DJI message types."""
    
    TELEMETRY = 0x01
    COMMAND = 0x02
    VIDEO_STREAM = 0x03
    FLIGHT_CONTROL = 0x04
    GIMBAL_CONTROL = 0x05
    CAMERA_CONTROL = 0x06
    BATTERY_INFO = 0x07
    GPS_DATA = 0x08
    STATUS_UPDATE = 0x09
    HEARTBEAT = 0x0A
    HANDSHAKE = 0x0B
    ACKNOWLEDGMENT = 0x0C
    ERROR_REPORT = 0x0D
    FIRMWARE_UPDATE = 0x0E
    UNKNOWN = 0xFF


@dataclass
class DJIProtocolConfig:
    """Configuration for DJI protocol detection and parsing."""
    
    # Detection parameters
    enable_ocusync_detection: bool = True
    enable_lightbridge_detection: bool = True
    enable_wifi_detection: bool = True
    detection_confidence_threshold: float = 0.8
    
    # Protocol-specific settings
    ocusync_frequencies: List[float] = field(default_factory=lambda: [2.4e9, 5.8e9])
    lightbridge_frequencies: List[float] = field(default_factory=lambda: [2.4e9])
    wifi_frequencies: List[float] = field(default_factory=lambda: [2.4e9, 5.8e9])
    
    # Parsing options
    decode_telemetry: bool = True
    decode_commands: bool = True
    extract_flight_data: bool = True
    handle_encryption: bool = False  # Basic encryption detection only
    
    # Output options
    include_raw_data: bool = True
    include_signal_analysis: bool = True
    verbose_parsing: bool = False


@dataclass
class DJIPacketStructure:
    """DJI packet structure information."""
    
    variant: DJIProtocolVariant
    sync_pattern: bytes
    header_length: int
    payload_length: int
    checksum_length: int
    total_length: int
    
    # Protocol-specific fields
    sequence_number: Optional[int] = None
    message_type: Optional[DJIMessageType] = None
    source_id: Optional[int] = None
    target_id: Optional[int] = None
    encryption_flag: bool = False
    
    # Quality metrics
    structure_confidence: float = 0.0
    checksum_valid: bool = False


@dataclass
class DJIFlightData:
    """Extracted DJI flight data."""
    
    # Position and orientation
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[float] = None
    height_above_takeoff: Optional[float] = None
    
    # Velocity
    velocity_x: Optional[float] = None
    velocity_y: Optional[float] = None
    velocity_z: Optional[float] = None
    
    # Attitude
    pitch: Optional[float] = None
    roll: Optional[float] = None
    yaw: Optional[float] = None
    
    # System status
    battery_percentage: Optional[int] = None
    signal_strength: Optional[int] = None
    gps_satellite_count: Optional[int] = None
    flight_mode: Optional[str] = None
    
    # Timestamps
    flight_time: Optional[int] = None  # Seconds since takeoff
    timestamp: Optional[int] = None     # System timestamp


class DJIProtocolPlugin(BaseProtocolPlugin if CORE_AVAILABLE else object):
    """
    DJI protocol plugin for comprehensive DJI drone communication analysis.
    
    This plugin provides detection, parsing, and encoding capabilities for
    various DJI communication protocols used in their drone systems.
    
    Example:
        >>> plugin = DJIProtocolPlugin()
        >>> result = plugin.detect(iq_samples)
        >>> if result.detected:
        ...     parsed = plugin.parse_packet(packet_bytes)
        ...     flight_data = parsed.parsed_data.get('flight_data')
    """
    
    def __init__(self, config: Optional[DJIProtocolConfig] = None) -> None:
        """
        Initialize DJI protocol plugin.
        
        Args:
            config: Plugin configuration (uses defaults if None)
        """
        self.config = config or DJIProtocolConfig()
        
        # Initialize signal processor
        if CORE_AVAILABLE:
            self.signal_processor = SignalProcessor()
        else:
            self.signal_processor = None
        
        # DJI protocol signatures and patterns
        self.protocol_signatures = {
            # OcuSync patterns
            DJIProtocolVariant.OCUSYNC_1: {
                'sync_patterns': [b'\x55\xAA', b'\xAA\x55'],
                'header_length': 10,
                'min_packet_size': 15,
                'max_packet_size': 2048,
                'frequency_bands': [2.4e9, 5.8e9]
            },
            DJIProtocolVariant.OCUSYNC_2: {
                'sync_patterns': [b'\x55\xAA\x04', b'\xAA\x55\x04'],
                'header_length': 12,
                'min_packet_size': 18,
                'max_packet_size': 2048,
                'frequency_bands': [2.4e9, 5.8e9]
            },
            DJIProtocolVariant.OCUSYNC_3: {
                'sync_patterns': [b'\x55\xAA\x06', b'\xAA\x55\x06'],
                'header_length': 14,
                'min_packet_size': 20,
                'max_packet_size': 4096,
                'frequency_bands': [2.4e9, 5.8e9]
            },
            # Lightbridge patterns
            DJIProtocolVariant.LIGHTBRIDGE: {
                'sync_patterns': [b'\x55\xAA\x02', b'\xFE\xDC'],
                'header_length': 8,
                'min_packet_size': 12,
                'max_packet_size': 1024,
                'frequency_bands': [2.4e9]
            },
            # WiFi-based patterns
            DJIProtocolVariant.WIFI_LEGACY: {
                'sync_patterns': [b'\x55\xAA\x01', b'\xCC\x33'],
                'header_length': 6,
                'min_packet_size': 10,
                'max_packet_size': 1500,
                'frequency_bands': [2.4e9]
            },
            DJIProtocolVariant.WIFI_ENHANCED: {
                'sync_patterns': [b'\x55\xAA\x05', b'\xDD\x22'],
                'header_length': 8,
                'min_packet_size': 12,
                'max_packet_size': 1500,
                'frequency_bands': [2.4e9, 5.8e9]
            }
        }
        
        # Message type patterns
        self.message_type_patterns = {
            DJIMessageType.TELEMETRY: [0x01, 0x81],
            DJIMessageType.COMMAND: [0x02, 0x82],
            DJIMessageType.FLIGHT_CONTROL: [0x04, 0x84],
            DJIMessageType.HEARTBEAT: [0x0A, 0x8A],
            DJIMessageType.GPS_DATA: [0x08, 0x88]
        }
        
        # Statistics
        self.detection_stats = {
            'packets_analyzed': 0,
            'dji_packets_detected': 0,
            'variant_detections': {variant.value: 0 for variant in DJIProtocolVariant},
            'message_type_counts': {msg_type.value: 0 for msg_type in DJIMessageType},
            'avg_confidence': 0.0,
            'flight_data_extracted': 0
        }
        
        logger.info("Initialized DJI protocol plugin")
    
    def get_name(self) -> str:
        """Get plugin name."""
        return "DJI Protocol"
    
    def get_version(self) -> str:
        """Get plugin version."""
        return "2.0.0"
    
    def get_supported_protocols(self) -> List[str]:
        """Get list of supported DJI protocol variants."""
        return [variant.value for variant in DJIProtocolVariant]
    
    def detect(
        self,
        iq_samples: IQSamples,
        signal_metadata: Optional[Dict[str, Any]] = None
    ) -> ProtocolDetectionResult:
        """
        Detect DJI protocols in IQ samples.
        
        Args:
            iq_samples: Complex IQ samples to analyze
            signal_metadata: Optional signal processing metadata
            
        Returns:
            Detection result with confidence and DJI-specific metadata
        """
        try:
            if len(iq_samples) < 1000:  # Need sufficient samples
                return ProtocolDetectionResult(
                    detected=False,
                    confidence=0.0,
                    protocol_name="dji",
                    metadata={"error": "Insufficient samples for DJI detection"}
                )
            
            # Signal analysis
            signal_analysis = {}
            if self.config.include_signal_analysis and CORE_AVAILABLE:
                signal_analysis = analyze_signal_quality(iq_samples)
            
            # Convert to packet data for pattern analysis
            packet_candidates = self._extract_packet_candidates(iq_samples)
            
            if not packet_candidates:
                return ProtocolDetectionResult(
                    detected=False,
                    confidence=0.2,
                    protocol_name="dji",
                    metadata={
                        "signal_analysis": signal_analysis,
                        "reason": "No valid packet candidates found"
                    }
                )
            
            # Analyze each candidate packet
            best_detection = None
            best_confidence = 0.0
            
            for packet_data in packet_candidates:
                detection = self._analyze_dji_packet(packet_data, signal_analysis)
                if detection['confidence'] > best_confidence:
                    best_confidence = detection['confidence']
                    best_detection = detection
            
            # Update statistics
            self._update_detection_stats(best_confidence, best_detection)
            
            detected = best_confidence >= self.config.detection_confidence_threshold
            
            metadata = {
                "signal_analysis": signal_analysis,
                "packet_candidates_analyzed": len(packet_candidates),
                "best_detection": best_detection,
                "dji_variant": best_detection.get('variant', 'unknown') if detected else None
            }
            
            return ProtocolDetectionResult(
                detected=detected,
                confidence=best_confidence,
                protocol_name="dji",
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"DJI detection failed: {e}")
            return ProtocolDetectionResult(
                detected=False,
                confidence=0.0,
                protocol_name="dji",
                metadata={"error": str(e)}
            )
    
    def parse_packet(
        self,
        packet_data: bytes,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProtocolParseResult:
        """
        Parse DJI packet data.
        
        Args:
            packet_data: Raw packet bytes
            metadata: Optional parsing metadata
            
        Returns:
            Parse result with extracted DJI information
        """
        try:
            if len(packet_data) < 8:
                return ProtocolParseResult(
                    success=False,
                    protocol_name="dji",
                    metadata={"error": "Packet too short for DJI protocol"}
                )
            
            # Identify DJI variant
            variant = self._identify_dji_variant(packet_data)
            if variant == DJIProtocolVariant.UNKNOWN_DJI:
                logger.warning("Could not identify specific DJI variant")
            
            # Parse packet structure
            structure = self._parse_dji_structure(packet_data, variant)
            
            # Extract header fields
            header_fields = self._extract_header_fields(packet_data, structure)
            
            # Extract and decode payload
            payload_data = self._extract_payload(packet_data, structure)
            decoded_payload = self._decode_payload(payload_data, structure, header_fields)
            
            # Extract flight data if available
            flight_data = None
            if (self.config.extract_flight_data and 
                structure.message_type in [DJIMessageType.TELEMETRY, DJIMessageType.GPS_DATA]):
                flight_data = self._extract_flight_data(decoded_payload, structure)
            
            # Validate packet
            validation_result = self._validate_dji_packet(packet_data, structure)
            
            # Compile parsed data
            parsed_data = {
                "variant": variant.value,
                "message_type": structure.message_type.value if structure.message_type else "unknown",
                "header": header_fields,
                "payload": decoded_payload,
                "structure": structure.__dict__
            }
            
            if flight_data:
                parsed_data["flight_data"] = flight_data.__dict__
                self.detection_stats['flight_data_extracted'] += 1
            
            # Compile metadata
            parse_metadata = {
                "packet_length": len(packet_data),
                "dji_variant": variant.value,
                "structure_confidence": structure.structure_confidence,
                "checksum_valid": validation_result.get("checksum_valid", False),
                "encryption_detected": structure.encryption_flag,
                "validation": validation_result
            }
            
            if self.config.include_raw_data:
                parse_metadata["raw_packet_hex"] = packet_data.hex()
            
            return ProtocolParseResult(
                success=validation_result.get("is_valid", True),
                protocol_name="dji",
                parsed_data=parsed_data,
                metadata=parse_metadata
            )
            
        except Exception as e:
            logger.error(f"DJI packet parsing failed: {e}")
            return ProtocolParseResult(
                success=False,
                protocol_name="dji",
                metadata={"error": str(e)}
            )
    
    def encode_packet(
        self,
        packet_fields: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bytes:
        """
        Encode DJI packet from fields.
        
        Args:
            packet_fields: Dictionary containing packet fields
            metadata: Optional encoding metadata
            
        Returns:
            Encoded DJI packet bytes
        """
        try:
            # Get variant (default to OcuSync v2)
            variant_name = packet_fields.get("variant", "ocusync_v2")
            variant = DJIProtocolVariant(variant_name)
            
            signature = self.protocol_signatures.get(variant)
            if not signature:
                raise ValueError(f"Unsupported DJI variant: {variant_name}")
            
            packet = bytearray()
            
            # Add sync pattern
            sync_pattern = packet_fields.get("sync_pattern")
            if sync_pattern:
                packet.extend(sync_pattern)
            else:
                packet.extend(signature['sync_patterns'][0])
            
            # Prepare header
            header = self._encode_dji_header(packet_fields, variant)
            packet.extend(header)
            
            # Add payload
            payload = packet_fields.get("payload", b"")
            if isinstance(payload, str):
                payload = payload.encode('utf-8')
            elif isinstance(payload, dict):
                payload = self._encode_dji_payload(payload, variant)
            
            packet.extend(payload)
            
            # Calculate and add checksum
            checksum_type = packet_fields.get("checksum_type", "crc16")
            checksum = self._calculate_dji_checksum(packet, checksum_type)
            packet.extend(checksum)
            
            return bytes(packet)
            
        except Exception as e:
            logger.error(f"DJI packet encoding failed: {e}")
            raise ValueError(f"Failed to encode DJI packet: {e}")
    
    def _extract_packet_candidates(self, iq_samples: IQSamples) -> List[bytes]:
        """Extract potential DJI packets from IQ samples."""
        candidates = []
        
        try:
            # Simple envelope detection and demodulation
            magnitude = np.abs(iq_samples)
            
            # Find potential packet regions
            threshold = np.mean(magnitude) * 1.5
            active_regions = magnitude > threshold
            
            # Find edges
            edges = np.diff(active_regions.astype(int))
            starts = np.where(edges == 1)[0]
            ends = np.where(edges == -1)[0]
            
            # Process each region
            for start, end in zip(starts, ends):
                if end - start < 100:  # Too short
                    continue
                
                region_samples = iq_samples[start:end]
                
                # Simple OOK demodulation
                samples_per_bit = max(1, len(region_samples) // (len(region_samples) // 8))
                decimated = magnitude[start:end:samples_per_bit]
                bit_threshold = np.mean(decimated)
                bits = (decimated > bit_threshold).astype(np.uint8)
                
                # Convert to bytes
                if len(bits) >= 8:
                    # Pad to byte boundary
                    if len(bits) % 8 != 0:
                        bits = np.pad(bits, (0, 8 - len(bits) % 8), 'constant')
                    
                    packet_bytes = np.packbits(bits).tobytes()
                    if len(packet_bytes) >= 8:  # Minimum DJI packet size
                        candidates.append(packet_bytes)
            
        except Exception as e:
            logger.debug(f"Packet extraction failed: {e}")
        
        return candidates[:10]  # Limit candidates to prevent excessive processing
    
    def _analyze_dji_packet(
        self,
        packet_data: bytes,
        signal_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze packet for DJI protocol patterns."""
        analysis = {
            'confidence': 0.0,
            'variant': DJIProtocolVariant.UNKNOWN_DJI,
            'sync_pattern_found': False,
            'structure_valid': False,
            'message_type_detected': False
        }
        
        confidence_factors = []
        
        # Check for DJI sync patterns
        for variant, signature in self.protocol_signatures.items():
            for sync_pattern in signature['sync_patterns']:
                if packet_data.startswith(sync_pattern):
                    analysis['sync_pattern_found'] = True
                    analysis['variant'] = variant
                    confidence_factors.append(0.4)  # Strong indicator
                    break
            if analysis['sync_pattern_found']:
                break
        
        # Check packet length constraints
        if analysis['variant'] != DJIProtocolVariant.UNKNOWN_DJI:
            signature = self.protocol_signatures[analysis['variant']]
            packet_len = len(packet_data)
            if signature['min_packet_size'] <= packet_len <= signature['max_packet_size']:
                analysis['structure_valid'] = True
                confidence_factors.append(0.2)
        
        # Check for DJI message type patterns
        if len(packet_data) > 8:
            # Look for message type indicators in typical header positions
            for i in range(2, min(8, len(packet_data))):
                byte_val = packet_data[i]
                for msg_type, patterns in self.message_type_patterns.items():
                    if byte_val in patterns:
                        analysis['message_type_detected'] = True
                        analysis['detected_message_type'] = msg_type
                        confidence_factors.append(0.2)
                        break
                if analysis['message_type_detected']:
                    break
        
        # Check for DJI-specific byte patterns
        # DJI packets often have specific byte sequences
        dji_indicators = [
            b'\x55\xAA',  # Common DJI sync
            b'\x00\x01',  # Common in headers
            b'\xFF\xFE',  # EOF patterns
        ]
        
        for indicator in dji_indicators:
            if indicator in packet_data:
                confidence_factors.append(0.1)
                break
        
        # Signal quality bonus
        snr = signal_analysis.get('estimated_snr_db', 0)
        if snr > 15:  # Good SNR
            confidence_factors.append(0.1)
        elif snr > 5:
            confidence_factors.append(0.05)
        
        analysis['confidence'] = min(1.0, sum(confidence_factors))
        return analysis
    
    def _identify_dji_variant(self, packet_data: bytes) -> DJIProtocolVariant:
        """Identify specific DJI protocol variant."""
        for variant, signature in self.protocol_signatures.items():
            for sync_pattern in signature['sync_patterns']:
                if packet_data.startswith(sync_pattern):
                    return variant
        
        # Fallback analysis
        if packet_data.startswith(b'\x55\xAA'):
            if len(packet_data) > 2:
                version_byte = packet_data[2] if len(packet_data) > 2 else 0
                if version_byte == 0x04:
                    return DJIProtocolVariant.OCUSYNC_2
                elif version_byte == 0x06:
                    return DJIProtocolVariant.OCUSYNC_3
                elif version_byte <= 0x02:
                    return DJIProtocolVariant.OCUSYNC_1
        
        return DJIProtocolVariant.UNKNOWN_DJI
    
    def _parse_dji_structure(
        self,
        packet_data: bytes,
        variant: DJIProtocolVariant
    ) -> DJIPacketStructure:
        """Parse DJI packet structure."""
        signature = self.protocol_signatures.get(variant)
        if not signature:
            signature = self.protocol_signatures[DJIProtocolVariant.OCUSYNC_1]  # Default
        
        # Find sync pattern
        sync_pattern = b''
        for pattern in signature['sync_patterns']:
            if packet_data.startswith(pattern):
                sync_pattern = pattern
                break
        
        header_len = signature['header_length']
        sync_len = len(sync_pattern)
        
        # Extract basic structure info
        structure = DJIPacketStructure(
            variant=variant,
            sync_pattern=sync_pattern,
            header_length=header_len,
            payload_length=0,
            checksum_length=2,  # Typical DJI checksum length
            total_length=len(packet_data)
        )
        
        # Parse header for additional info
        if len(packet_data) >= header_len:
            header_data = packet_data[sync_len:header_len]
            
            # Extract sequence number (typically at offset 0-1 after sync)
            if len(header_data) >= 2:
                structure.sequence_number = struct.unpack('<H', header_data[:2])[0]
            
            # Extract message type (varies by variant)
            msg_type_offset = 2 if variant == DJIProtocolVariant.OCUSYNC_1 else 3
            if len(header_data) > msg_type_offset:
                msg_type_byte = header_data[msg_type_offset]
                structure.message_type = self._decode_message_type(msg_type_byte)
            
            # Extract payload length if present
            len_offset = 4 if variant in [DJIProtocolVariant.OCUSYNC_2, DJIProtocolVariant.OCUSYNC_3] else 3
            if len(header_data) > len_offset + 1:
                structure.payload_length = struct.unpack('<H', header_data[len_offset:len_offset+2])[0]
            else:
                structure.payload_length = len(packet_data) - header_len - structure.checksum_length
            
            # Check for encryption flag
            if len(header_data) > 6:
                flags_byte = header_data[6]
                structure.encryption_flag = bool(flags_byte & 0x80)
        
        # Calculate structure confidence
        confidence_factors = []
        if sync_pattern:
            confidence_factors.append(0.4)
        if structure.message_type != DJIMessageType.UNKNOWN:
            confidence_factors.append(0.3)
        if 0 < structure.payload_length < 2000:  # Reasonable payload size
            confidence_factors.append(0.2)
        if structure.total_length == header_len + structure.payload_length + structure.checksum_length:
            confidence_factors.append(0.1)
        
        structure.structure_confidence = sum(confidence_factors)
        
        return structure
    
    def _extract_header_fields(
        self,
        packet_data: bytes,
        structure: DJIPacketStructure
    ) -> Dict[str, Any]:
        """Extract header fields from DJI packet."""
        fields = {}
        
        if len(packet_data) < structure.header_length:
            return fields
        
        sync_len = len(structure.sync_pattern)
        header_data = packet_data[sync_len:structure.header_length]
        
        try:
            # Common header fields
            if len(header_data) >= 2:
                fields['sequence_number'] = struct.unpack('<H', header_data[:2])[0]
            
            if len(header_data) >= 3:
                fields['message_type'] = header_data[2]
                fields['message_type_name'] = structure.message_type.value if structure.message_type else "unknown"
            
            if len(header_data) >= 6:
                fields['payload_length'] = struct.unpack('<H', header_data[3:5])[0]
                fields['flags'] = header_data[5]
            
            if len(header_data) >= 8:
                fields['source_id'] = header_data[6]
                fields['target_id'] = header_data[7]
            
            # Variant-specific fields
            if structure.variant == DJIProtocolVariant.OCUSYNC_2:
                if len(header_data) >= 10:
                    fields['timestamp'] = struct.unpack('<L', header_data[8:12])[0]
            elif structure.variant == DJIProtocolVariant.OCUSYNC_3:
                if len(header_data) >= 12:
                    fields['extended_flags'] = struct.unpack('<H', header_data[8:10])[0]
                    fields['timestamp'] = struct.unpack('<L', header_data[10:14])[0]
            
        except struct.error as e:
            logger.debug(f"Header parsing error: {e}")
        
        return fields
    
    def _extract_payload(self, packet_data: bytes, structure: DJIPacketStructure) -> bytes:
        """Extract payload from DJI packet."""
        payload_start = structure.header_length
        payload_end = structure.total_length - structure.checksum_length
        
        if payload_start >= payload_end or payload_start >= len(packet_data):
            return b''
        
        return packet_data[payload_start:payload_end]
    
    def _decode_payload(
        self,
        payload_data: bytes,
        structure: DJIPacketStructure,
        header_fields: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Decode DJI payload based on message type."""
        decoded = {
            "raw_payload": payload_data,
            "payload_length": len(payload_data)
        }
        
        if not payload_data or structure.encryption_flag:
            decoded["encrypted"] = structure.encryption_flag
            return decoded
        
        try:
            # Decode based on message type
            if structure.message_type == DJIMessageType.TELEMETRY:
                decoded.update(self._decode_telemetry_payload(payload_data))
            elif structure.message_type == DJIMessageType.GPS_DATA:
                decoded.update(self._decode_gps_payload(payload_data))
            elif structure.message_type == DJIMessageType.BATTERY_INFO:
                decoded.update(self._decode_battery_payload(payload_data))
            elif structure.message_type == DJIMessageType.FLIGHT_CONTROL:
                decoded.update(self._decode_flight_control_payload(payload_data))
            else:
                # Generic payload analysis
                decoded.update(self._analyze_generic_payload(payload_data))
                
        except Exception as e:
            logger.debug(f"Payload decoding error: {e}")
            decoded["decode_error"] = str(e)
        
        return decoded
    
    def _decode_telemetry_payload(self, payload_data: bytes) -> Dict[str, Any]:
        """Decode DJI telemetry payload."""
        telemetry = {}
        
        if len(payload_data) < 20:
            return telemetry
        
        try:
            # Basic telemetry structure (varies by drone model)
            offset = 0
            
            # Flight mode (1 byte)
            if offset < len(payload_data):
                flight_mode_byte = payload_data[offset]
                telemetry['flight_mode_code'] = flight_mode_byte
                telemetry['flight_mode'] = self._decode_flight_mode(flight_mode_byte)
                offset += 1
            
            # Battery percentage (1 byte)
            if offset < len(payload_data):
                telemetry['battery_percentage'] = payload_data[offset]
                offset += 1
            
            # Signal strength (1 byte)
            if offset < len(payload_data):
                telemetry['signal_strength'] = payload_data[offset]
                offset += 1
            
            # GPS satellite count (1 byte)
            if offset < len(payload_data):
                telemetry['gps_satellites'] = payload_data[offset]
                offset += 1
            
            # Altitude (4 bytes, float)
            if offset + 4 <= len(payload_data):
                telemetry['altitude_m'] = struct.unpack('<f', payload_data[offset:offset+4])[0]
                offset += 4
            
            # Velocity components (3 x 4 bytes, floats)
            if offset + 12 <= len(payload_data):
                vx, vy, vz = struct.unpack('<fff', payload_data[offset:offset+12])
                telemetry['velocity_x_ms'] = vx
                telemetry['velocity_y_ms'] = vy
                telemetry['velocity_z_ms'] = vz
                offset += 12
                
        except struct.error as e:
            logger.debug(f"Telemetry decoding error: {e}")
        
        return telemetry
    
    def _decode_gps_payload(self, payload_data: bytes) -> Dict[str, Any]:
        """Decode DJI GPS payload."""
        gps_data = {}
        
        if len(payload_data) < 16:
            return gps_data
        
        try:
            # GPS coordinates (8 bytes each for lat/lon)
            if len(payload_data) >= 16:
                lat_raw, lon_raw = struct.unpack('<LL', payload_data[:8])
                # Convert from DJI format (scaled integers)
                gps_data['latitude'] = lat_raw / 1e7
                gps_data['longitude'] = lon_raw / 1e7
            
            # Altitude and other GPS data
            if len(payload_data) >= 20:
                altitude_raw = struct.unpack('<L', payload_data[16:20])[0]
                gps_data['gps_altitude_m'] = altitude_raw / 1000.0  # mm to m
            
            if len(payload_data) >= 22:
                gps_data['gps_satellite_count'] = payload_data[20]
                gps_data['gps_fix_quality'] = payload_data[21]
                
        except struct.error as e:
            logger.debug(f"GPS decoding error: {e}")
        
        return gps_data
    
    def _decode_battery_payload(self, payload_data: bytes) -> Dict[str, Any]:
        """Decode DJI battery information payload."""
        battery_data = {}
        
        if len(payload_data) < 8:
            return battery_data
        
        try:
            # Battery voltage (2 bytes)
            if len(payload_data) >= 2:
                voltage_raw = struct.unpack('<H', payload_data[:2])[0]
                battery_data['voltage_mv'] = voltage_raw
                battery_data['voltage_v'] = voltage_raw / 1000.0
            
            # Current (2 bytes)
            if len(payload_data) >= 4:
                current_raw = struct.unpack('<H', payload_data[2:4])[0]
                battery_data['current_ma'] = current_raw
            
            # Remaining capacity (1 byte)
            if len(payload_data) >= 5:
                battery_data['remaining_percentage'] = payload_data[4]
            
            # Temperature (1 byte)
            if len(payload_data) >= 6:
                temp_raw = payload_data[5]
                battery_data['temperature_c'] = temp_raw - 40  # Offset encoding
            
            # Cycle count (2 bytes)
            if len(payload_data) >= 8:
                battery_data['cycle_count'] = struct.unpack('<H', payload_data[6:8])[0]
                
        except struct.error as e:
            logger.debug(f"Battery decoding error: {e}")
        
        return battery_data
    
    def _decode_flight_control_payload(self, payload_data: bytes) -> Dict[str, Any]:
        """Decode DJI flight control payload."""
        control_data = {}
        
        if len(payload_data) < 12:
            return control_data
        
        try:
            # Attitude (pitch, roll, yaw) - 3 x 4 bytes
            if len(payload_data) >= 12:
                pitch, roll, yaw = struct.unpack('<fff', payload_data[:12])
                control_data['pitch_deg'] = math.degrees(pitch)
                control_data['roll_deg'] = math.degrees(roll)
                control_data['yaw_deg'] = math.degrees(yaw)
            
            # Control inputs (4 channels) - 4 x 2 bytes
            if len(payload_data) >= 20:
                channels = struct.unpack('<HHHH', payload_data[12:20])
                control_data['throttle'] = channels[0]
                control_data['aileron'] = channels[1]
                control_data['elevator'] = channels[2]
                control_data['rudder'] = channels[3]
                
        except struct.error as e:
            logger.debug(f"Flight control decoding error: {e}")
        
        return control_data
    
    def _analyze_generic_payload(self, payload_data: bytes) -> Dict[str, Any]:
        """Analyze unknown payload content."""
        analysis = {
            "payload_type": "unknown",
            "entropy": 0.0,
            "printable_ratio": 0.0,
            "null_bytes": 0,
            "repeating_patterns": []
        }
        
        if not payload_data:
            return analysis
        
        # Calculate entropy
        byte_counts = np.bincount(np.frombuffer(payload_data, dtype=np.uint8), minlength=256)
        probabilities = byte_counts / len(payload_data)
        probabilities = probabilities[probabilities > 0]
        analysis['entropy'] = float(-np.sum(probabilities * np.log2(probabilities)))
        
        # Calculate printable ratio
        printable_count = sum(1 for b in payload_data if 32 <= b <= 126)
        analysis['printable_ratio'] = printable_count / len(payload_data)
        
        # Count null bytes
        analysis['null_bytes'] = payload_data.count(0)
        
        return analysis
    
    def _extract_flight_data(
        self,
        decoded_payload: Dict[str, Any],
        structure: DJIPacketStructure
    ) -> Optional[DJIFlightData]:
        """Extract flight data from decoded payload."""
        if not decoded_payload:
            return None
        
        flight_data = DJIFlightData()
        
        # Extract from telemetry
        if 'altitude_m' in decoded_payload:
            flight_data.altitude = decoded_payload['altitude_m']
        
        if 'velocity_x_ms' in decoded_payload:
            flight_data.velocity_x = decoded_payload['velocity_x_ms']
            flight_data.velocity_y = decoded_payload['velocity_y_ms']
            flight_data.velocity_z = decoded_payload['velocity_z_ms']
        
        if 'battery_percentage' in decoded_payload:
            flight_data.battery_percentage = decoded_payload['battery_percentage']
        
        if 'signal_strength' in decoded_payload:
            flight_data.signal_strength = decoded_payload['signal_strength']
        
        if 'gps_satellites' in decoded_payload:
            flight_data.gps_satellite_count = decoded_payload['gps_satellites']
        
        if 'flight_mode' in decoded_payload:
            flight_data.flight_mode = decoded_payload['flight_mode']
        
        # Extract from GPS data
        if 'latitude' in decoded_payload:
            flight_data.latitude = decoded_payload['latitude']
            flight_data.longitude = decoded_payload['longitude']
        
        # Extract from flight control
        if 'pitch_deg' in decoded_payload:
            flight_data.pitch = decoded_payload['pitch_deg']
            flight_data.roll = decoded_payload['roll_deg']
            flight_data.yaw = decoded_payload['yaw_deg']
        
        return flight_data
    
    def _decode_message_type(self, msg_type_byte: int) -> DJIMessageType:
        """Decode message type from byte value."""
        for msg_type, patterns in self.message_type_patterns.items():
            if msg_type_byte in patterns:
                return msg_type
        
        return DJIMessageType.UNKNOWN
    
    def _decode_flight_mode(self, mode_byte: int) -> str:
        """Decode flight mode from byte value."""
        flight_modes = {
            0x00: "Manual",
            0x01: "Atti", 
            0x02: "GPS",
            0x03: "IOC",
            0x04: "Sport",
            0x05: "Gentle",
            0x06: "RTH",
            0x07: "Landing",
            0x08: "ActiveTrack",
            0x09: "TapFly",
            0x0A: "Cinematic",
            0x0B: "Tripod"
        }
        
        return flight_modes.get(mode_byte, f"Unknown_{mode_byte:02X}")
    
    def _validate_dji_packet(
        self,
        packet_data: bytes,
        structure: DJIPacketStructure
    ) -> Dict[str, Any]:
        """Validate DJI packet integrity."""
        validation = {
            "is_valid": True,
            "checksum_valid": False,
            "structure_valid": True,
            "issues": []
        }
        
        # Basic structure validation
        expected_length = structure.header_length + structure.payload_length + structure.checksum_length
        if len(packet_data) != expected_length:
            validation["issues"].append(f"Length mismatch: expected {expected_length}, got {len(packet_data)}")
            validation["structure_valid"] = False
        
        # Checksum validation
        if len(packet_data) >= structure.checksum_length:
            checksum_start = len(packet_data) - structure.checksum_length
            received_checksum = packet_data[checksum_start:]
            calculated_checksum = self._calculate_dji_checksum(packet_data[:-structure.checksum_length])
            
            validation["checksum_valid"] = received_checksum == calculated_checksum
            if not validation["checksum_valid"]:
                validation["issues"].append("Checksum validation failed")
        
        # Overall validity
        validation["is_valid"] = validation["structure_valid"] and len(validation["issues"]) == 0
        
        return validation
    
    def _calculate_dji_checksum(
        self,
        data: bytes,
        checksum_type: str = "crc16"
    ) -> bytes:
        """Calculate DJI checksum."""
        if checksum_type == "crc16":
            # Simplified CRC16 (DJI uses specific polynomial)
            crc = 0xFFFF
            for byte in data:
                crc ^= byte
                for _ in range(8):
                    if crc & 1:
                        crc = (crc >> 1) ^ 0xA001
                    else:
                        crc >>= 1
            return struct.pack('<H', crc)
        else:
            # Simple sum checksum fallback
            checksum = sum(data) & 0xFFFF
            return struct.pack('<H', checksum)
    
    def _encode_dji_header(
        self,
        packet_fields: Dict[str, Any],
        variant: DJIProtocolVariant
    ) -> bytes:
        """Encode DJI packet header."""
        header = bytearray()
        
        # Sequence number
        seq_num = packet_fields.get("sequence_number", 0)
        header.extend(struct.pack('<H', seq_num))
        
        # Message type
        msg_type = packet_fields.get("message_type", DJIMessageType.UNKNOWN.value)
        if isinstance(msg_type, str):
            msg_type = getattr(DJIMessageType, msg_type.upper(), DJIMessageType.UNKNOWN).value
        header.append(msg_type)
        
        # Payload length
        payload_len = packet_fields.get("payload_length", 0)
        header.extend(struct.pack('<H', payload_len))
        
        # Flags
        flags = packet_fields.get("flags", 0)
        header.append(flags)
        
        # Source and target IDs
        header.append(packet_fields.get("source_id", 0))
        header.append(packet_fields.get("target_id", 0))
        
        # Variant-specific fields
        if variant in [DJIProtocolVariant.OCUSYNC_2, DJIProtocolVariant.OCUSYNC_3]:
            timestamp = packet_fields.get("timestamp", 0)
            header.extend(struct.pack('<L', timestamp))
        
        if variant == DJIProtocolVariant.OCUSYNC_3:
            ext_flags = packet_fields.get("extended_flags", 0)
            header.extend(struct.pack('<H', ext_flags))
        
        return bytes(header)
    
    def _encode_dji_payload(
        self,
        payload_fields: Dict[str, Any],
        variant: DJIProtocolVariant
    ) -> bytes:
        """Encode DJI payload from fields."""
        payload = bytearray()
        
        # Encode based on message type
        msg_type = payload_fields.get("message_type", "unknown")
        
        if msg_type == "telemetry":
            # Encode telemetry payload
            payload.append(payload_fields.get("flight_mode_code", 0))
            payload.append(payload_fields.get("battery_percentage", 0))
            payload.append(payload_fields.get("signal_strength", 0))
            payload.append(payload_fields.get("gps_satellites", 0))
            
            altitude = payload_fields.get("altitude_m", 0.0)
            payload.extend(struct.pack('<f', altitude))
            
            vx = payload_fields.get("velocity_x_ms", 0.0)
            vy = payload_fields.get("velocity_y_ms", 0.0)
            vz = payload_fields.get("velocity_z_ms", 0.0)
            payload.extend(struct.pack('<fff', vx, vy, vz))
            
        elif msg_type == "gps_data":
            # Encode GPS payload
            lat = payload_fields.get("latitude", 0.0)
            lon = payload_fields.get("longitude", 0.0)
            payload.extend(struct.pack('<LL', int(lat * 1e7), int(lon * 1e7)))
            
            alt = payload_fields.get("gps_altitude_m", 0.0)
            payload.extend(struct.pack('<L', int(alt * 1000)))
            
            payload.append(payload_fields.get("gps_satellite_count", 0))
            payload.append(payload_fields.get("gps_fix_quality", 0))
        
        else:
            # Generic payload
            raw_payload = payload_fields.get("raw_data", b"")
            if isinstance(raw_payload, str):
                raw_payload = raw_payload.encode('utf-8')
            payload.extend(raw_payload)
        
        return bytes(payload)
    
    def _update_detection_stats(
        self,
        confidence: float,
        detection_result: Optional[Dict[str, Any]]
    ) -> None:
        """Update plugin detection statistics."""
        self.detection_stats['packets_analyzed'] += 1
        
        if confidence >= self.config.detection_confidence_threshold:
            self.detection_stats['dji_packets_detected'] += 1
            
            if detection_result:
                variant = detection_result.get('variant', DJIProtocolVariant.UNKNOWN_DJI)
                if isinstance(variant, DJIProtocolVariant):
                    self.detection_stats['variant_detections'][variant.value] += 1
        
        # Update average confidence
        total = self.detection_stats['packets_analyzed']
        prev_avg = self.detection_stats['avg_confidence']
        self.detection_stats['avg_confidence'] = (prev_avg * (total - 1) + confidence) / total
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get plugin detection statistics."""
        stats = self.detection_stats.copy()
        
        if stats['packets_analyzed'] > 0:
            stats['dji_detection_rate'] = stats['dji_packets_detected'] / stats['packets_analyzed']
        else:
            stats['dji_detection_rate'] = 0.0
        
        return stats


# Factory function for creating DJI plugin instances
def create_dji_plugin(config: Optional[DJIProtocolConfig] = None) -> DJIProtocolPlugin:
    """Create a DJI protocol plugin instance."""
    return DJIProtocolPlugin(config)


# Convenience function for quick DJI packet analysis
def analyze_dji_packet(packet_bytes: bytes) -> Dict[str, Any]:
    """
    Quick analysis of potential DJI packet.
    
    Args:
        packet_bytes: Raw packet data
        
    Returns:
        Analysis results including flight data if available
    """
    plugin = DJIProtocolPlugin()
    result = plugin.parse_packet(packet_bytes)
    
    return {
        "success": result.success,
        "protocol": result.protocol_name,
        "dji_variant": result.metadata.get("dji_variant"),
        "message_type": result.parsed_data.get("message_type") if result.parsed_data else None,
        "flight_data": result.parsed_data.get("flight_data") if result.parsed_data else None,
        "metadata": result.metadata
    }


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    print("=== DJI Protocol Plugin Demo ===")
    
    # Create plugin
    config = DJIProtocolConfig(
        extract_flight_data=True,
        decode_telemetry=True
    )
    plugin = DJIProtocolPlugin(config)
    
    # Test DJI-like packets
    test_packets = [
        # OcuSync v1 telemetry packet
        b'\x55\xAA\x12\x34\x01\x1A\x00\x00\x01\x02\x64\x5A\x08\x03\x00\x00\x80\x3F\x00\x00\x00\x40\x00\x00\x40\x40\x12\x34',
        
        # OcuSync v2 GPS packet  
        b'\x55\xAA\x04\x56\x78\x08\x18\x00\x00\x03\x04\x12\x34\x56\x78\x9A\xBC\xDE\xF0\x12\x34\x56\x78\x08\x02\xAB\xCD',
        
        # Lightbridge packet
        b'\x55\xAA\x02\xAB\xCD\x02\x10\x00\x00\x01\x50\x32\x14\xFF\x00\x11\x22\x33\x44\xEF\x12',
        
        # Non-DJI packet
        b'\xFE\x21\x00\x01\x01\x00' + b'\x42' * 33 + b'\x12\x34'
    ]
    
    packet_names = ["OcuSync v1 Telemetry", "OcuSync v2 GPS", "Lightbridge", "Non-DJI"]
    
    for name, packet in zip(packet_names, test_packets):
        print(f"\n--- Testing {name} packet ({len(packet)} bytes) ---")
        
        # Parse packet
        result = plugin.parse_packet(packet)
        
        print(f"Parse success: {result.success}")
        
        if result.success and result.parsed_data:
            variant = result.parsed_data.get("variant", "unknown")
            msg_type = result.parsed_data.get("message_type", "unknown")
            
            print(f"DJI variant: {variant}")
            print(f"Message type: {msg_type}")
            
            # Show flight data if available
            flight_data = result.parsed_data.get("flight_data")
            if flight_data:
                print("Flight data extracted:")
                for key, value in flight_data.items():
                    if value is not None:
                        print(f"  {key}: {value}")
            
            # Show structure info
            structure = result.parsed_data.get("structure", {})
            confidence = structure.get("structure_confidence", 0)
            print(f"Structure confidence: {confidence:.2f}")
        else:
            print("Not a valid DJI packet")
    
    # Show plugin statistics
    print(f"\n--- Plugin Statistics ---")
    stats = plugin.get_detection_statistics()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")