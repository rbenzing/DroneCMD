#!/usr/bin/env python3
"""
Parrot Drone Protocol Plugin

Implementation of protocol detection and parsing for Parrot drone communication.
Supports multiple Parrot protocol variants including ARSDK, legacy AR.Drone,
and newer Anafi/Bebop protocols.

Parrot Protocol Characteristics:
- ARSDK (AR.Drone SDK) protocol family
- JSON-based command structures over UDP/TCP
- Binary telemetry protocols
- WiFi-based communication (typically 2.4/5 GHz)
- Multi-frame packet structures
- CRC32 checksums for data integrity

Supported Models:
- AR.Drone series (legacy)
- Bebop series
- Anafi series
- Disco fixed-wing
- Mambo/Swing mini drones

Protocol References:
- ARSDK documentation
- Parrot Ground SDK
- Open-source implementations (libARSAL, libARNetworkAL)
"""

from __future__ import annotations

import json
import logging
import struct
import time
import zlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

# Import from plugin framework
try:
    from ..base import BaseProtocolPlugin, ProtocolDetectionResult, ProtocolParseResult
    from ...core.signal_processing import SignalProcessor, analyze_signal_quality
    PLUGIN_FRAMEWORK_AVAILABLE = True
except ImportError:
    PLUGIN_FRAMEWORK_AVAILABLE = False
    # Fallback base class
    class BaseProtocolPlugin:
        pass

# Configure module logger
logger = logging.getLogger(__name__)

# Type aliases
IQSamples = npt.NDArray[np.complex64]


class ParrotProtocolType(Enum):
    """Parrot protocol variants."""
    
    ARSDK_V1 = "arsdk_v1"      # Legacy AR.Drone protocol
    ARSDK_V3 = "arsdk_v3"      # Modern ARSDK protocol
    ANAFI = "anafi"            # Anafi-specific protocol
    BEBOP = "bebop"            # Bebop-specific protocol
    JSON_COMMAND = "json_cmd"   # JSON command protocol
    TELEMETRY = "telemetry"    # Binary telemetry data
    VIDEO_STREAM = "video"     # Video streaming protocol


class ParrotFrameType(Enum):
    """ARSDK frame types."""
    
    UNINITIALIZED = 0
    ACK = 1
    DATA = 2
    LOW_LATENCY_DATA = 3
    DATA_WITH_ACK = 4


class ParrotCommandClass(Enum):
    """Parrot command classes."""
    
    COMMON = 0
    ARDRONE3 = 1
    MINIDRONE = 2
    JUMPINGSUMO = 3
    SKYCONTROLLER = 4
    MAPPER = 5
    DEBUG = 6
    FOLLOW_ME = 7
    WIFI = 8
    GENERIC = 9
    CONTROLLER_INFO = 10


@dataclass
class ParrotProtocolConfig:
    """Configuration for Parrot protocol detection and parsing."""
    
    # Detection parameters
    detection_threshold: float = 0.8
    min_packet_length: int = 7
    max_packet_length: int = 65535
    
    # Protocol-specific settings
    enable_json_parsing: bool = True
    enable_crc_validation: bool = True
    strict_frame_validation: bool = True
    
    # Parsing options
    decode_json_commands: bool = True
    include_raw_data: bool = False
    track_sequence_numbers: bool = True
    
    # Detection patterns
    wifi_ssid_patterns: List[str] = field(default_factory=lambda: [
        "ardrone2_",
        "Bebop",
        "Anafi",
        "Disco-",
        "Mambo_",
        "Swing_"
    ])


@dataclass
class ParrotFrame:
    """Represents a parsed Parrot protocol frame."""
    
    # Frame header
    frame_type: ParrotFrameType
    frame_id: int
    frame_seq: int
    frame_size: int
    
    # Command data
    project: int
    command_class: int
    command_id: int
    payload: bytes
    
    # Validation
    crc32: Optional[int] = None
    crc_valid: bool = False
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    raw_frame: bytes = b""


class ParrotProtocolPlugin(BaseProtocolPlugin):
    """
    Parrot drone protocol plugin with comprehensive ARSDK support.
    
    This plugin implements detection and parsing for various Parrot drone
    protocols including legacy AR.Drone and modern ARSDK-based systems.
    
    Key Features:
    - Multi-protocol support (ARSDK v1/v3, JSON commands, telemetry)
    - CRC32 validation for data integrity
    - Sequence number tracking
    - JSON command parsing
    - WiFi network detection patterns
    - Comprehensive frame structure parsing
    
    Example:
        >>> plugin = ParrotProtocolPlugin()
        >>> result = plugin.detect(iq_samples)
        >>> if result.detected:
        ...     parsed = plugin.parse_packet(packet_bytes)
        ...     print(f"Parrot frame: {parsed.parsed_data['frame_type']}")
    """
    
    # Parrot protocol constants
    ARSDK_HEADER_SIZE = 7
    ARSDK_MAGIC_BYTE = 0x02
    
    # Known Parrot sync patterns
    SYNC_PATTERNS = [
        b'\x02\x00',              # ARSDK frame start
        b'\x02\x01',              # ARSDK ACK frame
        b'\x02\x02',              # ARSDK data frame
        b'\x02\x03',              # ARSDK low-latency data
        b'\x02\x04',              # ARSDK data with ACK
        b'{"class":',             # JSON command start
        b'{"project":',           # JSON project command
    ]
    
    # Command mappings for common Parrot commands
    COMMON_COMMANDS = {
        (0, 0, 0): "AllStates",
        (0, 0, 1): "AllSettings", 
        (0, 0, 2): "AllSettingsChanged",
        (0, 0, 3): "AllStatesChanged",
        (0, 2, 0): "CurrentDate",
        (0, 2, 1): "CurrentTime",
        (0, 4, 0): "Reboot",
        (0, 5, 0): "WifiSettings",
        (1, 0, 0): "FlatTrim",
        (1, 0, 1): "TakeOff",
        (1, 0, 2): "PCMD",  # Piloting command
        (1, 0, 3): "Landing",
        (1, 0, 4): "Emergency",
        (1, 1, 0): "FlyingState",
        (1, 1, 1): "AlertState",
        (1, 25, 0): "PictureSettings",
        (1, 25, 1): "VideoSettings",
    }
    
    def __init__(self, config: Optional[ParrotProtocolConfig] = None) -> None:
        """Initialize Parrot protocol plugin."""
        self.config = config or ParrotProtocolConfig()
        
        # Initialize signal processor
        if PLUGIN_FRAMEWORK_AVAILABLE:
            self.signal_processor = SignalProcessor()
        else:
            self.signal_processor = None
        
        # Sequence tracking
        self.sequence_numbers = {}
        self.frame_history = []
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'successful_parses': 0,
            'crc_failures': 0,
            'json_commands': 0,
            'frame_type_counts': {},
            'command_counts': {},
            'sequence_errors': 0
        }
        
        logger.info("Initialized Parrot protocol plugin")
    
    def get_name(self) -> str:
        """Get plugin name."""
        return "Parrot ARSDK Protocol"
    
    def get_version(self) -> str:
        """Get plugin version."""
        return "2.0.0"
    
    def get_supported_protocols(self) -> List[str]:
        """Get list of supported protocol identifiers."""
        return [
            "parrot", "arsdk", "ardrone", "bebop", "anafi", 
            "disco", "mambo", "swing", "parrot_json"
        ]
    
    def detect(
        self,
        iq_samples: IQSamples,
        signal_metadata: Optional[Dict[str, Any]] = None
    ) -> ProtocolDetectionResult:
        """
        Detect Parrot protocol in IQ samples.
        
        Args:
            iq_samples: Complex IQ samples to analyze
            signal_metadata: Optional signal processing metadata
            
        Returns:
            Detection result with confidence and protocol information
        """
        try:
            # Signal quality analysis
            signal_quality = {}
            if self.signal_processor and len(iq_samples) > 0:
                signal_quality = analyze_signal_quality(iq_samples)
            
            # Convert IQ to packet data for analysis
            packet_data = self._iq_to_packet_data(iq_samples)
            
            if len(packet_data) < self.config.min_packet_length:
                return ProtocolDetectionResult(
                    detected=False,
                    confidence=0.0,
                    protocol_name="parrot",
                    metadata={
                        "signal_quality": signal_quality,
                        "reason": "Insufficient packet data",
                        "packet_length": len(packet_data)
                    }
                )
            
            # Pattern-based detection
            pattern_confidence = self._detect_sync_patterns(packet_data)
            
            # Structure-based detection
            structure_confidence = self._detect_arsdk_structure(packet_data)
            
            # JSON command detection
            json_confidence = self._detect_json_commands(packet_data)
            
            # WiFi network pattern detection
            wifi_confidence = 0.0
            if signal_metadata and 'wifi_networks' in signal_metadata:
                wifi_confidence = self._detect_wifi_patterns(signal_metadata['wifi_networks'])
            
            # Calculate overall confidence
            confidence_factors = [
                pattern_confidence * 0.4,
                structure_confidence * 0.3,
                json_confidence * 0.2,
                wifi_confidence * 0.1
            ]
            
            overall_confidence = sum(confidence_factors)
            
            # Signal quality boost
            if signal_quality.get('estimated_snr_db', 0) > 15:
                overall_confidence += 0.1
            
            # Determine detection result
            detected = overall_confidence >= self.config.detection_threshold
            
            # Determine specific protocol variant
            protocol_variant = self._determine_protocol_variant(
                packet_data, pattern_confidence, json_confidence
            )
            
            metadata = {
                "signal_quality": signal_quality,
                "pattern_confidence": pattern_confidence,
                "structure_confidence": structure_confidence,
                "json_confidence": json_confidence,
                "wifi_confidence": wifi_confidence,
                "protocol_variant": protocol_variant,
                "packet_length": len(packet_data),
                "confidence_breakdown": {
                    "patterns": pattern_confidence,
                    "structure": structure_confidence,
                    "json": json_confidence,
                    "wifi": wifi_confidence
                }
            }
            
            return ProtocolDetectionResult(
                detected=detected,
                confidence=overall_confidence,
                protocol_name="parrot",
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Parrot detection failed: {e}")
            return ProtocolDetectionResult(
                detected=False,
                confidence=0.0,
                protocol_name="parrot",
                metadata={"error": str(e)}
            )
    
    def parse_packet(
        self,
        packet_data: bytes,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProtocolParseResult:
        """
        Parse Parrot protocol packet.
        
        Args:
            packet_data: Raw packet bytes
            metadata: Optional parsing metadata
            
        Returns:
            Parse result with extracted Parrot protocol information
        """
        try:
            self.stats['frames_processed'] += 1
            
            # Validate packet length
            if len(packet_data) < self.config.min_packet_length:
                return ProtocolParseResult(
                    success=False,
                    protocol_name="parrot",
                    metadata={"error": f"Packet too short: {len(packet_data)} bytes"}
                )
            
            # Determine packet type
            if packet_data.startswith(b'{"') or b'"class":' in packet_data[:50]:
                # JSON command packet
                return self._parse_json_command(packet_data)
            elif len(packet_data) >= self.ARSDK_HEADER_SIZE and packet_data[0] == self.ARSDK_MAGIC_BYTE:
                # ARSDK binary frame
                return self._parse_arsdk_frame(packet_data)
            else:
                # Try generic Parrot parsing
                return self._parse_generic_parrot(packet_data)
            
        except Exception as e:
            logger.error(f"Parrot packet parsing failed: {e}")
            return ProtocolParseResult(
                success=False,
                protocol_name="parrot",
                metadata={"error": str(e)}
            )
    
    def encode_packet(
        self,
        packet_fields: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bytes:
        """
        Encode Parrot protocol packet.
        
        Args:
            packet_fields: Dictionary containing packet fields
            metadata: Optional encoding metadata
            
        Returns:
            Encoded packet bytes
        """
        try:
            packet_type = packet_fields.get("type", "arsdk")
            
            if packet_type == "json":
                return self._encode_json_command(packet_fields)
            elif packet_type == "arsdk":
                return self._encode_arsdk_frame(packet_fields)
            else:
                raise ValueError(f"Unknown Parrot packet type: {packet_type}")
                
        except Exception as e:
            logger.error(f"Parrot packet encoding failed: {e}")
            raise ValueError(f"Failed to encode Parrot packet: {e}")
    
    def _iq_to_packet_data(self, iq_samples: IQSamples) -> bytes:
        """Convert IQ samples to packet data."""
        if len(iq_samples) == 0:
            return b""
        
        # For WiFi-based Parrot communication, we'd typically see
        # 802.11 frames. This is a simplified conversion.
        
        try:
            # Use magnitude for simple demodulation
            magnitude = np.abs(iq_samples)
            
            # Adaptive threshold
            threshold = np.mean(magnitude) + np.std(magnitude)
            
            # Convert to bits
            bits = (magnitude > threshold).astype(np.uint8)
            
            # Decimate to reasonable bit rate (assume ~1 Mbps for WiFi data)
            if len(iq_samples) > 1000:
                decimation = len(iq_samples) // 1000
                bits = bits[::decimation]
            
            # Pad to byte boundary
            if len(bits) % 8 != 0:
                bits = np.pad(bits, (0, 8 - len(bits) % 8), 'constant')
            
            # Pack to bytes
            if len(bits) > 0:
                return np.packbits(bits).tobytes()
            else:
                return b""
                
        except Exception as e:
            logger.debug(f"IQ conversion failed: {e}")
            return b""
    
    def _detect_sync_patterns(self, packet_data: bytes) -> float:
        """Detect Parrot sync patterns in packet data."""
        confidence = 0.0
        
        for pattern in self.SYNC_PATTERNS:
            if pattern in packet_data:
                confidence += 0.3
                
                # Boost confidence if pattern is at the beginning
                if packet_data.startswith(pattern):
                    confidence += 0.2
        
        return min(1.0, confidence)
    
    def _detect_arsdk_structure(self, packet_data: bytes) -> float:
        """Detect ARSDK frame structure."""
        if len(packet_data) < self.ARSDK_HEADER_SIZE:
            return 0.0
        
        confidence = 0.0
        
        # Check magic byte
        if packet_data[0] == self.ARSDK_MAGIC_BYTE:
            confidence += 0.4
        
        # Check frame type
        if len(packet_data) > 1:
            frame_type = packet_data[1]
            if 0 <= frame_type <= 4:  # Valid ARSDK frame types
                confidence += 0.2
        
        # Check frame ID and sequence
        if len(packet_data) >= 4:
            frame_id = packet_data[2]
            frame_seq = packet_data[3]
            if frame_id < 256 and frame_seq < 256:  # Reasonable values
                confidence += 0.1
        
        # Check frame size consistency
        if len(packet_data) >= 7:
            frame_size = struct.unpack('<I', packet_data[4:8])[0]
            if frame_size == len(packet_data):
                confidence += 0.3
            elif abs(frame_size - len(packet_data)) <= 4:  # Allow small discrepancy
                confidence += 0.1
        
        return confidence
    
    def _detect_json_commands(self, packet_data: bytes) -> float:
        """Detect JSON command structures."""
        try:
            # Look for JSON-like patterns
            text = packet_data.decode('utf-8', errors='ignore')
            
            confidence = 0.0
            
            # Check for JSON structure
            if text.strip().startswith('{') and text.strip().endswith('}'):
                confidence += 0.3
                
                # Check for Parrot-specific JSON fields
                parrot_fields = ['class', 'project', 'id', 'name', 'arg']
                for field in parrot_fields:
                    if f'"{field}":' in text:
                        confidence += 0.1
                
                # Try to parse as JSON
                try:
                    json.loads(text)
                    confidence += 0.2
                except json.JSONDecodeError:
                    confidence = max(0.0, confidence - 0.1)
            
            return min(1.0, confidence)
            
        except UnicodeDecodeError:
            return 0.0
    
    def _detect_wifi_patterns(self, wifi_networks: List[str]) -> float:
        """Detect Parrot WiFi network patterns."""
        confidence = 0.0
        
        for network in wifi_networks:
            for pattern in self.config.wifi_ssid_patterns:
                if pattern.lower() in network.lower():
                    confidence += 0.5
                    break
        
        return min(1.0, confidence)
    
    def _determine_protocol_variant(
        self,
        packet_data: bytes,
        pattern_conf: float,
        json_conf: float
    ) -> str:
        """Determine specific Parrot protocol variant."""
        if json_conf > 0.5:
            return "json_command"
        elif len(packet_data) >= 7 and packet_data[0] == self.ARSDK_MAGIC_BYTE:
            return "arsdk_v3"
        elif b"ardrone" in packet_data.lower():
            return "arsdk_v1"
        elif b"bebop" in packet_data.lower():
            return "bebop"
        elif b"anafi" in packet_data.lower():
            return "anafi"
        else:
            return "unknown"
    
    def _parse_arsdk_frame(self, packet_data: bytes) -> ProtocolParseResult:
        """Parse ARSDK binary frame."""
        try:
            if len(packet_data) < self.ARSDK_HEADER_SIZE:
                raise ValueError("Packet too short for ARSDK frame")
            
            # Parse header
            magic = packet_data[0]
            frame_type = ParrotFrameType(packet_data[1])
            frame_id = packet_data[2]
            frame_seq = packet_data[3]
            frame_size = struct.unpack('<I', packet_data[4:8])[0]
            
            # Validate frame size
            if frame_size != len(packet_data):
                logger.warning(f"Frame size mismatch: declared {frame_size}, actual {len(packet_data)}")
            
            # Extract payload
            payload_start = self.ARSDK_HEADER_SIZE
            if frame_size > len(packet_data):
                payload_end = len(packet_data)
            else:
                payload_end = frame_size
            
            payload = packet_data[payload_start:payload_end]
            
            # Parse command data from payload
            command_data = {}
            if len(payload) >= 4:
                project = payload[0]
                command_class = payload[1]
                command_id = struct.unpack('<H', payload[2:4])[0]
                command_args = payload[4:]
                
                command_data = {
                    "project": project,
                    "command_class": command_class, 
                    "command_id": command_id,
                    "args": command_args
                }
                
                # Look up command name
                command_key = (project, command_class, command_id)
                command_name = self.COMMON_COMMANDS.get(command_key, "Unknown")
                command_data["command_name"] = command_name
            
            # CRC validation (if present)
            crc_valid = True
            if self.config.enable_crc_validation and len(payload) >= 4:
                # CRC is typically at the end of the payload
                # This is a simplified check
                try:
                    expected_crc = struct.unpack('<I', payload[-4:])[0]
                    calculated_crc = zlib.crc32(packet_data[:-4]) & 0xffffffff
                    crc_valid = (expected_crc == calculated_crc)
                    if not crc_valid:
                        self.stats['crc_failures'] += 1
                except:
                    crc_valid = False
            
            # Update sequence tracking
            if self.config.track_sequence_numbers:
                self._update_sequence_tracking(frame_id, frame_seq)
            
            # Create frame object
            frame = ParrotFrame(
                frame_type=frame_type,
                frame_id=frame_id,
                frame_seq=frame_seq,
                frame_size=frame_size,
                project=command_data.get("project", 0),
                command_class=command_data.get("command_class", 0),
                command_id=command_data.get("command_id", 0),
                payload=payload,
                crc_valid=crc_valid,
                raw_frame=packet_data
            )
            
            # Update statistics
            self.stats['successful_parses'] += 1
            frame_type_name = frame_type.name
            self.stats['frame_type_counts'][frame_type_name] = (
                self.stats['frame_type_counts'].get(frame_type_name, 0) + 1
            )
            
            command_name = command_data.get("command_name", "Unknown")
            self.stats['command_counts'][command_name] = (
                self.stats['command_counts'].get(command_name, 0) + 1
            )
            
            # Prepare result
            parsed_data = {
                "frame_type": frame_type.name,
                "frame_id": frame_id,
                "frame_sequence": frame_seq,
                "frame_size": frame_size,
                "command_data": command_data,
                "payload_length": len(payload),
                "crc_valid": crc_valid
            }
            
            if self.config.include_raw_data:
                parsed_data["raw_frame"] = packet_data.hex()
                parsed_data["raw_payload"] = payload.hex()
            
            metadata = {
                "protocol_variant": "arsdk",
                "magic_byte": f"0x{magic:02X}",
                "frame_object": frame,
                "validation": {
                    "size_match": frame_size == len(packet_data),
                    "crc_valid": crc_valid,
                    "sequence_valid": True  # Would check against expected sequence
                }
            }
            
            return ProtocolParseResult(
                success=True,
                protocol_name="parrot",
                parsed_data=parsed_data,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"ARSDK frame parsing failed: {e}")
            return ProtocolParseResult(
                success=False,
                protocol_name="parrot",
                metadata={"error": f"ARSDK parsing failed: {e}"}
            )
    
    def _parse_json_command(self, packet_data: bytes) -> ProtocolParseResult:
        """Parse JSON command packet."""
        try:
            # Decode as text
            text = packet_data.decode('utf-8', errors='ignore').strip()
            
            # Parse JSON
            command_data = json.loads(text)
            
            # Extract Parrot-specific fields
            parsed_data = {
                "format": "json",
                "command_class": command_data.get("class"),
                "project": command_data.get("project"),
                "command_id": command_data.get("id"),
                "command_name": command_data.get("name"),
                "arguments": command_data.get("arg", {}),
                "full_command": command_data
            }
            
            # Update statistics
            self.stats['successful_parses'] += 1
            self.stats['json_commands'] += 1
            
            metadata = {
                "protocol_variant": "json_command",
                "original_text": text,
                "json_valid": True
            }
            
            return ProtocolParseResult(
                success=True,
                protocol_name="parrot",
                parsed_data=parsed_data,
                metadata=metadata
            )
            
        except json.JSONDecodeError as e:
            return ProtocolParseResult(
                success=False,
                protocol_name="parrot",
                metadata={"error": f"Invalid JSON: {e}"}
            )
        except Exception as e:
            return ProtocolParseResult(
                success=False,
                protocol_name="parrot",
                metadata={"error": f"JSON parsing failed: {e}"}
            )
    
    def _parse_generic_parrot(self, packet_data: bytes) -> ProtocolParseResult:
        """Parse generic Parrot packet (fallback)."""
        # Use generic parsing for unknown Parrot formats
        parsed_data = {
            "format": "generic",
            "packet_length": len(packet_data),
            "data_preview": packet_data[:32].hex() if len(packet_data) > 32 else packet_data.hex()
        }
        
        # Try to extract some basic information
        if len(packet_data) >= 4:
            parsed_data["first_dword"] = struct.unpack('<I', packet_data[:4])[0]
        
        metadata = {
            "protocol_variant": "generic",
            "parsing_method": "fallback"
        }
        
        return ProtocolParseResult(
            success=True,
            protocol_name="parrot",
            parsed_data=parsed_data,
            metadata=metadata
        )
    
    def _encode_arsdk_frame(self, fields: Dict[str, Any]) -> bytes:
        """Encode ARSDK frame."""
        frame_type = fields.get("frame_type", ParrotFrameType.DATA)
        frame_id = fields.get("frame_id", 0)
        frame_seq = fields.get("frame_seq", 0)
        
        # Build command payload
        project = fields.get("project", 0)
        command_class = fields.get("command_class", 0)
        command_id = fields.get("command_id", 0)
        args = fields.get("args", b"")
        
        if isinstance(args, str):
            args = args.encode('utf-8')
        
        # Build payload
        payload = struct.pack('<BBH', project, command_class, command_id) + args
        
        # Calculate frame size
        frame_size = self.ARSDK_HEADER_SIZE + len(payload)
        
        # Build header
        header = struct.pack('<BBBBI', 
                           self.ARSDK_MAGIC_BYTE,
                           frame_type.value if isinstance(frame_type, ParrotFrameType) else frame_type,
                           frame_id,
                           frame_seq,
                           frame_size)
        
        return header + payload
    
    def _encode_json_command(self, fields: Dict[str, Any]) -> bytes:
        """Encode JSON command."""
        command = {
            "class": fields.get("command_class", "common"),
            "project": fields.get("project", "common"),
            "id": fields.get("command_id", 0),
            "name": fields.get("command_name", "AllStates"),
            "arg": fields.get("arguments", {})
        }
        
        json_text = json.dumps(command, separators=(',', ':'))
        return json_text.encode('utf-8')
    
    def _update_sequence_tracking(self, frame_id: int, frame_seq: int) -> None:
        """Update sequence number tracking."""
        if frame_id not in self.sequence_numbers:
            self.sequence_numbers[frame_id] = frame_seq
        else:
            expected_seq = (self.sequence_numbers[frame_id] + 1) % 256
            if frame_seq != expected_seq:
                self.stats['sequence_errors'] += 1
                logger.debug(f"Sequence error on frame {frame_id}: expected {expected_seq}, got {frame_seq}")
            self.sequence_numbers[frame_id] = frame_seq
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get plugin statistics."""
        stats = self.stats.copy()
        
        if stats['frames_processed'] > 0:
            stats['success_rate'] = stats['successful_parses'] / stats['frames_processed']
            stats['crc_failure_rate'] = stats['crc_failures'] / stats['frames_processed']
            stats['sequence_error_rate'] = stats['sequence_errors'] / stats['frames_processed']
        else:
            stats['success_rate'] = 0.0
            stats['crc_failure_rate'] = 0.0
            stats['sequence_error_rate'] = 0.0
        
        stats['active_sequences'] = len(self.sequence_numbers)
        return stats
    
    def reset_statistics(self) -> None:
        """Reset plugin statistics."""
        self.stats = {
            'frames_processed': 0,
            'successful_parses': 0,
            'crc_failures': 0,
            'json_commands': 0,
            'frame_type_counts': {},
            'command_counts': {},
            'sequence_errors': 0
        }
        self.sequence_numbers.clear()


# Factory function
def create_parrot_plugin(config: Optional[ParrotProtocolConfig] = None) -> ParrotProtocolPlugin:
    """Create Parrot protocol plugin instance."""
    return ParrotProtocolPlugin(config)


# Convenience function for command encoding
def create_parrot_command(
    command_name: str,
    project: str = "common", 
    command_class: str = "common",
    **kwargs
) -> bytes:
    """
    Create a Parrot ARSDK command.
    
    Args:
        command_name: Name of the command
        project: Project name (common, ardrone3, etc.)
        command_class: Command class
        **kwargs: Command arguments
        
    Returns:
        Encoded command bytes
    """
    plugin = ParrotProtocolPlugin()
    
    fields = {
        "type": "json",
        "command_name": command_name,
        "project": project,
        "command_class": command_class,
        "arguments": kwargs
    }
    
    return plugin.encode_packet(fields)


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    print("=== Parrot Protocol Plugin Demo ===")
    
    # Create plugin
    config = ParrotProtocolConfig(
        enable_json_parsing=True,
        enable_crc_validation=True,
        track_sequence_numbers=True
    )
    plugin = ParrotProtocolPlugin(config)
    
    print(f"Plugin: {plugin.get_name()} v{plugin.get_version()}")
    print(f"Supported: {plugin.get_supported_protocols()}")
    
    # Test JSON command
    print("\n--- Testing JSON Command ---")
    json_command = b'{"class":"common","project":"common","id":0,"name":"AllStates","arg":{}}'
    
    result = plugin.parse_packet(json_command)
    print(f"JSON Parse Success: {result.success}")
    if result.success:
        print(f"Command: {result.parsed_data.get('command_name')}")
        print(f"Class: {result.parsed_data.get('command_class')}")
    
    # Test ARSDK frame
    print("\n--- Testing ARSDK Frame ---")
    # Create a mock ARSDK frame
    arsdk_frame = plugin.encode_packet({
        "type": "arsdk",
        "frame_type": ParrotFrameType.DATA,
        "frame_id": 1,
        "frame_seq": 0,
        "project": 0,
        "command_class": 0,
        "command_id": 0,
        "args": b""
    })
    
    result = plugin.parse_packet(arsdk_frame)
    print(f"ARSDK Parse Success: {result.success}")
    if result.success:
        print(f"Frame Type: {result.parsed_data.get('frame_type')}")
        print(f"Command: {result.parsed_data.get('command_data', {}).get('command_name', 'Unknown')}")
    
    # Show statistics
    print(f"\n--- Plugin Statistics ---")
    stats = plugin.get_statistics()
    for key, value in stats.items():
        if not isinstance(value, dict):
            print(f"{key}: {value}")