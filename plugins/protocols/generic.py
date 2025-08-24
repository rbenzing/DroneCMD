#!/usr/bin/env python3
"""
Generic Protocol Plugin

A flexible protocol plugin that can handle unknown or generic wireless protocols.
This plugin serves as both a fallback for unrecognized protocols and a template
for creating new protocol-specific plugins.

Key Features:
- Statistical analysis of packet structure
- Generic packet segmentation and parsing
- Configurable detection patterns
- Extensible framework for unknown protocols
- Integration with signal processing and classification systems

This plugin implements the base protocol interface and provides reasonable
defaults for handling arbitrary wireless protocols.
"""

from __future__ import annotations

import logging
import struct
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

# Import from core modules
try:
    from ...core.signal_processing import SignalProcessor, detect_packets, analyze_signal_quality
    from ..base import BaseProtocolPlugin, ProtocolDetectionResult, ProtocolParseResult
    CORE_AVAILABLE = True
except ImportError:
    # Fallback for development
    CORE_AVAILABLE = False
    warnings.warn("Core modules not available, using fallback implementations")

# Configure module logger
logger = logging.getLogger(__name__)

# Type aliases
IQSamples = npt.NDArray[np.complex64]
BitStream = npt.NDArray[np.uint8]


@dataclass
class GenericProtocolConfig:
    """Configuration for generic protocol detection and parsing."""
    
    # Detection parameters
    min_packet_length: int = 8
    max_packet_length: int = 2048
    detection_threshold: float = 0.7
    enable_statistical_analysis: bool = True
    
    # Parsing parameters
    assume_header_size: int = 4
    assume_payload_max: int = 1024
    enable_structure_detection: bool = True
    
    # Signal processing
    enable_signal_analysis: bool = True
    require_preamble: bool = False
    preamble_patterns: List[bytes] = field(default_factory=list)
    
    # Output options
    include_raw_data: bool = True
    include_statistics: bool = True
    confidence_threshold: float = 0.5


@dataclass
class GenericPacketStructure:
    """Detected structure of a generic packet."""
    
    total_length: int
    estimated_header_length: int
    estimated_payload_length: int
    estimated_checksum_length: int
    
    # Detected patterns
    sync_pattern: Optional[bytes] = None
    length_field_offset: Optional[int] = None
    length_field_value: Optional[int] = None
    
    # Statistical properties
    entropy: float = 0.0
    byte_distribution: Dict[int, int] = field(default_factory=dict)
    repeating_patterns: List[Tuple[bytes, int]] = field(default_factory=list)
    
    # Confidence metrics
    structure_confidence: float = 0.0
    pattern_confidence: float = 0.0


class GenericProtocolPlugin(BaseProtocolPlugin if CORE_AVAILABLE else object):
    """
    Generic protocol plugin for handling unknown wireless protocols.
    
    This plugin provides a comprehensive framework for analyzing and parsing
    unknown or generic wireless protocols using statistical analysis and
    pattern detection techniques.
    
    Example:
        >>> plugin = GenericProtocolPlugin()
        >>> result = plugin.detect(iq_samples)
        >>> if result.detected:
        ...     parsed = plugin.parse_packet(packet_bytes)
        ...     print(f"Generic packet: {parsed.metadata['structure']}")
    """
    
    def __init__(self, config: Optional[GenericProtocolConfig] = None) -> None:
        """
        Initialize generic protocol plugin.
        
        Args:
            config: Plugin configuration (uses defaults if None)
        """
        self.config = config or GenericProtocolConfig()
        
        # Initialize signal processor if available
        if CORE_AVAILABLE:
            self.signal_processor = SignalProcessor()
        else:
            self.signal_processor = None
        
        # Common protocol patterns (extensible)
        self.known_patterns = {
            'mavlink_v1': b'\xFE',
            'mavlink_v2': b'\xFD', 
            'dji_sync': b'\x55\xAA',
            'ieee802154': b'\x7E',
            'bluetooth_le': b'\xAA\xD6',
            'wifi_beacon': b'\x80\x00',
        }
        
        # Add user-defined patterns
        for i, pattern in enumerate(self.config.preamble_patterns):
            self.known_patterns[f'user_pattern_{i}'] = pattern
        
        # Statistics
        self.detection_stats = {
            'packets_analyzed': 0,
            'successful_detections': 0,
            'pattern_matches': {},
            'avg_confidence': 0.0
        }
        
        logger.info("Initialized generic protocol plugin")
    
    def get_name(self) -> str:
        """Get plugin name."""
        return "Generic Protocol"
    
    def get_version(self) -> str:
        """Get plugin version."""
        return "1.0.0"
    
    def get_supported_protocols(self) -> List[str]:
        """Get list of supported protocol identifiers."""
        return ["unknown", "generic", "custom"] + list(self.known_patterns.keys())
    
    def detect(
        self,
        iq_samples: IQSamples,
        signal_metadata: Optional[Dict[str, Any]] = None
    ) -> ProtocolDetectionResult:
        """
        Detect if samples contain a recognizable protocol structure.
        
        Args:
            iq_samples: Complex IQ samples to analyze
            signal_metadata: Optional signal processing metadata
            
        Returns:
            Detection result with confidence and metadata
        """
        start_time = np.datetime64('now')
        
        try:
            # Basic validation
            if len(iq_samples) < self.config.min_packet_length * 8:  # Assume ~8 samples per bit
                return ProtocolDetectionResult(
                    detected=False,
                    confidence=0.0,
                    protocol_name="generic",
                    metadata={"error": "Insufficient samples for analysis"}
                )
            
            # Signal quality analysis
            signal_quality = {}
            if self.config.enable_signal_analysis and CORE_AVAILABLE:
                signal_quality = analyze_signal_quality(iq_samples)
            
            # Detect packet regions
            packet_regions = self._detect_packet_regions(iq_samples)
            
            if not packet_regions:
                return ProtocolDetectionResult(
                    detected=False,
                    confidence=0.2,
                    protocol_name="generic",
                    metadata={
                        "signal_quality": signal_quality,
                        "packet_regions": 0,
                        "reason": "No packet-like regions detected"
                    }
                )
            
            # Analyze the strongest packet region
            best_region = max(packet_regions, key=lambda r: r[1] - r[0])
            packet_samples = iq_samples[best_region[0]:best_region[1]]
            
            # Convert to bits for analysis
            packet_bits = self._samples_to_bits(packet_samples)
            packet_bytes = self._bits_to_bytes(packet_bits)
            
            # Pattern analysis
            pattern_results = self._analyze_patterns(packet_bytes)
            
            # Statistical analysis
            stats_results = {}
            if self.config.enable_statistical_analysis:
                stats_results = self._statistical_analysis(packet_bytes)
            
            # Calculate overall confidence
            confidence = self._calculate_detection_confidence(
                signal_quality, pattern_results, stats_results, packet_regions
            )
            
            # Update statistics
            self._update_detection_stats(confidence, pattern_results)
            
            # Determine if detection is positive
            detected = confidence >= self.config.detection_threshold
            
            metadata = {
                "signal_quality": signal_quality,
                "packet_regions": len(packet_regions),
                "best_region_length": len(packet_samples),
                "pattern_analysis": pattern_results,
                "statistical_analysis": stats_results,
                "processing_time_ms": float((np.datetime64('now') - start_time) / np.timedelta64(1, 'ms'))
            }
            
            return ProtocolDetectionResult(
                detected=detected,
                confidence=confidence,
                protocol_name="generic",
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return ProtocolDetectionResult(
                detected=False,
                confidence=0.0,
                protocol_name="generic",
                metadata={"error": str(e)}
            )
    
    def parse_packet(
        self,
        packet_data: bytes,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProtocolParseResult:
        """
        Parse packet data using generic analysis techniques.
        
        Args:
            packet_data: Raw packet bytes
            metadata: Optional parsing metadata
            
        Returns:
            Parse result with extracted information
        """
        start_time = np.datetime64('now')
        
        try:
            # Validate packet length
            if len(packet_data) < self.config.min_packet_length:
                return ProtocolParseResult(
                    success=False,
                    protocol_name="generic",
                    metadata={"error": f"Packet too short: {len(packet_data)} bytes"}
                )
            
            if len(packet_data) > self.config.max_packet_length:
                logger.warning(f"Packet truncated from {len(packet_data)} to {self.config.max_packet_length} bytes")
                packet_data = packet_data[:self.config.max_packet_length]
            
            # Analyze packet structure
            structure = self._analyze_packet_structure(packet_data)
            
            # Extract fields based on detected structure
            fields = self._extract_generic_fields(packet_data, structure)
            
            # Validate extracted data
            validation_result = self._validate_generic_packet(fields, structure)
            
            # Calculate parse confidence
            parse_confidence = self._calculate_parse_confidence(structure, validation_result)
            
            # Prepare metadata
            parse_metadata = {
                "packet_length": len(packet_data),
                "structure": structure.__dict__,
                "validation": validation_result,
                "confidence": parse_confidence,
                "processing_time_ms": float((np.datetime64('now') - start_time) / np.timedelta64(1, 'ms'))
            }
            
            if self.config.include_raw_data:
                parse_metadata["raw_data_hex"] = packet_data.hex()
            
            if self.config.include_statistics:
                parse_metadata["statistics"] = self._packet_statistics(packet_data)
            
            return ProtocolParseResult(
                success=validation_result.get("is_valid", True),
                protocol_name="generic",
                parsed_data=fields,
                metadata=parse_metadata
            )
            
        except Exception as e:
            logger.error(f"Packet parsing failed: {e}")
            return ProtocolParseResult(
                success=False,
                protocol_name="generic", 
                metadata={"error": str(e)}
            )
    
    def encode_packet(
        self,
        packet_fields: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bytes:
        """
        Encode packet fields into raw bytes (generic implementation).
        
        Args:
            packet_fields: Dictionary of packet fields to encode
            metadata: Optional encoding metadata
            
        Returns:
            Encoded packet bytes
            
        Note:
            This is a basic implementation for generic packets.
            Protocol-specific plugins should override this method.
        """
        try:
            # Basic generic encoding
            packet = bytearray()
            
            # Add sync pattern if specified
            if "sync_pattern" in packet_fields:
                packet.extend(packet_fields["sync_pattern"])
            
            # Add length field if specified
            if "length" in packet_fields:
                length = packet_fields["length"]
                if length <= 255:
                    packet.append(length)
                else:
                    packet.extend(struct.pack(">H", length))  # Big-endian 16-bit
            
            # Add header fields
            if "header" in packet_fields:
                if isinstance(packet_fields["header"], bytes):
                    packet.extend(packet_fields["header"])
                elif isinstance(packet_fields["header"], dict):
                    # Encode header dictionary as key-value pairs
                    for key, value in packet_fields["header"].items():
                        if isinstance(value, int) and 0 <= value <= 255:
                            packet.append(value)
                        elif isinstance(value, bytes):
                            packet.extend(value)
            
            # Add payload
            if "payload" in packet_fields:
                payload = packet_fields["payload"]
                if isinstance(payload, bytes):
                    packet.extend(payload)
                elif isinstance(payload, str):
                    packet.extend(payload.encode('utf-8'))
                elif isinstance(payload, list):
                    packet.extend(bytes(payload))
            
            # Add checksum if specified
            if "checksum" in packet_fields:
                checksum = packet_fields["checksum"]
                if isinstance(checksum, int):
                    if checksum <= 255:
                        packet.append(checksum)
                    else:
                        packet.extend(struct.pack(">H", checksum))
                elif isinstance(checksum, bytes):
                    packet.extend(checksum)
            elif metadata and metadata.get("auto_checksum", False):
                # Auto-generate simple checksum
                checksum = sum(packet) & 0xFF
                packet.append(checksum)
            
            return bytes(packet)
            
        except Exception as e:
            logger.error(f"Packet encoding failed: {e}")
            raise ValueError(f"Failed to encode generic packet: {e}")
    
    def _detect_packet_regions(self, iq_samples: IQSamples) -> List[Tuple[int, int]]:
        """Detect potential packet regions in IQ samples."""
        if CORE_AVAILABLE:
            return detect_packets(iq_samples, threshold=0.1, min_gap=100)
        else:
            # Fallback detection
            power = np.abs(iq_samples) ** 2
            threshold = np.mean(power) * 2
            active = power > threshold
            
            # Find edges
            edges = np.diff(active.astype(int))
            starts = np.where(edges == 1)[0]
            ends = np.where(edges == -1)[0]
            
            # Pair starts and ends
            regions = []
            for start, end in zip(starts, ends):
                if end - start > 100:  # Minimum region size
                    regions.append((int(start), int(end)))
            
            return regions
    
    def _samples_to_bits(self, iq_samples: IQSamples) -> BitStream:
        """Convert IQ samples to bit stream using simple threshold."""
        # Simple OOK demodulation
        magnitude = np.abs(iq_samples)
        threshold = np.mean(magnitude)
        
        # Decimate to approximate bit rate
        samples_per_bit = max(1, len(iq_samples) // (len(iq_samples) // 8))
        decimated = magnitude[::samples_per_bit]
        
        # Threshold to bits
        bits = (decimated > threshold).astype(np.uint8)
        return bits
    
    def _bits_to_bytes(self, bits: BitStream) -> bytes:
        """Convert bit stream to bytes."""
        # Pad to byte boundary
        if len(bits) % 8 != 0:
            padding = 8 - (len(bits) % 8)
            bits = np.pad(bits, (0, padding), 'constant')
        
        # Pack bits to bytes
        return np.packbits(bits).tobytes()
    
    def _analyze_patterns(self, packet_bytes: bytes) -> Dict[str, Any]:
        """Analyze packet for known patterns."""
        results = {
            "known_patterns_found": [],
            "sync_candidates": [],
            "repeating_sequences": []
        }
        
        # Check for known patterns
        for pattern_name, pattern_bytes in self.known_patterns.items():
            if pattern_bytes in packet_bytes:
                offset = packet_bytes.find(pattern_bytes)
                results["known_patterns_found"].append({
                    "name": pattern_name,
                    "pattern": pattern_bytes.hex(),
                    "offset": offset
                })
        
        # Look for potential sync patterns (repeated bytes at start)
        if len(packet_bytes) >= 2:
            for i in range(min(4, len(packet_bytes) - 1)):
                if packet_bytes[i] == packet_bytes[i + 1]:
                    results["sync_candidates"].append({
                        "byte": f"0x{packet_bytes[i]:02X}",
                        "offset": i,
                        "length": 2
                    })
        
        # Find repeating sequences
        for seq_len in [2, 3, 4]:
            if len(packet_bytes) >= seq_len * 2:
                for i in range(len(packet_bytes) - seq_len * 2 + 1):
                    seq = packet_bytes[i:i + seq_len]
                    if seq == packet_bytes[i + seq_len:i + seq_len * 2]:
                        results["repeating_sequences"].append({
                            "sequence": seq.hex(),
                            "offset": i,
                            "length": seq_len
                        })
        
        return results
    
    def _statistical_analysis(self, packet_bytes: bytes) -> Dict[str, Any]:
        """Perform statistical analysis of packet content."""
        if len(packet_bytes) == 0:
            return {}
        
        # Basic statistics
        byte_array = np.frombuffer(packet_bytes, dtype=np.uint8)
        
        stats = {
            "length": len(packet_bytes),
            "mean": float(np.mean(byte_array)),
            "std": float(np.std(byte_array)),
            "min": int(np.min(byte_array)),
            "max": int(np.max(byte_array)),
            "unique_bytes": len(np.unique(byte_array)),
            "entropy": self._calculate_entropy(byte_array)
        }
        
        # Byte distribution
        byte_counts = np.bincount(byte_array, minlength=256)
        stats["most_common_byte"] = int(np.argmax(byte_counts))
        stats["most_common_count"] = int(np.max(byte_counts))
        
        # Randomness indicators
        stats["zero_bytes"] = int(np.sum(byte_array == 0))
        stats["ff_bytes"] = int(np.sum(byte_array == 255))
        
        return stats
    
    def _calculate_entropy(self, data: np.ndarray) -> float:
        """Calculate Shannon entropy of data."""
        if len(data) == 0:
            return 0.0
        
        _, counts = np.unique(data, return_counts=True)
        probabilities = counts / len(data)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return float(entropy)
    
    def _analyze_packet_structure(self, packet_bytes: bytes) -> GenericPacketStructure:
        """Analyze packet structure using heuristics."""
        structure = GenericPacketStructure(
            total_length=len(packet_bytes),
            estimated_header_length=self.config.assume_header_size,
            estimated_payload_length=0,
            estimated_checksum_length=1  # Assume 1-byte checksum
        )
        
        if len(packet_bytes) < 4:
            return structure
        
        # Look for length fields in first few bytes
        for i in range(min(4, len(packet_bytes) - 1)):
            potential_length = packet_bytes[i]
            if potential_length > 0 and potential_length <= len(packet_bytes):
                structure.length_field_offset = i
                structure.length_field_value = potential_length
                break
        
        # Estimate payload length
        header_len = structure.estimated_header_length
        checksum_len = structure.estimated_checksum_length
        structure.estimated_payload_length = max(0, len(packet_bytes) - header_len - checksum_len)
        
        # Look for sync patterns
        if len(packet_bytes) >= 2:
            # Check for common sync patterns
            if packet_bytes[0] == packet_bytes[1]:
                structure.sync_pattern = packet_bytes[:2]
            elif len(packet_bytes) >= 3 and packet_bytes[0] == packet_bytes[2]:
                structure.sync_pattern = packet_bytes[:1]
        
        # Calculate entropy for structure assessment
        byte_array = np.frombuffer(packet_bytes, dtype=np.uint8)
        structure.entropy = self._calculate_entropy(byte_array)
        
        # Byte distribution
        structure.byte_distribution = dict(zip(*np.unique(byte_array, return_counts=True)))
        
        # Structure confidence based on patterns found
        confidence_factors = []
        
        if structure.sync_pattern is not None:
            confidence_factors.append(0.3)
        
        if structure.length_field_offset is not None:
            confidence_factors.append(0.2)
        
        if 1.0 < structure.entropy < 7.0:  # Reasonable entropy range
            confidence_factors.append(0.2)
        
        if len(structure.byte_distribution) > 5:  # Reasonable byte diversity
            confidence_factors.append(0.1)
        
        structure.structure_confidence = sum(confidence_factors)
        
        return structure
    
    def _extract_generic_fields(
        self,
        packet_bytes: bytes,
        structure: GenericPacketStructure
    ) -> Dict[str, Any]:
        """Extract generic fields based on analyzed structure."""
        fields = {
            "packet_length": len(packet_bytes),
            "raw_data": packet_bytes
        }
        
        if len(packet_bytes) == 0:
            return fields
        
        # Extract sync pattern if detected
        if structure.sync_pattern:
            fields["sync_pattern"] = structure.sync_pattern
        
        # Extract potential header
        header_len = min(structure.estimated_header_length, len(packet_bytes))
        if header_len > 0:
            fields["header"] = packet_bytes[:header_len]
        
        # Extract length field if detected
        if structure.length_field_offset is not None:
            offset = structure.length_field_offset
            if offset < len(packet_bytes):
                fields["length_field"] = packet_bytes[offset]
        
        # Extract payload
        payload_start = header_len
        payload_end = len(packet_bytes) - structure.estimated_checksum_length
        if payload_start < payload_end:
            fields["payload"] = packet_bytes[payload_start:payload_end]
        
        # Extract potential checksum
        if structure.estimated_checksum_length > 0:
            checksum_start = len(packet_bytes) - structure.estimated_checksum_length
            fields["checksum"] = packet_bytes[checksum_start:]
        
        # Add human-readable representations
        if "header" in fields:
            fields["header_hex"] = fields["header"].hex()
        
        if "payload" in fields:
            fields["payload_hex"] = fields["payload"].hex()
            # Try to decode as text
            try:
                fields["payload_text"] = fields["payload"].decode('utf-8', errors='ignore')
            except:
                pass
        
        if "checksum" in fields:
            fields["checksum_hex"] = fields["checksum"].hex()
        
        return fields
    
    def _validate_generic_packet(
        self,
        fields: Dict[str, Any],
        structure: GenericPacketStructure
    ) -> Dict[str, Any]:
        """Validate extracted packet fields."""
        validation = {
            "is_valid": True,
            "validation_score": 0.0,
            "issues": []
        }
        
        score_factors = []
        
        # Check if packet has reasonable structure
        if "header" in fields and len(fields["header"]) > 0:
            score_factors.append(0.2)
        
        if "payload" in fields and len(fields["payload"]) > 0:
            score_factors.append(0.3)
        
        # Check length consistency
        if "length_field" in fields:
            declared_length = fields["length_field"]
            actual_length = fields["packet_length"]
            if abs(declared_length - actual_length) <= 2:  # Allow small discrepancies
                score_factors.append(0.2)
            else:
                validation["issues"].append(f"Length mismatch: declared {declared_length}, actual {actual_length}")
        
        # Check entropy (not too random, not too structured)
        if 0.5 <= structure.entropy <= 7.5:
            score_factors.append(0.1)
        else:
            validation["issues"].append(f"Unusual entropy: {structure.entropy:.2f}")
        
        # Check for reasonable byte distribution
        if len(structure.byte_distribution) >= 3:
            score_factors.append(0.1)
        
        # Simple checksum validation (if applicable)
        if "checksum" in fields and "payload" in fields:
            calculated_checksum = sum(fields["payload"]) & 0xFF
            received_checksum = fields["checksum"][0] if fields["checksum"] else 0
            if calculated_checksum == received_checksum:
                score_factors.append(0.1)
            else:
                validation["issues"].append("Checksum validation failed")
        
        validation["validation_score"] = sum(score_factors)
        validation["is_valid"] = validation["validation_score"] >= 0.3 and len(validation["issues"]) == 0
        
        return validation
    
    def _calculate_detection_confidence(
        self,
        signal_quality: Dict[str, Any],
        pattern_results: Dict[str, Any],
        stats_results: Dict[str, Any],
        packet_regions: List[Tuple[int, int]]
    ) -> float:
        """Calculate overall detection confidence."""
        confidence_factors = []
        
        # Signal quality contribution
        if signal_quality:
            snr = signal_quality.get("estimated_snr_db", 0)
            if snr > 10:
                confidence_factors.append(0.3)
            elif snr > 5:
                confidence_factors.append(0.2)
            elif snr > 0:
                confidence_factors.append(0.1)
        
        # Pattern detection contribution
        if pattern_results.get("known_patterns_found"):
            confidence_factors.append(0.4)
        elif pattern_results.get("sync_candidates"):
            confidence_factors.append(0.2)
        
        # Statistical analysis contribution
        if stats_results:
            entropy = stats_results.get("entropy", 0)
            if 1.0 <= entropy <= 7.0:  # Reasonable entropy
                confidence_factors.append(0.2)
            
            unique_ratio = stats_results.get("unique_bytes", 0) / 256
            if 0.1 <= unique_ratio <= 0.8:  # Good byte diversity
                confidence_factors.append(0.1)
        
        # Packet region contribution
        if len(packet_regions) > 0:
            confidence_factors.append(0.1)
        
        return min(1.0, sum(confidence_factors))
    
    def _calculate_parse_confidence(
        self,
        structure: GenericPacketStructure,
        validation: Dict[str, Any]
    ) -> float:
        """Calculate parsing confidence."""
        base_confidence = validation.get("validation_score", 0.0)
        structure_confidence = structure.structure_confidence
        
        # Combine confidences
        combined = (base_confidence + structure_confidence) / 2
        
        # Penalize for validation issues
        issue_penalty = len(validation.get("issues", [])) * 0.1
        
        return max(0.0, min(1.0, combined - issue_penalty))
    
    def _packet_statistics(self, packet_bytes: bytes) -> Dict[str, Any]:
        """Generate comprehensive packet statistics."""
        if len(packet_bytes) == 0:
            return {}
        
        byte_array = np.frombuffer(packet_bytes, dtype=np.uint8)
        
        return {
            "total_bytes": len(packet_bytes),
            "entropy": self._calculate_entropy(byte_array),
            "mean_value": float(np.mean(byte_array)),
            "std_value": float(np.std(byte_array)),
            "min_value": int(np.min(byte_array)),
            "max_value": int(np.max(byte_array)),
            "unique_bytes": len(np.unique(byte_array)),
            "zero_bytes": int(np.sum(byte_array == 0)),
            "printable_bytes": sum(1 for b in packet_bytes if 32 <= b <= 126),
            "most_frequent_byte": f"0x{np.argmax(np.bincount(byte_array)):02X}",
            "byte_distribution_entropy": self._calculate_entropy(np.bincount(byte_array))
        }
    
    def _update_detection_stats(
        self,
        confidence: float,
        pattern_results: Dict[str, Any]
    ) -> None:
        """Update plugin detection statistics."""
        self.detection_stats["packets_analyzed"] += 1
        
        if confidence >= self.config.detection_threshold:
            self.detection_stats["successful_detections"] += 1
        
        # Update pattern statistics
        for pattern_info in pattern_results.get("known_patterns_found", []):
            pattern_name = pattern_info["name"]
            self.detection_stats["pattern_matches"][pattern_name] = (
                self.detection_stats["pattern_matches"].get(pattern_name, 0) + 1
            )
        
        # Update average confidence
        total_detections = self.detection_stats["packets_analyzed"]
        prev_avg = self.detection_stats["avg_confidence"]
        self.detection_stats["avg_confidence"] = (
            (prev_avg * (total_detections - 1) + confidence) / total_detections
        )
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get plugin detection statistics."""
        stats = self.detection_stats.copy()
        
        if stats["packets_analyzed"] > 0:
            stats["success_rate"] = stats["successful_detections"] / stats["packets_analyzed"]
        else:
            stats["success_rate"] = 0.0
        
        return stats


# Factory function for creating generic plugin instances
def create_generic_plugin(config: Optional[GenericProtocolConfig] = None) -> GenericProtocolPlugin:
    """Create a generic protocol plugin instance."""
    return GenericProtocolPlugin(config)


# Convenience function for quick protocol analysis
def analyze_unknown_packet(packet_bytes: bytes) -> Dict[str, Any]:
    """
    Quick analysis of unknown packet using generic plugin.
    
    Args:
        packet_bytes: Raw packet data
        
    Returns:
        Analysis results
    """
    plugin = GenericProtocolPlugin()
    result = plugin.parse_packet(packet_bytes)
    
    return {
        "success": result.success,
        "protocol": result.protocol_name,
        "parsed_data": result.parsed_data,
        "metadata": result.metadata
    }


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    print("=== Generic Protocol Plugin Demo ===")
    
    # Create plugin
    config = GenericProtocolConfig(
        enable_statistical_analysis=True,
        enable_structure_detection=True,
        include_statistics=True
    )
    plugin = GenericProtocolPlugin(config)
    
    # Test packets
    test_packets = [
        # MAVLink-like packet
        b'\xFE\x21\x00\x01\x01\x00' + b'\x42' * 33 + b'\x12\x34',
        
        # DJI-like packet  
        b'\x55\xAA\x27\x10' + b'\x33' * 35 + b'\x56\x78',
        
        # Generic data packet
        b'\x7E\x10Hello World\x00\x00\x00\x00\xDE',
        
        # Random data
        bytes(np.random.randint(0, 256, 32))
    ]
    
    packet_names = ["MAVLink-like", "DJI-like", "Generic", "Random"]
    
    for name, packet in zip(packet_names, test_packets):
        print(f"\n--- Testing {name} packet ({len(packet)} bytes) ---")
        
        # Parse packet
        result = plugin.parse_packet(packet)
        
        print(f"Parse success: {result.success}")
        print(f"Protocol: {result.protocol_name}")
        
        if result.parsed_data:
            print(f"Fields found: {list(result.parsed_data.keys())}")
            
            if "payload" in result.parsed_data:
                payload = result.parsed_data["payload"]
                print(f"Payload length: {len(payload) if payload else 0} bytes")
        
        if result.metadata:
            structure = result.metadata.get("structure", {})
            confidence = result.metadata.get("confidence", 0)
            print(f"Structure confidence: {confidence:.2f}")
            print(f"Estimated header length: {structure.get('estimated_header_length', 0)}")
    
    # Show statistics
    print(f"\n--- Plugin Statistics ---")
    stats = plugin.get_detection_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")