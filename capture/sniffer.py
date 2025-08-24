#!/usr/bin/env python3
"""
Simplified Packet Sniffing Interface

This module provides a simplified interface for packet sniffing operations,
acting as an adapter layer between simple usage patterns and the enhanced
core systems for parsing and classification.

Key Features:
- Simple packet sniffing from IQ files
- Automatic protocol detection and classification
- Integration with enhanced parsing systems
- Backward compatible API
- Progressive feature exposure

Usage:
    >>> sniffer = PacketSniffer("capture.iq")
    >>> packets = sniffer.sniff()
    >>> for packet in packets:
    ...     print(f"Protocol: {packet['protocol']}, Length: {packet['length']}")
"""

from __future__ import annotations

import logging
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt

# Import from core and capture modules
try:
    from .manager import CaptureManager
    from ..core.parsing import EnhancedPacketParser, ParserConfig
    from ..core.classification import EnhancedProtocolClassifier, ClassifierConfig
    from ..core.signal_processing import detect_packets, QualityMonitor
    from ..core.demodulation import DemodulationEngine, DemodConfig, ModulationScheme
    from ..utils.fileio import read_iq_file
    ENHANCED_AVAILABLE = True
except ImportError:
    # Fallback for development/testing
    ENHANCED_AVAILABLE = False
    warnings.warn("Enhanced modules not available, using fallback implementations")

# Configure module logger
logger = logging.getLogger(__name__)

# Type aliases
IQSamples = npt.NDArray[np.complex64]
SniffedPacket = Dict[str, Any]


class PacketSnifferError(Exception):
    """Exception raised by packet sniffer operations."""
    pass


class PacketSniffer:
    """
    Simplified packet sniffer for extracting and analyzing packets from IQ data.
    
    This class provides an easy-to-use interface for packet sniffing operations
    while leveraging the enhanced core systems for parsing and classification.
    
    Example:
        >>> sniffer = PacketSniffer("drone_capture.iq")
        >>> packets = sniffer.sniff(threshold=0.05)
        >>> print(f"Found {len(packets)} packets")
        >>> 
        >>> # Analyze specific protocols
        >>> mavlink_packets = sniffer.filter_by_protocol("mavlink")
        >>> dji_packets = sniffer.filter_by_protocol("dji")
    """
    
    def __init__(
        self,
        iq_file: Optional[Union[str, Path]] = None,
        sample_rate: float = 2.048e6,
        **kwargs: Any
    ) -> None:
        """
        Initialize packet sniffer.
        
        Args:
            iq_file: Path to IQ file to analyze
            sample_rate: Sample rate of IQ data in Hz
            **kwargs: Additional configuration parameters
        """
        self.iq_file = Path(iq_file) if iq_file else None
        self.sample_rate = sample_rate
        self.iq_data = None
        
        # Configuration
        self.enable_classification = kwargs.get('enable_classification', True)
        self.enable_parsing = kwargs.get('enable_parsing', True)
        self.enable_demodulation = kwargs.get('enable_demodulation', True)
        
        # Enhanced components (if available)
        self._parser = None
        self._classifier = None
        self._demodulator = None
        self._quality_monitor = None
        
        # Results cache
        self._sniffed_packets = []
        self._last_sniff_params = None
        
        # Statistics
        self.stats = {
            'packets_sniffed': 0,
            'protocols_detected': {},
            'total_processing_time': 0.0,
            'last_sniff_time': None
        }
        
        # Initialize enhanced components
        if ENHANCED_AVAILABLE:
            self._init_enhanced_components()
        
        # Load IQ file if provided
        if self.iq_file:
            self.load_file(self.iq_file)
        
        logger.info(f"Initialized packet sniffer for {iq_file or 'manual data'}")
    
    def _init_enhanced_components(self) -> None:
        """Initialize enhanced core components."""
        try:
            # Initialize parser
            if self.enable_parsing:
                parser_config = ParserConfig(
                    enable_protocol_detection=True,
                    enable_error_correction=True,
                    enable_quality_assessment=True
                )
                self._parser = EnhancedPacketParser(parser_config)
            
            # Initialize classifier
            if self.enable_classification:
                classifier_config = ClassifierConfig(
                    enable_ensemble=True,
                    confidence_threshold=0.7,
                    performance_monitoring=True
                )
                self._classifier = EnhancedProtocolClassifier(classifier_config)
            
            # Initialize quality monitor
            self._quality_monitor = QualityMonitor()
            
            logger.debug("Enhanced components initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize enhanced components: {e}")
            self._parser = None
            self._classifier = None
            self._quality_monitor = None
    
    def load_file(self, iq_file: Union[str, Path]) -> None:
        """
        Load IQ data from file.
        
        Args:
            iq_file: Path to IQ file
            
        Raises:
            PacketSnifferError: If file cannot be loaded
        """
        try:
            self.iq_file = Path(iq_file)
            self.iq_data = read_iq_file(str(iq_file))
            
            # Clear previous results
            self._sniffed_packets = []
            self._last_sniff_params = None
            
            logger.info(f"Loaded {len(self.iq_data):,} IQ samples from {iq_file}")
            
        except Exception as e:
            logger.error(f"Failed to load IQ file {iq_file}: {e}")
            raise PacketSnifferError(f"Failed to load IQ file: {e}") from e
    
    def set_iq_data(self, iq_data: IQSamples, sample_rate: Optional[float] = None) -> None:
        """
        Set IQ data directly.
        
        Args:
            iq_data: Complex IQ samples
            sample_rate: Sample rate in Hz (optional)
        """
        self.iq_data = iq_data
        if sample_rate is not None:
            self.sample_rate = sample_rate
        
        # Clear previous results
        self._sniffed_packets = []
        self._last_sniff_params = None
        
        logger.info(f"Set IQ data: {len(iq_data):,} samples")
    
    def sniff(
        self,
        threshold: float = 0.05,
        min_packet_length: int = 1000,
        max_packets: Optional[int] = None,
        enable_demodulation: bool = True,
        modulation_schemes: Optional[List[str]] = None
    ) -> List[SniffedPacket]:
        """
        Sniff packets from IQ data.
        
        Args:
            threshold: Detection threshold for packet detection
            min_packet_length: Minimum packet length in samples
            max_packets: Maximum number of packets to process
            enable_demodulation: Enable demodulation of detected packets
            modulation_schemes: List of modulation schemes to try
            
        Returns:
            List of sniffed packet dictionaries
            
        Raises:
            PacketSnifferError: If sniffing fails
        """
        if self.iq_data is None:
            raise PacketSnifferError("No IQ data available")
        
        # Check if we can reuse cached results
        current_params = (threshold, min_packet_length, max_packets, enable_demodulation)
        if self._last_sniff_params == current_params and self._sniffed_packets:
            logger.debug("Returning cached sniff results")
            return self._sniffed_packets
        
        start_time = time.time()
        
        try:
            # Step 1: Detect packet regions
            packet_regions = self._detect_packet_regions(threshold, min_packet_length)
            
            if max_packets is not None:
                packet_regions = packet_regions[:max_packets]
            
            logger.info(f"Detected {len(packet_regions)} packet regions")
            
            # Step 2: Process each packet
            sniffed_packets = []
            
            for i, (start, end) in enumerate(packet_regions):
                try:
                    packet_iq = self.iq_data[start:end]
                    packet_info = self._process_packet(
                        packet_iq, i, start, end, enable_demodulation, modulation_schemes
                    )
                    sniffed_packets.append(packet_info)
                    
                except Exception as e:
                    logger.warning(f"Failed to process packet {i}: {e}")
                    # Add error packet
                    sniffed_packets.append({
                        'packet_id': i,
                        'start_sample': start,
                        'end_sample': end,
                        'length_samples': end - start,
                        'error': str(e),
                        'valid': False
                    })
            
            # Update cache and statistics
            self._sniffed_packets = sniffed_packets
            self._last_sniff_params = current_params
            
            processing_time = time.time() - start_time
            self._update_statistics(sniffed_packets, processing_time)
            
            logger.info(f"Sniffed {len(sniffed_packets)} packets in {processing_time:.2f}s")
            return sniffed_packets
            
        except Exception as e:
            logger.error(f"Packet sniffing failed: {e}")
            raise PacketSnifferError(f"Packet sniffing failed: {e}") from e
    
    def _detect_packet_regions(
        self,
        threshold: float,
        min_length: int
    ) -> List[tuple[int, int]]:
        """Detect packet regions in IQ data."""
        if ENHANCED_AVAILABLE:
            # Use enhanced detection
            return detect_packets(self.iq_data, threshold, min_length)
        else:
            # Fallback detection
            return self._simple_packet_detection(threshold, min_length)
    
    def _process_packet(
        self,
        packet_iq: IQSamples,
        packet_id: int,
        start_sample: int,
        end_sample: int,
        enable_demodulation: bool,
        modulation_schemes: Optional[List[str]]
    ) -> SniffedPacket:
        """Process individual packet with full analysis."""
        packet_info = {
            'packet_id': packet_id,
            'start_sample': start_sample,
            'end_sample': end_sample,
            'length_samples': len(packet_iq),
            'duration_s': len(packet_iq) / self.sample_rate,
            'valid': True,
            'timestamp': time.time()
        }
        
        # Signal quality analysis
        if self._quality_monitor:
            quality_metrics = self._quality_monitor.update(packet_iq)
            packet_info['signal_quality'] = quality_metrics
        else:
            # Basic quality metrics
            avg_power = np.mean(np.abs(packet_iq) ** 2)
            packet_info['signal_quality'] = {
                'signal_power_dbfs': 10 * np.log10(avg_power + 1e-12)
            }
        
        # Try demodulation if enabled
        demod_results = {}
        packet_bytes = None
        
        if enable_demodulation:
            if modulation_schemes is None:
                modulation_schemes = ['ook', 'fsk', 'psk']
            
            demod_results = self._try_demodulation(packet_iq, modulation_schemes)
            
            # Use best demodulation result
            best_scheme = None
            best_snr = -np.inf
            
            for scheme, result in demod_results.items():
                if result.get('snr_db', -np.inf) > best_snr:
                    best_snr = result['snr_db']
                    best_scheme = scheme
                    packet_bytes = result.get('packet_bytes')
            
            packet_info['demodulation'] = {
                'best_scheme': best_scheme,
                'best_snr_db': best_snr,
                'schemes_tried': list(demod_results.keys()),
                'results': demod_results
            }
        
        # Protocol classification
        if packet_bytes and self._classifier:
            try:
                classification_result = self._classifier.classify(packet_bytes)
                
                if hasattr(classification_result, 'predicted_protocol'):
                    packet_info['protocol'] = {
                        'detected': classification_result.predicted_protocol,
                        'confidence': classification_result.confidence,
                        'alternatives': classification_result.top_k_predictions[:3]
                    }
                else:
                    packet_info['protocol'] = {
                        'detected': str(classification_result),
                        'confidence': 0.5,
                        'alternatives': []
                    }
                    
            except Exception as e:
                logger.debug(f"Protocol classification failed: {e}")
                packet_info['protocol'] = {'error': str(e)}
        
        # Packet parsing (if we have bytes and parser)
        if packet_bytes and self._parser:
            try:
                parse_result = self._parser.parse_packet(packet_bytes)
                packet_info['parsed'] = {
                    'success': parse_result.is_valid,
                    'protocol': parse_result.detected_protocol,
                    'fields': parse_result.parsed_fields,
                    'confidence': parse_result.confidence
                }
                
            except Exception as e:
                logger.debug(f"Packet parsing failed: {e}")
                packet_info['parsed'] = {'error': str(e)}
        
        # Add raw data info
        packet_info['raw_data'] = {
            'iq_samples': len(packet_iq),
            'packet_bytes': len(packet_bytes) if packet_bytes else 0,
            'has_iq_data': True,
            'has_packet_bytes': packet_bytes is not None
        }
        
        return packet_info
    
    def _try_demodulation(
        self,
        packet_iq: IQSamples,
        schemes: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Try demodulation with different schemes."""
        results = {}
        
        if not ENHANCED_AVAILABLE:
            # Simple fallback demodulation
            return self._simple_demodulation(packet_iq, schemes)
        
        # Map scheme names to enums
        scheme_map = {
            'ook': ModulationScheme.OOK,
            'ask': ModulationScheme.ASK,
            'fsk': ModulationScheme.FSK,
            'gfsk': ModulationScheme.GFSK,
            'psk': ModulationScheme.PSK,
            'bpsk': ModulationScheme.BPSK,
            'qpsk': ModulationScheme.QPSK
        }
        
        for scheme_name in schemes:
            scheme_enum = scheme_map.get(scheme_name.lower())
            if scheme_enum is None:
                continue
            
            try:
                # Create demodulation config
                demod_config = DemodConfig(
                    scheme=scheme_enum,
                    sample_rate_hz=self.sample_rate,
                    bitrate_bps=9600,  # Standard bitrate
                    enable_adaptive_threshold=True
                )
                
                # Create and run demodulator
                demodulator = DemodulationEngine(demod_config)
                demod_result = demodulator.demodulate(packet_iq)
                
                if demod_result.is_valid:
                    # Convert bits to bytes
                    bits = demod_result.bits
                    if len(bits) >= 8:
                        # Pad to byte boundary
                        if len(bits) % 8 != 0:
                            padding = 8 - (len(bits) % 8)
                            bits = np.pad(bits, (0, padding), 'constant')
                        
                        packet_bytes = np.packbits(bits).tobytes()
                    else:
                        packet_bytes = b''
                    
                    results[scheme_name] = {
                        'success': True,
                        'bits_decoded': len(demod_result.bits),
                        'packet_bytes': packet_bytes,
                        'snr_db': demod_result.snr_db or 0.0,
                        'signal_power_dbfs': demod_result.signal_power_dbfs or -60.0,
                        'timing_offset': demod_result.timing_offset_samples or 0.0,
                        'processing_time_ms': demod_result.processing_time_ms
                    }
                else:
                    results[scheme_name] = {
                        'success': False,
                        'error': demod_result.error_message or 'Demodulation failed'
                    }
                    
            except Exception as e:
                results[scheme_name] = {
                    'success': False,
                    'error': str(e)
                }
        
        return results
    
    def filter_by_protocol(self, protocol: str) -> List[SniffedPacket]:
        """
        Filter sniffed packets by detected protocol.
        
        Args:
            protocol: Protocol name to filter by
            
        Returns:
            List of packets matching the protocol
        """
        if not self._sniffed_packets:
            logger.warning("No sniffed packets available, run sniff() first")
            return []
        
        filtered = []
        for packet in self._sniffed_packets:
            detected_protocol = packet.get('protocol', {}).get('detected', '').lower()
            if protocol.lower() in detected_protocol:
                filtered.append(packet)
        
        logger.info(f"Filtered {len(filtered)} packets for protocol '{protocol}'")
        return filtered
    
    def filter_by_quality(
        self,
        min_snr_db: Optional[float] = None,
        min_power_dbfs: Optional[float] = None
    ) -> List[SniffedPacket]:
        """
        Filter sniffed packets by signal quality.
        
        Args:
            min_snr_db: Minimum SNR in dB
            min_power_dbfs: Minimum power in dBFS
            
        Returns:
            List of packets meeting quality criteria
        """
        if not self._sniffed_packets:
            logger.warning("No sniffed packets available, run sniff() first")
            return []
        
        filtered = []
        for packet in self._sniffed_packets:
            # Check SNR
            if min_snr_db is not None:
                best_snr = packet.get('demodulation', {}).get('best_snr_db', -np.inf)
                if best_snr < min_snr_db:
                    continue
            
            # Check power
            if min_power_dbfs is not None:
                power = packet.get('signal_quality', {}).get('signal_power_dbfs', -np.inf)
                if power < min_power_dbfs:
                    continue
            
            filtered.append(packet)
        
        logger.info(f"Filtered {len(filtered)} packets by quality criteria")
        return filtered
    
    def get_protocol_summary(self) -> Dict[str, Any]:
        """
        Get summary of detected protocols.
        
        Returns:
            Dictionary with protocol statistics
        """
        if not self._sniffed_packets:
            return {}
        
        protocol_counts = {}
        total_packets = len(self._sniffed_packets)
        
        for packet in self._sniffed_packets:
            protocol = packet.get('protocol', {}).get('detected', 'unknown')
            protocol_counts[protocol] = protocol_counts.get(protocol, 0) + 1
        
        # Calculate percentages
        protocol_percentages = {
            proto: (count / total_packets) * 100
            for proto, count in protocol_counts.items()
        }
        
        return {
            'total_packets': total_packets,
            'protocols_detected': len(protocol_counts),
            'protocol_counts': protocol_counts,
            'protocol_percentages': protocol_percentages,
            'most_common': max(protocol_counts.items(), key=lambda x: x[1]) if protocol_counts else None
        }
    
    def export_packets(
        self,
        filename: Union[str, Path],
        format: str = 'json',
        include_raw_data: bool = False
    ) -> None:
        """
        Export sniffed packets to file.
        
        Args:
            filename: Output filename
            format: Export format ('json', 'csv')
            include_raw_data: Include raw IQ/byte data
        """
        if not self._sniffed_packets:
            raise PacketSnifferError("No packets to export")
        
        filepath = Path(filename)
        
        # Prepare data for export
        export_data = []
        for packet in self._sniffed_packets:
            if include_raw_data:
                export_packet = packet.copy()
            else:
                # Remove heavy data
                export_packet = {k: v for k, v in packet.items() 
                               if k not in ['raw_data']}
        
        try:
            if format.lower() == 'json':
                import json
                with open(filepath, 'w') as f:
                    json.dump(self._sniffed_packets, f, indent=2, default=str)
            
            elif format.lower() == 'csv':
                import csv
                
                # Flatten packet data for CSV
                flattened_data = []
                for packet in self._sniffed_packets:
                    flat_packet = self._flatten_packet_for_csv(packet)
                    flattened_data.append(flat_packet)
                
                if flattened_data:
                    fieldnames = flattened_data[0].keys()
                    with open(filepath, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(flattened_data)
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            logger.info(f"Exported {len(self._sniffed_packets)} packets to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export packets: {e}")
            raise PacketSnifferError(f"Export failed: {e}") from e
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get packet sniffer statistics."""
        stats = self.stats.copy()
        stats.update({
            'iq_file': str(self.iq_file) if self.iq_file else None,
            'sample_rate_mhz': self.sample_rate / 1e6,
            'iq_samples_loaded': len(self.iq_data) if self.iq_data is not None else 0,
            'packets_cached': len(self._sniffed_packets),
            'enhanced_mode': ENHANCED_AVAILABLE,
            'components_enabled': {
                'parsing': self._parser is not None,
                'classification': self._classifier is not None,
                'demodulation': self.enable_demodulation,
                'quality_monitoring': self._quality_monitor is not None
            }
        })
        
        return stats
    
    def _update_statistics(self, packets: List[SniffedPacket], processing_time: float) -> None:
        """Update sniffer statistics."""
        self.stats['packets_sniffed'] = len(packets)
        self.stats['total_processing_time'] = processing_time
        self.stats['last_sniff_time'] = time.time()
        
        # Count protocols
        protocol_counts = {}
        for packet in packets:
            protocol = packet.get('protocol', {}).get('detected', 'unknown')
            protocol_counts[protocol] = protocol_counts.get(protocol, 0) + 1
        
        self.stats['protocols_detected'] = protocol_counts
    
    def _simple_packet_detection(
        self,
        threshold: float,
        min_length: int
    ) -> List[tuple[int, int]]:
        """Simple fallback packet detection."""
        power = np.abs(self.iq_data)
        active = power > (threshold * np.max(power))
        
        # Find edges
        edges = np.diff(active.astype(int))
        starts = np.where(edges == 1)[0] + 1
        ends = np.where(edges == -1)[0] + 1
        
        # Handle edge cases
        if len(ends) > 0 and (len(starts) == 0 or starts[0] > ends[0]):
            starts = np.insert(starts, 0, 0)
        if len(starts) > 0 and (len(ends) == 0 or ends[-1] < starts[-1]):
            ends = np.append(ends, len(active))
        
        # Filter short packets
        packets = []
        for start, end in zip(starts, ends):
            if end - start >= min_length:
                packets.append((int(start), int(end)))
        
        return packets
    
    def _simple_demodulation(
        self,
        packet_iq: IQSamples,
        schemes: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Simple fallback demodulation."""
        results = {}
        
        for scheme in schemes:
            if scheme.lower() in ['ook', 'ask']:
                # Simple envelope detection
                envelope = np.abs(packet_iq)
                threshold = np.mean(envelope) * 1.2
                bits = (envelope > threshold).astype(np.uint8)
                
                if len(bits) >= 8:
                    # Convert to bytes
                    if len(bits) % 8 != 0:
                        bits = bits[:-(len(bits) % 8)]
                    packet_bytes = np.packbits(bits).tobytes()
                else:
                    packet_bytes = b''
                
                results[scheme] = {
                    'success': True,
                    'bits_decoded': len(bits),
                    'packet_bytes': packet_bytes,
                    'snr_db': 10.0,  # Placeholder
                    'signal_power_dbfs': -20.0,  # Placeholder
                    'method': 'envelope_detection'
                }
            else:
                results[scheme] = {
                    'success': False,
                    'error': f'Scheme {scheme} not supported in fallback mode'
                }
        
        return results
    
    def _flatten_packet_for_csv(self, packet: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten nested packet dictionary for CSV export."""
        flat = {}
        
        # Basic fields
        for key in ['packet_id', 'start_sample', 'end_sample', 'length_samples', 'duration_s', 'valid']:
            flat[key] = packet.get(key, '')
        
        # Signal quality
        quality = packet.get('signal_quality', {})
        flat['signal_power_dbfs'] = quality.get('signal_power_dbfs', '')
        flat['snr_db'] = quality.get('snr_db', '')
        
        # Protocol
        protocol = packet.get('protocol', {})
        flat['detected_protocol'] = protocol.get('detected', '')
        flat['protocol_confidence'] = protocol.get('confidence', '')
        
        # Demodulation
        demod = packet.get('demodulation', {})
        flat['best_scheme'] = demod.get('best_scheme', '')
        flat['best_snr_db'] = demod.get('best_snr_db', '')
        
        # Parsed data
        parsed = packet.get('parsed', {})
        flat['parse_success'] = parsed.get('success', '')
        flat['parsed_protocol'] = parsed.get('protocol', '')
        
        return flat


# Backward compatibility function
def create_packet_sniffer(iq_file: Union[str, Path]) -> PacketSniffer:
    """Create packet sniffer with backward compatibility."""
    return PacketSniffer(iq_file)


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    print("=== Packet Sniffer Demo ===")
    
    try:
        # Create test IQ data since we might not have real files
        print("Creating test IQ data...")
        duration = 1.0  # 1 second
        sample_rate = 2.048e6
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create test signal with packets
        signal = np.zeros(len(t), dtype=np.complex64)
        
        # Add some "packets" - bursts of modulated signal
        packet_starts = [0.1, 0.3, 0.5, 0.7]  # Start times
        packet_duration = 0.05  # 50ms packets
        
        for start_time in packet_starts:
            start_idx = int(start_time * sample_rate)
            end_idx = int((start_time + packet_duration) * sample_rate)
            
            # Simple OOK signal (alternating on/off)
            packet_t = t[start_idx:end_idx]
            data_bits = np.random.randint(0, 2, len(packet_t) // 100)
            modulated = np.repeat(data_bits, 100)[:len(packet_t)]
            signal[start_idx:end_idx] = modulated * np.exp(1j * 2 * np.pi * 1000 * packet_t)
        
        # Add noise
        noise = 0.1 * (np.random.normal(0, 1, len(signal)) + 
                      1j * np.random.normal(0, 1, len(signal)))
        signal += noise
        
        print(f"Generated {len(signal):,} IQ samples with {len(packet_starts)} test packets")
        
        # Create sniffer and set data
        sniffer = PacketSniffer(sample_rate=sample_rate)
        sniffer.set_iq_data(signal, sample_rate)
        
        # Sniff packets
        print("\nSniffing packets...")
        packets = sniffer.sniff(threshold=0.2, min_packet_length=1000)
        
        print(f"Found {len(packets)} packets:")
        for i, packet in enumerate(packets):
            print(f"  Packet {i}: {packet['length_samples']} samples, "
                  f"{packet['duration_s']*1000:.1f}ms")
            
            if 'demodulation' in packet:
                best_scheme = packet['demodulation'].get('best_scheme', 'none')
                best_snr = packet['demodulation'].get('best_snr_db', 0)
                print(f"    Best demod: {best_scheme} (SNR: {best_snr:.1f} dB)")
            
            if 'protocol' in packet:
                detected = packet['protocol'].get('detected', 'unknown')
                confidence = packet['protocol'].get('confidence', 0)
                print(f"    Protocol: {detected} (confidence: {confidence:.2f})")
        
        # Get protocol summary
        summary = sniffer.get_protocol_summary()
        print(f"\nProtocol Summary:")
        print(f"  Total packets: {summary.get('total_packets', 0)}")
        print(f"  Protocols detected: {summary.get('protocols_detected', 0)}")
        
        for protocol, count in summary.get('protocol_counts', {}).items():
            percentage = summary.get('protocol_percentages', {}).get(protocol, 0)
            print(f"    {protocol}: {count} packets ({percentage:.1f}%)")
        
        # Show statistics
        stats = sniffer.get_statistics()
        print(f"\nSniffer Statistics:")
        print(f"  Enhanced mode: {stats['enhanced_mode']}")
        print(f"  Processing time: {stats['total_processing_time']:.3f}s")
        print(f"  IQ samples: {stats['iq_samples_loaded']:,}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()