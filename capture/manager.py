#!/usr/bin/env python3
"""
Simplified Capture Manager

This module provides a simplified interface to the enhanced live capture system
while maintaining backward compatibility with existing code. It acts as an
adapter layer that exposes simple methods while leveraging the advanced
features of the core capture system.

Key Features:
- Backward compatible API
- Automatic configuration management  
- Simplified error handling
- Integration with enhanced core systems
- Progressive feature exposure

Usage:
    >>> manager = CaptureManager()
    >>> manager.set_frequency(2.44e9)
    >>> filename = manager.capture_to_file("test.iq", duration=10)
    >>> packets = manager.extract_packets(threshold=0.05)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

# Import from core modules
try:
    from ..core.capture import EnhancedLiveCapture, SDRConfig, SDRPlatform, GainMode, CaptureFormat
    from ..core.signal_processing import detect_packets, find_preamble, SignalProcessor, QualityMonitor
    from ..utils.fileio import read_iq_file, write_iq_file
    CORE_AVAILABLE = True
except ImportError:
    # Fallback for development/testing
    CORE_AVAILABLE = False
    warnings.warn("Core modules not available, using fallback implementations")

# Configure module logger
logger = logging.getLogger(__name__)

# Type aliases
IQSamples = npt.NDArray[np.complex64]
FrequencyHz = float
SampleRateHz = float


class CaptureManagerError(Exception):
    """Exception raised by capture manager operations."""
    pass


class CaptureManager:
    """
    Simplified capture manager providing easy-to-use SDR capture functionality.
    
    This class wraps the enhanced live capture system with a simple API
    that's backward compatible with existing code while providing access
    to advanced features when needed.
    
    Example:
        >>> manager = CaptureManager()
        >>> manager.set_frequency(100.1e6)  # FM radio
        >>> filename = manager.capture_to_file("fm_capture.iq", duration=30)
        >>> iq_data = manager.load_file(filename)
        >>> packets = manager.extract_packets(iq_data)
    """
    
    def __init__(
        self,
        platform: str = "rtl_sdr",
        sample_rate: SampleRateHz = 2.048e6,
        device_index: int = 0,
        **kwargs: Any
    ) -> None:
        """
        Initialize capture manager with simplified configuration.
        
        Args:
            platform: SDR platform ("rtl_sdr", "hackrf", "airspy", etc.)
            sample_rate: Sample rate in Hz
            device_index: Device index for multiple SDRs
            **kwargs: Additional configuration parameters
        """
        # Store configuration
        self.platform = platform
        self.sample_rate = sample_rate
        self.device_index = device_index
        self.frequency = kwargs.get('frequency', 100.1e6)
        self.gain_mode = kwargs.get('gain_mode', 'auto')
        self.gain_db = kwargs.get('gain_db', None)
        
        # Enhanced components (if available)
        self._enhanced_capture = None
        self._signal_processor = None
        self._quality_monitor = None
        self._is_connected = False
        
        # Initialize enhanced components
        if CORE_AVAILABLE:
            self._init_enhanced_components()
        
        # Fallback data storage
        self._iq_data = None
        self._captured_files = []
        
        # Statistics
        self.stats = {
            'captures_performed': 0,
            'total_samples_captured': 0,
            'total_capture_time': 0.0,
            'last_capture_time': None
        }
        
        logger.info(f"Initialized capture manager: {platform} @ {sample_rate/1e6:.3f} MHz")
    
    def _init_enhanced_components(self) -> None:
        """Initialize enhanced core components."""
        try:
            # Map platform strings to enums
            platform_map = {
                'rtl_sdr': SDRPlatform.RTL_SDR,
                'rtl_sdr_tcp': SDRPlatform.RTL_SDR_TCP,
                'hackrf': SDRPlatform.HACKRF,
                'airspy': SDRPlatform.AIRSPY,
                'sdrplay': SDRPlatform.SDRPLAY
            }
            
            platform_enum = platform_map.get(self.platform.lower(), SDRPlatform.RTL_SDR)
            
            # Map gain mode
            gain_mode_map = {
                'auto': GainMode.AUTO,
                'manual': GainMode.MANUAL,
                'agc': GainMode.AGC
            }
            gain_mode_enum = gain_mode_map.get(str(self.gain_mode).lower(), GainMode.AUTO)
            
            # Create SDR configuration
            self._sdr_config = SDRConfig(
                platform=platform_enum,
                frequency_hz=self.frequency,
                sample_rate_hz=self.sample_rate,
                gain_mode=gain_mode_enum,
                gain_db=self.gain_db,
                device_index=self.device_index,
                validate_hardware_limits=True
            )
            
            # Initialize components (but don't connect yet)
            self._signal_processor = SignalProcessor()
            self._quality_monitor = QualityMonitor()
            
            logger.debug("Enhanced components initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize enhanced components: {e}")
            self._signal_processor = None
            self._quality_monitor = None
    
    def set_frequency(self, frequency_hz: FrequencyHz) -> None:
        """
        Set center frequency for capture operations.
        
        Args:
            frequency_hz: Center frequency in Hz
        """
        self.frequency = frequency_hz
        
        if CORE_AVAILABLE and self._sdr_config:
            # Update SDR configuration
            self._sdr_config = SDRConfig(
                **{**self._sdr_config.__dict__, 'frequency_hz': frequency_hz}
            )
        
        logger.info(f"Set frequency to {frequency_hz/1e6:.3f} MHz")
    
    def set_sample_rate(self, sample_rate_hz: SampleRateHz) -> None:
        """
        Set sample rate for capture operations.
        
        Args:
            sample_rate_hz: Sample rate in Hz
        """
        self.sample_rate = sample_rate_hz
        
        if CORE_AVAILABLE and self._sdr_config:
            self._sdr_config = SDRConfig(
                **{**self._sdr_config.__dict__, 'sample_rate_hz': sample_rate_hz}
            )
        
        logger.info(f"Set sample rate to {sample_rate_hz/1e6:.3f} MHz")
    
    def set_gain(self, gain_db: Optional[float] = None, mode: str = 'auto') -> None:
        """
        Set gain configuration.
        
        Args:
            gain_db: Manual gain in dB (None for auto)
            mode: Gain mode ('auto', 'manual', 'agc')
        """
        self.gain_mode = mode
        self.gain_db = gain_db
        
        if CORE_AVAILABLE and self._sdr_config:
            gain_mode_map = {
                'auto': GainMode.AUTO,
                'manual': GainMode.MANUAL,
                'agc': GainMode.AGC
            }
            gain_mode_enum = gain_mode_map.get(mode.lower(), GainMode.AUTO)
            
            self._sdr_config = SDRConfig(
                **{**self._sdr_config.__dict__, 
                   'gain_mode': gain_mode_enum, 'gain_db': gain_db}
            )
        
        logger.info(f"Set gain: {mode} mode" + (f", {gain_db} dB" if gain_db else ""))
    
    def connect(self) -> bool:
        """
        Connect to SDR hardware.
        
        Returns:
            True if connection successful, False otherwise
        """
        if not CORE_AVAILABLE:
            logger.warning("Enhanced capture not available, using fallback mode")
            self._is_connected = True
            return True
        
        try:
            # Create enhanced capture instance
            self._enhanced_capture = EnhancedLiveCapture(self._sdr_config)
            
            # Connect asynchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                loop.run_until_complete(self._enhanced_capture.connect())
                self._is_connected = True
                logger.info("Connected to SDR hardware")
                return True
                
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Failed to connect to SDR: {e}")
            self._is_connected = False
            return False
    
    def disconnect(self) -> None:
        """Disconnect from SDR hardware."""
        if self._enhanced_capture and self._is_connected:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    loop.run_until_complete(self._enhanced_capture.disconnect())
                finally:
                    loop.close()
                    
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")
            
            self._enhanced_capture = None
        
        self._is_connected = False
        logger.info("Disconnected from SDR hardware")
    
    def capture(
        self,
        duration: float,
        auto_connect: bool = True
    ) -> IQSamples:
        """
        Capture IQ samples for specified duration.
        
        Args:
            duration: Capture duration in seconds
            auto_connect: Automatically connect if not connected
            
        Returns:
            Complex IQ samples
            
        Raises:
            CaptureManagerError: If capture fails
        """
        if auto_connect and not self._is_connected:
            if not self.connect():
                raise CaptureManagerError("Failed to connect to SDR")
        
        if not self._is_connected:
            raise CaptureManagerError("Not connected to SDR hardware")
        
        start_time = time.time()
        
        try:
            if CORE_AVAILABLE and self._enhanced_capture:
                # Use enhanced capture
                num_samples = int(self.sample_rate * duration)
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    samples, metadata = loop.run_until_complete(
                        self._enhanced_capture.capture_samples(num_samples)
                    )
                    
                    # Update statistics
                    self._update_capture_stats(samples, time.time() - start_time)
                    
                    # Store for later access
                    self._iq_data = samples
                    
                    logger.info(f"Captured {len(samples):,} samples in {duration:.1f}s")
                    return samples
                    
                finally:
                    loop.close()
            else:
                # Fallback simulation
                logger.warning("Using simulated capture data")
                num_samples = int(self.sample_rate * duration)
                samples = self._generate_test_signal(num_samples)
                self._iq_data = samples
                self._update_capture_stats(samples, time.time() - start_time)
                return samples
                
        except Exception as e:
            logger.error(f"Capture failed: {e}")
            raise CaptureManagerError(f"Capture failed: {e}") from e
    
    def capture_to_file(
        self,
        filename: Union[str, Path],
        duration: float,
        auto_connect: bool = True,
        include_metadata: bool = True
    ) -> str:
        """
        Capture IQ samples and save directly to file.
        
        Args:
            filename: Output filename
            duration: Capture duration in seconds
            auto_connect: Automatically connect if not connected
            include_metadata: Include metadata sidecar file
            
        Returns:
            Path to saved file
            
        Raises:
            CaptureManagerError: If capture fails
        """
        filepath = Path(filename)
        
        if auto_connect and not self._is_connected:
            if not self.connect():
                raise CaptureManagerError("Failed to connect to SDR")
        
        start_time = time.time()
        
        try:
            if CORE_AVAILABLE and self._enhanced_capture:
                # Use enhanced capture with direct file saving
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    metadata = loop.run_until_complete(
                        self._enhanced_capture.save_capture_to_file(
                            filepath=filepath,
                            num_samples=int(self.sample_rate * duration),
                            format_output=CaptureFormat.COMPLEX64,
                            include_metadata=include_metadata
                        )
                    )
                    
                    # Update statistics
                    self._update_capture_stats(None, time.time() - start_time, metadata.total_samples)
                    
                finally:
                    loop.close()
            else:
                # Fallback: capture then save
                samples = self.capture(duration, auto_connect=False)
                write_iq_file(str(filepath), samples)
                
                # Create simple metadata
                if include_metadata:
                    metadata_file = filepath.with_suffix('.json')
                    import json
                    metadata = {
                        'frequency_hz': self.frequency,
                        'sample_rate_hz': self.sample_rate,
                        'duration_s': duration,
                        'samples': len(samples),
                        'timestamp': datetime.now().isoformat()
                    }
                    with open(metadata_file, 'w') as f:
                        json.dump(metadata, f, indent=2)
            
            # Track captured files
            self._captured_files.append(str(filepath))
            
            logger.info(f"Saved capture to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to save capture: {e}")
            raise CaptureManagerError(f"Failed to save capture: {e}") from e
    
    def load_file(self, filepath: Union[str, Path]) -> IQSamples:
        """
        Load IQ data from file.
        
        Args:
            filepath: Path to IQ file
            
        Returns:
            Complex IQ samples
            
        Raises:
            CaptureManagerError: If file cannot be loaded
        """
        try:
            samples = read_iq_file(str(filepath))
            self._iq_data = samples
            logger.info(f"Loaded {len(samples):,} samples from {filepath}")
            return samples
            
        except Exception as e:
            logger.error(f"Failed to load file {filepath}: {e}")
            raise CaptureManagerError(f"Failed to load file: {e}") from e
    
    def get_iq_data(self) -> Optional[IQSamples]:
        """
        Get the most recently captured/loaded IQ data.
        
        Returns:
            IQ samples or None if no data available
        """
        return self._iq_data
    
    def extract_packets(
        self,
        iq_data: Optional[IQSamples] = None,
        threshold: float = 0.05,
        min_gap: int = 1000
    ) -> List[IQSamples]:
        """
        Extract packet-like segments from IQ data.
        
        Args:
            iq_data: IQ data to process (uses stored data if None)
            threshold: Detection threshold
            min_gap: Minimum gap between packets
            
        Returns:
            List of packet IQ segments
            
        Raises:
            CaptureManagerError: If no data available
        """
        if iq_data is None:
            iq_data = self._iq_data
        
        if iq_data is None:
            raise CaptureManagerError("No IQ data available")
        
        try:
            # Use enhanced detection if available
            if CORE_AVAILABLE:
                packet_regions = detect_packets(iq_data, threshold, min_gap)
            else:
                # Fallback detection
                packet_regions = self._simple_packet_detection(iq_data, threshold, min_gap)
            
            # Extract packet segments
            packets = []
            for start, end in packet_regions:
                packets.append(iq_data[start:end])
            
            logger.info(f"Extracted {len(packets)} packets from {len(iq_data):,} samples")
            return packets
            
        except Exception as e:
            logger.error(f"Packet extraction failed: {e}")
            raise CaptureManagerError(f"Packet extraction failed: {e}") from e
    
    def find_preambles(
        self,
        preamble_pattern: npt.NDArray,
        iq_data: Optional[IQSamples] = None,
        threshold: float = 0.8
    ) -> List[int]:
        """
        Find preamble patterns in IQ data.
        
        Args:
            preamble_pattern: Known preamble pattern
            iq_data: IQ data to search (uses stored data if None)
            threshold: Correlation threshold
            
        Returns:
            List of indices where preambles were found
        """
        if iq_data is None:
            iq_data = self._iq_data
        
        if iq_data is None:
            raise CaptureManagerError("No IQ data available")
        
        try:
            if CORE_AVAILABLE:
                indices = find_preamble(iq_data, preamble_pattern, threshold)
            else:
                # Fallback preamble detection
                indices = self._simple_preamble_detection(iq_data, preamble_pattern, threshold)
            
            logger.info(f"Found {len(indices)} preamble matches")
            return indices.tolist()
            
        except Exception as e:
            logger.error(f"Preamble search failed: {e}")
            raise CaptureManagerError(f"Preamble search failed: {e}") from e
    
    def analyze_signal_quality(
        self,
        iq_data: Optional[IQSamples] = None
    ) -> Dict[str, float]:
        """
        Analyze signal quality metrics.
        
        Args:
            iq_data: IQ data to analyze (uses stored data if None)
            
        Returns:
            Dictionary of quality metrics
        """
        if iq_data is None:
            iq_data = self._iq_data
        
        if iq_data is None:
            raise CaptureManagerError("No IQ data available")
        
        try:
            if CORE_AVAILABLE and self._quality_monitor:
                metrics = self._quality_monitor.update(iq_data)
            else:
                # Basic quality analysis
                metrics = self._basic_quality_analysis(iq_data)
            
            logger.debug(f"Analyzed signal quality: {len(metrics)} metrics")
            return metrics
            
        except Exception as e:
            logger.error(f"Quality analysis failed: {e}")
            return {}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get capture manager statistics."""
        stats = self.stats.copy()
        stats.update({
            'platform': self.platform,
            'frequency_mhz': self.frequency / 1e6,
            'sample_rate_mhz': self.sample_rate / 1e6,
            'is_connected': self._is_connected,
            'captured_files': len(self._captured_files),
            'has_data': self._iq_data is not None,
            'data_samples': len(self._iq_data) if self._iq_data is not None else 0
        })
        
        if CORE_AVAILABLE and self._enhanced_capture:
            stats['enhanced_mode'] = True
        else:
            stats['enhanced_mode'] = False
        
        return stats
    
    def scan_spectrum(
        self,
        start_freq: FrequencyHz,
        stop_freq: FrequencyHz,
        step_size: FrequencyHz = 1e6,
        dwell_time: float = 0.1
    ) -> Dict[FrequencyHz, float]:
        """
        Perform spectrum scan across frequency range.
        
        Args:
            start_freq: Start frequency in Hz
            stop_freq: Stop frequency in Hz
            step_size: Frequency step size in Hz
            dwell_time: Time to spend at each frequency
            
        Returns:
            Dictionary mapping frequency to power level
        """
        if not self._is_connected:
            if not self.connect():
                raise CaptureManagerError("Failed to connect for spectrum scan")
        
        spectrum = {}
        original_freq = self.frequency
        
        try:
            freq = start_freq
            while freq <= stop_freq:
                # Set frequency
                self.set_frequency(freq)
                
                # Capture short sample
                samples = self.capture(dwell_time, auto_connect=False)
                
                # Calculate power
                power_dbfs = 10 * np.log10(np.mean(np.abs(samples) ** 2) + 1e-12)
                spectrum[freq] = power_dbfs
                
                freq += step_size
                
                logger.debug(f"Scanned {freq/1e6:.1f} MHz: {power_dbfs:.1f} dBFS")
            
            logger.info(f"Completed spectrum scan: {len(spectrum)} points")
            return spectrum
            
        finally:
            # Restore original frequency
            self.set_frequency(original_freq)
    
    def _update_capture_stats(
        self,
        samples: Optional[IQSamples],
        duration: float,
        sample_count: Optional[int] = None
    ) -> None:
        """Update capture statistics."""
        self.stats['captures_performed'] += 1
        self.stats['total_capture_time'] += duration
        self.stats['last_capture_time'] = datetime.now()
        
        if samples is not None:
            self.stats['total_samples_captured'] += len(samples)
        elif sample_count is not None:
            self.stats['total_samples_captured'] += sample_count
    
    def _generate_test_signal(self, num_samples: int) -> IQSamples:
        """Generate test signal for fallback mode."""
        t = np.arange(num_samples) / self.sample_rate
        
        # Create a mix of tones and noise
        signal = (0.3 * np.exp(1j * 2 * np.pi * 1000 * t) +  # 1 kHz tone
                 0.2 * np.exp(1j * 2 * np.pi * 2000 * t) +   # 2 kHz tone
                 0.1 * (np.random.normal(0, 1, num_samples) + 
                       1j * np.random.normal(0, 1, num_samples)))  # Noise
        
        return signal.astype(np.complex64)
    
    def _simple_packet_detection(
        self,
        iq_data: IQSamples,
        threshold: float,
        min_gap: int
    ) -> List[Tuple[int, int]]:
        """Simple packet detection fallback."""
        power = np.abs(iq_data)
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
        
        # Filter short bursts
        packets = []
        for start, end in zip(starts, ends):
            if end - start > min_gap:
                packets.append((int(start), int(end)))
        
        return packets
    
    def _simple_preamble_detection(
        self,
        iq_data: IQSamples,
        pattern: npt.NDArray,
        threshold: float
    ) -> npt.NDArray:
        """Simple preamble detection fallback."""
        # Basic correlation
        correlation = np.correlate(np.abs(iq_data), np.abs(pattern), mode='valid')
        normalized = correlation / np.max(correlation)
        hits = np.where(normalized > threshold)[0]
        return hits
    
    def _basic_quality_analysis(self, iq_data: IQSamples) -> Dict[str, float]:
        """Basic quality analysis fallback."""
        power = np.abs(iq_data) ** 2
        avg_power = np.mean(power)
        peak_power = np.max(power)
        
        return {
            'signal_power_dbfs': 10 * np.log10(avg_power + 1e-12),
            'peak_power_dbfs': 10 * np.log10(peak_power + 1e-12),
            'crest_factor_db': 10 * np.log10(peak_power / (avg_power + 1e-12)),
            'total_samples': len(iq_data)
        }
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


# Backward compatibility functions

def create_capture_manager(**kwargs) -> CaptureManager:
    """Create capture manager with backward compatibility."""
    return CaptureManager(**kwargs)


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Simple usage example
    print("=== Capture Manager Demo ===")
    
    try:
        # Create manager
        manager = CaptureManager(sample_rate=2.048e6)
        manager.set_frequency(100.1e6)  # FM radio
        
        # Test connection (may fail without hardware)
        connected = manager.connect()
        print(f"Connection: {'Success' if connected else 'Failed (using simulation)'}")
        
        # Capture data
        samples = manager.capture(duration=1.0)  # 1 second
        print(f"Captured: {len(samples):,} samples")
        
        # Analyze quality
        quality = manager.analyze_signal_quality()
        print(f"Signal power: {quality.get('signal_power_dbfs', 0):.1f} dBFS")
        
        # Extract packets
        packets = manager.extract_packets(threshold=0.1)
        print(f"Found: {len(packets)} packets")
        
        # Show statistics
        stats = manager.get_statistics()
        print(f"Statistics: {stats['captures_performed']} captures performed")
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        if 'manager' in locals():
            manager.disconnect()