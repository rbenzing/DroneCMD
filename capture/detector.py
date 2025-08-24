#!/usr/bin/env python3
"""
Basic Signal Detection Interface

This module provides simplified interfaces for signal detection operations,
consolidating detection functions from various sources and providing both
simple function-based interfaces and class-based detectors.

Key Features:
- Signal presence detection
- Preamble/sync pattern detection  
- Signal characterization and analysis
- Multiple detection algorithms
- Integration with enhanced signal processing
- Backward compatible function interfaces

Usage:
    >>> detector = SignalDetector()
    >>> signals = detector.detect_signals(iq_data, threshold=0.05)
    >>> 
    >>> # Find specific patterns
    >>> preamble = np.array([1,0,1,0,1,0,1,0])
    >>> hits = detector.find_preambles(iq_data, preamble)
"""

from __future__ import annotations

import logging
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from scipy.signal import correlate, find_peaks, welch

# Import from core modules
try:
    from ..core.signal_processing import (
        SignalProcessor, QualityMonitor, detect_packets, find_preamble,
        analyze_signal_quality
    )
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
RealSamples = npt.NDArray[np.float32]
DetectionResult = Dict[str, Any]
FrequencyHz = float
PowerDb = float


class SignalDetectorError(Exception):
    """Exception raised by signal detector operations."""
    pass


class SignalDetector:
    """
    Advanced signal detector with multiple detection algorithms.
    
    This class provides comprehensive signal detection capabilities
    including signal presence detection, pattern matching, and
    signal characterization.
    
    Example:
        >>> detector = SignalDetector()
        >>> iq_data = detector.load_file("capture.iq")
        >>> 
        >>> # Detect signal regions
        >>> signals = detector.detect_signals(threshold=0.05)
        >>> print(f"Found {len(signals)} signal regions")
        >>> 
        >>> # Characterize signals
        >>> for signal in signals:
        ...     analysis = detector.analyze_signal(signal['iq_data'])
        ...     print(f"Power: {analysis['power_dbfs']:.1f} dBFS")
    """
    
    def __init__(
        self,
        sample_rate: float = 2.048e6,
        enable_enhanced_features: bool = True,
        **kwargs: Any
    ) -> None:
        """
        Initialize signal detector.
        
        Args:
            sample_rate: Sample rate in Hz
            enable_enhanced_features: Use enhanced core components
            **kwargs: Additional configuration parameters
        """
        self.sample_rate = sample_rate
        self.enable_enhanced = enable_enhanced_features and ENHANCED_AVAILABLE
        
        # Configuration
        self.config = {
            'min_signal_length': kwargs.get('min_signal_length', 1000),
            'max_signal_length': kwargs.get('max_signal_length', 1000000),
            'noise_floor_estimation': kwargs.get('noise_floor_estimation', 'percentile'),
            'detection_algorithms': kwargs.get('detection_algorithms', ['threshold', 'energy', 'correlation']),
            'quality_analysis': kwargs.get('quality_analysis', True)
        }
        
        # Enhanced components
        self._signal_processor = None
        self._quality_monitor = None
        
        if self.enable_enhanced:
            self._init_enhanced_components()
        
        # Detection cache
        self._detection_cache = {}
        self._last_iq_hash = None
        
        # Statistics
        self.stats = {
            'detections_performed': 0,
            'signals_detected': 0,
            'preambles_found': 0,
            'total_processing_time': 0.0
        }
        
        logger.info(f"Initialized signal detector (enhanced: {self.enable_enhanced})")
    
    def _init_enhanced_components(self) -> None:
        """Initialize enhanced core components."""
        try:
            self._signal_processor = SignalProcessor()
            self._quality_monitor = QualityMonitor()
            logger.debug("Enhanced components initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize enhanced components: {e}")
            self.enable_enhanced = False
    
    def load_file(self, filename: str) -> IQSamples:
        """
        Load IQ data from file.
        
        Args:
            filename: Path to IQ file
            
        Returns:
            Complex IQ samples
        """
        try:
            iq_data = read_iq_file(filename)
            logger.info(f"Loaded {len(iq_data):,} IQ samples from {filename}")
            return iq_data
        except Exception as e:
            logger.error(f"Failed to load file {filename}: {e}")
            raise SignalDetectorError(f"Failed to load file: {e}") from e
    
    def detect_signals(
        self,
        iq_data: IQSamples,
        threshold: float = 0.05,
        method: str = 'auto',
        min_length: Optional[int] = None,
        max_signals: Optional[int] = None
    ) -> List[DetectionResult]:
        """
        Detect signal regions in IQ data.
        
        Args:
            iq_data: Complex IQ samples
            threshold: Detection threshold
            method: Detection method ('auto', 'threshold', 'energy', 'adaptive')
            min_length: Minimum signal length in samples
            max_signals: Maximum number of signals to detect
            
        Returns:
            List of detected signal regions with metadata
        """
        if len(iq_data) == 0:
            return []
        
        start_time = time.time()
        
        # Use configured minimum length if not specified
        if min_length is None:
            min_length = self.config['min_signal_length']
        
        try:
            # Choose detection method
            if method == 'auto':
                method = self._select_optimal_method(iq_data, threshold)
            
            # Perform detection
            if method == 'threshold':
                regions = self._threshold_detection(iq_data, threshold, min_length)
            elif method == 'energy':
                regions = self._energy_detection(iq_data, threshold, min_length)
            elif method == 'adaptive':
                regions = self._adaptive_detection(iq_data, threshold, min_length)
            elif method == 'enhanced' and self.enable_enhanced:
                regions = self._enhanced_detection(iq_data, threshold, min_length)
            else:
                # Fallback to threshold detection
                regions = self._threshold_detection(iq_data, threshold, min_length)
            
            # Limit number of signals if requested
            if max_signals is not None:
                regions = regions[:max_signals]
            
            # Build detection results
            signals = []
            for i, (start, end) in enumerate(regions):
                signal_iq = iq_data[start:end]
                
                # Basic signal info
                signal_info = {
                    'signal_id': i,
                    'start_sample': start,
                    'end_sample': end,
                    'length_samples': end - start,
                    'duration_s': (end - start) / self.sample_rate,
                    'detection_method': method,
                    'iq_data': signal_iq
                }
                
                # Add signal analysis if enabled
                if self.config['quality_analysis']:
                    analysis = self.analyze_signal(signal_iq)
                    signal_info['analysis'] = analysis
                
                signals.append(signal_info)
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_detection_stats(len(signals), processing_time)
            
            logger.info(f"Detected {len(signals)} signals using {method} method in {processing_time:.3f}s")
            return signals
            
        except Exception as e:
            logger.error(f"Signal detection failed: {e}")
            raise SignalDetectorError(f"Signal detection failed: {e}") from e
    
    def find_preambles(
        self,
        iq_data: IQSamples,
        preamble_pattern: Union[npt.NDArray, List],
        threshold: float = 0.8,
        method: str = 'correlation'
    ) -> List[Dict[str, Any]]:
        """
        Find preamble patterns in IQ data.
        
        Args:
            iq_data: Complex IQ samples
            preamble_pattern: Known preamble pattern (bits or samples)
            threshold: Detection threshold (0-1)
            method: Detection method ('correlation', 'enhanced')
            
        Returns:
            List of preamble detections with metadata
        """
        if len(iq_data) == 0 or len(preamble_pattern) == 0:
            return []
        
        start_time = time.time()
        
        try:
            # Convert pattern to numpy array
            pattern = np.array(preamble_pattern)
            
            # Use enhanced detection if available
            if method == 'enhanced' and self.enable_enhanced:
                indices = find_preamble(iq_data, pattern, threshold)
            else:
                indices = self._correlation_preamble_detection(iq_data, pattern, threshold)
            
            # Build preamble results
            preambles = []
            for i, idx in enumerate(indices):
                preamble_info = {
                    'preamble_id': i,
                    'start_index': int(idx),
                    'pattern_length': len(pattern),
                    'detection_method': method,
                    'correlation_peak': self._calculate_correlation_peak(iq_data, pattern, idx),
                    'snr_estimate': self._estimate_local_snr(iq_data, idx, len(pattern))
                }
                preambles.append(preamble_info)
            
            # Update statistics
            processing_time = time.time() - start_time
            self.stats['preambles_found'] += len(preambles)
            self.stats['total_processing_time'] += processing_time
            
            logger.info(f"Found {len(preambles)} preambles in {processing_time:.3f}s")
            return preambles
            
        except Exception as e:
            logger.error(f"Preamble detection failed: {e}")
            raise SignalDetectorError(f"Preamble detection failed: {e}") from e
    
    def analyze_signal(
        self,
        iq_data: IQSamples,
        include_spectral: bool = True,
        include_modulation: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive signal analysis.
        
        Args:
            iq_data: Complex IQ samples
            include_spectral: Include spectral analysis
            include_modulation: Include modulation analysis
            
        Returns:
            Dictionary with analysis results
        """
        if len(iq_data) == 0:
            return {}
        
        try:
            analysis = {}
            
            # Basic power metrics
            power_linear = np.mean(np.abs(iq_data) ** 2)
            analysis['power_linear'] = float(power_linear)
            analysis['power_dbfs'] = float(10 * np.log10(power_linear + 1e-12))
            analysis['peak_power_dbfs'] = float(10 * np.log10(np.max(np.abs(iq_data) ** 2) + 1e-12))
            
            # Crest factor
            if power_linear > 0:
                peak_power = np.max(np.abs(iq_data) ** 2)
                analysis['crest_factor_db'] = float(10 * np.log10(peak_power / power_linear))
            
            # Use enhanced analysis if available
            if self.enable_enhanced:
                enhanced_analysis = analyze_signal_quality(iq_data, self.sample_rate)
                analysis.update(enhanced_analysis)
            else:
                # Basic SNR estimation
                analysis['snr_estimate_db'] = self._simple_snr_estimation(iq_data)
            
            # Spectral analysis
            if include_spectral and len(iq_data) >= 256:
                spectral_analysis = self._spectral_analysis(iq_data)
                analysis['spectral'] = spectral_analysis
            
            # Modulation characteristics
            if include_modulation:
                mod_analysis = self._modulation_analysis(iq_data)
                analysis['modulation'] = mod_analysis
            
            return analysis
            
        except Exception as e:
            logger.warning(f"Signal analysis failed: {e}")
            return {'error': str(e)}
    
    def characterize_noise(
        self,
        iq_data: IQSamples,
        method: str = 'percentile'
    ) -> Dict[str, float]:
        """
        Characterize noise properties of the signal.
        
        Args:
            iq_data: Complex IQ samples
            method: Noise estimation method
            
        Returns:
            Noise characteristics
        """
        if len(iq_data) == 0:
            return {}
        
        power = np.abs(iq_data) ** 2
        
        noise_char = {}
        
        if method == 'percentile':
            # Use lower percentiles as noise estimate
            noise_char['noise_floor_dbfs'] = 10 * np.log10(np.percentile(power, 10) + 1e-12)
            noise_char['noise_variance'] = float(np.percentile(power, 25))
            
        elif method == 'minimum':
            # Use minimum statistics
            noise_char['noise_floor_dbfs'] = 10 * np.log10(np.min(power) + 1e-12)
            noise_char['noise_variance'] = float(np.var(power[power < np.percentile(power, 50)]))
        
        # Noise distribution analysis
        noise_samples = power[power < np.percentile(power, 75)]
        if len(noise_samples) > 10:
            noise_char['noise_mean'] = float(np.mean(noise_samples))
            noise_char['noise_std'] = float(np.std(noise_samples))
            
            # Test for Gaussianity (simple kurtosis test)
            from scipy import stats
            noise_char['noise_kurtosis'] = float(stats.kurtosis(noise_samples))
            noise_char['is_gaussian_like'] = abs(noise_char['noise_kurtosis']) < 1.0
        
        return noise_char
    
    def detect_frequency_shifts(
        self,
        iq_data: IQSamples,
        window_size: int = 1024,
        overlap: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Detect frequency shifts or hops in the signal.
        
        Args:
            iq_data: Complex IQ samples
            window_size: Analysis window size
            overlap: Window overlap ratio
            
        Returns:
            List of detected frequency shifts
        """
        if len(iq_data) < window_size * 2:
            return []
        
        try:
            shifts = []
            step_size = int(window_size * (1 - overlap))
            
            prev_center_freq = None
            
            for i in range(0, len(iq_data) - window_size, step_size):
                window = iq_data[i:i + window_size]
                
                # Estimate center frequency of this window
                center_freq = self._estimate_center_frequency(window)
                
                if prev_center_freq is not None:
                    freq_shift = center_freq - prev_center_freq
                    
                    # Detect significant shifts (threshold: 10% of sample rate)
                    shift_threshold = self.sample_rate * 0.1
                    
                    if abs(freq_shift) > shift_threshold:
                        shift_info = {
                            'start_sample': i,
                            'frequency_shift_hz': freq_shift,
                            'from_freq_hz': prev_center_freq,
                            'to_freq_hz': center_freq,
                            'timestamp_s': i / self.sample_rate
                        }
                        shifts.append(shift_info)
                
                prev_center_freq = center_freq
            
            logger.info(f"Detected {len(shifts)} frequency shifts")
            return shifts
            
        except Exception as e:
            logger.warning(f"Frequency shift detection failed: {e}")
            return []
    
    def get_detection_summary(self) -> Dict[str, Any]:
        """Get summary of detection operations."""
        return {
            'detector_config': self.config.copy(),
            'enhanced_mode': self.enable_enhanced,
            'sample_rate_mhz': self.sample_rate / 1e6,
            'statistics': self.stats.copy(),
            'supported_methods': [
                'threshold', 'energy', 'adaptive',
                'enhanced' if self.enable_enhanced else 'enhanced (unavailable)'
            ]
        }
    
    # Private methods for different detection algorithms
    
    def _select_optimal_method(self, iq_data: IQSamples, threshold: float) -> str:
        """Select optimal detection method based on signal characteristics."""
        # Simple heuristic based on signal properties
        snr_estimate = self._simple_snr_estimation(iq_data)
        
        if self.enable_enhanced:
            return 'enhanced'
        elif snr_estimate > 10:  # High SNR
            return 'threshold'
        elif snr_estimate > 0:   # Medium SNR
            return 'energy'
        else:                    # Low SNR
            return 'adaptive'
    
    def _threshold_detection(
        self,
        iq_data: IQSamples,
        threshold: float,
        min_length: int
    ) -> List[Tuple[int, int]]:
        """Simple threshold-based detection."""
        power = np.abs(iq_data) ** 2
        
        # Adaptive threshold if needed
        if threshold < 1.0:
            threshold = threshold * np.max(power)
        
        active = power > threshold
        
        # Find signal regions
        edges = np.diff(active.astype(int))
        starts = np.where(edges == 1)[0] + 1
        ends = np.where(edges == -1)[0] + 1
        
        # Handle edge cases
        if len(ends) > 0 and (len(starts) == 0 or starts[0] > ends[0]):
            starts = np.insert(starts, 0, 0)
        if len(starts) > 0 and (len(ends) == 0 or ends[-1] < starts[-1]):
            ends = np.append(ends, len(active))
        
        # Filter by minimum length
        regions = []
        for start, end in zip(starts, ends):
            if end - start >= min_length:
                regions.append((int(start), int(end)))
        
        return regions
    
    def _energy_detection(
        self,
        iq_data: IQSamples,
        threshold: float,
        min_length: int
    ) -> List[Tuple[int, int]]:
        """Energy-based detection with smoothing."""
        power = np.abs(iq_data) ** 2
        
        # Smooth power estimate
        window_size = max(10, min_length // 10)
        smoothed_power = np.convolve(power, np.ones(window_size)/window_size, mode='same')
        
        # Energy threshold
        if threshold < 1.0:
            noise_floor = np.percentile(smoothed_power, 25)
            threshold = noise_floor + threshold * (np.max(smoothed_power) - noise_floor)
        
        active = smoothed_power > threshold
        
        # Find regions (same as threshold method)
        edges = np.diff(active.astype(int))
        starts = np.where(edges == 1)[0] + 1
        ends = np.where(edges == -1)[0] + 1
        
        if len(ends) > 0 and (len(starts) == 0 or starts[0] > ends[0]):
            starts = np.insert(starts, 0, 0)
        if len(starts) > 0 and (len(ends) == 0 or ends[-1] < starts[-1]):
            ends = np.append(ends, len(active))
        
        regions = []
        for start, end in zip(starts, ends):
            if end - start >= min_length:
                regions.append((int(start), int(end)))
        
        return regions
    
    def _adaptive_detection(
        self,
        iq_data: IQSamples,
        threshold: float,
        min_length: int
    ) -> List[Tuple[int, int]]:
        """Adaptive detection with local statistics."""
        power = np.abs(iq_data) ** 2
        
        # Local adaptive threshold
        window_size = min_length
        adaptive_threshold = np.zeros_like(power)
        
        for i in range(len(power)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(power), i + window_size // 2)
            local_power = power[start_idx:end_idx]
            
            local_mean = np.mean(local_power)
            local_std = np.std(local_power)
            adaptive_threshold[i] = local_mean + threshold * local_std
        
        active = power > adaptive_threshold
        
        # Find regions
        edges = np.diff(active.astype(int))
        starts = np.where(edges == 1)[0] + 1
        ends = np.where(edges == -1)[0] + 1
        
        if len(ends) > 0 and (len(starts) == 0 or starts[0] > ends[0]):
            starts = np.insert(starts, 0, 0)
        if len(starts) > 0 and (len(ends) == 0 or ends[-1] < starts[-1]):
            ends = np.append(ends, len(active))
        
        regions = []
        for start, end in zip(starts, ends):
            if end - start >= min_length:
                regions.append((int(start), int(end)))
        
        return regions
    
    def _enhanced_detection(
        self,
        iq_data: IQSamples,
        threshold: float,
        min_length: int
    ) -> List[Tuple[int, int]]:
        """Enhanced detection using core signal processing."""
        try:
            regions = detect_packets(iq_data, threshold, min_length)
            return [(int(start), int(end)) for start, end in regions]
        except Exception as e:
            logger.warning(f"Enhanced detection failed, falling back: {e}")
            return self._threshold_detection(iq_data, threshold, min_length)
    
    def _correlation_preamble_detection(
        self,
        iq_data: IQSamples,
        pattern: npt.NDArray,
        threshold: float
    ) -> npt.NDArray:
        """Correlation-based preamble detection."""
        # Handle complex vs real data
        if np.iscomplexobj(iq_data) and not np.iscomplexobj(pattern):
            # Convert pattern to complex or use magnitude
            search_data = np.abs(iq_data)
            search_pattern = np.abs(pattern).astype(np.float32)
        elif not np.iscomplexobj(iq_data) and np.iscomplexobj(pattern):
            search_data = iq_data.astype(np.float32)
            search_pattern = np.abs(pattern).astype(np.float32)
        else:
            search_data = iq_data
            search_pattern = pattern
        
        # Normalize for better correlation
        if len(search_data) > 0 and len(search_pattern) > 0:
            search_data = search_data / (np.linalg.norm(search_data) + 1e-12)
            search_pattern = search_pattern / (np.linalg.norm(search_pattern) + 1e-12)
        
        # Cross-correlation
        correlation = correlate(search_data, search_pattern, mode='valid')
        
        # Find peaks above threshold
        if len(correlation) > 0:
            max_corr = np.max(np.abs(correlation))
            if max_corr > 0:
                normalized_corr = np.abs(correlation) / max_corr
                hits = np.where(normalized_corr > threshold)[0]
                return hits
        
        return np.array([], dtype=np.int32)
    
    def _calculate_correlation_peak(
        self,
        iq_data: IQSamples,
        pattern: npt.NDArray,
        index: int
    ) -> float:
        """Calculate correlation peak value at specific index."""
        try:
            if index + len(pattern) > len(iq_data):
                return 0.0
            
            segment = iq_data[index:index + len(pattern)]
            
            # Simple correlation calculation
            if np.iscomplexobj(segment) != np.iscomplexobj(pattern):
                segment = np.abs(segment)
                pattern = np.abs(pattern)
            
            corr = np.abs(np.sum(segment * np.conj(pattern)))
            norm = np.linalg.norm(segment) * np.linalg.norm(pattern)
            
            return float(corr / (norm + 1e-12))
            
        except Exception:
            return 0.0
    
    def _estimate_local_snr(
        self,
        iq_data: IQSamples,
        center_idx: int,
        window_size: int
    ) -> float:
        """Estimate SNR around a specific location."""
        try:
            # Define signal and noise regions
            signal_start = max(0, center_idx - window_size // 2)
            signal_end = min(len(iq_data), center_idx + window_size // 2)
            
            # Noise regions (before and after signal)
            noise_before = max(0, signal_start - window_size)
            noise_after = min(len(iq_data), signal_end + window_size)
            
            signal_power = np.mean(np.abs(iq_data[signal_start:signal_end]) ** 2)
            
            # Combine noise from both regions
            noise_samples = np.concatenate([
                iq_data[noise_before:signal_start],
                iq_data[signal_end:noise_after]
            ])
            
            if len(noise_samples) > 0:
                noise_power = np.mean(np.abs(noise_samples) ** 2)
                if noise_power > 0:
                    return float(10 * np.log10(signal_power / noise_power))
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _simple_snr_estimation(self, iq_data: IQSamples) -> float:
        """Simple SNR estimation using percentiles."""
        power = np.abs(iq_data) ** 2
        signal_power = np.mean(power)
        noise_power = np.percentile(power, 25)
        
        if noise_power > 0:
            return 10 * np.log10(signal_power / noise_power)
        return 0.0
    
    def _spectral_analysis(self, iq_data: IQSamples) -> Dict[str, Any]:
        """Spectral analysis of signal."""
        try:
            freqs, psd = welch(iq_data, fs=self.sample_rate, nperseg=min(1024, len(iq_data)//4))
            
            # Find peak frequency
            peak_idx = np.argmax(psd)
            peak_freq = freqs[peak_idx]
            
            # Occupied bandwidth (99% power)
            cumulative_power = np.cumsum(psd)
            total_power = cumulative_power[-1]
            
            low_idx = np.where(cumulative_power >= 0.005 * total_power)[0]
            high_idx = np.where(cumulative_power >= 0.995 * total_power)[0]
            
            if len(low_idx) > 0 and len(high_idx) > 0:
                occupied_bw = freqs[high_idx[0]] - freqs[low_idx[0]]
            else:
                occupied_bw = 0.0
            
            return {
                'peak_frequency_hz': float(peak_freq),
                'peak_power_db': float(10 * np.log10(psd[peak_idx] + 1e-12)),
                'occupied_bandwidth_hz': float(abs(occupied_bw)),
                'spectral_centroid_hz': float(np.sum(freqs * psd) / np.sum(psd)) if np.sum(psd) > 0 else 0.0
            }
            
        except Exception as e:
            logger.debug(f"Spectral analysis failed: {e}")
            return {}
    
    def _modulation_analysis(self, iq_data: IQSamples) -> Dict[str, Any]:
        """Basic modulation characteristic analysis."""
        try:
            amplitude = np.abs(iq_data)
            phase = np.angle(iq_data)
            
            # Amplitude statistics
            amp_var = np.var(amplitude)
            amp_mean = np.mean(amplitude)
            amp_cv = np.sqrt(amp_var) / amp_mean if amp_mean > 0 else 0
            
            # Phase statistics
            phase_diff = np.diff(np.unwrap(phase))
            phase_var = np.var(phase_diff)
            
            # Simple modulation hints
            modulation_hints = {}
            
            if amp_cv > 0.3:
                modulation_hints['amplitude_modulated'] = True
                modulation_hints['likely_ask_ook'] = True
            else:
                modulation_hints['amplitude_modulated'] = False
            
            if phase_var > 0.1:
                modulation_hints['phase_modulated'] = True
                modulation_hints['likely_psk'] = True
            else:
                modulation_hints['phase_modulated'] = False
            
            if not modulation_hints.get('amplitude_modulated', False) and not modulation_hints.get('phase_modulated', False):
                modulation_hints['likely_fsk'] = True
            
            return {
                'amplitude_cv': float(amp_cv),
                'phase_variance': float(phase_var),
                'modulation_hints': modulation_hints
            }
            
        except Exception as e:
            logger.debug(f"Modulation analysis failed: {e}")
            return {}
    
    def _estimate_center_frequency(self, iq_data: IQSamples) -> float:
        """Estimate center frequency of a signal segment."""
        try:
            # Use FFT to find peak frequency
            fft_result = np.fft.fftshift(np.fft.fft(iq_data))
            freqs = np.fft.fftshift(np.fft.fftfreq(len(iq_data), 1/self.sample_rate))
            
            peak_idx = np.argmax(np.abs(fft_result))
            return freqs[peak_idx]
            
        except Exception:
            return 0.0
    
    def _update_detection_stats(self, num_signals: int, processing_time: float) -> None:
        """Update detection statistics."""
        self.stats['detections_performed'] += 1
        self.stats['signals_detected'] += num_signals
        self.stats['total_processing_time'] += processing_time


# Backward compatibility functions

def detect_packets(
    iq_samples: IQSamples,
    threshold: float = 0.05,
    min_gap: int = 1000
) -> List[Tuple[int, int]]:
    """
    Detect packet regions (backward compatible).
    
    Args:
        iq_samples: Complex IQ samples
        threshold: Detection threshold
        min_gap: Minimum gap between packets
        
    Returns:
        List of (start, end) tuples
    """
    detector = SignalDetector()
    signals = detector.detect_signals(iq_samples, threshold, min_length=min_gap)
    return [(s['start_sample'], s['end_sample']) for s in signals]


def find_preamble(
    iq_data: IQSamples,
    preamble: npt.NDArray,
    threshold: float = 0.8
) -> npt.NDArray:
    """
    Find preamble pattern (backward compatible).
    
    Args:
        iq_data: Complex IQ samples
        preamble: Preamble pattern
        threshold: Correlation threshold
        
    Returns:
        Array of indices where preamble was found
    """
    detector = SignalDetector()
    preambles = detector.find_preambles(iq_data, preamble, threshold)
    return np.array([p['start_index'] for p in preambles], dtype=np.int32)


def calculate_power(iq_data: IQSamples) -> float:
    """Calculate signal power (backward compatible)."""
    if len(iq_data) == 0:
        return 0.0
    return float(np.mean(np.abs(iq_data) ** 2))


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    print("=== Signal Detector Demo ===")
    
    try:
        # Create test signal with multiple components
        sample_rate = 2.048e6
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Base noise
        noise = 0.1 * (np.random.normal(0, 1, len(t)) + 1j * np.random.normal(0, 1, len(t)))
        signal = noise.copy()
        
        # Add signal bursts at different times
        burst_times = [0.2, 0.4, 0.6, 0.8]
        burst_duration = 0.05  # 50ms bursts
        
        print(f"Creating test signal with {len(burst_times)} signal bursts...")
        
        for i, start_time in enumerate(burst_times):
            start_idx = int(start_time * sample_rate)
            end_idx = int((start_time + burst_duration) * sample_rate)
            
            # Different frequency for each burst
            freq = 1000 + i * 500  # 1kHz, 1.5kHz, 2kHz, 2.5kHz
            burst_t = t[start_idx:end_idx]
            
            # Modulated signal (simple PSK)
            data_rate = 1000  # 1kbps
            bit_duration = 1.0 / data_rate
            samples_per_bit = int(sample_rate * bit_duration)
            
            num_bits = len(burst_t) // samples_per_bit
            bits = np.random.randint(0, 2, num_bits)
            
            modulated = np.zeros(len(burst_t), dtype=np.complex64)
            for bit_idx, bit in enumerate(bits):
                bit_start = bit_idx * samples_per_bit
                bit_end = min(bit_start + samples_per_bit, len(burst_t))
                if bit_start < len(burst_t):
                    phase = 0 if bit else np.pi
                    bit_signal = np.exp(1j * (2 * np.pi * freq * burst_t[bit_start:bit_end] + phase))
                    modulated[bit_start:bit_end] = bit_signal
            
            # Add to main signal
            signal[start_idx:end_idx] += 0.5 * modulated
        
        print(f"Generated {len(signal):,} IQ samples")
        
        # Create detector and analyze
        detector = SignalDetector(sample_rate=sample_rate)
        
        # Test signal detection
        print("\n=== Signal Detection ===")
        detected_signals = detector.detect_signals(signal, threshold=0.1, method='auto')
        
        print(f"Detected {len(detected_signals)} signals:")
        for sig in detected_signals:
            start_time = sig['start_sample'] / sample_rate
            duration = sig['duration_s']
            power = sig['analysis']['power_dbfs']
            print(f"  Signal {sig['signal_id']}: {start_time:.3f}s, {duration*1000:.1f}ms, {power:.1f} dBFS")
        
        # Test preamble detection
        print("\n=== Preamble Detection ===")
        # Create a simple preamble pattern
        preamble_pattern = np.array([1, 0, 1, 0, 1, 0, 1, 0])  # Simple alternating pattern
        
        # Convert to complex pattern (BPSK-like)
        complex_preamble = np.array([1 if b else -1 for b in preamble_pattern], dtype=np.complex64)
        
        preambles = detector.find_preambles(signal, complex_preamble, threshold=0.6)
        print(f"Found {len(preambles)} preamble matches:")
        
        for preamble in preambles[:5]:  # Show first 5
            start_time = preamble['start_index'] / sample_rate
            correlation = preamble['correlation_peak']
            snr = preamble['snr_estimate']
            print(f"  Preamble {preamble['preamble_id']}: {start_time:.3f}s, "
                  f"corr: {correlation:.3f}, SNR: {snr:.1f} dB")
        
        # Test noise characterization
        print("\n=== Noise Analysis ===")
        noise_char = detector.characterize_noise(signal[:100000])  # First 100k samples
        print("Noise characteristics:")
        for key, value in noise_char.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
        
        # Test frequency shift detection
        print("\n=== Frequency Shift Detection ===")
        freq_shifts = detector.detect_frequency_shifts(signal, window_size=2048)
        print(f"Detected {len(freq_shifts)} frequency shifts:")
        
        for shift in freq_shifts[:3]:  # Show first 3
            time_s = shift['timestamp_s']
            shift_hz = shift['frequency_shift_hz']
            print(f"  Shift at {time_s:.3f}s: {shift_hz:.0f} Hz")
        
        # Show summary
        print("\n=== Detection Summary ===")
        summary = detector.get_detection_summary()
        print(f"Enhanced mode: {summary['enhanced_mode']}")
        print(f"Total processing time: {summary['statistics']['total_processing_time']:.3f}s")
        print(f"Detections performed: {summary['statistics']['detections_performed']}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()