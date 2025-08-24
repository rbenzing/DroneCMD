#!/usr/bin/env python3
"""
Consolidated Signal Processing Module

This module consolidates signal processing utilities from multiple sources:
- Enhanced demodulation signal processing
- Capture signal utilities 
- Utils signal processing functions

Provides both simple function interfaces and advanced class-based APIs
following the progressive enhancement principle.

Integration Points:
- Core demodulation systems
- Live capture quality monitoring
- FHSS signal generation
- Protocol classification
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from scipy import signal, stats
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import hilbert, butter, sosfilt, find_peaks, correlate

# Configure module logger
logger = logging.getLogger(__name__)

# Type aliases
IQSamples = npt.NDArray[np.complex64]
RealSamples = npt.NDArray[np.float32]
QualityMetrics = Dict[str, float]
FrequencyHz = float
SampleRateHz = float
PowerDb = float


class SignalProcessingError(Exception):
    """Custom exception for signal processing errors."""
    pass


# =============================================================================
# ADVANCED SIGNAL PROCESSOR CLASS
# =============================================================================

class SignalProcessor:
    """
    Advanced signal processing class with comprehensive functionality.
    
    Consolidates advanced signal processing from the enhanced modules
    while providing clean interfaces for common operations.
    """
    
    def __init__(self) -> None:
        """Initialize signal processor."""
        self._filter_cache = {}
        self._agc_state = {}
    
    def normalize(
        self,
        signal_data: Union[IQSamples, RealSamples],
        method: str = 'peak',
        target_level: float = 1.0,
        remove_dc: bool = True
    ) -> Union[IQSamples, RealSamples]:
        """
        Advanced signal normalization with multiple methods.
        
        Args:
            signal_data: Input signal samples
            method: Normalization method ('peak', 'rms', 'energy', 'quantile')
            target_level: Target level after normalization
            remove_dc: Remove DC component before normalization
            
        Returns:
            Normalized signal samples
        """
        if len(signal_data) == 0:
            return signal_data
        
        # Remove DC if requested
        if remove_dc:
            signal_data = signal_data - np.mean(signal_data)
        
        # Calculate normalization factor
        if method == 'peak':
            norm_factor = np.max(np.abs(signal_data))
        elif method == 'rms':
            norm_factor = np.sqrt(np.mean(np.abs(signal_data) ** 2))
        elif method == 'energy':
            norm_factor = np.sqrt(np.sum(np.abs(signal_data) ** 2))
        elif method == 'quantile':
            norm_factor = np.percentile(np.abs(signal_data), 95)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        if norm_factor == 0:
            return signal_data
        
        # Apply normalization with clipping protection
        normalized = signal_data * (target_level / norm_factor)
        max_val = np.max(np.abs(normalized))
        if max_val > 1.0:
            normalized = normalized / max_val
        
        return normalized.astype(signal_data.dtype)
    
    def apply_agc(
        self,
        samples: Union[IQSamples, RealSamples],
        target_power: float = 1.0,
        attack_time: float = 0.001,
        release_time: float = 0.1,
        sample_rate: float = 1.0,
        channel_id: str = "default"
    ) -> Union[IQSamples, RealSamples]:
        """
        Apply Automatic Gain Control (AGC) with state memory.
        
        Args:
            samples: Input signal samples
            target_power: Target average power level
            attack_time: AGC attack time constant in seconds
            release_time: AGC release time constant in seconds
            sample_rate: Sample rate in Hz
            channel_id: Channel identifier for state tracking
            
        Returns:
            AGC-controlled signal samples
        """
        if len(samples) == 0:
            return samples
        
        # Calculate power
        power = np.abs(samples) ** 2
        
        # Get or initialize AGC state
        if channel_id not in self._agc_state:
            self._agc_state[channel_id] = {'avg_power': power[0]}
        
        avg_power_prev = self._agc_state[channel_id]['avg_power']
        
        # Design AGC filter coefficients
        attack_alpha = 1.0 - np.exp(-1.0 / (attack_time * sample_rate))
        release_alpha = 1.0 - np.exp(-1.0 / (release_time * sample_rate))
        
        # Apply AGC with state tracking
        avg_power = np.zeros(len(power))
        avg_power[0] = avg_power_prev
        
        for i in range(1, len(power)):
            alpha = attack_alpha if power[i] > avg_power[i-1] else release_alpha
            avg_power[i] = alpha * power[i] + (1 - alpha) * avg_power[i-1]
        
        # Update state
        self._agc_state[channel_id]['avg_power'] = avg_power[-1]
        
        # Calculate and apply gain
        gain = np.sqrt(target_power / (avg_power + 1e-12))
        gain = np.clip(gain, 0.1, 10.0)  # Limit gain range
        
        return samples * gain
    
    def design_filter(
        self,
        filter_type: str,
        cutoff_freq: float,
        sample_rate: float,
        order: int = 4,
        **kwargs: Any
    ) -> npt.NDArray[np.float64]:
        """
        Design digital filter with caching.
        
        Args:
            filter_type: Filter type ('butterworth', 'chebyshev1', 'elliptic', etc.)
            cutoff_freq: Cutoff frequency in Hz
            sample_rate: Sample rate in Hz
            order: Filter order
            **kwargs: Additional filter parameters
            
        Returns:
            Filter coefficients in SOS format
        """
        cache_key = (filter_type, cutoff_freq, sample_rate, order, tuple(sorted(kwargs.items())))
        
        if cache_key in self._filter_cache:
            return self._filter_cache[cache_key]
        
        nyquist = sample_rate / 2
        normalized_cutoff = cutoff_freq / nyquist
        
        if normalized_cutoff >= 1.0:
            warnings.warn(f"Cutoff frequency {cutoff_freq} Hz exceeds Nyquist frequency")
            normalized_cutoff = 0.99
        
        # Design filter based on type
        if filter_type.lower() == 'butterworth':
            sos = butter(order, normalized_cutoff, btype='low', output='sos')
        elif filter_type.lower() == 'chebyshev1':
            ripple = kwargs.get('ripple', 0.5)
            sos = signal.cheby1(order, ripple, normalized_cutoff, btype='low', output='sos')
        elif filter_type.lower() == 'elliptic':
            ripple = kwargs.get('ripple', 0.5)
            attenuation = kwargs.get('attenuation', 40)
            sos = signal.ellip(order, ripple, attenuation, normalized_cutoff, btype='low', output='sos')
        else:
            # Default to Butterworth
            sos = butter(order, normalized_cutoff, btype='low', output='sos')
        
        self._filter_cache[cache_key] = sos
        return sos
    
    def estimate_carrier_frequency(
        self,
        iq_samples: IQSamples,
        sample_rate: float,
        method: str = "fft_peak"
    ) -> float:
        """
        Estimate carrier frequency from IQ samples.
        
        Args:
            iq_samples: Complex IQ samples
            sample_rate: Sample rate in Hz
            method: Estimation method ('fft_peak', 'autocorr')
            
        Returns:
            Estimated carrier frequency in Hz
        """
        if len(iq_samples) < 64:
            return 0.0
        
        if method == "fft_peak":
            fft_result = fftshift(fft(iq_samples))
            freqs = fftshift(fftfreq(len(iq_samples), 1/sample_rate))
            peak_idx = np.argmax(np.abs(fft_result))
            return freqs[peak_idx]
        
        elif method == "autocorr":
            autocorr = correlate(iq_samples, iq_samples, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            peaks, _ = find_peaks(np.abs(autocorr[1:]), height=np.max(np.abs(autocorr)) * 0.1)
            if len(peaks) > 0:
                period_samples = peaks[0] + 1
                return sample_rate / period_samples
        
        return 0.0
    
    def estimate_snr(
        self,
        signal_data: Union[IQSamples, RealSamples],
        method: str = "moment"
    ) -> float:
        """
        Estimate Signal-to-Noise Ratio using various methods.
        
        Args:
            signal_data: Input signal samples
            method: Estimation method ('moment', 'percentile', 'spectral')
            
        Returns:
            Estimated SNR in dB
        """
        if len(signal_data) < 10:
            return -np.inf
        
        power = np.abs(signal_data) ** 2
        
        if method == "moment":
            # Use second and fourth moments
            m2 = np.mean(power)
            m4 = np.mean(power ** 2)
            
            if m2 > 0:
                kurtosis_ratio = m4 / (m2 ** 2)
                if kurtosis_ratio > 2.1:  # Signal + noise
                    noise_var = m2 / kurtosis_ratio
                    signal_var = m2 - noise_var
                    if signal_var > 0 and noise_var > 0:
                        return 10 * np.log10(signal_var / noise_var)
        
        elif method == "percentile":
            noise_power = np.percentile(power, 25)
            signal_power = np.mean(power)
            if noise_power > 0:
                return 10 * np.log10(signal_power / noise_power)
        
        elif method == "spectral":
            # Use spectral analysis for noise floor estimation
            if len(signal_data) >= 256:
                freqs, psd = signal.welch(signal_data, nperseg=min(256, len(signal_data)//4))
                noise_floor = np.percentile(psd, 10)
                signal_power = np.mean(psd)
                if noise_floor > 0:
                    return 10 * np.log10(signal_power / noise_floor)
        
        return 0.0
    
    def calculate_power_dbfs(self, signal_data: Union[IQSamples, RealSamples]) -> float:
        """Calculate signal power in dBFS."""
        if len(signal_data) == 0:
            return -np.inf
        
        power_linear = np.mean(np.abs(signal_data) ** 2)
        return 10 * np.log10(power_linear + 1e-12)
    
    def frequency_shift(
        self,
        iq_data: IQSamples,
        shift_hz: float,
        sample_rate: float
    ) -> IQSamples:
        """Apply frequency shift to IQ data."""
        if len(iq_data) == 0:
            return iq_data
        
        t = np.arange(len(iq_data)) / sample_rate
        shift_signal = np.exp(2j * np.pi * shift_hz * t)
        return (iq_data * shift_signal).astype(np.complex64)


class QualityMonitor:
    """
    Real-time signal quality monitoring system.
    
    Provides continuous analysis of signal characteristics with
    history tracking and anomaly detection.
    """
    
    def __init__(self, history_length: int = 1000) -> None:
        """Initialize quality monitor."""
        self.history_length = history_length
        self.power_history = deque(maxlen=history_length)
        self.noise_history = deque(maxlen=history_length)
        self.snr_history = deque(maxlen=history_length)
        self.processor = SignalProcessor()
        
        # Statistics
        self.total_samples = 0
        self.dropped_samples = 0
    
    def update(self, samples: Union[IQSamples, RealSamples]) -> QualityMetrics:
        """
        Update quality metrics with new samples.
        
        Args:
            samples: New signal samples
            
        Returns:
            Current quality metrics dictionary
        """
        if len(samples) == 0:
            return {}
        
        # Basic power calculations
        power_dbfs = self.processor.calculate_power_dbfs(samples)
        snr_db = self.processor.estimate_snr(samples)
        
        # Estimate noise floor
        power = np.abs(samples) ** 2
        fft_power = np.abs(fft(samples)) ** 2
        noise_floor_dbfs = 10 * np.log10(np.percentile(fft_power, 10) + 1e-12)
        
        # Update histories
        self.power_history.append(power_dbfs)
        self.noise_history.append(noise_floor_dbfs)
        if snr_db > -50:  # Valid SNR
            self.snr_history.append(snr_db)
        
        # Update sample counts
        self.total_samples += len(samples)
        
        # Calculate metrics
        metrics = {
            'signal_level_dbfs': power_dbfs,
            'noise_floor_dbfs': noise_floor_dbfs,
            'snr_db': snr_db,
            'total_samples': self.total_samples,
            'dropped_samples': self.dropped_samples
        }
        
        # Add statistical metrics if enough history
        if len(self.power_history) > 10:
            metrics.update({
                'avg_signal_level_dbfs': float(np.mean(self.power_history)),
                'signal_std_db': float(np.std(self.power_history)),
                'avg_snr_db': float(np.mean(self.snr_history)) if self.snr_history else 0.0,
                'sample_loss_rate': (self.dropped_samples / self.total_samples) * 100.0
            })
        
        return metrics
    
    def detect_anomalies(self, current_metrics: QualityMetrics) -> List[str]:
        """Detect signal anomalies based on current metrics."""
        anomalies = []
        
        if len(self.power_history) < 10:
            return anomalies
        
        current_power = current_metrics.get('signal_level_dbfs', 0)
        power_history = list(self.power_history)[:-1]  # Exclude current
        
        if power_history:
            avg_power = np.mean(power_history)
            power_std = np.std(power_history)
            
            # Sudden power changes
            if abs(current_power - avg_power) > 3 * power_std:
                anomalies.append(f"Power anomaly: {current_power:.1f} dBFS (avg: {avg_power:.1f})")
            
            # Clipping detection
            if current_power > -3:
                anomalies.append(f"Potential clipping: {current_power:.1f} dBFS")
            
            # Very low signal
            if current_power < -80:
                anomalies.append(f"Low signal: {current_power:.1f} dBFS")
        
        return anomalies


# =============================================================================
# SIGNAL DETECTION AND PACKET UTILITIES
# =============================================================================

def detect_packets(
    iq_samples: Union[IQSamples, RealSamples],
    threshold: float = 0.05,
    min_gap: int = 1000
) -> List[Tuple[int, int]]:
    """
    Detect regions of signal activity based on amplitude thresholding.
    
    Args:
        iq_samples: Input signal samples
        threshold: Detection threshold (relative to signal level)
        min_gap: Minimum gap between packets in samples
        
    Returns:
        List of (start, end) tuples representing packet regions
    """
    if len(iq_samples) == 0:
        return []
    
    # Calculate envelope/power
    if np.iscomplexobj(iq_samples):
        power = np.abs(iq_samples)
    else:
        power = np.abs(iq_samples)
    
    # Adaptive threshold if needed
    if threshold < 1.0:
        threshold = threshold * np.max(power)
    
    # Find active regions
    active = power > threshold
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


def find_preamble(
    iq_data: Union[IQSamples, RealSamples],
    preamble_pattern: npt.NDArray,
    threshold: float = 0.8
) -> npt.NDArray[np.int32]:
    """
    Find preamble pattern in signal using correlation.
    
    Args:
        iq_data: Input signal data
        preamble_pattern: Known preamble pattern
        threshold: Correlation threshold (0-1)
        
    Returns:
        Array of indices where preamble was found
    """
    if len(iq_data) == 0 or len(preamble_pattern) == 0:
        return np.array([], dtype=np.int32)
    
    # Ensure same type
    if np.iscomplexobj(iq_data) != np.iscomplexobj(preamble_pattern):
        if np.iscomplexobj(iq_data):
            iq_data = np.abs(iq_data)
        else:
            preamble_pattern = np.abs(preamble_pattern)
    
    # Normalize inputs
    iq_norm = iq_data / (np.linalg.norm(iq_data) + 1e-12)
    pattern_norm = preamble_pattern / (np.linalg.norm(preamble_pattern) + 1e-12)
    
    # Cross-correlation
    correlation = correlate(iq_norm, pattern_norm, mode='valid')
    
    # Find peaks above threshold
    max_corr = np.max(np.abs(correlation))
    if max_corr == 0:
        return np.array([], dtype=np.int32)
    
    normalized_corr = np.abs(correlation) / max_corr
    hits = np.where(normalized_corr > threshold)[0]
    
    return hits.astype(np.int32)


# =============================================================================
# SIMPLE FUNCTION INTERFACES (BACKWARD COMPATIBILITY)
# =============================================================================

def normalize_signal(iq_data: npt.NDArray) -> npt.NDArray:
    """
    Simple signal normalization (backward compatible).
    
    Args:
        iq_data: Complex IQ data array
        
    Returns:
        Normalized IQ data
    """
    processor = SignalProcessor()
    return processor.normalize(iq_data, method='peak')


def iq_to_complex(
    i_samples: Union[List, npt.NDArray],
    q_samples: Union[List, npt.NDArray]
) -> npt.NDArray:
    """
    Convert separate I and Q samples to complex array (backward compatible).
    
    Args:
        i_samples: In-phase samples
        q_samples: Quadrature samples
        
    Returns:
        Complex IQ array
    """
    return np.array(i_samples) + 1j * np.array(q_samples)


def calculate_power(iq_data: npt.NDArray) -> float:
    """Calculate average power (backward compatible)."""
    if len(iq_data) == 0:
        return 0.0
    return float(np.mean(np.abs(iq_data) ** 2))


def frequency_shift(
    iq_data: npt.NDArray,
    shift_hz: float,
    sample_rate: float
) -> npt.NDArray:
    """Apply frequency shift (backward compatible)."""
    processor = SignalProcessor()
    return processor.frequency_shift(iq_data, shift_hz, sample_rate)


# =============================================================================
# ADVANCED ANALYSIS FUNCTIONS
# =============================================================================

def analyze_signal_quality(
    iq_data: Union[IQSamples, RealSamples],
    sample_rate: Optional[float] = None
) -> QualityMetrics:
    """
    Comprehensive signal quality analysis.
    
    Args:
        iq_data: Signal samples to analyze
        sample_rate: Sample rate in Hz (for spectral analysis)
        
    Returns:
        Dictionary of quality metrics
    """
    if len(iq_data) == 0:
        return {}
    
    processor = SignalProcessor()
    metrics = {}
    
    # Basic power metrics
    metrics['signal_power_dbfs'] = processor.calculate_power_dbfs(iq_data)
    metrics['peak_power_dbfs'] = 10 * np.log10(np.max(np.abs(iq_data) ** 2) + 1e-12)
    
    # SNR estimation
    metrics['estimated_snr_db'] = processor.estimate_snr(iq_data)
    
    # Crest factor
    avg_power = np.mean(np.abs(iq_data) ** 2)
    peak_power = np.max(np.abs(iq_data) ** 2)
    if avg_power > 0:
        metrics['crest_factor_db'] = 10 * np.log10(peak_power / avg_power)
    
    # Frequency domain analysis if sample rate provided
    if sample_rate is not None:
        try:
            freqs, psd = signal.welch(iq_data, fs=sample_rate, nperseg=min(1024, len(iq_data)//4))
            
            # Occupied bandwidth (99% power)
            cumulative_power = np.cumsum(psd)
            total_power = cumulative_power[-1]
            
            low_idx = np.where(cumulative_power >= 0.005 * total_power)[0]
            high_idx = np.where(cumulative_power >= 0.995 * total_power)[0]
            
            if len(low_idx) > 0 and len(high_idx) > 0:
                occupied_bw = freqs[high_idx[0]] - freqs[low_idx[0]]
                metrics['occupied_bandwidth_hz'] = float(abs(occupied_bw))
            
            # Spectral centroid
            if np.sum(psd) > 0:
                spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
                metrics['spectral_centroid_hz'] = float(spectral_centroid)
            
        except Exception as e:
            logger.debug(f"Spectral analysis failed: {e}")
    
    return metrics


def correct_iq_imbalance(iq_data: IQSamples) -> IQSamples:
    """
    Correct IQ imbalance using statistical methods.
    
    Args:
        iq_data: Complex IQ data with potential imbalance
        
    Returns:
        IQ data with corrected imbalance
    """
    if len(iq_data) < 100:
        return iq_data
    
    i_samples = np.real(iq_data)
    q_samples = np.imag(iq_data)
    
    # Amplitude imbalance correction
    i_power = np.var(i_samples)
    q_power = np.var(q_samples)
    
    if i_power > 0 and q_power > 0:
        amplitude_correction = np.sqrt(i_power / q_power)
        q_corrected = q_samples * amplitude_correction
    else:
        q_corrected = q_samples
    
    # Phase imbalance correction
    cross_correlation = np.mean(i_samples * q_corrected)
    if abs(cross_correlation) > 0.1:
        phase_correction = -np.arctan2(cross_correlation, np.var(q_corrected))
        q_corrected = (q_corrected * np.cos(phase_correction) - 
                      i_samples * np.sin(phase_correction))
    
    return (i_samples + 1j * q_corrected).astype(np.complex64)


def estimate_modulation_scheme(
    iq_data: IQSamples,
    sample_rate: float
) -> Dict[str, Any]:
    """
    Estimate modulation scheme using signal characteristics.
    
    Args:
        iq_data: Complex IQ data
        sample_rate: Sample rate in Hz
        
    Returns:
        Dictionary with modulation analysis results
    """
    if len(iq_data) < 100:
        return {'error': 'Insufficient data'}
    
    results = {}
    
    # Amplitude variation analysis
    amplitude = np.abs(iq_data)
    phase = np.angle(iq_data)
    
    amp_var = np.var(amplitude)
    amp_mean = np.mean(amplitude)
    
    if amp_mean > 0:
        amp_cv = np.sqrt(amp_var) / amp_mean
        
        if amp_cv < 0.1:
            results['amplitude_modulation'] = 'Low (PSK/FSK likely)'
        elif amp_cv > 0.3:
            results['amplitude_modulation'] = 'High (ASK/QAM likely)'
        else:
            results['amplitude_modulation'] = 'Medium (Unknown)'
    
    # Phase discontinuity analysis
    phase_diff = np.diff(np.unwrap(phase))
    phase_jumps = np.sum(np.abs(phase_diff) > np.pi/4)
    
    results['phase_discontinuities'] = int(phase_jumps)
    results['phase_modulated'] = phase_jumps > len(iq_data) * 0.01
    
    # Simple scheme estimation
    if amp_cv < 0.1 and results['phase_modulated']:
        results['likely_scheme'] = 'PSK'
    elif amp_cv > 0.3 and not results['phase_modulated']:
        results['likely_scheme'] = 'ASK/OOK'
    elif amp_cv < 0.1 and not results['phase_modulated']:
        results['likely_scheme'] = 'FSK'
    else:
        results['likely_scheme'] = 'QAM or Unknown'
    
    return results


def create_test_signal(
    signal_type: str,
    sample_rate: float,
    duration: float,
    **parameters: Any
) -> IQSamples:
    """
    Create test signals for validation and testing.
    
    Args:
        signal_type: Type of signal ('sine', 'chirp', 'noise', 'psk', 'fsk')
        sample_rate: Sample rate in Hz
        duration: Signal duration in seconds
        **parameters: Signal-specific parameters
        
    Returns:
        Generated IQ test signal
    """
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples)
    
    if signal_type == 'sine':
        frequency = parameters.get('frequency', 1000.0)
        amplitude = parameters.get('amplitude', 1.0)
        phase = parameters.get('phase', 0.0)
        signal_data = amplitude * np.exp(1j * (2 * np.pi * frequency * t + phase))
        
    elif signal_type == 'chirp':
        f_start = parameters.get('f_start', 0.0)
        f_end = parameters.get('f_end', sample_rate / 4)
        amplitude = parameters.get('amplitude', 1.0)
        
        instantaneous_freq = f_start + (f_end - f_start) * t / duration
        phase = 2 * np.pi * np.cumsum(instantaneous_freq) / sample_rate
        signal_data = amplitude * np.exp(1j * phase)
        
    elif signal_type == 'noise':
        noise_power = parameters.get('power', 1.0)
        signal_data = (np.random.normal(0, np.sqrt(noise_power/2), num_samples) + 
                      1j * np.random.normal(0, np.sqrt(noise_power/2), num_samples))
        
    elif signal_type == 'psk':
        carrier_freq = parameters.get('carrier_freq', 1000.0)
        symbol_rate = parameters.get('symbol_rate', 100.0)
        num_phases = parameters.get('num_phases', 2)
        
        symbols_per_second = symbol_rate
        total_symbols = int(duration * symbols_per_second)
        symbols = np.random.randint(0, num_phases, total_symbols)
        
        samples_per_symbol = int(sample_rate / symbol_rate)
        signal_data = np.zeros(num_samples, dtype=np.complex64)
        
        for i, symbol in enumerate(symbols):
            start_idx = i * samples_per_symbol
            end_idx = min(start_idx + samples_per_symbol, num_samples)
            if start_idx >= num_samples:
                break
                
            phase = 2 * np.pi * symbol / num_phases
            t_symbol = t[start_idx:end_idx]
            signal_data[start_idx:end_idx] = np.exp(1j * (2 * np.pi * carrier_freq * t_symbol + phase))
            
    else:
        raise ValueError(f"Unknown signal type: {signal_type}")
    
    return signal_data.astype(np.complex64)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def convert_power_units(
    value: float,
    from_unit: str,
    to_unit: str,
    reference_impedance: float = 50.0
) -> float:
    """Convert between different power units."""
    # Convert to watts first
    if from_unit.lower() == 'dbm':
        watts = 10 ** ((value - 30) / 10)
    elif from_unit.lower() == 'dbw':
        watts = 10 ** (value / 10)
    elif from_unit.lower() == 'watts':
        watts = value
    elif from_unit.lower() == 'milliwatts':
        watts = value / 1000
    elif from_unit.lower() == 'dbfs':
        watts = 10 ** (value / 10)
    else:
        raise ValueError(f"Unknown source unit: {from_unit}")
    
    # Convert to target unit
    if to_unit.lower() == 'dbm':
        return 10 * np.log10(watts * 1000)
    elif to_unit.lower() == 'dbw':
        return 10 * np.log10(watts)
    elif to_unit.lower() == 'watts':
        return watts
    elif to_unit.lower() == 'milliwatts':
        return watts * 1000
    elif to_unit.lower() == 'dbfs':
        return 10 * np.log10(watts)
    else:
        raise ValueError(f"Unknown target unit: {to_unit}")


# Factory function for easy processor creation
def create_signal_processor() -> SignalProcessor:
    """Create a new signal processor instance."""
    return SignalProcessor()


def create_quality_monitor(history_length: int = 1000) -> QualityMonitor:
    """Create a new quality monitor instance."""
    return QualityMonitor(history_length)


# Main function for testing
if __name__ == "__main__":
    # Test signal processor
    processor = SignalProcessor()
    
    # Create test signal
    test_signal = create_test_signal('psk', 48000, 0.1, num_phases=4)
    print(f"Created test signal: {len(test_signal)} samples")
    
    # Test normalization
    normalized = processor.normalize(test_signal, method='rms')
    print(f"Normalized signal power: {processor.calculate_power_dbfs(normalized):.1f} dBFS")
    
    # Test quality analysis
    quality = analyze_signal_quality(test_signal, sample_rate=48000)
    print(f"Quality metrics: {len(quality)} parameters")
    
    # Test modulation detection
    mod_result = estimate_modulation_scheme(test_signal, 48000)
    print(f"Modulation analysis: {mod_result.get('likely_scheme', 'Unknown')}")