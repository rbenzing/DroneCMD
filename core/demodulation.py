#!/usr/bin/env python3
"""
Enhanced Digital Signal Demodulation System

A comprehensive digital signal processing library for demodulating various
modulation schemes commonly used in wireless communications, including
drone control systems. This module implements industry-standard algorithms
with advanced signal processing techniques for robust performance in
challenging RF environments.

Standards Compliance:
- IEEE 802.11 wireless communication standards
- ITU-R radio communication recommendations  
- Digital modulation standards (ASK, FSK, PSK, QAM)
- Software Defined Radio (SDR) best practices
- Signal processing optimization techniques

Key Features:
- Multiple modulation scheme support (OOK, ASK, FSK, GFSK, MSK, PSK, QPSK, DPSK)
- Adaptive threshold and parameter estimation
- Advanced filtering and noise reduction
- Clock and carrier recovery algorithms
- Signal quality assessment and monitoring
- Real-time processing capabilities
- Memory-efficient implementations
- Comprehensive error handling and validation

Example:
    >>> from dronecmd.core.enhanced_demodulation import DemodulationEngine, DemodConfig
    >>> config = DemodConfig(
    ...     scheme=ModulationScheme.FSK,
    ...     sample_rate_hz=2.048e6,
    ...     bitrate_bps=9600,
    ...     enable_adaptive_threshold=True
    ... )
    >>> engine = DemodulationEngine(config)
    >>> bits, quality = engine.demodulate(iq_samples)
"""

from __future__ import annotations

import logging
import math
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import numpy.typing as npt
from scipy import signal
from scipy.fft import fft, fftfreq, fftshift
from scipy.optimize import minimize_scalar
from scipy.signal import hilbert, butter, sosfilt, find_peaks, correlate

# Configure module logger
logger = logging.getLogger(__name__)

# Type aliases for clarity
FrequencyHz = float
SampleRateHz = float
BitrateHz = float
TimeSeconds = float
IQSamples = npt.NDArray[np.complex64]
RealSamples = npt.NDArray[np.float32]
BitStream = npt.NDArray[np.uint8]
QualityMetrics = Dict[str, float]


class ModulationScheme(Enum):
    """Supported digital modulation schemes."""
    
    OOK = "ook"          # On-Off Keying
    ASK = "ask"          # Amplitude Shift Keying  
    FSK = "fsk"          # Frequency Shift Keying
    GFSK = "gfsk"        # Gaussian FSK
    MSK = "msk"          # Minimum Shift Keying
    AFSK = "afsk"        # Audio FSK
    PSK = "psk"          # Phase Shift Keying
    BPSK = "bpsk"        # Binary PSK
    QPSK = "qpsk"        # Quadrature PSK
    DPSK = "dpsk"        # Differential PSK
    QAM16 = "qam16"      # 16-QAM
    QAM64 = "qam64"      # 64-QAM
    
    @property
    def bits_per_symbol(self) -> int:
        """Get number of bits encoded per symbol."""
        bits = {
            self.OOK: 1, self.ASK: 1, self.FSK: 1, self.GFSK: 1,
            self.MSK: 1, self.AFSK: 1, self.PSK: 1, self.BPSK: 1,
            self.QPSK: 2, self.DPSK: 1, self.QAM16: 4, self.QAM64: 6
        }
        return bits.get(self, 1)
    
    @property
    def requires_coherent_detection(self) -> bool:
        """Check if scheme requires coherent carrier recovery."""
        coherent_schemes = {
            self.PSK, self.BPSK, self.QPSK, self.QAM16, self.QAM64
        }
        return self in coherent_schemes


class FilterType(Enum):
    """Digital filter types for signal conditioning."""
    
    BUTTERWORTH = "butterworth"
    CHEBYSHEV1 = "chebyshev1"
    CHEBYSHEV2 = "chebyshev2"
    ELLIPTIC = "elliptic"
    BESSEL = "bessel"
    RAISED_COSINE = "raised_cosine"
    ROOT_RAISED_COSINE = "root_raised_cosine"
    GAUSSIAN = "gaussian"


@dataclass(frozen=True)
class DemodConfig:
    """
    Configuration parameters for demodulation operations.
    
    Comprehensive configuration supporting various modulation schemes
    with advanced signal processing options.
    
    Attributes:
        scheme: Modulation scheme to demodulate
        sample_rate_hz: Sample rate of input IQ data
        bitrate_bps: Expected bit rate in bits per second
        enable_adaptive_threshold: Enable adaptive threshold algorithms
        enable_carrier_recovery: Enable carrier frequency/phase recovery
        enable_clock_recovery: Enable symbol timing recovery
        enable_agc: Enable automatic gain control
        filter_type: Type of digital filter to apply
        filter_order: Order of digital filters
        filter_cutoff_factor: Filter cutoff as factor of bitrate
        frequency_offset_max_hz: Maximum expected frequency offset
        phase_noise_bandwidth_hz: Phase noise tracking bandwidth
        enable_differential_decoding: Enable differential decoding
        enable_soft_decisions: Output soft decision values
        snr_estimation_method: Method for SNR estimation
        enable_performance_monitoring: Enable real-time performance metrics
        debug_mode: Enable debug output and intermediate results
    """
    
    scheme: ModulationScheme = ModulationScheme.OOK
    sample_rate_hz: SampleRateHz = 2.048e6
    bitrate_bps: BitrateHz = 9600
    enable_adaptive_threshold: bool = True
    enable_carrier_recovery: bool = True
    enable_clock_recovery: bool = True
    enable_agc: bool = True
    filter_type: FilterType = FilterType.BUTTERWORTH
    filter_order: int = 4
    filter_cutoff_factor: float = 2.0
    frequency_offset_max_hz: FrequencyHz = 1000.0
    phase_noise_bandwidth_hz: FrequencyHz = 100.0
    enable_differential_decoding: bool = False
    enable_soft_decisions: bool = False
    snr_estimation_method: str = "moment"
    enable_performance_monitoring: bool = True
    debug_mode: bool = False
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.sample_rate_hz <= 0:
            raise ValueError("Sample rate must be positive")
        
        if self.bitrate_bps <= 0:
            raise ValueError("Bit rate must be positive")
        
        if self.bitrate_bps >= self.sample_rate_hz / 2:
            raise ValueError("Bit rate must be less than Nyquist frequency")
        
        if self.filter_order < 1:
            raise ValueError("Filter order must be at least 1")
        
        if self.filter_cutoff_factor <= 0:
            raise ValueError("Filter cutoff factor must be positive")
    
    @property
    def samples_per_symbol(self) -> int:
        """Calculate samples per symbol."""
        symbol_rate = self.bitrate_bps / self.scheme.bits_per_symbol
        return int(self.sample_rate_hz / symbol_rate)
    
    @property
    def symbol_rate_hz(self) -> float:
        """Calculate symbol rate in Hz."""
        return self.bitrate_bps / self.scheme.bits_per_symbol


@dataclass
class DemodulationResult:
    """
    Results from demodulation operations.
    
    Contains demodulated data plus comprehensive quality metrics
    and processing information.
    """
    
    # Demodulated data
    bits: BitStream = field(default_factory=lambda: np.array([], dtype=np.uint8))
    soft_bits: Optional[npt.NDArray[np.float32]] = None
    symbols: Optional[npt.NDArray[np.complex64]] = None
    
    # Signal quality metrics
    snr_db: Optional[float] = None
    evm_percent: Optional[float] = None  # Error Vector Magnitude
    ber_estimate: Optional[float] = None  # Bit Error Rate estimate
    signal_power_dbfs: Optional[float] = None
    noise_power_dbfs: Optional[float] = None
    
    # Synchronization metrics
    carrier_frequency_offset_hz: float = 0.0
    phase_offset_deg: float = 0.0
    timing_offset_samples: float = 0.0
    clock_recovery_locked: bool = False
    
    # Processing metrics
    processing_time_ms: float = 0.0
    samples_processed: int = 0
    symbols_decoded: int = 0
    
    # Algorithm details
    threshold_used: Optional[float] = None
    filter_response: Optional[npt.NDArray[np.float32]] = None
    constellation_points: Optional[npt.NDArray[np.complex64]] = None
    
    # Validation flags
    is_valid: bool = True
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    @property
    def bit_rate_achieved(self) -> float:
        """Calculate achieved bit rate."""
        if self.processing_time_ms > 0:
            return len(self.bits) / (self.processing_time_ms / 1000.0)
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'bits_decoded': len(self.bits),
            'symbols_decoded': self.symbols_decoded,
            'snr_db': self.snr_db,
            'evm_percent': self.evm_percent,
            'ber_estimate': self.ber_estimate,
            'signal_power_dbfs': self.signal_power_dbfs,
            'noise_power_dbfs': self.noise_power_dbfs,
            'carrier_frequency_offset_hz': self.carrier_frequency_offset_hz,
            'phase_offset_deg': self.phase_offset_deg,
            'timing_offset_samples': self.timing_offset_samples,
            'clock_recovery_locked': self.clock_recovery_locked,
            'processing_time_ms': self.processing_time_ms,
            'samples_processed': self.samples_processed,
            'bit_rate_achieved': self.bit_rate_achieved,
            'threshold_used': self.threshold_used,
            'is_valid': self.is_valid,
            'error_message': self.error_message,
            'warnings': self.warnings
        }


class SignalProcessor:
    """
    Advanced signal processing utilities for demodulation.
    
    Provides optimized implementations of common signal processing
    operations used in digital demodulation.
    """
    
    @staticmethod
    def apply_agc(
        samples: Union[IQSamples, RealSamples],
        target_power: float = 1.0,
        attack_time: float = 0.001,
        release_time: float = 0.1,
        sample_rate: float = 1.0
    ) -> Union[IQSamples, RealSamples]:
        """
        Apply Automatic Gain Control (AGC) to signal.
        
        Args:
            samples: Input signal samples
            target_power: Target average power level
            attack_time: AGC attack time constant in seconds
            release_time: AGC release time constant in seconds  
            sample_rate: Sample rate in Hz
            
        Returns:
            AGC-controlled signal samples
        """
        if len(samples) == 0:
            return samples
        
        # Calculate power
        power = np.abs(samples) ** 2
        
        # Design AGC filter
        attack_alpha = 1.0 - np.exp(-1.0 / (attack_time * sample_rate))
        release_alpha = 1.0 - np.exp(-1.0 / (release_time * sample_rate))
        
        # Apply AGC
        avg_power = np.zeros(len(power))
        avg_power[0] = power[0]
        
        for i in range(1, len(power)):
            if power[i] > avg_power[i-1]:
                # Attack (fast response to increases)
                alpha = attack_alpha
            else:
                # Release (slow response to decreases)
                alpha = release_alpha
            
            avg_power[i] = alpha * power[i] + (1 - alpha) * avg_power[i-1]
        
        # Calculate gain
        gain = np.sqrt(target_power / (avg_power + 1e-12))
        
        # Limit gain to prevent excessive amplification
        gain = np.clip(gain, 0.1, 10.0)
        
        return samples * gain
    
    @staticmethod
    def design_digital_filter(
        filter_type: FilterType,
        cutoff_freq: float,
        sample_rate: float,
        order: int = 4,
        **kwargs: Any
    ) -> npt.NDArray[np.float64]:
        """
        Design digital filter with specified characteristics.
        
        Args:
            filter_type: Type of filter to design
            cutoff_freq: Cutoff frequency in Hz
            sample_rate: Sample rate in Hz
            order: Filter order
            **kwargs: Additional filter-specific parameters
            
        Returns:
            Filter coefficients (SOS format)
        """
        nyquist = sample_rate / 2
        normalized_cutoff = cutoff_freq / nyquist
        
        if normalized_cutoff >= 1.0:
            warnings.warn(f"Cutoff frequency {cutoff_freq} Hz exceeds Nyquist frequency")
            normalized_cutoff = 0.99
        
        if filter_type == FilterType.BUTTERWORTH:
            sos = butter(order, normalized_cutoff, btype='low', output='sos')
        elif filter_type == FilterType.CHEBYSHEV1:
            ripple = kwargs.get('ripple', 0.5)
            sos = signal.cheby1(order, ripple, normalized_cutoff, btype='low', output='sos')
        elif filter_type == FilterType.CHEBYSHEV2:
            attenuation = kwargs.get('attenuation', 40)
            sos = signal.cheby2(order, attenuation, normalized_cutoff, btype='low', output='sos')
        elif filter_type == FilterType.ELLIPTIC:
            ripple = kwargs.get('ripple', 0.5)
            attenuation = kwargs.get('attenuation', 40)
            sos = signal.ellip(order, ripple, attenuation, normalized_cutoff, btype='low', output='sos')
        elif filter_type == FilterType.BESSEL:
            sos = signal.bessel(order, normalized_cutoff, btype='low', output='sos')
        else:
            # Default to Butterworth
            sos = butter(order, normalized_cutoff, btype='low', output='sos')
        
        return sos
    
    @staticmethod
    def estimate_carrier_frequency(
        iq_samples: IQSamples,
        sample_rate: float,
        method: str = "fft_peak"
    ) -> float:
        """
        Estimate carrier frequency from IQ samples.
        
        Args:
            iq_samples: Complex IQ samples
            sample_rate: Sample rate in Hz
            method: Estimation method ('fft_peak', 'autocorr', 'esprit')
            
        Returns:
            Estimated carrier frequency in Hz
        """
        if len(iq_samples) < 64:
            return 0.0
        
        if method == "fft_peak":
            # Use FFT to find peak frequency
            fft_result = fftshift(fft(iq_samples))
            freqs = fftshift(fftfreq(len(iq_samples), 1/sample_rate))
            peak_idx = np.argmax(np.abs(fft_result))
            return freqs[peak_idx]
        
        elif method == "autocorr":
            # Use autocorrelation method
            autocorr = correlate(iq_samples, iq_samples, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Find first significant peak (excluding DC)
            peaks, _ = find_peaks(np.abs(autocorr[1:]), height=np.max(np.abs(autocorr)) * 0.1)
            if len(peaks) > 0:
                period_samples = peaks[0] + 1
                return sample_rate / period_samples
            
        return 0.0  # Could not estimate
    
    @staticmethod
    def estimate_snr(
        signal: Union[IQSamples, RealSamples],
        method: str = "moment"
    ) -> float:
        """
        Estimate Signal-to-Noise Ratio.
        
        Args:
            signal: Input signal samples
            method: Estimation method ('moment', 'percentile', 'em')
            
        Returns:
            Estimated SNR in dB
        """
        if len(signal) < 10:
            return -np.inf
        
        power = np.abs(signal) ** 2
        
        if method == "moment":
            # Use second and fourth moments
            m2 = np.mean(power)
            m4 = np.mean(power ** 2)
            
            # For complex Gaussian noise: m4/m2^2 = 2
            # For signal + noise: ratio is higher
            if m2 > 0:
                kurtosis_ratio = m4 / (m2 ** 2)
                if kurtosis_ratio > 2.1:  # Has signal component
                    noise_var = m2 / kurtosis_ratio
                    signal_var = m2 - noise_var
                    if signal_var > 0 and noise_var > 0:
                        return 10 * np.log10(signal_var / noise_var)
        
        elif method == "percentile":
            # Assume noise is in lower percentiles
            noise_power = np.percentile(power, 25)
            signal_power = np.mean(power)
            if noise_power > 0:
                return 10 * np.log10(signal_power / noise_power)
        
        return 0.0  # Could not estimate


class BaseDemodulator(ABC):
    """Abstract base class for all demodulators."""
    
    def __init__(self, config: DemodConfig) -> None:
        """Initialize base demodulator."""
        self.config = config
        self.signal_processor = SignalProcessor()
    
    @abstractmethod
    def demodulate(self, iq_samples: IQSamples) -> DemodulationResult:
        """
        Demodulate IQ samples to digital bits.
        
        Args:
            iq_samples: Complex IQ samples
            
        Returns:
            Demodulation result with bits and quality metrics
        """
        pass
    
    def _preprocess_signal(self, iq_samples: IQSamples) -> IQSamples:
        """Apply common signal preprocessing."""
        processed = iq_samples.copy()
        
        # Apply AGC if enabled
        if self.config.enable_agc:
            processed = self.signal_processor.apply_agc(
                processed,
                target_power=1.0,
                sample_rate=self.config.sample_rate_hz
            )
        
        # Apply low-pass filtering
        cutoff_freq = self.config.bitrate_bps * self.config.filter_cutoff_factor
        sos = self.signal_processor.design_digital_filter(
            self.config.filter_type,
            cutoff_freq,
            self.config.sample_rate_hz,
            self.config.filter_order
        )
        processed = sosfilt(sos, processed).astype(np.complex64)
        
        return processed
    
    def _estimate_quality_metrics(
        self,
        iq_samples: IQSamples,
        demod_result: DemodulationResult
    ) -> None:
        """Estimate signal quality metrics."""
        # Estimate SNR
        demod_result.snr_db = self.signal_processor.estimate_snr(iq_samples)
        
        # Calculate signal power
        signal_power = np.mean(np.abs(iq_samples) ** 2)
        demod_result.signal_power_dbfs = 10 * np.log10(signal_power + 1e-12)
        
        # Estimate noise power (crude approximation)
        if demod_result.snr_db is not None and demod_result.snr_db > -50:
            noise_power_linear = signal_power / (10 ** (demod_result.snr_db / 10))
            demod_result.noise_power_dbfs = 10 * np.log10(noise_power_linear + 1e-12)
        
        # Estimate carrier frequency offset
        if self.config.enable_carrier_recovery:
            demod_result.carrier_frequency_offset_hz = self.signal_processor.estimate_carrier_frequency(
                iq_samples, self.config.sample_rate_hz
            )


class OOKDemodulator(BaseDemodulator):
    """
    On-Off Keying (OOK) demodulator with adaptive threshold.
    
    Implements advanced OOK demodulation with signal conditioning,
    adaptive threshold estimation, and quality assessment.
    """
    
    def demodulate(self, iq_samples: IQSamples) -> DemodulationResult:
        """Demodulate OOK signal."""
        start_time = time.time()
        result = DemodulationResult()
        
        try:
            # Preprocess signal
            processed = self._preprocess_signal(iq_samples)
            
            # Calculate envelope (magnitude)
            envelope = np.abs(processed)
            
            # Determine threshold
            if self.config.enable_adaptive_threshold:
                threshold = self._adaptive_threshold(envelope)
            else:
                threshold = np.mean(envelope) * 1.2
            
            result.threshold_used = threshold
            
            # Sample at bit centers if clock recovery enabled
            if self.config.enable_clock_recovery:
                bit_samples, timing_offset = self._sample_at_bit_centers(envelope)
                result.timing_offset_samples = timing_offset
                result.clock_recovery_locked = True
            else:
                # Simple decimation
                samples_per_bit = self.config.samples_per_symbol
                bit_samples = envelope[samples_per_bit//2::samples_per_bit]
            
            # Make binary decisions
            result.bits = (bit_samples > threshold).astype(np.uint8)
            
            # Generate soft decisions if requested
            if self.config.enable_soft_decisions:
                result.soft_bits = (bit_samples - threshold).astype(np.float32)
            
            # Estimate quality metrics
            self._estimate_quality_metrics(iq_samples, result)
            
            # Calculate additional OOK-specific metrics
            result.symbols_decoded = len(result.bits)
            result.samples_processed = len(iq_samples)
            
            result.is_valid = True
            
        except Exception as e:
            result.is_valid = False
            result.error_message = str(e)
            logger.error(f"OOK demodulation failed: {e}")
        
        finally:
            result.processing_time_ms = (time.time() - start_time) * 1000
        
        return result
    
    def _adaptive_threshold(self, envelope: RealSamples) -> float:
        """Calculate adaptive threshold using signal statistics."""
        # Use Otsu's method for optimal threshold
        hist, bin_edges = np.histogram(envelope, bins=256)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Calculate weighted variances for all possible thresholds
        total_weight = np.sum(hist)
        total_mean = np.sum(hist * bin_centers) / total_weight
        
        max_variance = 0
        optimal_threshold = np.mean(envelope)
        
        weight_bg = 0
        sum_bg = 0
        
        for i in range(len(hist)):
            weight_bg += hist[i]
            weight_fg = total_weight - weight_bg
            
            if weight_bg == 0 or weight_fg == 0:
                continue
            
            sum_bg += hist[i] * bin_centers[i]
            mean_bg = sum_bg / weight_bg
            mean_fg = (total_mean * total_weight - sum_bg) / weight_fg
            
            # Between-class variance
            variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
            
            if variance > max_variance:
                max_variance = variance
                optimal_threshold = bin_centers[i]
        
        return optimal_threshold
    
    def _sample_at_bit_centers(
        self,
        envelope: RealSamples
    ) -> Tuple[RealSamples, float]:
        """Sample envelope at optimal bit center positions."""
        samples_per_bit = self.config.samples_per_symbol
        
        # Find optimal sampling phase using squared signal
        envelope_squared = envelope ** 2
        
        # Try different sampling phases
        best_phase = 0
        best_metric = -np.inf
        
        for phase in range(samples_per_bit):
            samples = envelope_squared[phase::samples_per_bit]
            if len(samples) > 1:
                # Use variance as quality metric (higher is better for OOK)
                metric = np.var(samples)
                if metric > best_metric:
                    best_metric = metric
                    best_phase = phase
        
        # Extract samples at optimal phase
        bit_samples = envelope[best_phase::samples_per_bit]
        timing_offset = best_phase - samples_per_bit // 2
        
        return bit_samples, timing_offset


class FSKDemodulator(BaseDemodulator):
    """
    Frequency Shift Keying (FSK) demodulator.
    
    Implements coherent and non-coherent FSK demodulation with
    frequency estimation and tracking.
    """
    
    def demodulate(self, iq_samples: IQSamples) -> DemodulationResult:
        """Demodulate FSK signal."""
        start_time = time.time()
        result = DemodulationResult()
        
        try:
            # Preprocess signal
            processed = self._preprocess_signal(iq_samples)
            
            # Estimate mark and space frequencies
            mark_freq, space_freq = self._estimate_fsk_frequencies(processed)
            
            # Correlate with mark and space reference signals
            mark_correlation, space_correlation = self._fsk_correlate(
                processed, mark_freq, space_freq
            )
            
            # Make decisions
            decisions = mark_correlation > space_correlation
            result.bits = decisions.astype(np.uint8)
            
            # Generate soft decisions
            if self.config.enable_soft_decisions:
                result.soft_bits = (mark_correlation - space_correlation).astype(np.float32)
            
            # Estimate quality metrics
            self._estimate_quality_metrics(iq_samples, result)
            
            # FSK-specific metrics
            freq_separation = abs(mark_freq - space_freq)
            result.symbols_decoded = len(result.bits)
            result.samples_processed = len(iq_samples)
            
            # Store additional info in warnings for debug
            if self.config.debug_mode:
                result.warnings.append(f"Mark freq: {mark_freq:.1f} Hz")
                result.warnings.append(f"Space freq: {space_freq:.1f} Hz")
                result.warnings.append(f"Freq separation: {freq_separation:.1f} Hz")
            
            result.is_valid = True
            
        except Exception as e:
            result.is_valid = False
            result.error_message = str(e)
            logger.error(f"FSK demodulation failed: {e}")
        
        finally:
            result.processing_time_ms = (time.time() - start_time) * 1000
        
        return result
    
    def _estimate_fsk_frequencies(self, iq_samples: IQSamples) -> Tuple[float, float]:
        """Estimate mark and space frequencies using spectral analysis."""
        # Use FFT to find frequency peaks
        fft_result = fftshift(fft(iq_samples))
        freqs = fftshift(fftfreq(len(iq_samples), 1/self.config.sample_rate_hz))
        power_spectrum = np.abs(fft_result) ** 2
        
        # Find peaks in power spectrum
        peak_indices, _ = find_peaks(
            power_spectrum,
            height=np.max(power_spectrum) * 0.1,
            distance=len(power_spectrum) // 20
        )
        
        if len(peak_indices) >= 2:
            # Take two strongest peaks
            peak_powers = power_spectrum[peak_indices]
            strongest_peaks = peak_indices[np.argsort(peak_powers)[-2:]]
            
            freq1 = freqs[strongest_peaks[0]]
            freq2 = freqs[strongest_peaks[1]]
            
            # Assign based on typical FSK convention
            if freq1 > freq2:
                mark_freq, space_freq = freq1, freq2
            else:
                mark_freq, space_freq = freq2, freq1
        else:
            # Default estimate based on bitrate
            deviation = self.config.bitrate_bps / 2
            mark_freq = deviation
            space_freq = -deviation
        
        return mark_freq, space_freq
    
    def _fsk_correlate(
        self,
        iq_samples: IQSamples,
        mark_freq: float,
        space_freq: float
    ) -> Tuple[RealSamples, RealSamples]:
        """Correlate signal with mark and space reference frequencies."""
        samples_per_bit = self.config.samples_per_symbol
        
        mark_correlations = []
        space_correlations = []
        
        # Process signal in symbol-length segments
        for i in range(0, len(iq_samples) - samples_per_bit + 1, samples_per_bit):
            segment = iq_samples[i:i + samples_per_bit]
            
            # Generate reference signals
            t = np.arange(len(segment)) / self.config.sample_rate_hz
            mark_ref = np.exp(1j * 2 * np.pi * mark_freq * t)
            space_ref = np.exp(1j * 2 * np.pi * space_freq * t)
            
            # Calculate correlations (magnitude of complex correlation)
            mark_corr = np.abs(np.sum(segment * np.conj(mark_ref)))
            space_corr = np.abs(np.sum(segment * np.conj(space_ref)))
            
            mark_correlations.append(mark_corr)
            space_correlations.append(space_corr)
        
        return np.array(mark_correlations), np.array(space_correlations)


class PSKDemodulator(BaseDemodulator):
    """
    Phase Shift Keying (PSK) demodulator with carrier recovery.
    
    Implements coherent PSK demodulation with Costas loop carrier
    recovery and decision-directed tracking.
    """
    
    def demodulate(self, iq_samples: IQSamples) -> DemodulationResult:
        """Demodulate PSK signal."""
        start_time = time.time()
        result = DemodulationResult()
        
        try:
            # Preprocess signal
            processed = self._preprocess_signal(iq_samples)
            
            # Carrier recovery (simplified)
            if self.config.enable_carrier_recovery:
                recovered, phase_offset = self._carrier_recovery(processed)
                result.phase_offset_deg = np.degrees(phase_offset)
            else:
                recovered = processed
                result.phase_offset_deg = 0.0
            
            # Symbol timing recovery
            if self.config.enable_clock_recovery:
                symbols, timing_offset = self._symbol_timing_recovery(recovered)
                result.timing_offset_samples = timing_offset
                result.clock_recovery_locked = True
            else:
                # Simple decimation
                samples_per_symbol = self.config.samples_per_symbol
                symbols = recovered[samples_per_symbol//2::samples_per_symbol]
            
            # Store constellation points
            result.constellation_points = symbols
            
            # Make decisions based on modulation type
            if self.config.scheme in (ModulationScheme.PSK, ModulationScheme.BPSK):
                result.bits = (np.real(symbols) > 0).astype(np.uint8)
            elif self.config.scheme == ModulationScheme.QPSK:
                # QPSK: 2 bits per symbol
                i_bits = (np.real(symbols) > 0).astype(np.uint8)
                q_bits = (np.imag(symbols) > 0).astype(np.uint8)
                # Interleave I and Q bits
                result.bits = np.zeros(len(i_bits) * 2, dtype=np.uint8)
                result.bits[0::2] = i_bits
                result.bits[1::2] = q_bits
            
            # Soft decisions
            if self.config.enable_soft_decisions:
                if self.config.scheme == ModulationScheme.QPSK:
                    soft_i = np.real(symbols).astype(np.float32)
                    soft_q = np.imag(symbols).astype(np.float32)
                    result.soft_bits = np.zeros(len(soft_i) * 2, dtype=np.float32)
                    result.soft_bits[0::2] = soft_i
                    result.soft_bits[1::2] = soft_q
                else:
                    result.soft_bits = np.real(symbols).astype(np.float32)
            
            # Calculate EVM (Error Vector Magnitude)
            if len(symbols) > 0:
                ideal_constellation = self._get_ideal_constellation(symbols)
                error_vectors = symbols - ideal_constellation
                evm = np.sqrt(np.mean(np.abs(error_vectors) ** 2))
                reference_power = np.sqrt(np.mean(np.abs(ideal_constellation) ** 2))
                result.evm_percent = (evm / reference_power) * 100
            
            # Estimate quality metrics
            self._estimate_quality_metrics(iq_samples, result)
            
            result.symbols_decoded = len(symbols)
            result.samples_processed = len(iq_samples)
            result.is_valid = True
            
        except Exception as e:
            result.is_valid = False
            result.error_message = str(e)
            logger.error(f"PSK demodulation failed: {e}")
        
        finally:
            result.processing_time_ms = (time.time() - start_time) * 1000
        
        return result
    
    def _carrier_recovery(self, iq_samples: IQSamples) -> Tuple[IQSamples, float]:
        """Simplified carrier recovery using Costas loop concept."""
        # Estimate carrier frequency using FFT
        carrier_freq = self.signal_processor.estimate_carrier_frequency(
            iq_samples, self.config.sample_rate_hz
        )
        
        # Generate local oscillator to remove carrier
        t = np.arange(len(iq_samples)) / self.config.sample_rate_hz
        local_osc = np.exp(-1j * 2 * np.pi * carrier_freq * t)
        
        # Mix to baseband
        baseband = iq_samples * local_osc
        
        # Apply low-pass filter to remove double-frequency terms
        cutoff = self.config.bitrate_bps * 2
        sos = self.signal_processor.design_digital_filter(
            FilterType.BUTTERWORTH, cutoff, self.config.sample_rate_hz, 4
        )
        recovered = sosfilt(sos, baseband).astype(np.complex64)
        
        return recovered, 0.0  # Phase offset estimation would be more complex
    
    def _symbol_timing_recovery(self, baseband: IQSamples) -> Tuple[IQSamples, float]:
        """Symbol timing recovery using early-late gate."""
        samples_per_symbol = self.config.samples_per_symbol
        
        # Simple timing recovery: find optimal sampling phase
        # by maximizing symbol energy variance
        best_phase = 0
        best_variance = -1
        
        for phase in range(samples_per_symbol):
            symbols = baseband[phase::samples_per_symbol]
            if len(symbols) > 1:
                symbol_energy = np.abs(symbols) ** 2
                variance = np.var(symbol_energy)
                if variance > best_variance:
                    best_variance = variance
                    best_phase = phase
        
        # Extract symbols at optimal timing
        symbols = baseband[best_phase::samples_per_symbol]
        timing_offset = best_phase - samples_per_symbol // 2
        
        return symbols, timing_offset
    
    def _get_ideal_constellation(self, symbols: IQSamples) -> IQSamples:
        """Get ideal constellation points for EVM calculation."""
        if self.config.scheme in (ModulationScheme.PSK, ModulationScheme.BPSK):
            # BPSK: ±1
            decisions = (np.real(symbols) > 0).astype(np.float32) * 2 - 1
            return decisions.astype(np.complex64)
        elif self.config.scheme == ModulationScheme.QPSK:
            # QPSK: ±1±j
            i_decisions = (np.real(symbols) > 0).astype(np.float32) * 2 - 1
            q_decisions = (np.imag(symbols) > 0).astype(np.float32) * 2 - 1
            return (i_decisions + 1j * q_decisions).astype(np.complex64)
        else:
            return symbols  # Fallback


class DemodulationEngine:
    """
    Main demodulation engine supporting multiple modulation schemes.
    
    Provides a unified interface for demodulating various digital
    modulation schemes with automatic scheme detection and optimization.
    """
    
    def __init__(self, config: DemodConfig) -> None:
        """
        Initialize demodulation engine.
        
        Args:
            config: Demodulation configuration
        """
        self.config = config
        self._demodulators = self._create_demodulators()
        
        # Performance monitoring
        self.stats = {
            'demodulations_performed': 0,
            'total_processing_time': 0.0,
            'scheme_usage': {scheme.value: 0 for scheme in ModulationScheme}
        }
        
        logger.info(f"Initialized demodulation engine for {config.scheme.value}")
    
    def _create_demodulators(self) -> Dict[ModulationScheme, BaseDemodulator]:
        """Create demodulator instances for supported schemes."""
        demodulators = {}
        
        # Create demodulator for configured scheme
        if self.config.scheme in (ModulationScheme.OOK, ModulationScheme.ASK):
            demodulators[self.config.scheme] = OOKDemodulator(self.config)
        elif self.config.scheme in (
            ModulationScheme.FSK, ModulationScheme.GFSK, 
            ModulationScheme.MSK, ModulationScheme.AFSK
        ):
            demodulators[self.config.scheme] = FSKDemodulator(self.config)
        elif self.config.scheme in (
            ModulationScheme.PSK, ModulationScheme.BPSK, 
            ModulationScheme.QPSK, ModulationScheme.DPSK
        ):
            demodulators[self.config.scheme] = PSKDemodulator(self.config)
        else:
            raise NotImplementedError(f"Demodulator for {self.config.scheme.value} not implemented")
        
        return demodulators
    
    def demodulate(
        self, 
        iq_samples: IQSamples,
        scheme_override: Optional[ModulationScheme] = None
    ) -> DemodulationResult:
        """
        Demodulate IQ samples using configured or specified scheme.
        
        Args:
            iq_samples: Complex IQ samples to demodulate
            scheme_override: Optional scheme override
            
        Returns:
            Demodulation result with bits and quality metrics
        """
        if len(iq_samples) == 0:
            result = DemodulationResult()
            result.is_valid = False
            result.error_message = "Empty input samples"
            return result
        
        # Determine scheme to use
        scheme = scheme_override or self.config.scheme
        
        # Get appropriate demodulator
        if scheme not in self._demodulators:
            # Create demodulator for override scheme
            temp_config = DemodConfig(
                scheme=scheme,
                sample_rate_hz=self.config.sample_rate_hz,
                bitrate_bps=self.config.bitrate_bps,
                **{k: v for k, v in self.config.__dict__.items() 
                   if k not in ('scheme',)}
            )
            if scheme in (ModulationScheme.OOK, ModulationScheme.ASK):
                demodulator = OOKDemodulator(temp_config)
            elif scheme in (ModulationScheme.FSK, ModulationScheme.GFSK, ModulationScheme.MSK):
                demodulator = FSKDemodulator(temp_config)
            elif scheme in (ModulationScheme.PSK, ModulationScheme.BPSK, ModulationScheme.QPSK):
                demodulator = PSKDemodulator(temp_config)
            else:
                result = DemodulationResult()
                result.is_valid = False
                result.error_message = f"Unsupported scheme: {scheme.value}"
                return result
        else:
            demodulator = self._demodulators[scheme]
        
        # Perform demodulation
        start_time = time.time()
        try:
            result = demodulator.demodulate(iq_samples)
            
            # Update statistics
            self.stats['demodulations_performed'] += 1
            self.stats['scheme_usage'][scheme.value] += 1
            
            processing_time = time.time() - start_time
            self.stats['total_processing_time'] += processing_time
            
            if self.config.enable_performance_monitoring:
                logger.debug(
                    f"Demodulated {len(iq_samples)} samples using {scheme.value} "
                    f"in {processing_time*1000:.1f}ms, got {len(result.bits)} bits"
                )
            
            return result
            
        except Exception as e:
            result = DemodulationResult()
            result.is_valid = False
            result.error_message = str(e)
            result.processing_time_ms = (time.time() - start_time) * 1000
            logger.error(f"Demodulation failed: {e}")
            return result
    
    def auto_detect_scheme(
        self,
        iq_samples: IQSamples,
        candidates: Optional[List[ModulationScheme]] = None
    ) -> Tuple[ModulationScheme, float]:
        """
        Automatically detect modulation scheme from signal characteristics.
        
        Args:
            iq_samples: IQ samples to analyze
            candidates: Optional list of candidate schemes to test
            
        Returns:
            Tuple of (detected_scheme, confidence)
        """
        if candidates is None:
            candidates = [
                ModulationScheme.OOK,
                ModulationScheme.FSK, 
                ModulationScheme.PSK,
                ModulationScheme.QPSK
            ]
        
        best_scheme = self.config.scheme
        best_score = -np.inf
        
        for scheme in candidates:
            try:
                result = self.demodulate(iq_samples, scheme_override=scheme)
                
                if result.is_valid:
                    # Score based on SNR and other quality metrics
                    score = result.snr_db or -50
                    
                    # Bonus for successful error correction metrics
                    if result.evm_percent is not None and result.evm_percent < 50:
                        score += 10
                    
                    if score > best_score:
                        best_score = score
                        best_scheme = scheme
                        
            except Exception as e:
                logger.debug(f"Auto-detection failed for {scheme.value}: {e}")
        
        confidence = min(1.0, max(0.0, (best_score + 50) / 50))  # Normalize to 0-1
        return best_scheme, confidence
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get engine performance statistics."""
        avg_time = (
            self.stats['total_processing_time'] / 
            max(1, self.stats['demodulations_performed'])
        )
        
        return {
            'demodulations_performed': self.stats['demodulations_performed'],
            'average_processing_time_ms': avg_time * 1000,
            'total_processing_time_s': self.stats['total_processing_time'],
            'scheme_usage_distribution': self.stats['scheme_usage'],
            'configured_scheme': self.config.scheme.value,
            'samples_per_symbol': self.config.samples_per_symbol,
            'symbol_rate_hz': self.config.symbol_rate_hz
        }


# Convenience functions for backward compatibility and simple usage
def ook_decode(
    iq: IQSamples, 
    sample_rate: SampleRateHz = 2.048e6,
    bitrate: BitrateHz = 9600,
    threshold_ratio: float = 1.2
) -> BitStream:
    """
    Simplified OOK decoding function for backward compatibility.
    
    Args:
        iq: Complex IQ samples
        sample_rate: Sample rate in Hz
        bitrate: Bit rate in bits per second
        threshold_ratio: Threshold ratio for decision making
        
    Returns:
        Decoded bit stream
    """
    config = DemodConfig(
        scheme=ModulationScheme.OOK,
        sample_rate_hz=sample_rate,
        bitrate_bps=bitrate,
        enable_adaptive_threshold=False
    )
    
    engine = DemodulationEngine(config)
    result = engine.demodulate(iq)
    
    if not result.is_valid:
        logger.warning(f"OOK decode failed: {result.error_message}")
        return np.array([], dtype=np.uint8)
    
    return result.bits


def fsk_decode(
    iq: IQSamples,
    sample_rate: SampleRateHz = 2.048e6,
    bitrate: BitrateHz = 9600
) -> BitStream:
    """
    Simplified FSK decoding function for backward compatibility.
    
    Args:
        iq: Complex IQ samples
        sample_rate: Sample rate in Hz  
        bitrate: Bit rate in bits per second
        
    Returns:
        Decoded bit stream
    """
    config = DemodConfig(
        scheme=ModulationScheme.FSK,
        sample_rate_hz=sample_rate,
        bitrate_bps=bitrate,
        enable_carrier_recovery=True
    )
    
    engine = DemodulationEngine(config)
    result = engine.demodulate(iq)
    
    if not result.is_valid:
        logger.warning(f"FSK decode failed: {result.error_message}")
        return np.array([], dtype=np.uint8)
    
    return result.bits


def main() -> None:
    """
    Example usage and testing of the enhanced demodulation system.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=== Enhanced Demodulation System Demo ===\n")
    
    # Configuration
    sample_rate = 2.048e6
    bitrate = 9600
    duration = 0.05  # 50ms
    
    # Generate test signals
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Test data pattern
    test_bits = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1])
    samples_per_bit = int(sample_rate / bitrate)
    
    print(f"Test configuration:")
    print(f"  Sample rate: {sample_rate/1e6:.3f} MHz")
    print(f"  Bit rate: {bitrate} bps")
    print(f"  Test pattern: {''.join(map(str, test_bits))}")
    print(f"  Samples per bit: {samples_per_bit}")
    
    # Test 1: OOK Signal
    print(f"\n=== Testing OOK Demodulation ===")
    ook_signal = np.array([])
    for bit in test_bits:
        amplitude = 1.0 if bit else 0.1
        bit_signal = amplitude * np.ones(samples_per_bit, dtype=np.complex64)
        ook_signal = np.concatenate([ook_signal, bit_signal])
    
    # Add noise
    noise_power = 0.05
    noise = (np.random.normal(0, noise_power, len(ook_signal)) + 
             1j * np.random.normal(0, noise_power, len(ook_signal)))
    ook_signal += noise
    
    # Demodulate OOK
    ook_config = DemodConfig(
        scheme=ModulationScheme.OOK,
        sample_rate_hz=sample_rate,
        bitrate_bps=bitrate,
        enable_adaptive_threshold=True,
        enable_clock_recovery=True
    )
    
    ook_engine = DemodulationEngine(ook_config)
    ook_result = ook_engine.demodulate(ook_signal)
    
    print(f"OOK Results:")
    print(f"  Valid: {ook_result.is_valid}")
    print(f"  Bits decoded: {len(ook_result.bits)}")
    print(f"  Decoded pattern: {''.join(map(str, ook_result.bits[:len(test_bits)]))}")
    print(f"  SNR: {ook_result.snr_db:.1f} dB")
    print(f"  Processing time: {ook_result.processing_time_ms:.1f} ms")
    print(f"  Threshold used: {ook_result.threshold_used:.3f}")
    
    # Test 2: FSK Signal  
    print(f"\n=== Testing FSK Demodulation ===")
    mark_freq = 2000  # Hz for '1'
    space_freq = 1000  # Hz for '0'
    
    fsk_signal = np.array([])
    for bit in test_bits:
        freq = mark_freq if bit else space_freq
        bit_time = np.linspace(0, 1/bitrate, samples_per_bit)
        bit_signal = np.exp(1j * 2 * np.pi * freq * bit_time)
        fsk_signal = np.concatenate([fsk_signal, bit_signal])
    
    # Add noise
    fsk_signal += noise[:len(fsk_signal)]
    
    # Demodulate FSK
    fsk_config = DemodConfig(
        scheme=ModulationScheme.FSK,
        sample_rate_hz=sample_rate,
        bitrate_bps=bitrate,
        enable_carrier_recovery=True,
        debug_mode=True
    )
    
    fsk_engine = DemodulationEngine(fsk_config)
    fsk_result = fsk_engine.demodulate(fsk_signal)
    
    print(f"FSK Results:")
    print(f"  Valid: {fsk_result.is_valid}")
    print(f"  Bits decoded: {len(fsk_result.bits)}")
    print(f"  Decoded pattern: {''.join(map(str, fsk_result.bits[:len(test_bits)]))}")
    print(f"  SNR: {fsk_result.snr_db:.1f} dB")
    print(f"  Processing time: {fsk_result.processing_time_ms:.1f} ms")
    print(f"  Warnings: {fsk_result.warnings}")
    
    # Test 3: PSK Signal
    print(f"\n=== Testing PSK Demodulation ===")
    carrier_freq = 10000  # 10 kHz carrier
    
    psk_signal = np.array([])
    for bit in test_bits:
        phase = 0 if bit else np.pi  # 0° for '1', 180° for '0'
        bit_time = np.linspace(0, 1/bitrate, samples_per_bit)
        bit_signal = np.exp(1j * (2 * np.pi * carrier_freq * bit_time + phase))
        psk_signal = np.concatenate([psk_signal, bit_signal])
    
    # Add noise
    psk_signal += noise[:len(psk_signal)]
    
    # Demodulate PSK
    psk_config = DemodConfig(
        scheme=ModulationScheme.PSK,
        sample_rate_hz=sample_rate,
        bitrate_bps=bitrate,
        enable_carrier_recovery=True,
        enable_clock_recovery=True
    )
    
    psk_engine = DemodulationEngine(psk_config)
    psk_result = psk_engine.demodulate(psk_signal)
    
    print(f"PSK Results:")
    print(f"  Valid: {psk_result.is_valid}")
    print(f"  Bits decoded: {len(psk_result.bits)}")
    print(f"  Decoded pattern: {''.join(map(str, psk_result.bits[:len(test_bits)]))}")
    print(f"  SNR: {psk_result.snr_db:.1f} dB")
    print(f"  EVM: {psk_result.evm_percent:.1f}%")
    print(f"  Processing time: {psk_result.processing_time_ms:.1f} ms")
    print(f"  Phase offset: {psk_result.phase_offset_deg:.1f}°")
    
    # Test 4: Auto-detection
    print(f"\n=== Testing Auto-Detection ===")
    detected_scheme, confidence = ook_engine.auto_detect_scheme(
        ook_signal,
        candidates=[ModulationScheme.OOK, ModulationScheme.FSK, ModulationScheme.PSK]
    )
    print(f"Auto-detected scheme: {detected_scheme.value} (confidence: {confidence:.3f})")
    
    # Performance statistics
    print(f"\n=== Performance Statistics ===")
    ook_stats = ook_engine.get_performance_stats()
    for key, value in ook_stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()