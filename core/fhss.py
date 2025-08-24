#!/usr/bin/env python3
"""
Consolidated FHSS (Frequency Hopping Spread Spectrum) Engine

This module consolidates the FHSS implementations from fhss_engine.py and signal_tools.py
providing both advanced FCC-compliant features and simple interfaces for common use cases.

The module follows the progressive enhancement principle:
- SimpleFHSS: Easy-to-use interface for basic FHSS operations
- EnhancedFHSSEngine: Full-featured implementation with compliance monitoring
- Factory functions: Automatic selection based on requirements

Integration with existing code:
- Maintains backward compatibility with signal_tools.py FHSSEngine
- Provides upgrade path to enhanced features
- Supports both simple and complex use cases
"""

from __future__ import annotations

import logging
import math
import secrets
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

# Configure module logger
logger = logging.getLogger(__name__)

# Type aliases
FrequencyHz = float
TimeSeconds = float
IQSamples = npt.NDArray[np.complex64]
HopIndex = int


class FHSSBand(Enum):
    """FCC-defined frequency bands for FHSS operation per CFR 47 Part 15.247."""
    
    ISM_900_MHz = "902-928 MHz"
    ISM_2_4_GHz = "2400-2483.5 MHz" 
    ISM_5_8_GHz = "5725-5850 MHz"
    
    @property
    def frequency_range(self) -> Tuple[FrequencyHz, FrequencyHz]:
        """Get the frequency range for this band in Hz."""
        ranges = {
            self.ISM_900_MHz: (902e6, 928e6),
            self.ISM_2_4_GHz: (2400e6, 2483.5e6),
            self.ISM_5_8_GHz: (5725e6, 5850e6)
        }
        return ranges[self]
    
    @property
    def min_channels(self) -> int:
        """Get minimum required channels per FCC 15.247."""
        minimums = {
            self.ISM_900_MHz: 25,
            self.ISM_2_4_GHz: 75,
            self.ISM_5_8_GHz: 75
        }
        return minimums[self]


@dataclass(frozen=True)
class FHSSConfig:
    """
    Unified FHSS configuration supporting both simple and advanced use cases.
    
    For simple use cases, only center_freq_hz and channel_spacing_hz are needed.
    Advanced features can be enabled through additional parameters.
    """
    
    # Core parameters (required)
    center_freq_hz: FrequencyHz
    channel_spacing_hz: FrequencyHz = 1e6
    hop_count: int = 8
    
    # Simple interface parameters
    seed: Optional[int] = None
    hop_sequence: Optional[List[HopIndex]] = None
    
    # Advanced parameters (enhanced features)
    band: Optional[FHSSBand] = None
    validate_fcc_compliance: bool = False
    enable_advanced_features: bool = False
    
    def __post_init__(self) -> None:
        """Validate configuration and set defaults."""
        # Auto-detect band if not specified but compliance is requested
        if self.validate_fcc_compliance and self.band is None:
            self._auto_detect_band()
        
        # Validate basic parameters
        if self.center_freq_hz <= 0:
            raise ValueError("center_freq_hz must be positive")
        if self.channel_spacing_hz <= 0:
            raise ValueError("channel_spacing_hz must be positive")
        if self.hop_count < 1:
            raise ValueError("hop_count must be at least 1")
    
    def _auto_detect_band(self) -> None:
        """Auto-detect FCC band based on center frequency."""
        for band in FHSSBand:
            freq_min, freq_max = band.frequency_range
            if freq_min <= self.center_freq_hz <= freq_max:
                object.__setattr__(self, 'band', band)
                return
        
        warnings.warn(
            f"Center frequency {self.center_freq_hz/1e6:.1f} MHz "
            f"does not fall within any FCC ISM band"
        )


@dataclass
class HopFrame:
    """Represents a single frequency hop frame."""
    
    frequency_hz: FrequencyHz
    iq_samples: IQSamples
    duration_s: TimeSeconds
    hop_index: HopIndex
    chunk_data: bytes = field(repr=False)


class PulseShapeFilter:
    """Optimized pulse shaping filter with caching."""
    
    _filter_cache: Dict[Tuple[float, int, int], npt.NDArray[np.float32]] = {}
    
    @classmethod
    def raised_cosine_filter(
        cls,
        beta: float,
        span: int,
        sps: int
    ) -> npt.NDArray[np.float32]:
        """Generate raised cosine pulse shaping filter."""
        if not (0.0 <= beta <= 1.0):
            raise ValueError(f"Beta must be in [0, 1], got {beta}")
        
        cache_key = (beta, span, sps)
        if cache_key in cls._filter_cache:
            return cls._filter_cache[cache_key]
        
        # Generate filter (implementation from enhanced_fhss_engine.py)
        N = span * sps
        t = np.arange(-N//2, N//2 + 1, dtype=np.float32) / sps
        h = np.zeros_like(t)
        
        # Vectorized computation
        mask_zero = (t == 0.0)
        mask_special = (np.abs(t) == 1/(4*beta)) & (beta != 0.0)
        mask_normal = ~(mask_zero | mask_special)
        
        if np.any(mask_zero):
            h[mask_zero] = 1.0 - beta + (4*beta/math.pi)
        
        if np.any(mask_special) and beta != 0.0:
            h[mask_special] = (beta / math.sqrt(2)) * (
                ((1 + 2/math.pi) * np.sin(math.pi/(4*beta))) +
                ((1 - 2/math.pi) * np.cos(math.pi/(4*beta)))
            )
        
        if np.any(mask_normal):
            t_normal = t[mask_normal]
            num = (np.sin(math.pi * t_normal * (1 - beta)) + 
                   4*beta*t_normal*np.cos(math.pi * t_normal * (1 + beta)))
            den = math.pi * t_normal * (1 - (4*beta*t_normal)**2)
            h[mask_normal] = num / den
        
        # Normalize
        h = h / np.sqrt(np.sum(h**2))
        cls._filter_cache[cache_key] = h
        return h


class FHSSCore:
    """
    Core FHSS functionality shared between simple and enhanced implementations.
    
    This class contains the essential FHSS algorithms without compliance
    monitoring or advanced features.
    """
    
    def __init__(self, config: FHSSConfig) -> None:
        """Initialize core FHSS functionality."""
        self.config = config
        self._generate_channel_frequencies()
        self._initialize_hop_sequence()
    
    def _generate_channel_frequencies(self) -> None:
        """Generate frequency list for all hopping channels."""
        half_span = (self.config.hop_count - 1) / 2.0
        self.channel_frequencies = np.array([
            self.config.center_freq_hz + (i - half_span) * self.config.channel_spacing_hz
            for i in range(self.config.hop_count)
        ], dtype=np.float64)
    
    def _initialize_hop_sequence(self) -> None:
        """Initialize hop sequence generator."""
        if self.config.hop_sequence is not None:
            self.base_sequence = list(self.config.hop_sequence)
        else:
            self.rng = np.random.default_rng(self.config.seed)
            base = np.arange(self.config.hop_count)
            self.rng.shuffle(base)
            self.base_sequence = base.tolist()
    
    def generate_hop_sequence(self, length: int, seed: Optional[int] = None) -> List[HopIndex]:
        """Generate hop sequence of specified length."""
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = getattr(self, 'rng', np.random.default_rng())
        
        sequence = []
        while len(sequence) < length:
            block = self.base_sequence.copy()
            rng.shuffle(block)
            sequence.extend(block)
        
        return sequence[:length]
    
    def split_packet_into_chunks(self, packet: bytes, hop_count: int) -> List[bytes]:
        """Split packet into balanced chunks."""
        if hop_count <= 0:
            raise ValueError("hop_count must be positive")
        
        if not packet:
            return [b""] * hop_count
        
        packet_len = len(packet)
        chunk_size, remainder = divmod(packet_len, hop_count)
        
        chunks = []
        offset = 0
        for i in range(hop_count):
            take = chunk_size + (1 if i < remainder else 0)
            chunks.append(packet[offset:offset + take])
            offset += take
        
        return chunks
    
    @staticmethod
    def bytes_to_bits(data: bytes) -> npt.NDArray[np.int8]:
        """Convert bytes to bit array."""
        if not data:
            return np.array([], dtype=np.int8)
        return np.unpackbits(np.frombuffer(data, dtype=np.uint8)).astype(np.int8)
    
    @staticmethod
    def bits_to_bytes(bits: npt.NDArray[np.int8]) -> bytes:
        """Convert bit array back to bytes."""
        if bits.size == 0:
            return b""
        
        if bits.size % 8 != 0:
            padding = 8 - (bits.size % 8)
            bits = np.pad(bits, (0, padding), 'constant')
        
        return np.packbits(bits.astype(np.uint8)).tobytes()
    
    def bpsk_modulate(
        self,
        bits: npt.NDArray[np.int8],
        sample_rate: int,
        bitrate: int,
        rc_beta: float = 0.25,
        rc_span: int = 6
    ) -> IQSamples:
        """BPSK modulation with raised cosine pulse shaping."""
        sps = sample_rate // bitrate
        if sps < 2:
            raise ValueError("Sample rate too low for bitrate")
        
        if bits.size == 0:
            return np.array([], dtype=np.complex64)
        
        # BPSK mapping: 0 -> -1, 1 -> +1
        symbols = 2 * bits.astype(np.float32) - 1
        
        # Upsample
        upsampled = np.zeros(len(symbols) * sps, dtype=np.float32)
        upsampled[::sps] = symbols
        
        # Pulse shape
        rc_filter = PulseShapeFilter.raised_cosine_filter(rc_beta, rc_span, sps)
        shaped = np.convolve(upsampled, rc_filter, mode='same')
        
        # Convert to complex baseband
        return (shaped + 1j * np.zeros_like(shaped)).astype(np.complex64)


class SimpleFHSS:
    """
    Simple FHSS interface for basic frequency hopping operations.
    
    Provides an easy-to-use interface compatible with the original
    signal_tools.py FHSSEngine while leveraging the enhanced backend.
    
    Example:
        >>> fhss = SimpleFHSS(center_freq_hz=2.44e9, hops=8)
        >>> frames = fhss.prepare_transmit_frames(b"Hello World")
    """
    
    def __init__(
        self,
        center_freq_hz: FrequencyHz,
        channel_spacing_hz: FrequencyHz = 1e6,
        hops: int = 8,
        seed: Optional[int] = None,
        hop_sequence: Optional[List[int]] = None
    ) -> None:
        """
        Initialize simple FHSS engine.
        
        Args:
            center_freq_hz: Center frequency in Hz
            channel_spacing_hz: Channel spacing in Hz
            hops: Number of hop channels
            seed: Random seed for reproducible sequences
            hop_sequence: Optional explicit hop sequence
        """
        config = FHSSConfig(
            center_freq_hz=center_freq_hz,
            channel_spacing_hz=channel_spacing_hz,
            hop_count=hops,
            seed=seed,
            hop_sequence=hop_sequence
        )
        
        self._core = FHSSCore(config)
        self.center = center_freq_hz
        self.spacing = channel_spacing_hz
        self.hops = hops
        
        # Backward compatibility properties
        self.channels = self._core.channel_frequencies
        self.hop_sequence = self._core.base_sequence
    
    def generate_hop_map(self, length: int, seed: Optional[int] = None) -> List[int]:
        """Generate hop sequence (backward compatible)."""
        return self._core.generate_hop_sequence(length, seed)
    
    def split_packet_into_hops(self, packet: bytes, hop_count: int) -> List[bytes]:
        """Split packet into chunks (backward compatible)."""
        return self._core.split_packet_into_chunks(packet, hop_count)
    
    def bpsk_modulate_bits(
        self,
        bits: np.ndarray,
        sample_rate: int,
        bitrate: int,
        rc_beta: float = 0.25,
        rc_span: int = 6
    ) -> np.ndarray:
        """BPSK modulation (backward compatible)."""
        return self._core.bpsk_modulate(bits, sample_rate, bitrate, rc_beta, rc_span)
    
    def prepare_transmit_frames(
        self,
        packet: bytes,
        sample_rate: int = 2_000_000,
        bitrate: int = 1000,
        hop_duration_ms: Optional[float] = None,
        rc_beta: float = 0.25,
        rc_span: int = 6,
        seed: Optional[int] = None
    ) -> List[Tuple[float, np.ndarray, float]]:
        """
        Prepare transmit frames (backward compatible interface).
        
        Returns:
            List of (freq_hz, iq_samples, duration_s) tuples
        """
        # Use enhanced engine for the actual work
        enhanced = EnhancedFHSSEngine(self._core.config)
        frames = enhanced.prepare_transmit_frames(
            packet, sample_rate, bitrate, hop_duration_ms, 
            None, rc_beta, rc_span, seed
        )
        
        # Convert to backward compatible format
        return [(f.frequency_hz, f.iq_samples, f.duration_s) for f in frames]


class EnhancedFHSSEngine:
    """
    Enhanced FHSS engine with FCC compliance and advanced features.
    
    This is the full-featured implementation from the original enhanced_fhss_engine.py
    with additional integration capabilities.
    """
    
    def __init__(self, config: FHSSConfig) -> None:
        """Initialize enhanced FHSS engine."""
        self.config = config
        self._core = FHSSCore(config)
        
        # Compliance monitoring if enabled
        if config.validate_fcc_compliance:
            self._validate_fcc_compliance()
        
        logger.info(
            f"Initialized enhanced FHSS: {config.center_freq_hz/1e6:.1f} MHz, "
            f"{config.hop_count} channels, {config.channel_spacing_hz/1e3:.1f} kHz spacing"
        )
    
    def _validate_fcc_compliance(self) -> None:
        """Validate FCC compliance if band is specified."""
        if self.config.band is None:
            return
        
        # Check frequency range
        freq_min, freq_max = self.config.band.frequency_range
        total_span = (self.config.hop_count - 1) * self.config.channel_spacing_hz
        min_freq = self.config.center_freq_hz - total_span / 2
        max_freq = self.config.center_freq_hz + total_span / 2
        
        if min_freq < freq_min or max_freq > freq_max:
            raise ValueError(
                f"Frequency range [{min_freq/1e6:.1f}, {max_freq/1e6:.1f}] MHz "
                f"exceeds band limits {self.config.band.value}"
            )
        
        # Check minimum channel count
        if self.config.hop_count < self.config.band.min_channels:
            raise ValueError(
                f"Hop count {self.config.hop_count} below minimum "
                f"{self.config.band.min_channels} for band {self.config.band.value}"
            )
    
    @property
    def frequency_span_hz(self) -> FrequencyHz:
        """Get total frequency span of hop pattern."""
        return self._core.channel_frequencies[-1] - self._core.channel_frequencies[0]
    
    def prepare_transmit_frames(
        self,
        packet: bytes,
        sample_rate: int = 2_000_000,
        bitrate: int = 100_000,
        hop_duration_ms: Optional[float] = None,
        max_chunk_size: Optional[int] = None,
        rc_beta: float = 0.25,
        rc_span: int = 6,
        seed: Optional[int] = None
    ) -> List[HopFrame]:
        """
        Prepare comprehensive transmit frames.
        
        Returns:
            List of HopFrame objects with full metadata
        """
        if not packet:
            raise ValueError("Packet cannot be empty")
        
        # Determine hop count
        if max_chunk_size is not None:
            min_hops = math.ceil(len(packet) / max_chunk_size)
            hop_count = max(min_hops, min(self.config.hop_count, 16))
        else:
            target_chunk_size = max(16, min(len(packet) // 4, 256))
            hop_count = min(
                self.config.hop_count,
                max(1, math.ceil(len(packet) / target_chunk_size))
            )
        
        # Generate hop sequence and split packet
        hop_indices = self._core.generate_hop_sequence(hop_count, seed)
        chunks = self._core.split_packet_into_chunks(packet, hop_count)
        
        # Create frames
        frames = []
        for hop_idx, chunk in zip(hop_indices, chunks):
            bits = self._core.bytes_to_bits(chunk)
            if bits.size == 0:
                bits = np.zeros(8, dtype=np.int8)  # Minimal filler
            
            iq_samples = self._core.bpsk_modulate(
                bits, sample_rate, bitrate, rc_beta, rc_span
            )
            
            duration_s = (hop_duration_ms / 1000.0 if hop_duration_ms is not None 
                         else bits.size / bitrate)
            
            frame = HopFrame(
                frequency_hz=float(self._core.channel_frequencies[hop_idx]),
                iq_samples=iq_samples,
                duration_s=duration_s,
                hop_index=hop_idx,
                chunk_data=chunk
            )
            frames.append(frame)
        
        return frames
    
    def get_channel_utilization_stats(self, hop_sequence: List[HopIndex]) -> Dict[str, Any]:
        """Analyze channel utilization for compliance verification."""
        if not hop_sequence:
            return {"error": "Empty hop sequence"}
        
        usage_counts = np.bincount(hop_sequence, minlength=self.config.hop_count)
        total_hops = len(hop_sequence)
        usage_percentages = (usage_counts / total_hops) * 100
        
        return {
            "total_hops": total_hops,
            "channels_used": np.count_nonzero(usage_counts),
            "usage_counts": usage_counts.tolist(),
            "usage_percentages": usage_percentages.tolist(),
            "min_usage_percent": float(np.min(usage_percentages)),
            "max_usage_percent": float(np.max(usage_percentages)),
            "mean_usage_percent": float(np.mean(usage_percentages)),
            "std_usage_percent": float(np.std(usage_percentages)),
            "fcc_equal_usage_compliance": float(np.std(usage_percentages)) < 10.0
        }


# Factory functions for backward compatibility and ease of use

def create_fhss_engine(
    center_freq_hz: FrequencyHz,
    channel_spacing_hz: FrequencyHz = 1e6,
    hops: int = 8,
    simple: bool = True,
    **kwargs: Any
) -> Union[SimpleFHSS, EnhancedFHSSEngine]:
    """
    Factory function to create appropriate FHSS engine.
    
    Args:
        center_freq_hz: Center frequency in Hz
        channel_spacing_hz: Channel spacing in Hz  
        hops: Number of hop channels
        simple: If True, return SimpleFHSS; if False, return EnhancedFHSSEngine
        **kwargs: Additional configuration parameters
        
    Returns:
        FHSS engine instance
    """
    if simple:
        # Filter kwargs for SimpleFHSS
        simple_kwargs = {k: v for k, v in kwargs.items() 
                        if k in ['seed', 'hop_sequence']}
        return SimpleFHSS(center_freq_hz, channel_spacing_hz, hops, **simple_kwargs)
    else:
        # Create enhanced engine
        config = FHSSConfig(
            center_freq_hz=center_freq_hz,
            channel_spacing_hz=channel_spacing_hz,
            hop_count=hops,
            enable_advanced_features=True,
            **kwargs
        )
        return EnhancedFHSSEngine(config)


def create_fcc_compliant_fhss(
    band: FHSSBand,
    center_freq_hz: Optional[FrequencyHz] = None,
    **kwargs: Any
) -> EnhancedFHSSEngine:
    """
    Create FCC-compliant FHSS engine for specified band.
    
    Args:
        band: FCC band to operate in
        center_freq_hz: Center frequency (auto-calculated if None)
        **kwargs: Additional configuration parameters
        
    Returns:
        Enhanced FHSS engine with compliance validation
    """
    if center_freq_hz is None:
        # Use center of band
        freq_min, freq_max = band.frequency_range
        center_freq_hz = (freq_min + freq_max) / 2
    
    config = FHSSConfig(
        center_freq_hz=center_freq_hz,
        hop_count=band.min_channels,
        band=band,
        validate_fcc_compliance=True,
        enable_advanced_features=True,
        **kwargs
    )
    
    return EnhancedFHSSEngine(config)


# Backward compatibility alias (from signal_tools.py)
FHSSEngine = SimpleFHSS


# Example usage
if __name__ == "__main__":
    # Simple usage (backward compatible)
    simple_fhss = SimpleFHSS(center_freq_hz=2.44e9, hops=8)
    frames = simple_fhss.prepare_transmit_frames(b"Hello World")
    print(f"Simple FHSS: {len(frames)} frames generated")
    
    # Enhanced usage with FCC compliance
    enhanced_fhss = create_fcc_compliant_fhss(
        FHSSBand.ISM_2_4_GHz,
        channel_spacing_hz=1e6
    )
    enhanced_frames = enhanced_fhss.prepare_transmit_frames(b"Hello Enhanced World")
    print(f"Enhanced FHSS: {len(enhanced_frames)} frames generated")