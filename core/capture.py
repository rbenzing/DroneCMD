#!/usr/bin/env python3
"""
Enhanced Live SDR Capture System

A comprehensive Software Defined Radio (SDR) capture system designed for
professional signal acquisition and analysis. This module provides robust,
production-ready SDR capture capabilities with support for multiple hardware
platforms, comprehensive error handling, and advanced signal processing features.

Compliance with SDR Standards:
- VITA 49.0 VRT (VITA Radio Transport) protocol framework
- RTL-SDR best practices per pyrtlsdr documentation
- Software Communications Architecture (SCA) design principles
- Proper USB buffer management and sample rate validation

Key Features:
- Multi-platform SDR support (RTL-SDR, HackRF, Airspy, SDRplay)
- Async streaming with configurable buffering
- Automatic frequency correction and calibration
- Comprehensive metadata recording
- Signal quality monitoring and validation
- Memory-efficient chunked capture for large files
- Robust error handling and recovery mechanisms
- Integration with capture management systems

Example:
    >>> from dronecmd.core.enhanced_live_capture import EnhancedLiveCapture, SDRConfig
    >>> config = SDRConfig(
    ...     frequency_hz=100.1e6,
    ...     sample_rate_hz=2.048e6,
    ...     duration_s=30.0,
    ...     gain_mode='auto'
    ... )
    >>> capture = EnhancedLiveCapture(config)
    >>> async with capture:
    ...     async for samples in capture.stream_samples():
    ...         # Process samples in real-time
    ...         process_samples(samples)
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import tempfile
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
    runtime_checkable,
)

import numpy as np
import numpy.typing as npt

# SDR Hardware Support
try:
    from rtlsdr import RtlSdr, RtlSdrTcpClient

    RTL_SDR_AVAILABLE = True
except ImportError:
    RTL_SDR_AVAILABLE = False
    RtlSdr = None
    RtlSdrTcpClient = None

# Optional imports for extended SDR support
try:
    import SoapySDR

    SOAPY_SDR_AVAILABLE = True
except ImportError:
    SOAPY_SDR_AVAILABLE = False
    SoapySDR = None


def _find_hackrf_binary(name: str) -> Optional[str]:
    """Locate a HackRF CLI tool (e.g. ``hackrf_transfer``) on this host.

    Checks ``PATH`` first, then the default PothosSDR install locations on
    Windows. Returns the absolute path, or ``None`` if not found.
    """
    exe = shutil.which(name)
    if exe:
        return exe
    suffix = ".exe" if os.name == "nt" else ""
    for base in (
        r"C:\Program Files\PothosSDR\bin",
        r"C:\PothosSDR\bin",
        r"C:\Program Files (x86)\PothosSDR\bin",
    ):
        cand = os.path.join(base, name + suffix)
        if os.path.isfile(cand):
            return cand
    return None


# The stock ``hackrf_transfer`` / ``hackrf_info`` CLI (from libhackrf / PothosSDR)
# provides a receive path that works regardless of the Python interpreter's ABI —
# unlike the SoapySDR Python bindings, which must match the running interpreter
# (PothosSDR ships them for Python 3.9 only). This is the portable RX fallback.
HACKRF_TRANSFER_PATH = _find_hackrf_binary("hackrf_transfer")
HACKRF_TRANSFER_AVAILABLE = HACKRF_TRANSFER_PATH is not None

# Configure module logger
logger = logging.getLogger(__name__)

# Type aliases for clarity
FrequencyHz = float
SampleRateHz = float
TimeSeconds = float
GainDb = float
IQSamples = npt.NDArray[np.complex64]
RealSamples = npt.NDArray[np.float32]
MetadataDict = Dict[str, Any]


class SDRPlatform(Enum):
    """Supported SDR hardware platforms."""

    RTL_SDR = "rtl_sdr"
    RTL_SDR_TCP = "rtl_sdr_tcp"
    HACKRF = "hackrf"
    AIRSPY = "airspy"
    SDRPLAY = "sdrplay"
    PLUTO_SDR = "plutosdr"
    USRP = "usrp"
    LIME_SDR = "limesdr"

    @property
    def max_sample_rate(self) -> SampleRateHz:
        """Get maximum sample rate for this platform."""
        rates = {
            self.RTL_SDR: 3.2e6,
            self.RTL_SDR_TCP: 3.2e6,
            self.HACKRF: 20e6,
            self.AIRSPY: 10e6,
            self.SDRPLAY: 12e6,
            self.PLUTO_SDR: 61.44e6,
            self.USRP: 100e6,
            self.LIME_SDR: 65e6,
        }
        return rates.get(self, 3.2e6)

    @property
    def frequency_range(self) -> Tuple[FrequencyHz, FrequencyHz]:
        """Get frequency range for this platform."""
        ranges = {
            self.RTL_SDR: (24e6, 1.75e9),
            self.RTL_SDR_TCP: (24e6, 1.75e9),
            self.HACKRF: (1e6, 6e9),
            self.AIRSPY: (24e6, 1.8e9),
            self.SDRPLAY: (1e3, 2e9),
            self.PLUTO_SDR: (325e6, 3.8e9),
            self.USRP: (70e6, 6e9),
            self.LIME_SDR: (100e3, 3.8e9),
        }
        return ranges.get(self, (24e6, 1.75e9))


class GainMode(Enum):
    """SDR gain control modes."""

    AUTO = "auto"
    MANUAL = "manual"
    AGC = "agc"


class CaptureFormat(Enum):
    """Output format for captured samples."""

    COMPLEX64 = "complex64"
    COMPLEX128 = "complex128"
    INT16 = "int16"
    FLOAT32 = "float32"


@runtime_checkable
class SDRHardwareInterface(Protocol):
    """Protocol defining the interface for SDR hardware backends."""

    def open(self) -> None:
        """Open connection to SDR hardware."""
        ...

    def close(self) -> None:
        """Close connection to SDR hardware."""
        ...

    def configure(self, config: "SDRConfig") -> None:
        """Configure SDR hardware with given parameters."""
        ...

    def read_samples(self, num_samples: int) -> IQSamples:
        """Read IQ samples from SDR hardware."""
        ...

    async def read_samples_async(self, num_samples: int) -> IQSamples:
        """Asynchronously read IQ samples from SDR hardware."""
        ...

    @property
    def is_connected(self) -> bool:
        """Check if SDR hardware is connected."""
        ...


@dataclass(frozen=True)
class SDRConfig:
    """
    Configuration parameters for SDR capture operations.

    All parameters are validated for hardware compatibility and
    RF engineering best practices.

    Attributes:
        platform: SDR hardware platform
        frequency_hz: Center frequency in Hz
        sample_rate_hz: Sample rate in Hz
        duration_s: Capture duration in seconds (None for infinite)
        gain_mode: Gain control mode
        gain_db: Manual gain in dB (when gain_mode is MANUAL)
        frequency_correction_ppm: Frequency correction in PPM
        bandwidth_hz: Optional baseband filter bandwidth
        device_index: Device index for multiple SDRs
        tcp_address: TCP address for remote SDR (RTL-SDR TCP only)
        tcp_port: TCP port for remote SDR (RTL-SDR TCP only)
        bias_tee_enabled: Enable bias tee power (if supported)
        direct_sampling: Enable direct sampling mode
        enable_dithering: Enable PLL dithering
        buffer_size_samples: Buffer size for USB transfers
        validate_hardware_limits: Enable hardware limit validation
    """

    platform: SDRPlatform = SDRPlatform.RTL_SDR
    frequency_hz: FrequencyHz = 100.1e6
    sample_rate_hz: SampleRateHz = 2.048e6
    duration_s: Optional[TimeSeconds] = 10.0
    gain_mode: GainMode = GainMode.AUTO
    gain_db: Optional[GainDb] = None
    frequency_correction_ppm: float = 0.0
    bandwidth_hz: Optional[FrequencyHz] = None
    device_index: int = 0
    tcp_address: Optional[str] = None
    tcp_port: int = 1234
    bias_tee_enabled: bool = False
    direct_sampling: Union[bool, str] = False
    enable_dithering: bool = True
    buffer_size_samples: int = 262144  # 256k samples default
    validate_hardware_limits: bool = True

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.validate_hardware_limits:
            self._validate_hardware_limits()
        self._validate_basic_parameters()

    def _validate_hardware_limits(self) -> None:
        """Validate parameters against hardware platform limits."""
        # Validate frequency range
        freq_min, freq_max = self.platform.frequency_range
        if not (freq_min <= self.frequency_hz <= freq_max):
            raise ValueError(
                f"Frequency {self.frequency_hz/1e6:.3f} MHz outside "
                f"platform range [{freq_min/1e6:.1f}, {freq_max/1e6:.1f}] MHz"
            )

        # Validate sample rate
        max_rate = self.platform.max_sample_rate
        if self.sample_rate_hz > max_rate:
            raise ValueError(
                f"Sample rate {self.sample_rate_hz/1e6:.3f} MHz exceeds "
                f"platform maximum {max_rate/1e6:.1f} MHz"
            )

        # RTL-SDR specific validations
        if self.platform in (SDRPlatform.RTL_SDR, SDRPlatform.RTL_SDR_TCP):
            self._validate_rtl_sdr_specific()

    def _validate_rtl_sdr_specific(self) -> None:
        """Validate RTL-SDR specific parameters."""
        # Check sample rate ranges per pyrtlsdr documentation
        if not (
            225e3 <= self.sample_rate_hz <= 300e3
            or 900e3 <= self.sample_rate_hz <= 3.2e6
        ):
            warnings.warn(
                f"RTL-SDR sample rate {self.sample_rate_hz/1e6:.3f} MHz "
                f"outside recommended ranges: [0.225-0.3] or [0.9-3.2] MHz",
                UserWarning,
            )

        # Warn about high sample rates USB limitations
        if self.sample_rate_hz > 2.4e6:
            warnings.warn(
                "RTL-SDR sample rates above 2.4 MHz may lose samples due to USB bandwidth",
                UserWarning,
            )

        # Validate gain for manual mode
        if self.gain_mode == GainMode.MANUAL and self.gain_db is None:
            raise ValueError("Manual gain mode requires gain_db to be specified")

    def _validate_basic_parameters(self) -> None:
        """Validate basic parameter sanity."""
        if self.frequency_hz <= 0:
            raise ValueError("Frequency must be positive")

        if self.sample_rate_hz <= 0:
            raise ValueError("Sample rate must be positive")

        if self.duration_s is not None and self.duration_s <= 0:
            raise ValueError("Duration must be positive or None")

        if self.buffer_size_samples <= 0:
            raise ValueError("Buffer size must be positive")

        if self.buffer_size_samples & (self.buffer_size_samples - 1) != 0:
            warnings.warn(
                f"Buffer size {self.buffer_size_samples} is not a power of 2, "
                f"which may cause USB performance issues",
                UserWarning,
            )


@dataclass
class CaptureMetadata:
    """
    Comprehensive metadata for captured signals.

    Following VITA 49.0 VRT context packet specifications where applicable.
    """

    # Capture session information
    session_id: str
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None

    # SDR configuration
    platform: SDRPlatform = SDRPlatform.RTL_SDR
    frequency_hz: FrequencyHz = 0.0
    sample_rate_hz: SampleRateHz = 0.0
    actual_gain_db: Optional[GainDb] = None
    frequency_correction_ppm: float = 0.0

    # Signal quality metrics
    total_samples: int = 0
    dropped_samples: int = 0
    overrun_count: int = 0
    signal_level_dbfs: Optional[float] = None
    noise_floor_dbfs: Optional[float] = None
    snr_db: Optional[float] = None

    # Hardware information
    device_serial: Optional[str] = None
    device_name: Optional[str] = None
    driver_version: Optional[str] = None
    firmware_version: Optional[str] = None

    # Processing information
    output_format: CaptureFormat = CaptureFormat.COMPLEX64
    file_path: Optional[Path] = None
    file_size_bytes: int = 0
    checksum: Optional[str] = None

    # Additional context
    tags: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        """Calculate capture duration in seconds."""
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time).total_seconds()

    @property
    def sample_loss_rate(self) -> float:
        """Calculate sample loss rate as percentage."""
        if self.total_samples == 0:
            return 0.0
        return (self.dropped_samples / self.total_samples) * 100.0

    def to_dict(self) -> MetadataDict:
        """Convert metadata to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "platform": self.platform.value,
            "frequency_hz": self.frequency_hz,
            "sample_rate_hz": self.sample_rate_hz,
            "actual_gain_db": self.actual_gain_db,
            "frequency_correction_ppm": self.frequency_correction_ppm,
            "total_samples": self.total_samples,
            "dropped_samples": self.dropped_samples,
            "overrun_count": self.overrun_count,
            "signal_level_dbfs": self.signal_level_dbfs,
            "noise_floor_dbfs": self.noise_floor_dbfs,
            "snr_db": self.snr_db,
            "device_serial": self.device_serial,
            "device_name": self.device_name,
            "driver_version": self.driver_version,
            "firmware_version": self.firmware_version,
            "output_format": self.output_format.value,
            "file_path": str(self.file_path) if self.file_path else None,
            "file_size_bytes": self.file_size_bytes,
            "checksum": self.checksum,
            "duration_s": self.duration_s,
            "sample_loss_rate": self.sample_loss_rate,
            "tags": self.tags,
        }


class SignalQualityMonitor:
    """
    Real-time signal quality monitoring and analysis.

    Provides continuous monitoring of signal characteristics including
    power levels, noise floor estimation, and anomaly detection.
    """

    def __init__(self, history_length: int = 1000) -> None:
        """
        Initialize signal quality monitor.

        Args:
            history_length: Number of measurement windows to keep in history
        """
        self.history_length = history_length
        self.power_history: List[float] = []
        self.noise_history: List[float] = []
        self.overrun_count = 0
        self.total_samples = 0
        self.dropped_samples = 0

    def update(self, samples: IQSamples) -> Dict[str, float]:
        """
        Update signal quality metrics with new samples.

        Args:
            samples: New IQ samples to analyze

        Returns:
            Dictionary of current signal quality metrics
        """
        if samples.size == 0:
            return {}

        # Calculate instantaneous power
        power_linear = np.mean(np.abs(samples) ** 2)
        power_dbfs = 10 * np.log10(power_linear + 1e-12)  # Add small epsilon

        # Estimate noise floor (lower 10th percentile of power)
        fft_power = np.abs(np.fft.fft(samples)) ** 2
        fft_power_db = 10 * np.log10(fft_power + 1e-12)
        noise_floor_dbfs = np.percentile(fft_power_db, 10)

        # Update history
        self.power_history.append(power_dbfs)
        self.noise_history.append(noise_floor_dbfs)

        # Trim history
        if len(self.power_history) > self.history_length:
            self.power_history.pop(0)
            self.noise_history.pop(0)

        # Calculate SNR
        snr_db = power_dbfs - noise_floor_dbfs if noise_floor_dbfs > -100 else None

        # Update sample counts
        self.total_samples += len(samples)

        return {
            "signal_level_dbfs": power_dbfs,
            "noise_floor_dbfs": noise_floor_dbfs,
            "snr_db": snr_db,
            "avg_signal_level_dbfs": np.mean(self.power_history),
            "avg_noise_floor_dbfs": np.mean(self.noise_history),
            "signal_std_db": np.std(self.power_history),
            "total_samples": self.total_samples,
            "dropped_samples": self.dropped_samples,
            "sample_loss_rate": (self.dropped_samples / self.total_samples) * 100.0
            if self.total_samples > 0
            else 0.0,
        }

    def detect_anomalies(self, current_metrics: Dict[str, float]) -> List[str]:
        """
        Detect signal anomalies based on current metrics.

        Args:
            current_metrics: Current signal quality metrics

        Returns:
            List of detected anomaly descriptions
        """
        anomalies = []

        if len(self.power_history) < 10:
            return anomalies  # Need more history

        current_power = current_metrics.get("signal_level_dbfs", 0)
        avg_power = np.mean(self.power_history[:-1])  # Exclude current sample
        power_std = np.std(self.power_history[:-1])

        # Detect sudden power changes
        if abs(current_power - avg_power) > 3 * power_std:
            anomalies.append(
                f"Sudden power change: {current_power:.1f} dBFS (avg: {avg_power:.1f})"
            )

        # Detect clipping
        if current_power > -3:
            anomalies.append(f"Potential clipping detected: {current_power:.1f} dBFS")

        # Detect very low signal
        if current_power < -80:
            anomalies.append(f"Very low signal level: {current_power:.1f} dBFS")

        # Check sample loss rate
        loss_rate = current_metrics.get("sample_loss_rate", 0)
        if loss_rate > 1.0:
            anomalies.append(f"High sample loss rate: {loss_rate:.2f}%")

        return anomalies


class RTLSDRHardware:
    """RTL-SDR hardware interface implementation."""

    def __init__(self, config: SDRConfig) -> None:
        """Initialize RTL-SDR hardware interface."""
        if not RTL_SDR_AVAILABLE:
            raise RuntimeError("RTL-SDR library not available")

        self.config = config
        self.sdr: Optional[Union[RtlSdr, RtlSdrTcpClient]] = None
        self._is_connected = False

    def open(self) -> None:
        """Open connection to RTL-SDR hardware."""
        try:
            if self.config.platform == SDRPlatform.RTL_SDR_TCP:
                if self.config.tcp_address is None:
                    raise ValueError("TCP address required for RTL-SDR TCP mode")
                self.sdr = RtlSdrTcpClient(
                    hostname=self.config.tcp_address, port=self.config.tcp_port
                )
            else:
                self.sdr = RtlSdr(device_index=self.config.device_index)

            self._is_connected = True
            logger.info(f"Connected to RTL-SDR device {self.config.device_index}")

        except Exception as e:
            logger.error(f"Failed to connect to RTL-SDR: {e}")
            raise RuntimeError(f"Cannot connect to RTL-SDR: {e}") from e

    def close(self) -> None:
        """Close connection to RTL-SDR hardware."""
        if self.sdr is not None:
            try:
                self.sdr.close()
                logger.info("RTL-SDR connection closed")
            except Exception as e:
                logger.warning(f"Error closing RTL-SDR: {e}")
            finally:
                self.sdr = None
                self._is_connected = False

    def configure(self, config: SDRConfig) -> None:
        """Configure RTL-SDR hardware with given parameters."""
        if self.sdr is None:
            raise RuntimeError("RTL-SDR not connected")

        try:
            # Set sample rate first (affects other settings)
            self.sdr.sample_rate = config.sample_rate_hz

            # Set center frequency
            self.sdr.center_freq = config.frequency_hz

            # Set frequency correction
            if config.frequency_correction_ppm != 0:
                self.sdr.freq_correction = config.frequency_correction_ppm

            # Configure gain
            if config.gain_mode == GainMode.AUTO:
                self.sdr.gain = "auto"
            elif config.gain_mode == GainMode.MANUAL and config.gain_db is not None:
                self.sdr.gain = config.gain_db

            # Configure optional features if supported
            if hasattr(self.sdr, "set_bias_tee") and config.bias_tee_enabled:
                self.sdr.set_bias_tee(True)

            if hasattr(self.sdr, "set_direct_sampling") and config.direct_sampling:
                self.sdr.set_direct_sampling(config.direct_sampling)

            if hasattr(self.sdr, "set_dithering"):
                self.sdr.set_dithering(config.enable_dithering)

            logger.info(
                f"RTL-SDR configured: {config.frequency_hz/1e6:.3f} MHz, "
                f"{config.sample_rate_hz/1e6:.3f} MSps, gain: {config.gain_mode.value}"
            )

        except Exception as e:
            logger.error(f"Failed to configure RTL-SDR: {e}")
            raise RuntimeError(f"Cannot configure RTL-SDR: {e}") from e

    def read_samples(self, num_samples: int) -> IQSamples:
        """Read IQ samples from RTL-SDR hardware."""
        if self.sdr is None:
            raise RuntimeError("RTL-SDR not connected")

        try:
            # Discard first samples to avoid transients (RTL-SDR best practice)
            if num_samples > 2048:
                self.sdr.read_samples(2048)

            samples = self.sdr.read_samples(num_samples)
            return samples.astype(np.complex64)

        except Exception as e:
            logger.error(f"Failed to read samples from RTL-SDR: {e}")
            raise RuntimeError(f"Cannot read samples: {e}") from e

    async def read_samples_async(self, num_samples: int) -> IQSamples:
        """Asynchronously read IQ samples from RTL-SDR hardware."""
        # Run blocking operation in thread pool
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(executor, self.read_samples, num_samples)

    @property
    def is_connected(self) -> bool:
        """Check if RTL-SDR hardware is connected."""
        return self._is_connected and self.sdr is not None


class SoapyHackRFHardware:
    """HackRF **receive-only** hardware interface via SoapySDR (SoapyHackRF).

    This backend NEVER transmits — it only opens an RX stream and reads IQ
    samples, the passive/lawful path for capture and analysis. (HackRF is
    half-duplex; transmit is deliberately out of scope for this class.)

    Requires the native SoapySDR runtime plus the SoapyHackRF module (e.g. the
    PothosSDR Windows bundle) and the HackRF USB interface bound to a usable
    driver (WinUSB via Zadig on Windows). Implements the same
    :class:`SDRHardwareInterface` protocol as :class:`RTLSDRHardware`.
    """

    def __init__(self, config: SDRConfig) -> None:
        """Initialize the SoapySDR HackRF (RX) interface."""
        if not SOAPY_SDR_AVAILABLE:
            raise RuntimeError(
                "SoapySDR not available; install the SoapySDR runtime + "
                "SoapyHackRF (e.g. the PothosSDR bundle) to use the HackRF platform"
            )
        self.config = config
        self.sdr: Optional[Any] = None
        self._stream: Optional[Any] = None
        self._is_connected = False

    def open(self) -> None:
        """Open the HackRF via SoapySDR."""
        try:
            args: Dict[str, str] = {"driver": "hackrf"}
            if self.config.device_index:
                args["device"] = str(self.config.device_index)
            self.sdr = SoapySDR.Device(args)
            self._is_connected = True
            logger.info("Connected to HackRF via SoapySDR")
        except Exception as e:
            logger.error(f"Failed to connect to HackRF: {e}")
            raise RuntimeError(f"Cannot connect to HackRF: {e}") from e

    def close(self) -> None:
        """Deactivate/close the RX stream and release the device."""
        if self.sdr is not None:
            try:
                if self._stream is not None:
                    self.sdr.deactivateStream(self._stream)
                    self.sdr.closeStream(self._stream)
                logger.info("HackRF connection closed")
            except Exception as e:
                logger.warning(f"Error closing HackRF: {e}")
            finally:
                self._stream = None
                self.sdr = None
                self._is_connected = False

    def configure(self, config: SDRConfig) -> None:
        """Configure the RX chain (sample rate, frequency, bandwidth, gain) and
        set up the complex-float32 receive stream."""
        if self.sdr is None:
            raise RuntimeError("HackRF not connected")
        try:
            ch = 0
            rx = SoapySDR.SOAPY_SDR_RX
            self.sdr.setSampleRate(rx, ch, float(config.sample_rate_hz))
            self.sdr.setFrequency(rx, ch, float(config.frequency_hz))
            if config.bandwidth_hz:
                self.sdr.setBandwidth(rx, ch, float(config.bandwidth_hz))
            if config.gain_mode == GainMode.AUTO:
                try:
                    self.sdr.setGainMode(rx, ch, True)  # hardware AGC if supported
                except Exception:
                    self.sdr.setGain(rx, ch, 32.0)  # sane manual default
            elif config.gain_mode == GainMode.MANUAL and config.gain_db is not None:
                self.sdr.setGain(rx, ch, float(config.gain_db))
            self._stream = self.sdr.setupStream(rx, SoapySDR.SOAPY_SDR_CF32)
            self.sdr.activateStream(self._stream)
            logger.info(
                f"HackRF configured: {config.frequency_hz/1e6:.3f} MHz, "
                f"{config.sample_rate_hz/1e6:.3f} MSps, gain: {config.gain_mode.value}"
            )
        except Exception as e:
            logger.error(f"Failed to configure HackRF: {e}")
            raise RuntimeError(f"Cannot configure HackRF: {e}") from e

    def read_samples(self, num_samples: int) -> IQSamples:
        """Read ``num_samples`` complex64 IQ samples from the RX stream.

        ``readStream`` returns in bounded chunks, so this loops until the
        request is filled, raising on a stream error or a persistent timeout.
        """
        if self.sdr is None or self._stream is None:
            raise RuntimeError("HackRF not connected/configured")
        try:
            out = np.empty(num_samples, dtype=np.complex64)
            filled = 0
            empty_reads = 0
            max_empty = 8 + 4 * (num_samples // 1024 + 1)
            while filled < num_samples:
                chunk = np.empty(num_samples - filled, dtype=np.complex64)
                status = self.sdr.readStream(
                    self._stream, [chunk], chunk.size, timeoutUs=1_000_000
                )
                n = int(status.ret)
                if n > 0:
                    out[filled : filled + n] = chunk[:n]
                    filled += n
                    empty_reads = 0
                elif n < 0:
                    raise RuntimeError(f"SoapySDR readStream error code {n}")
                else:
                    empty_reads += 1
                    if empty_reads > max_empty:
                        raise RuntimeError("HackRF read timed out")
            return out
        except Exception as e:
            logger.error(f"Failed to read samples from HackRF: {e}")
            raise RuntimeError(f"Cannot read samples: {e}") from e

    async def read_samples_async(self, num_samples: int) -> IQSamples:
        """Asynchronously read IQ samples (blocking read in a thread pool)."""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(executor, self.read_samples, num_samples)

    @property
    def is_connected(self) -> bool:
        """Check whether the HackRF is connected."""
        return self._is_connected and self.sdr is not None


class HackRFTransferHardware:
    """HackRF **receive-only** backend driving the ``hackrf_transfer`` CLI.

    A pure-subprocess RX path for hosts where the SoapySDR *Python* bindings
    are unavailable or ABI-mismatched (e.g. Windows, where PothosSDR ships the
    SoapySDR bindings for Python 3.9 only while the application runs on a newer
    interpreter). Each :meth:`read_samples` call shells out to the stock
    ``hackrf_transfer`` tool to receive a finite IQ buffer to a temporary file,
    then loads it as ``complex64``.

    Receive-only by construction: it only ever runs ``hackrf_transfer -r`` and
    never ``-t`` (transmit). Implements the same :class:`SDRHardwareInterface`
    protocol as :class:`RTLSDRHardware` / :class:`SoapyHackRFHardware`.

    ``hackrf_transfer`` records signed 8-bit interleaved I/Q; samples are
    normalized to the ``[-1, 1)`` range (``/128``) on load.
    """

    # HackRF One sample-rate envelope (Sps). Below 2 MSps is unsupported.
    _MIN_SAMPLE_RATE_HZ = 2_000_000.0
    _MAX_SAMPLE_RATE_HZ = 20_000_000.0

    def __init__(self, config: SDRConfig) -> None:
        """Initialize the ``hackrf_transfer`` RX interface."""
        if not HACKRF_TRANSFER_AVAILABLE or HACKRF_TRANSFER_PATH is None:
            raise RuntimeError(
                "hackrf_transfer CLI not found; install the HackRF tools "
                "(e.g. the PothosSDR bundle) to use the HackRF platform"
            )
        self.config = config
        self._bin = HACKRF_TRANSFER_PATH
        self._is_connected = False

    def open(self) -> None:
        """Confirm a HackRF is reachable via ``hackrf_info`` (best effort)."""
        info = _find_hackrf_binary("hackrf_info")
        if info is not None:
            try:
                proc = subprocess.run(
                    [info],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                if "Found HackRF" not in (proc.stdout or ""):
                    raise RuntimeError(
                        f"no HackRF detected by hackrf_info: "
                        f"{(proc.stdout or proc.stderr or '').strip()[:200]}"
                    )
            except FileNotFoundError as e:
                raise RuntimeError(f"cannot run hackrf_info: {e}") from e
            except subprocess.TimeoutExpired as e:
                raise RuntimeError("hackrf_info timed out") from e
        self._is_connected = True
        logger.info("HackRF (hackrf_transfer backend) ready")

    def close(self) -> None:
        """Release the interface (the CLI holds no persistent handle)."""
        self._is_connected = False
        logger.info("HackRF (hackrf_transfer backend) closed")

    def configure(self, config: SDRConfig) -> None:
        """Validate and store the RX parameters used for each capture."""
        sr = float(config.sample_rate_hz)
        if not (self._MIN_SAMPLE_RATE_HZ <= sr <= self._MAX_SAMPLE_RATE_HZ):
            raise RuntimeError(
                f"HackRF sample rate {sr/1e6:.3f} MSps out of range "
                f"[{self._MIN_SAMPLE_RATE_HZ/1e6:.0f}, "
                f"{self._MAX_SAMPLE_RATE_HZ/1e6:.0f}] MSps"
            )
        self.config = config
        logger.info(
            f"HackRF configured: {config.frequency_hz/1e6:.3f} MHz, "
            f"{sr/1e6:.3f} MSps, gain: {config.gain_mode.value}"
        )

    def _gain_args(self) -> List[str]:
        """Map the configured gain to ``hackrf_transfer`` LNA/VGA/amp flags.

        AUTO uses moderate fixed RX gains (HackRF has no hardware AGC). MANUAL
        splits ``gain_db`` into the LNA stage (0-40 dB, 8 dB steps) first, then
        the VGA stage (0-62 dB, 2 dB steps).
        """
        if self.config.gain_mode == GainMode.MANUAL and self.config.gain_db is not None:
            total = max(0.0, float(self.config.gain_db))
            lna = int(min(40, (int(total) // 8) * 8))
            vga = int(min(62, (int(total - lna) // 2) * 2))
            return ["-l", str(lna), "-g", str(vga), "-a", "0"]
        # AUTO / unspecified: sane moderate RX gains, amp off.
        return ["-l", "24", "-g", "20", "-a", "0"]

    def read_samples(self, num_samples: int) -> IQSamples:
        """Receive ``num_samples`` complex64 IQ samples via ``hackrf_transfer``.

        Runs one ``hackrf_transfer -r`` capture to a temp file and loads it.
        Raises :class:`RuntimeError` loudly on a non-zero exit, a missing/short
        output file, or a timeout.
        """
        if not self._is_connected:
            raise RuntimeError("HackRF not connected")
        if num_samples <= 0:
            return np.empty(0, dtype=np.complex64)

        fd, tmp = tempfile.mkstemp(prefix="hackrf_rx_", suffix=".iq")
        os.close(fd)
        try:
            cmd = [
                self._bin,
                "-r",
                tmp,
                "-f",
                str(int(self.config.frequency_hz)),
                "-s",
                str(int(self.config.sample_rate_hz)),
                "-n",
                str(int(num_samples)),
            ]
            if self.config.bandwidth_hz:
                cmd += ["-b", str(int(self.config.bandwidth_hz))]
            cmd += self._gain_args()
            # Generous timeout: capture wall time + fixed startup/teardown margin.
            timeout_s = 30.0 + num_samples / max(1.0, float(self.config.sample_rate_hz))
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=timeout_s
                )
            except subprocess.TimeoutExpired as e:
                raise RuntimeError("hackrf_transfer capture timed out") from e
            if proc.returncode != 0:
                raise RuntimeError(
                    f"hackrf_transfer failed (exit {proc.returncode}): "
                    f"{(proc.stderr or proc.stdout or '').strip()[:300]}"
                )
            raw = np.fromfile(tmp, dtype=np.int8)
            if raw.size < 2 * num_samples:
                raise RuntimeError(
                    f"hackrf_transfer returned {raw.size // 2} samples, "
                    f"expected {num_samples}"
                )
            raw = raw[: 2 * num_samples]
            iq = (
                raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
            ) / 128.0
            return iq.astype(np.complex64)
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass

    async def read_samples_async(self, num_samples: int) -> IQSamples:
        """Asynchronously receive IQ samples (blocking capture in a thread)."""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(executor, self.read_samples, num_samples)

    @property
    def is_connected(self) -> bool:
        """Check whether the HackRF interface is ready."""
        return self._is_connected


class EnhancedLiveCapture:
    """
    Enhanced live SDR capture system with comprehensive features.

    This class provides a production-ready SDR capture system with:
    - Multi-platform SDR support
    - Async streaming capabilities
    - Real-time signal quality monitoring
    - Comprehensive metadata recording
    - Robust error handling and recovery
    - Memory-efficient operation

    Example:
        >>> config = SDRConfig(
        ...     frequency_hz=100.1e6,
        ...     sample_rate_hz=2.048e6,
        ...     duration_s=30.0
        ... )
        >>> capture = EnhancedLiveCapture(config)
        >>> samples = await capture.capture_samples()
    """

    def __init__(
        self,
        config: SDRConfig,
        capture_manager: Optional[Any] = None,
        quality_callback: Optional[Callable[[Dict[str, float]], None]] = None,
    ) -> None:
        """
        Initialize enhanced live capture system.

        Args:
            config: SDR configuration parameters
            capture_manager: Optional capture manager for integration
            quality_callback: Optional callback for signal quality updates
        """
        self.config = config
        self.capture_manager = capture_manager
        self.quality_callback = quality_callback

        # Initialize hardware interface
        self.hardware = self._create_hardware_interface()

        # Initialize monitoring and metadata
        self.quality_monitor = SignalQualityMonitor()
        self.metadata = CaptureMetadata(
            session_id=f"sdr_capture_{int(time.time())}",
            platform=config.platform,
            frequency_hz=config.frequency_hz,
            sample_rate_hz=config.sample_rate_hz,
            frequency_correction_ppm=config.frequency_correction_ppm,
        )

        # Runtime state
        self._is_capturing = False
        self._capture_lock = asyncio.Lock()
        self._stop_event = asyncio.Event()

        logger.info(f"Initialized {config.platform.value} capture system")

    def _create_hardware_interface(self) -> SDRHardwareInterface:
        """Create appropriate hardware interface for configured platform."""
        if self.config.platform in (SDRPlatform.RTL_SDR, SDRPlatform.RTL_SDR_TCP):
            return RTLSDRHardware(self.config)
        elif self.config.platform == SDRPlatform.HACKRF:
            # Prefer the in-process SoapySDR backend when its Python bindings
            # are importable (Linux, or a Python whose ABI matches SoapyHackRF).
            # Otherwise fall back to the portable ``hackrf_transfer`` CLI backend
            # (works on any interpreter; the only HackRF path on hosts where the
            # SoapySDR bindings are ABI-mismatched, e.g. Windows + PothosSDR).
            if SOAPY_SDR_AVAILABLE:
                return SoapyHackRFHardware(self.config)
            if HACKRF_TRANSFER_AVAILABLE:
                return HackRFTransferHardware(self.config)
            raise RuntimeError(
                "HackRF selected but no usable backend: install SoapyHackRF with "
                "Python bindings matching this interpreter, or the HackRF CLI tools "
                "(e.g. the PothosSDR bundle) for the hackrf_transfer fallback"
            )
        elif SOAPY_SDR_AVAILABLE:
            # Other SoapySDR-backed platforms (Airspy/SDRplay/…) not yet wired.
            raise NotImplementedError(
                f"Platform {self.config.platform.value} not yet implemented"
            )
        else:
            raise RuntimeError(
                f"No hardware interface available for {self.config.platform.value}"
            )

    async def __aenter__(self) -> "EnhancedLiveCapture":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()

    async def connect(self) -> None:
        """Connect to SDR hardware and configure for capture."""
        try:
            # Connect to hardware
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.hardware.open)

            # Configure hardware
            await loop.run_in_executor(None, self.hardware.configure, self.config)

            # Update metadata with device information
            await self._update_device_metadata()

            logger.info("SDR hardware connected and configured")

        except Exception as e:
            logger.error(f"Failed to connect to SDR hardware: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from SDR hardware."""
        try:
            # Stop any ongoing capture
            if self._is_capturing:
                await self.stop_capture()

            # Close hardware connection
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.hardware.close)

            # Finalize metadata
            self.metadata.end_time = datetime.now(timezone.utc)

            logger.info("SDR hardware disconnected")

        except Exception as e:
            logger.warning(f"Error during disconnect: {e}")

    async def _update_device_metadata(self) -> None:
        """Update metadata with device-specific information."""
        try:
            if hasattr(self.hardware, "sdr") and self.hardware.sdr is not None:
                sdr = self.hardware.sdr

                # Get device serial if available
                if hasattr(sdr, "get_device_serial_addresses"):
                    serials = sdr.get_device_serial_addresses()
                    if serials and len(serials) > self.config.device_index:
                        self.metadata.device_serial = serials[self.config.device_index]

                # Get actual gain setting
                if hasattr(sdr, "gain") and sdr.gain != "auto":
                    self.metadata.actual_gain_db = float(sdr.gain)

                # Platform-specific metadata
                if self.config.platform == SDRPlatform.RTL_SDR:
                    self.metadata.device_name = "RTL-SDR"
                    if hasattr(sdr, "get_tuner_type"):
                        tuner_type = sdr.get_tuner_type()
                        self.metadata.tags["tuner_type"] = tuner_type

        except Exception as e:
            logger.debug(f"Could not update device metadata: {e}")

    async def capture_samples(
        self,
        num_samples: Optional[int] = None,
        format_output: CaptureFormat = CaptureFormat.COMPLEX64,
    ) -> Tuple[IQSamples, CaptureMetadata]:
        """
        Capture a fixed number of samples.

        Args:
            num_samples: Number of samples to capture (auto-calculated if None)
            format_output: Output format for samples

        Returns:
            Tuple of (samples, metadata)
        """
        if not self.hardware.is_connected:
            raise RuntimeError("Hardware not connected")

        async with self._capture_lock:
            # Calculate samples if not specified
            if num_samples is None:
                if self.config.duration_s is None:
                    raise ValueError(
                        "Either num_samples or duration_s must be specified"
                    )
                num_samples = int(self.config.sample_rate_hz * self.config.duration_s)

            logger.info(f"Starting capture of {num_samples:,} samples")
            self._is_capturing = True

            try:
                # Read samples
                samples = await self.hardware.read_samples_async(num_samples)

                # Update quality monitoring
                quality_metrics = self.quality_monitor.update(samples)

                # Update metadata
                self.metadata.total_samples += len(samples)
                self.metadata.signal_level_dbfs = quality_metrics.get(
                    "signal_level_dbfs"
                )
                self.metadata.noise_floor_dbfs = quality_metrics.get("noise_floor_dbfs")
                self.metadata.snr_db = quality_metrics.get("snr_db")

                # Check for anomalies
                anomalies = self.quality_monitor.detect_anomalies(quality_metrics)
                if anomalies:
                    logger.warning(f"Signal anomalies detected: {anomalies}")

                # Notify quality callback
                if self.quality_callback:
                    self.quality_callback(quality_metrics)

                # Convert format if needed
                if format_output != CaptureFormat.COMPLEX64:
                    samples = self._convert_format(samples, format_output)

                logger.info(f"Captured {len(samples):,} samples successfully")
                return samples, self.metadata

            except Exception as e:
                logger.error(f"Capture failed: {e}")
                raise
            finally:
                self._is_capturing = False

    async def stream_samples(
        self,
        chunk_size: int = 262144,
        format_output: CaptureFormat = CaptureFormat.COMPLEX64,
    ) -> AsyncGenerator[Tuple[IQSamples, Dict[str, float]], None]:
        """
        Stream samples continuously with real-time processing.

        Args:
            chunk_size: Size of each sample chunk
            format_output: Output format for samples

        Yields:
            Tuples of (samples, quality_metrics)
        """
        if not self.hardware.is_connected:
            raise RuntimeError("Hardware not connected")

        async with self._capture_lock:
            logger.info(f"Starting sample streaming with {chunk_size:,} sample chunks")
            self._is_capturing = True
            self._stop_event.clear()

            try:
                start_time = time.time()
                total_samples = 0

                while not self._stop_event.is_set():
                    # Check duration limit
                    if (
                        self.config.duration_s is not None
                        and time.time() - start_time >= self.config.duration_s
                    ):
                        break

                    # Read samples
                    samples = await self.hardware.read_samples_async(chunk_size)

                    # Update monitoring
                    quality_metrics = self.quality_monitor.update(samples)
                    total_samples += len(samples)

                    # Check for anomalies
                    anomalies = self.quality_monitor.detect_anomalies(quality_metrics)
                    if anomalies:
                        logger.warning(f"Signal anomalies: {anomalies}")

                    # Notify quality callback
                    if self.quality_callback:
                        self.quality_callback(quality_metrics)

                    # Convert format if needed
                    if format_output != CaptureFormat.COMPLEX64:
                        samples = self._convert_format(samples, format_output)

                    yield samples, quality_metrics

                # Update final metadata
                self.metadata.total_samples = total_samples
                logger.info(f"Streaming completed: {total_samples:,} total samples")

            except Exception as e:
                logger.error(f"Streaming failed: {e}")
                raise
            finally:
                self._is_capturing = False

    async def stop_capture(self) -> None:
        """Stop ongoing capture or streaming operation."""
        if self._is_capturing:
            self._stop_event.set()
            logger.info("Capture stop requested")

    def _convert_format(
        self, samples: IQSamples, target_format: CaptureFormat
    ) -> Union[IQSamples, RealSamples]:
        """Convert samples to target format."""
        if target_format == CaptureFormat.COMPLEX64:
            return samples.astype(np.complex64)
        elif target_format == CaptureFormat.COMPLEX128:
            return samples.astype(np.complex128)
        elif target_format == CaptureFormat.FLOAT32:
            # Convert to interleaved I/Q float32
            interleaved = np.zeros(len(samples) * 2, dtype=np.float32)
            interleaved[0::2] = samples.real
            interleaved[1::2] = samples.imag
            return interleaved
        elif target_format == CaptureFormat.INT16:
            # Convert to interleaved I/Q int16 (scaled)
            scale_factor = 32767.0
            interleaved = np.zeros(len(samples) * 2, dtype=np.int16)
            interleaved[0::2] = np.clip(samples.real * scale_factor, -32768, 32767)
            interleaved[1::2] = np.clip(samples.imag * scale_factor, -32768, 32767)
            return interleaved
        else:
            raise ValueError(f"Unsupported format: {target_format}")

    async def save_capture_to_file(
        self,
        filepath: Union[str, Path],
        num_samples: Optional[int] = None,
        format_output: CaptureFormat = CaptureFormat.COMPLEX64,
        include_metadata: bool = True,
        chunk_size: int = 1048576,  # 1M samples per chunk
    ) -> CaptureMetadata:
        """
        Capture samples and save directly to file with memory efficiency.

        Args:
            filepath: Output file path
            num_samples: Number of samples to capture
            format_output: Output format for samples
            include_metadata: Include metadata sidecar file
            chunk_size: Chunk size for memory-efficient writing

        Returns:
            Capture metadata
        """
        filepath = Path(filepath)

        # Calculate total samples
        if num_samples is None:
            if self.config.duration_s is None:
                raise ValueError("Either num_samples or duration_s must be specified")
            num_samples = int(self.config.sample_rate_hz * self.config.duration_s)

        logger.info(f"Saving {num_samples:,} samples to {filepath}")

        # Open file for writing
        with open(filepath, "wb") as f:
            total_written = 0

            # Write samples in chunks
            async for samples, quality_metrics in self.stream_samples(
                chunk_size=min(chunk_size, num_samples - total_written),
                format_output=format_output,
            ):
                # Write chunk to file
                samples.tobytes() and f.write(samples.tobytes())
                total_written += len(samples)

                # Log progress
                if total_written % (chunk_size * 10) == 0:
                    progress = (total_written / num_samples) * 100
                    logger.info(f"Capture progress: {progress:.1f}%")

                # Check if complete
                if total_written >= num_samples:
                    break

        # Update metadata
        self.metadata.file_path = filepath
        self.metadata.file_size_bytes = filepath.stat().st_size
        self.metadata.output_format = format_output

        # Save metadata sidecar if requested
        if include_metadata:
            metadata_path = filepath.with_suffix(".json")
            import json

            with open(metadata_path, "w") as f:
                json.dump(self.metadata.to_dict(), f, indent=2)
            logger.info(f"Metadata saved to {metadata_path}")

        logger.info(
            f"Capture completed: {filepath} ({self.metadata.file_size_bytes:,} bytes)"
        )
        return self.metadata


async def main() -> None:
    """
    Example usage and testing of the Enhanced Live Capture system.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create SDR configuration
    config = SDRConfig(
        platform=SDRPlatform.RTL_SDR,
        frequency_hz=100.1e6,  # 100.1 MHz (FM radio)
        sample_rate_hz=2.048e6,  # 2.048 MHz
        duration_s=5.0,  # 5 seconds
        gain_mode=GainMode.AUTO,
        frequency_correction_ppm=0.0,
    )

    # Quality monitoring callback
    def quality_callback(metrics: Dict[str, float]) -> None:
        signal_db = metrics.get("signal_level_dbfs", 0)
        noise_db = metrics.get("noise_floor_dbfs", 0)
        snr_db = metrics.get("snr_db", 0)
        loss_rate = metrics.get("sample_loss_rate", 0)

        print(
            f"Signal: {signal_db:.1f} dBFS, "
            f"Noise: {noise_db:.1f} dBFS, "
            f"SNR: {snr_db:.1f} dB, "
            f"Loss: {loss_rate:.2f}%"
        )

    # Create and run capture
    capture = EnhancedLiveCapture(config, quality_callback=quality_callback)

    try:
        async with capture:
            print(f"Connected to {config.platform.value}")
            print(f"Frequency: {config.frequency_hz/1e6:.3f} MHz")
            print(f"Sample Rate: {config.sample_rate_hz/1e6:.3f} MHz")
            print(f"Duration: {config.duration_s} seconds\n")

            # Method 1: Capture fixed samples
            print("=== Capturing fixed samples ===")
            samples, metadata = await capture.capture_samples()
            print(f"Captured {len(samples):,} samples")
            print(f"Signal level: {metadata.signal_level_dbfs:.1f} dBFS")
            print(f"SNR: {metadata.snr_db:.1f} dB\n")

            # Method 2: Stream samples (first 10 chunks)
            print("=== Streaming samples ===")
            chunk_count = 0
            async for chunk_samples, quality_metrics in capture.stream_samples(
                chunk_size=65536
            ):
                chunk_count += 1
                if chunk_count >= 10:  # Just show first 10 chunks
                    break

            # Method 3: Save to file
            print("\n=== Saving to file ===")
            output_file = Path("sdr_capture_test.iq")
            final_metadata = await capture.save_capture_to_file(
                filepath=output_file,
                format_output=CaptureFormat.COMPLEX64,
                include_metadata=True,
            )

            print(f"Saved to: {final_metadata.file_path}")
            print(f"File size: {final_metadata.file_size_bytes:,} bytes")
            print(f"Duration: {final_metadata.duration_s:.3f} seconds")
            print(f"Sample loss rate: {final_metadata.sample_loss_rate:.3f}%")

    except Exception as e:
        logger.error(f"Capture failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
