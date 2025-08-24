#!/usr/bin/env python3
"""
Centralized Constants

This module contains all constants used throughout the DroneCmd framework.
Centralizing constants makes the codebase more maintainable and ensures
consistency across modules.

Categories:
- RF and Hardware Constants
- Protocol Constants
- Signal Processing Constants
- File and Data Constants
- Performance and Limits Constants
- Default Configuration Values
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# =============================================================================
# VERSION AND FRAMEWORK INFO
# =============================================================================

FRAMEWORK_VERSION = "2.0.0"
FRAMEWORK_NAME = "DroneCmd"
FRAMEWORK_AUTHOR = "DroneCmd Framework Team"
FRAMEWORK_LICENSE = "Educational/Research Use Only"
FRAMEWORK_URL = "https://github.com/dronecmd/dronecmd"

# Python version requirements
MIN_PYTHON_VERSION = (3, 8)
RECOMMENDED_PYTHON_VERSION = (3, 9)

# =============================================================================
# RF AND HARDWARE CONSTANTS
# =============================================================================

# Frequency constants (Hz)
class Frequencies:
    """RF frequency constants."""
    
    # ISM Band ranges (FCC Part 15.247)
    ISM_900_MHz_MIN = 902e6
    ISM_900_MHz_MAX = 928e6
    ISM_2_4_GHz_MIN = 2400e6
    ISM_2_4_GHz_MAX = 2483.5e6
    ISM_5_8_GHz_MIN = 5725e6
    ISM_5_8_GHz_MAX = 5850e6
    
    # Common drone frequencies
    DRONE_2_4_GHz_CENTER = 2.44e9
    DRONE_5_8_GHz_CENTER = 5.8e9
    
    # Wi-Fi channels (2.4 GHz)
    WIFI_CHANNEL_1 = 2.412e9
    WIFI_CHANNEL_6 = 2.437e9
    WIFI_CHANNEL_11 = 2.462e9
    
    # Amateur radio bands
    HAM_70CM_MIN = 420e6
    HAM_70CM_MAX = 450e6
    HAM_33CM_MIN = 902e6
    HAM_33CM_MAX = 928e6
    
    # Broadcast bands
    FM_RADIO_MIN = 88e6
    FM_RADIO_MAX = 108e6


# Sample rate constants (Hz)
class SampleRates:
    """Sample rate constants for various applications."""
    
    # Common SDR sample rates
    RTL_SDR_MIN = 225e3
    RTL_SDR_MAX = 3.2e6
    RTL_SDR_DEFAULT = 2.048e6
    
    HACKRF_MAX = 20e6
    HACKRF_DEFAULT = 10e6
    
    AIRSPY_MAX = 10e6
    AIRSPY_DEFAULT = 10e6
    
    # Audio sample rates
    AUDIO_8KHZ = 8000
    AUDIO_16KHZ = 16000
    AUDIO_44_1KHZ = 44100
    AUDIO_48KHZ = 48000
    
    # Common analysis rates
    ANALYSIS_1MHZ = 1e6
    ANALYSIS_2MHZ = 2e6
    ANALYSIS_10MHZ = 10e6


# Power constants (dBm, dBW)
class PowerLevels:
    """Power level constants."""
    
    # FCC limits
    FCC_PART_15_MAX_DBM = 30.0  # 1 Watt
    FCC_UNLICENSED_MAX_DBM = 20.0  # 100 mW
    
    # Typical device power levels
    WIFI_TYPICAL_DBM = 20.0
    BLUETOOTH_TYPICAL_DBM = 0.0
    DRONE_RC_TYPICAL_DBM = 20.0
    
    # Reference levels
    ONE_MILLIWATT_DBM = 0.0
    ONE_WATT_DBM = 30.0
    ONE_MICROWATT_DBM = -30.0
    
    # Dynamic range
    NOISE_FLOOR_TYPICAL_DBM = -100.0
    STRONG_SIGNAL_DBM = -20.0


# =============================================================================
# PROTOCOL CONSTANTS
# =============================================================================

class ProtocolConstants:
    """Constants for various drone protocols."""
    
    # MAVLink constants
    MAVLINK_V1_MAGIC = 0xFE
    MAVLINK_V2_MAGIC = 0xFD
    MAVLINK_V1_HEADER_LEN = 8
    MAVLINK_V2_HEADER_LEN = 12
    MAVLINK_MAX_PAYLOAD_LEN = 255
    MAVLINK_CHECKSUM_LEN = 2
    
    # DJI constants
    DJI_SYNC_BYTES = b'\x55\xAA'
    DJI_HEADER_LEN = 4
    DJI_MAX_PACKET_LEN = 1024
    
    # Generic packet constraints
    MIN_PACKET_LENGTH = 8
    MAX_PACKET_LENGTH = 2048
    DEFAULT_PACKET_LENGTH = 256
    
    # Protocol identifiers
    PROTOCOL_UNKNOWN = "unknown"
    PROTOCOL_MAVLINK = "mavlink"
    PROTOCOL_DJI = "dji"
    PROTOCOL_PARROT = "parrot"
    PROTOCOL_SKYDIO = "skydio"
    PROTOCOL_AUTEL = "autel"
    PROTOCOL_YUNEEC = "yuneec"
    PROTOCOL_GENERIC = "generic"


class ModulationConstants:
    """Constants for modulation schemes."""
    
    # Modulation types
    MODULATION_OOK = "ook"
    MODULATION_ASK = "ask"
    MODULATION_FSK = "fsk"
    MODULATION_GFSK = "gfsk"
    MODULATION_MSK = "msk"
    MODULATION_PSK = "psk"
    MODULATION_BPSK = "bpsk"
    MODULATION_QPSK = "qpsk"
    MODULATION_QAM16 = "qam16"
    MODULATION_QAM64 = "qam64"
    
    # Common bit rates (bps)
    BITRATE_1200 = 1200
    BITRATE_2400 = 2400
    BITRATE_4800 = 4800
    BITRATE_9600 = 9600
    BITRATE_19200 = 19200
    BITRATE_38400 = 38400
    BITRATE_57600 = 57600
    BITRATE_115200 = 115200
    
    # Symbol rates
    SYMBOL_RATE_LOW = 1000
    SYMBOL_RATE_MEDIUM = 10000
    SYMBOL_RATE_HIGH = 100000


# =============================================================================
# SIGNAL PROCESSING CONSTANTS
# =============================================================================

class SignalProcessing:
    """Signal processing constants."""
    
    # Filter parameters
    DEFAULT_FILTER_ORDER = 4
    MAX_FILTER_ORDER = 12
    DEFAULT_CUTOFF_FACTOR = 2.0
    
    # Window functions
    WINDOW_HAMMING = "hamming"
    WINDOW_HANNING = "hanning"
    WINDOW_BLACKMAN = "blackman"
    WINDOW_KAISER = "kaiser"
    
    # FFT sizes (powers of 2)
    FFT_SIZE_SMALL = 256
    FFT_SIZE_MEDIUM = 1024
    FFT_SIZE_LARGE = 4096
    FFT_SIZE_EXTRA_LARGE = 8192
    
    # Overlap factors for spectrograms
    OVERLAP_FACTOR_25 = 0.25
    OVERLAP_FACTOR_50 = 0.50
    OVERLAP_FACTOR_75 = 0.75
    
    # Noise floor estimation
    NOISE_PERCENTILE = 10  # Use 10th percentile for noise floor
    SIGNAL_PERCENTILE = 90  # Use 90th percentile for signal level
    
    # AGC parameters
    AGC_ATTACK_TIME_S = 0.001  # 1ms
    AGC_RELEASE_TIME_S = 0.1   # 100ms
    AGC_TARGET_POWER = 1.0
    AGC_MAX_GAIN = 10.0
    AGC_MIN_GAIN = 0.1
    
    # Threshold detection
    DEFAULT_DETECTION_THRESHOLD = 0.05
    SNR_THRESHOLD_DB = 6.0  # Minimum SNR for reliable detection
    
    # Quality metrics
    MIN_CONFIDENCE = 0.0
    MAX_CONFIDENCE = 1.0
    DEFAULT_CONFIDENCE_THRESHOLD = 0.7
    EXCELLENT_CONFIDENCE_THRESHOLD = 0.9


# =============================================================================
# FHSS CONSTANTS
# =============================================================================

class FHSSConstants:
    """Frequency Hopping Spread Spectrum constants."""
    
    # FCC Part 15.247 requirements
    MIN_CHANNELS_900MHZ = 25
    MIN_CHANNELS_2_4GHZ = 75
    MIN_CHANNELS_5_8GHZ = 75
    
    MAX_DWELL_TIME_S = 0.4  # 400ms maximum dwell time
    MIN_CHANNEL_SPACING_HZ = 25e3  # 25 kHz minimum
    
    # Typical channel spacings
    CHANNEL_SPACING_25KHZ = 25e3
    CHANNEL_SPACING_100KHZ = 100e3
    CHANNEL_SPACING_1MHZ = 1e6
    
    # Hop sequence parameters
    DEFAULT_HOP_COUNT = 8
    MAX_HOP_COUNT = 256
    MIN_HOP_COUNT = 2
    
    # Pulse shaping
    RAISED_COSINE_BETA_DEFAULT = 0.25
    RAISED_COSINE_SPAN_DEFAULT = 6
    PULSE_SHAPE_SAMPLES_PER_SYMBOL_MIN = 2


# =============================================================================
# FILE AND DATA CONSTANTS
# =============================================================================

class FileConstants:
    """File and data handling constants."""
    
    # File extensions
    IQ_FILE_EXTENSION = ".iq"
    METADATA_FILE_EXTENSION = ".json"
    LOG_FILE_EXTENSION = ".log"
    CONFIG_FILE_EXTENSION = ".yaml"
    
    # Data formats
    COMPLEX64_FORMAT = "complex64"
    COMPLEX128_FORMAT = "complex128"
    FLOAT32_FORMAT = "float32"
    INT16_FORMAT = "int16"
    
    # Chunk sizes for processing
    SMALL_CHUNK_SIZE = 1024
    MEDIUM_CHUNK_SIZE = 8192
    LARGE_CHUNK_SIZE = 65536
    HUGE_CHUNK_SIZE = 1048576  # 1MB
    
    # Buffer sizes
    USB_BUFFER_SIZE = 262144  # 256KB
    NETWORK_BUFFER_SIZE = 8192  # 8KB
    DISK_BUFFER_SIZE = 1048576  # 1MB


# Default directories
class Directories:
    """Default directory paths."""
    
    # Get user's home directory
    HOME_DIR = Path.home()
    
    # Framework directories
    DRONECMD_DIR = HOME_DIR / ".dronecmd"
    CONFIG_DIR = DRONECMD_DIR / "config"
    LOGS_DIR = DRONECMD_DIR / "logs"
    CAPTURES_DIR = DRONECMD_DIR / "captures"
    PLUGINS_DIR = DRONECMD_DIR / "plugins"
    MODELS_DIR = DRONECMD_DIR / "models"
    CACHE_DIR = DRONECMD_DIR / "cache"
    
    # Temporary directories
    TEMP_DIR = DRONECMD_DIR / "temp"
    
    # Data directories
    TEST_DATA_DIR = DRONECMD_DIR / "test_data"
    REFERENCE_DATA_DIR = DRONECMD_DIR / "reference"


# =============================================================================
# PERFORMANCE AND LIMITS CONSTANTS
# =============================================================================

class PerformanceLimits:
    """Performance and resource limits."""
    
    # Memory limits (bytes)
    MAX_SAMPLE_BUFFER_SIZE = 100 * 1024 * 1024  # 100MB
    MAX_PACKET_CACHE_SIZE = 10 * 1024 * 1024    # 10MB
    MAX_LOG_FILE_SIZE = 50 * 1024 * 1024        # 50MB
    
    # Processing limits
    MAX_CONCURRENT_THREADS = 8
    MAX_PROCESSING_TIME_S = 300  # 5 minutes
    MAX_CAPTURE_DURATION_S = 3600  # 1 hour
    
    # Rate limits
    MAX_PACKET_RATE_HZ = 10000
    MAX_CLASSIFICATION_RATE_HZ = 1000
    MAX_FILE_WRITE_RATE_MBS = 100  # MB/s
    
    # Queue sizes
    PACKET_QUEUE_SIZE = 1000
    SIGNAL_QUEUE_SIZE = 100
    LOG_QUEUE_SIZE = 10000
    
    # Timeout values (seconds)
    HARDWARE_TIMEOUT_S = 10
    NETWORK_TIMEOUT_S = 30
    PLUGIN_TIMEOUT_S = 5
    CLASSIFICATION_TIMEOUT_S = 1
    
    # Retry limits
    MAX_RETRY_ATTEMPTS = 3
    RETRY_DELAY_S = 1.0
    BACKOFF_MULTIPLIER = 2.0


class QualityThresholds:
    """Quality assessment thresholds."""
    
    # SNR thresholds (dB)
    SNR_POOR = 0
    SNR_FAIR = 6
    SNR_GOOD = 12
    SNR_EXCELLENT = 20
    
    # Confidence thresholds
    CONFIDENCE_LOW = 0.3
    CONFIDENCE_MEDIUM = 0.6
    CONFIDENCE_HIGH = 0.8
    CONFIDENCE_VERY_HIGH = 0.95
    
    # Error rates
    BER_EXCELLENT = 1e-6
    BER_GOOD = 1e-4
    BER_ACCEPTABLE = 1e-2
    BER_POOR = 1e-1
    
    # Signal power thresholds (dBFS)
    POWER_VERY_LOW = -80
    POWER_LOW = -60
    POWER_MEDIUM = -40
    POWER_HIGH = -20
    POWER_VERY_HIGH = -10
    
    # EVM thresholds (%)
    EVM_EXCELLENT = 1.0
    EVM_GOOD = 5.0
    EVM_ACCEPTABLE = 15.0
    EVM_POOR = 30.0


# =============================================================================
# DEFAULT CONFIGURATION VALUES
# =============================================================================

class DefaultConfig:
    """Default configuration values."""
    
    # SDR defaults
    DEFAULT_FREQUENCY_HZ = 100.1e6  # FM radio for testing
    DEFAULT_SAMPLE_RATE_HZ = SampleRates.RTL_SDR_DEFAULT
    DEFAULT_GAIN_MODE = "auto"
    DEFAULT_DEVICE_INDEX = 0
    
    # Capture defaults
    DEFAULT_CAPTURE_DURATION_S = 10.0
    DEFAULT_DETECTION_THRESHOLD = SignalProcessing.DEFAULT_DETECTION_THRESHOLD
    DEFAULT_MIN_PACKET_LENGTH = ProtocolConstants.MIN_PACKET_LENGTH
    DEFAULT_MAX_PACKET_LENGTH = ProtocolConstants.MAX_PACKET_LENGTH
    
    # Processing defaults
    DEFAULT_FILTER_ORDER = SignalProcessing.DEFAULT_FILTER_ORDER
    DEFAULT_CONFIDENCE_THRESHOLD = SignalProcessing.DEFAULT_CONFIDENCE_THRESHOLD
    DEFAULT_ENABLE_AGC = True
    DEFAULT_ENABLE_ADAPTIVE_THRESHOLD = True
    
    # Plugin defaults
    DEFAULT_PLUGIN_TIMEOUT_MS = PerformanceLimits.PLUGIN_TIMEOUT_S * 1000
    DEFAULT_MAX_PLUGINS = 50
    DEFAULT_ENABLE_PLUGIN_VALIDATION = True
    
    # Logging defaults
    DEFAULT_LOG_LEVEL = "INFO"
    DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DEFAULT_MAX_LOG_FILES = 10
    
    # Cache defaults
    DEFAULT_ENABLE_CACHING = True
    DEFAULT_CACHE_SIZE = 1000
    DEFAULT_CACHE_TTL_S = 3600  # 1 hour


# =============================================================================
# MATHEMATICAL CONSTANTS
# =============================================================================

class MathConstants:
    """Mathematical constants used in signal processing."""
    
    # Commonly used values
    PI = np.pi
    TWO_PI = 2 * np.pi
    SQRT_2 = np.sqrt(2)
    LN_2 = np.log(2)
    LN_10 = np.log(10)
    
    # dB conversion constants
    DB_FACTOR = 10.0
    DB_POWER_FACTOR = 20.0  # For voltage/current to dB
    
    # Noise and random process constants
    NOISE_SEED = 42  # Default seed for reproducible noise
    GAUSSIAN_SIGMA = 1.0
    
    # Numerical precision
    EPSILON = 1e-12  # Small value to avoid division by zero
    FLOAT32_EPSILON = np.finfo(np.float32).eps
    FLOAT64_EPSILON = np.finfo(np.float64).eps


# =============================================================================
# ERROR CODES
# =============================================================================

class ErrorCodes:
    """Standardized error codes for the framework."""
    
    # General errors (0-99)
    UNKNOWN_ERROR = "ERR000"
    INITIALIZATION_FAILED = "ERR001"
    INVALID_PARAMETER = "ERR002"
    MISSING_PARAMETER = "ERR003"
    
    # Hardware errors (100-199)
    HARDWARE_NOT_FOUND = "ERR100"
    HARDWARE_INIT_FAILED = "ERR101"
    CAPTURE_FAILED = "ERR102"
    TRANSMISSION_FAILED = "ERR103"
    SDR_CONNECTION_LOST = "ERR104"
    
    # Processing errors (200-299)
    PROCESSING_FAILED = "ERR200"
    DEMODULATION_FAILED = "ERR201"
    CLASSIFICATION_FAILED = "ERR202"
    PARSING_FAILED = "ERR203"
    INSUFFICIENT_DATA = "ERR204"
    
    # Plugin errors (300-399)
    PLUGIN_NOT_FOUND = "ERR300"
    PLUGIN_LOAD_FAILED = "ERR301"
    PLUGIN_VALIDATION_FAILED = "ERR302"
    PLUGIN_TIMEOUT = "ERR303"
    
    # Protocol errors (400-499)
    PROTOCOL_NOT_SUPPORTED = "ERR400"
    PACKET_FORMAT_ERROR = "ERR401"
    CHECKSUM_ERROR = "ERR402"
    SEQUENCE_ERROR = "ERR403"
    
    # Compliance errors (500-599)
    FCC_VIOLATION = "ERR500"
    POWER_LIMIT_EXCEEDED = "ERR501"
    FREQUENCY_VIOLATION = "ERR502"
    SAFETY_VIOLATION = "ERR503"


# =============================================================================
# UTILITY FUNCTIONS FOR CONSTANTS
# =============================================================================

def get_constant_by_name(name: str) -> any:
    """
    Get a constant by its name.
    
    Args:
        name: Name of the constant
        
    Returns:
        Constant value or None if not found
    """
    # Search through all classes in this module
    current_module = globals()
    
    for class_name, class_obj in current_module.items():
        if isinstance(class_obj, type) and hasattr(class_obj, name):
            return getattr(class_obj, name)
    
    return None


def list_constants_in_category(category_class: type) -> Dict[str, any]:
    """
    List all constants in a category class.
    
    Args:
        category_class: Class containing constants
        
    Returns:
        Dictionary of constant names and values
    """
    constants = {}
    
    for attr_name in dir(category_class):
        if not attr_name.startswith('_'):
            attr_value = getattr(category_class, attr_name)
            if not callable(attr_value):
                constants[attr_name] = attr_value
    
    return constants


def validate_frequency_range(frequency_hz: float, band: str = "ISM_2_4_GHz") -> bool:
    """
    Validate if frequency is within specified band.
    
    Args:
        frequency_hz: Frequency to validate
        band: Band to check against
        
    Returns:
        True if frequency is in band
    """
    if band == "ISM_2_4_GHz":
        return Frequencies.ISM_2_4_GHz_MIN <= frequency_hz <= Frequencies.ISM_2_4_GHz_MAX
    elif band == "ISM_900_MHz":
        return Frequencies.ISM_900_MHz_MIN <= frequency_hz <= Frequencies.ISM_900_MHz_MAX
    elif band == "ISM_5_8_GHz":
        return Frequencies.ISM_5_8_GHz_MIN <= frequency_hz <= Frequencies.ISM_5_8_GHz_MAX
    else:
        return False


def get_recommended_sample_rate(sdr_type: str) -> float:
    """
    Get recommended sample rate for SDR type.
    
    Args:
        sdr_type: Type of SDR device
        
    Returns:
        Recommended sample rate in Hz
    """
    sdr_type = sdr_type.lower()
    
    if "rtl" in sdr_type:
        return SampleRates.RTL_SDR_DEFAULT
    elif "hackrf" in sdr_type:
        return SampleRates.HACKRF_DEFAULT
    elif "airspy" in sdr_type:
        return SampleRates.AIRSPY_DEFAULT
    else:
        return SampleRates.RTL_SDR_DEFAULT  # Safe default


def create_directories() -> None:
    """Create default framework directories if they don't exist."""
    directories = [
        Directories.DRONECMD_DIR,
        Directories.CONFIG_DIR,
        Directories.LOGS_DIR,
        Directories.CAPTURES_DIR,
        Directories.PLUGINS_DIR,
        Directories.MODELS_DIR,
        Directories.CACHE_DIR,
        Directories.TEMP_DIR,
        Directories.TEST_DATA_DIR,
        Directories.REFERENCE_DATA_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Demo constants usage
    print("=== DroneCmd Constants Demo ===")
    
    print(f"Framework: {FRAMEWORK_NAME} v{FRAMEWORK_VERSION}")
    print(f"Author: {FRAMEWORK_AUTHOR}")
    print()
    
    print("Frequency Ranges:")
    print(f"  2.4 GHz ISM: {Frequencies.ISM_2_4_GHz_MIN/1e6:.1f} - {Frequencies.ISM_2_4_GHz_MAX/1e6:.1f} MHz")
    print(f"  5.8 GHz ISM: {Frequencies.ISM_5_8_GHz_MIN/1e6:.1f} - {Frequencies.ISM_5_8_GHz_MAX/1e6:.1f} MHz")
    print()
    
    print("Sample Rates:")
    print(f"  RTL-SDR Default: {SampleRates.RTL_SDR_DEFAULT/1e6:.3f} MHz")
    print(f"  HackRF Default: {SampleRates.HACKRF_DEFAULT/1e6:.1f} MHz")
    print()
    
    print("Protocol Constants:")
    protocols = list_constants_in_category(ProtocolConstants)
    for name, value in list(protocols.items())[:5]:  # Show first 5
        print(f"  {name}: {value}")
    print()
    
    print("Default Configuration:")
    print(f"  Frequency: {DefaultConfig.DEFAULT_FREQUENCY_HZ/1e6:.1f} MHz")
    print(f"  Sample Rate: {DefaultConfig.DEFAULT_SAMPLE_RATE_HZ/1e6:.3f} MHz")
    print(f"  Duration: {DefaultConfig.DEFAULT_CAPTURE_DURATION_S} seconds")
    print()
    
    # Test validation
    test_freq = 2.44e9
    is_valid = validate_frequency_range(test_freq, "ISM_2_4_GHz")
    print(f"Frequency {test_freq/1e6:.1f} MHz in 2.4 GHz ISM band: {is_valid}")
    
    # Test directory creation
    print("\nCreating framework directories...")
    create_directories()
    print("Directories created successfully.")