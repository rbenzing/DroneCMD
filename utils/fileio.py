#!/usr/bin/env python3
"""
Enhanced File I/O Utilities

This module provides comprehensive file I/O functionality for the dronecmd library,
supporting various signal formats, metadata handling, compression, and robust
error handling.

Supported Formats:
- IQ files (.iq, .raw, .bin) - Complex64, Complex128, Int16, Float32
- WAV files (.wav) - For audio-rate signals
- CSV files (.csv) - For tabular data export
- JSON/YAML (.json, .yaml) - For metadata and configuration
- HDF5 (.h5, .hdf5) - For large datasets with metadata
- SigMF (.sigmf-meta, .sigmf-data) - Signal Metadata Format

Key Features:
- Automatic format detection
- Metadata preservation and extraction
- Compression support (gzip, bzip2, lzma)
- Chunked reading for large files
- Progress reporting for long operations
- Robust error handling and validation
- Memory-efficient streaming
- Cross-platform compatibility
"""

from __future__ import annotations

import bz2
import gzip
import json
import lzma
import logging
import mmap
import os
import struct
import warnings
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union, BinaryIO
import hashlib

import numpy as np
import numpy.typing as npt

# Optional imports for extended functionality
try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    h5py = None

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None

try:
    import scipy.io.wavfile as wavfile
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    wavfile = None

# Configure module logger
logger = logging.getLogger(__name__)

# Type aliases
IQSamples = npt.NDArray[np.complex64]
RealSamples = npt.NDArray[np.float32]
FilePath = Union[str, Path]
ProgressCallback = Optional[callable]


class FileFormat(Enum):
    """Supported file formats."""
    
    # IQ data formats
    COMPLEX64 = "complex64"          # 32-bit I + 32-bit Q
    COMPLEX128 = "complex128"        # 64-bit I + 64-bit Q
    INT16_IQ = "int16_iq"           # 16-bit I + 16-bit Q interleaved
    FLOAT32_IQ = "float32_iq"       # 32-bit I + 32-bit Q interleaved
    
    # Audio formats
    WAV = "wav"                      # WAV audio file
    
    # Data formats
    CSV = "csv"                      # Comma-separated values
    JSON = "json"                    # JSON format
    YAML = "yaml"                    # YAML format
    
    # Advanced formats
    HDF5 = "hdf5"                    # HDF5 hierarchical format
    SIGMF = "sigmf"                  # Signal Metadata Format
    
    # Raw formats
    BINARY = "binary"                # Raw binary data
    TEXT = "text"                    # Plain text


class CompressionType(Enum):
    """Supported compression types."""
    
    NONE = "none"
    GZIP = "gzip"
    BZIP2 = "bzip2"
    LZMA = "lzma"


class FileIOError(Exception):
    """Custom exception for file I/O operations."""
    pass


class FileInfo:
    """Container for file information and metadata."""
    
    def __init__(self, filepath: FilePath):
        self.filepath = Path(filepath)
        self.format = None
        self.compression = CompressionType.NONE
        self.size_bytes = 0
        self.samples_count = 0
        self.sample_rate = None
        self.center_frequency = None
        self.timestamp = None
        self.checksum = None
        self.metadata = {}
        
        if self.filepath.exists():
            self._analyze_file()
    
    def _analyze_file(self) -> None:
        """Analyze file to determine format and properties."""
        try:
            self.size_bytes = self.filepath.stat().st_size
            self.format = _detect_file_format(self.filepath)
            self.compression = _detect_compression(self.filepath)
            
            # Try to extract metadata
            metadata_file = self._find_metadata_file()
            if metadata_file:
                self.metadata = _load_metadata_file(metadata_file)
                self._extract_metadata_fields()
            
        except Exception as e:
            logger.debug(f"Could not fully analyze file {self.filepath}: {e}")
    
    def _find_metadata_file(self) -> Optional[Path]:
        """Find associated metadata file."""
        base_path = self.filepath.with_suffix('')
        
        # Common metadata file extensions
        metadata_extensions = ['.json', '.yaml', '.yml', '.meta', '.sigmf-meta']
        
        for ext in metadata_extensions:
            metadata_path = base_path.with_suffix(ext)
            if metadata_path.exists():
                return metadata_path
        
        return None
    
    def _extract_metadata_fields(self) -> None:
        """Extract common fields from metadata."""
        if not self.metadata:
            return
        
        # Common field mappings
        field_mapping = {
            'sample_rate': ['sample_rate', 'sampleRate', 'fs', 'sample_rate_hz'],
            'center_frequency': ['center_frequency', 'centerFrequency', 'freq', 'frequency_hz'],
            'timestamp': ['timestamp', 'datetime', 'capture_time', 'start_time'],
            'samples_count': ['samples', 'sample_count', 'length', 'num_samples']
        }
        
        for attr, possible_keys in field_mapping.items():
            for key in possible_keys:
                if key in self.metadata:
                    setattr(self, attr, self.metadata[key])
                    break
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert file info to dictionary."""
        return {
            'filepath': str(self.filepath),
            'format': self.format.value if self.format else None,
            'compression': self.compression.value,
            'size_bytes': self.size_bytes,
            'samples_count': self.samples_count,
            'sample_rate': self.sample_rate,
            'center_frequency': self.center_frequency,
            'timestamp': self.timestamp,
            'checksum': self.checksum,
            'metadata': self.metadata
        }


def _detect_file_format(filepath: Path) -> FileFormat:
    """Detect file format from extension and content."""
    extension = filepath.suffix.lower()
    
    # Handle compressed files
    if extension in ['.gz', '.bz2', '.xz']:
        # Look at the extension before compression
        stem_ext = Path(filepath.stem).suffix.lower()
        if stem_ext:
            extension = stem_ext
    
    # Format mapping
    format_map = {
        '.iq': FileFormat.COMPLEX64,
        '.raw': FileFormat.COMPLEX64,
        '.bin': FileFormat.BINARY,
        '.cfile': FileFormat.COMPLEX64,
        '.cf32': FileFormat.COMPLEX64,
        '.cf64': FileFormat.COMPLEX128,
        '.cs16': FileFormat.INT16_IQ,
        '.wav': FileFormat.WAV,
        '.csv': FileFormat.CSV,
        '.json': FileFormat.JSON,
        '.yaml': FileFormat.YAML,
        '.yml': FileFormat.YAML,
        '.h5': FileFormat.HDF5,
        '.hdf5': FileFormat.HDF5,
        '.sigmf-data': FileFormat.SIGMF,
        '.txt': FileFormat.TEXT
    }
    
    detected_format = format_map.get(extension, FileFormat.BINARY)
    
    # Additional content-based detection for ambiguous files
    if detected_format == FileFormat.BINARY and filepath.exists():
        detected_format = _detect_format_by_content(filepath)
    
    return detected_format


def _detect_compression(filepath: Path) -> CompressionType:
    """Detect compression type from file extension."""
    extension = filepath.suffix.lower()
    
    compression_map = {
        '.gz': CompressionType.GZIP,
        '.bz2': CompressionType.BZIP2,
        '.xz': CompressionType.LZMA,
        '.lzma': CompressionType.LZMA
    }
    
    return compression_map.get(extension, CompressionType.NONE)


def _detect_format_by_content(filepath: Path) -> FileFormat:
    """Detect format by examining file content."""
    try:
        with open(filepath, 'rb') as f:
            header = f.read(16)
        
        # Check for common file signatures
        if header.startswith(b'RIFF') and b'WAVE' in header:
            return FileFormat.WAV
        elif header.startswith(b'\x89HDF'):
            return FileFormat.HDF5
        elif len(header) >= 8 and len(header) % 8 == 0:
            # Might be complex64 data
            return FileFormat.COMPLEX64
        elif len(header) >= 4 and len(header) % 4 == 0:
            # Might be int16 IQ data
            return FileFormat.INT16_IQ
        
    except Exception:
        pass
    
    return FileFormat.BINARY


def _open_file(filepath: Path, mode: str = 'rb') -> BinaryIO:
    """Open file with automatic compression detection."""
    compression = _detect_compression(filepath)
    
    if compression == CompressionType.GZIP:
        return gzip.open(filepath, mode)
    elif compression == CompressionType.BZIP2:
        return bz2.open(filepath, mode)
    elif compression == CompressionType.LZMA:
        return lzma.open(filepath, mode)
    else:
        return open(filepath, mode)


def _load_metadata_file(filepath: Path) -> Dict[str, Any]:
    """Load metadata from file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            if filepath.suffix.lower() == '.json':
                return json.load(f)
            elif filepath.suffix.lower() in ['.yaml', '.yml']:
                if YAML_AVAILABLE:
                    return yaml.safe_load(f)
                else:
                    logger.warning("YAML support not available")
                    return {}
            elif filepath.suffix.lower() in ['.sigmf-meta']:
                # SigMF metadata format (JSON-based)
                return json.load(f)
            else:
                return {}
    except Exception as e:
        logger.warning(f"Could not load metadata from {filepath}: {e}")
        return {}


def _save_metadata_file(filepath: Path, metadata: Dict[str, Any]) -> None:
    """Save metadata to file."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            if filepath.suffix.lower() == '.json':
                json.dump(metadata, f, indent=2, default=str)
            elif filepath.suffix.lower() in ['.yaml', '.yml']:
                if YAML_AVAILABLE:
                    yaml.dump(metadata, f, default_flow_style=False)
                else:
                    # Fallback to JSON
                    json.dump(metadata, f, indent=2, default=str)
            else:
                json.dump(metadata, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"Could not save metadata to {filepath}: {e}")


def _calculate_checksum(filepath: Path, algorithm: str = 'sha256') -> str:
    """Calculate file checksum."""
    hash_algo = hashlib.new(algorithm)
    
    try:
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_algo.update(chunk)
        return hash_algo.hexdigest()
    except Exception as e:
        logger.warning(f"Could not calculate checksum for {filepath}: {e}")
        return ""


# =============================================================================
# HIGH-LEVEL READ FUNCTIONS
# =============================================================================

def read_iq_file(
    filepath: FilePath,
    format_hint: Optional[FileFormat] = None,
    max_samples: Optional[int] = None,
    offset_samples: int = 0,
    progress_callback: ProgressCallback = None
) -> IQSamples:
    """
    Read IQ samples from file with automatic format detection.
    
    Args:
        filepath: Path to IQ file
        format_hint: Optional format hint for ambiguous files
        max_samples: Maximum number of samples to read
        offset_samples: Number of samples to skip from beginning
        progress_callback: Optional progress callback function
        
    Returns:
        Complex IQ samples
        
    Raises:
        FileIOError: If file cannot be read
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileIOError(f"File not found: {filepath}")
    
    # Detect format
    file_format = format_hint or _detect_file_format(filepath)
    
    logger.info(f"Reading IQ file: {filepath} (format: {file_format.value})")
    
    try:
        if file_format == FileFormat.COMPLEX64:
            return _read_complex64(filepath, max_samples, offset_samples, progress_callback)
        elif file_format == FileFormat.COMPLEX128:
            return _read_complex128(filepath, max_samples, offset_samples, progress_callback)
        elif file_format == FileFormat.INT16_IQ:
            return _read_int16_iq(filepath, max_samples, offset_samples, progress_callback)
        elif file_format == FileFormat.FLOAT32_IQ:
            return _read_float32_iq(filepath, max_samples, offset_samples, progress_callback)
        elif file_format == FileFormat.WAV:
            return _read_wav_file(filepath, max_samples, offset_samples)
        elif file_format == FileFormat.HDF5:
            return _read_hdf5_iq(filepath, max_samples, offset_samples)
        elif file_format == FileFormat.SIGMF:
            return _read_sigmf_data(filepath, max_samples, offset_samples)
        else:
            # Try as complex64 by default
            logger.warning(f"Unknown format {file_format}, trying as complex64")
            return _read_complex64(filepath, max_samples, offset_samples, progress_callback)
            
    except Exception as e:
        raise FileIOError(f"Failed to read IQ file {filepath}: {e}") from e


def _read_complex64(
    filepath: Path,
    max_samples: Optional[int],
    offset_samples: int,
    progress_callback: ProgressCallback
) -> IQSamples:
    """Read complex64 format file."""
    with _open_file(filepath, 'rb') as f:
        # Skip offset
        if offset_samples > 0:
            f.seek(offset_samples * 8)  # 8 bytes per complex64 sample
        
        # Determine how many samples to read
        if max_samples is None:
            remaining_bytes = f.seek(0, 2) - f.tell()  # Seek to end and back
            f.seek(-remaining_bytes, 2)  # Return to read position
            samples_to_read = remaining_bytes // 8
        else:
            samples_to_read = max_samples
        
        # Read in chunks for large files
        chunk_size = min(1024*1024, samples_to_read)  # 1M samples max per chunk
        samples = []
        samples_read = 0
        
        while samples_read < samples_to_read:
            current_chunk_size = min(chunk_size, samples_to_read - samples_read)
            chunk_bytes = current_chunk_size * 8
            
            data = f.read(chunk_bytes)
            if not data:
                break
            
            chunk_samples = np.frombuffer(data, dtype=np.complex64)
            samples.append(chunk_samples)
            samples_read += len(chunk_samples)
            
            # Progress callback
            if progress_callback:
                progress = samples_read / samples_to_read
                progress_callback(progress)
        
        if not samples:
            return np.array([], dtype=np.complex64)
        
        return np.concatenate(samples)


def _read_complex128(
    filepath: Path,
    max_samples: Optional[int],
    offset_samples: int,
    progress_callback: ProgressCallback
) -> IQSamples:
    """Read complex128 format and convert to complex64."""
    with _open_file(filepath, 'rb') as f:
        # Skip offset
        if offset_samples > 0:
            f.seek(offset_samples * 16)  # 16 bytes per complex128 sample
        
        # Read as complex128 then convert
        if max_samples is None:
            data = f.read()
            samples_complex128 = np.frombuffer(data, dtype=np.complex128)
        else:
            data = f.read(max_samples * 16)
            samples_complex128 = np.frombuffer(data, dtype=np.complex128)
        
        # Convert to complex64
        return samples_complex128.astype(np.complex64)


def _read_int16_iq(
    filepath: Path,
    max_samples: Optional[int],
    offset_samples: int,
    progress_callback: ProgressCallback
) -> IQSamples:
    """Read int16 interleaved IQ format."""
    with _open_file(filepath, 'rb') as f:
        # Skip offset (2 int16 values per sample)
        if offset_samples > 0:
            f.seek(offset_samples * 4)  # 4 bytes per IQ sample
        
        # Read int16 data
        if max_samples is None:
            data = f.read()
        else:
            data = f.read(max_samples * 4)
        
        int16_data = np.frombuffer(data, dtype=np.int16)
        
        # Ensure even number of values
        if len(int16_data) % 2 != 0:
            int16_data = int16_data[:-1]
        
        # Reshape and convert to complex
        iq_pairs = int16_data.reshape(-1, 2)
        i_samples = iq_pairs[:, 0].astype(np.float32) / 32768.0
        q_samples = iq_pairs[:, 1].astype(np.float32) / 32768.0
        
        return (i_samples + 1j * q_samples).astype(np.complex64)


def _read_float32_iq(
    filepath: Path,
    max_samples: Optional[int],
    offset_samples: int,
    progress_callback: ProgressCallback
) -> IQSamples:
    """Read float32 interleaved IQ format."""
    with _open_file(filepath, 'rb') as f:
        # Skip offset
        if offset_samples > 0:
            f.seek(offset_samples * 8)  # 8 bytes per IQ sample
        
        # Read float32 data
        if max_samples is None:
            data = f.read()
        else:
            data = f.read(max_samples * 8)
        
        float32_data = np.frombuffer(data, dtype=np.float32)
        
        # Ensure even number of values
        if len(float32_data) % 2 != 0:
            float32_data = float32_data[:-1]
        
        # Reshape to IQ pairs and convert to complex
        iq_pairs = float32_data.reshape(-1, 2)
        return (iq_pairs[:, 0] + 1j * iq_pairs[:, 1]).astype(np.complex64)


def _read_wav_file(
    filepath: Path,
    max_samples: Optional[int],
    offset_samples: int
) -> IQSamples:
    """Read WAV file and convert to IQ samples."""
    if not SCIPY_AVAILABLE:
        raise FileIOError("scipy required for WAV file support")
    
    try:
        sample_rate, data = wavfile.read(filepath)
        
        # Handle different WAV formats
        if data.ndim == 1:
            # Mono - treat as real signal, set Q=0
            real_samples = data.astype(np.float32)
            if data.dtype == np.int16:
                real_samples = real_samples / 32768.0
            elif data.dtype == np.int32:
                real_samples = real_samples / 2147483648.0
            
            iq_samples = real_samples + 1j * np.zeros_like(real_samples)
            
        elif data.ndim == 2 and data.shape[1] == 2:
            # Stereo - treat as I/Q channels
            i_samples = data[:, 0].astype(np.float32)
            q_samples = data[:, 1].astype(np.float32)
            
            if data.dtype == np.int16:
                i_samples = i_samples / 32768.0
                q_samples = q_samples / 32768.0
            elif data.dtype == np.int32:
                i_samples = i_samples / 2147483648.0
                q_samples = q_samples / 2147483648.0
            
            iq_samples = (i_samples + 1j * q_samples).astype(np.complex64)
        else:
            raise FileIOError(f"Unsupported WAV format: {data.shape}")
        
        # Apply offset and max_samples
        if offset_samples > 0:
            iq_samples = iq_samples[offset_samples:]
        
        if max_samples is not None:
            iq_samples = iq_samples[:max_samples]
        
        return iq_samples
        
    except Exception as e:
        raise FileIOError(f"Failed to read WAV file: {e}") from e


def _read_hdf5_iq(
    filepath: Path,
    max_samples: Optional[int],
    offset_samples: int
) -> IQSamples:
    """Read IQ data from HDF5 file."""
    if not HDF5_AVAILABLE:
        raise FileIOError("h5py required for HDF5 support")
    
    try:
        with h5py.File(filepath, 'r') as f:
            # Look for common dataset names
            dataset_names = ['iq', 'iq_data', 'samples', 'data', 'signal']
            dataset = None
            
            for name in dataset_names:
                if name in f:
                    dataset = f[name]
                    break
            
            if dataset is None:
                # Use first dataset found
                if len(f.keys()) > 0:
                    dataset = f[list(f.keys())[0]]
                else:
                    raise FileIOError("No datasets found in HDF5 file")
            
            # Handle slicing
            end_idx = None
            if max_samples is not None:
                end_idx = offset_samples + max_samples
            
            data = dataset[offset_samples:end_idx]
            
            # Convert to complex64 if needed
            if data.dtype != np.complex64:
                data = data.astype(np.complex64)
            
            return data
            
    except Exception as e:
        raise FileIOError(f"Failed to read HDF5 file: {e}") from e


def _read_sigmf_data(
    filepath: Path,
    max_samples: Optional[int],
    offset_samples: int
) -> IQSamples:
    """Read SigMF format data."""
    # SigMF data file
    data_file = filepath
    meta_file = filepath.with_suffix('.sigmf-meta')
    
    if not meta_file.exists():
        # Try to find metadata file
        meta_file = filepath.parent / (filepath.stem + '.sigmf-meta')
    
    if not meta_file.exists():
        logger.warning(f"SigMF metadata file not found for {filepath}")
        # Try to read as raw complex64
        return _read_complex64(filepath, max_samples, offset_samples, None)
    
    # Load metadata
    metadata = _load_metadata_file(meta_file)
    
    # Get data format from metadata
    global_meta = metadata.get('global', {})
    datatype = global_meta.get('core:datatype', 'cf32_le')
    
    # Map SigMF datatypes to our readers
    if datatype in ['cf32_le', 'cf32_be']:
        return _read_complex64(data_file, max_samples, offset_samples, None)
    elif datatype in ['cf64_le', 'cf64_be']:
        return _read_complex128(data_file, max_samples, offset_samples, None)
    elif datatype in ['ci16_le', 'ci16_be']:
        return _read_int16_iq(data_file, max_samples, offset_samples, None)
    else:
        logger.warning(f"Unknown SigMF datatype: {datatype}, trying as cf32")
        return _read_complex64(data_file, max_samples, offset_samples, None)


# =============================================================================
# HIGH-LEVEL WRITE FUNCTIONS
# =============================================================================

def write_iq_file(
    filepath: FilePath,
    iq_samples: IQSamples,
    file_format: FileFormat = FileFormat.COMPLEX64,
    compression: CompressionType = CompressionType.NONE,
    metadata: Optional[Dict[str, Any]] = None,
    progress_callback: ProgressCallback = None
) -> None:
    """
    Write IQ samples to file with specified format.
    
    Args:
        filepath: Output file path
        iq_samples: Complex IQ samples to write
        file_format: Output format
        compression: Compression type
        metadata: Optional metadata to save
        progress_callback: Optional progress callback
        
    Raises:
        FileIOError: If file cannot be written
    """
    filepath = Path(filepath)
    
    # Create directory if needed
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Writing IQ file: {filepath} (format: {file_format.value})")
    
    try:
        if file_format == FileFormat.COMPLEX64:
            _write_complex64(filepath, iq_samples, compression, progress_callback)
        elif file_format == FileFormat.COMPLEX128:
            _write_complex128(filepath, iq_samples, compression, progress_callback)
        elif file_format == FileFormat.INT16_IQ:
            _write_int16_iq(filepath, iq_samples, compression, progress_callback)
        elif file_format == FileFormat.FLOAT32_IQ:
            _write_float32_iq(filepath, iq_samples, compression, progress_callback)
        elif file_format == FileFormat.WAV:
            _write_wav_file(filepath, iq_samples, metadata)
        elif file_format == FileFormat.HDF5:
            _write_hdf5_iq(filepath, iq_samples, metadata)
        elif file_format == FileFormat.SIGMF:
            _write_sigmf_data(filepath, iq_samples, metadata)
        else:
            raise FileIOError(f"Unsupported output format: {file_format}")
        
        # Save metadata if provided
        if metadata:
            metadata_file = _get_metadata_filename(filepath, file_format)
            if metadata_file:
                # Add file information to metadata
                enhanced_metadata = metadata.copy()
                enhanced_metadata.update({
                    'samples': len(iq_samples),
                    'format': file_format.value,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'checksum': _calculate_checksum(filepath)
                })
                
                _save_metadata_file(metadata_file, enhanced_metadata)
        
        logger.info(f"Successfully wrote {len(iq_samples):,} samples to {filepath}")
        
    except Exception as e:
        raise FileIOError(f"Failed to write IQ file {filepath}: {e}") from e


def _write_complex64(
    filepath: Path,
    iq_samples: IQSamples,
    compression: CompressionType,
    progress_callback: ProgressCallback
) -> None:
    """Write complex64 format file."""
    # Add compression extension if needed
    if compression != CompressionType.NONE:
        if compression == CompressionType.GZIP:
            filepath = filepath.with_suffix(filepath.suffix + '.gz')
        elif compression == CompressionType.BZIP2:
            filepath = filepath.with_suffix(filepath.suffix + '.bz2')
        elif compression == CompressionType.LZMA:
            filepath = filepath.with_suffix(filepath.suffix + '.xz')
    
    with _open_file(filepath, 'wb') as f:
        # Convert to complex64 if needed
        if iq_samples.dtype != np.complex64:
            iq_samples = iq_samples.astype(np.complex64)
        
        # Write in chunks for large files
        chunk_size = 1024 * 1024  # 1M samples per chunk
        samples_written = 0
        total_samples = len(iq_samples)
        
        for i in range(0, total_samples, chunk_size):
            chunk = iq_samples[i:i + chunk_size]
            f.write(chunk.tobytes())
            samples_written += len(chunk)
            
            # Progress callback
            if progress_callback:
                progress = samples_written / total_samples
                progress_callback(progress)


def _write_complex128(
    filepath: Path,
    iq_samples: IQSamples,
    compression: CompressionType,
    progress_callback: ProgressCallback
) -> None:
    """Write complex128 format file."""
    # Convert to complex128
    samples_complex128 = iq_samples.astype(np.complex128)
    
    with _open_file(filepath, 'wb') as f:
        f.write(samples_complex128.tobytes())


def _write_int16_iq(
    filepath: Path,
    iq_samples: IQSamples,
    compression: CompressionType,
    progress_callback: ProgressCallback
) -> None:
    """Write int16 interleaved IQ format."""
    # Convert to int16 range
    i_samples = np.real(iq_samples) * 32767
    q_samples = np.imag(iq_samples) * 32767
    
    # Clip to int16 range
    i_samples = np.clip(i_samples, -32768, 32767).astype(np.int16)
    q_samples = np.clip(q_samples, -32768, 32767).astype(np.int16)
    
    # Interleave I and Q
    interleaved = np.zeros(len(iq_samples) * 2, dtype=np.int16)
    interleaved[0::2] = i_samples
    interleaved[1::2] = q_samples
    
    with _open_file(filepath, 'wb') as f:
        f.write(interleaved.tobytes())


def _write_float32_iq(
    filepath: Path,
    iq_samples: IQSamples,
    compression: CompressionType,
    progress_callback: ProgressCallback
) -> None:
    """Write float32 interleaved IQ format."""
    # Extract I and Q components
    i_samples = np.real(iq_samples).astype(np.float32)
    q_samples = np.imag(iq_samples).astype(np.float32)
    
    # Interleave I and Q
    interleaved = np.zeros(len(iq_samples) * 2, dtype=np.float32)
    interleaved[0::2] = i_samples
    interleaved[1::2] = q_samples
    
    with _open_file(filepath, 'wb') as f:
        f.write(interleaved.tobytes())


def _write_wav_file(
    filepath: Path,
    iq_samples: IQSamples,
    metadata: Optional[Dict[str, Any]]
) -> None:
    """Write WAV file format."""
    if not SCIPY_AVAILABLE:
        raise FileIOError("scipy required for WAV file support")
    
    # Extract sample rate from metadata
    sample_rate = 44100  # Default
    if metadata and 'sample_rate' in metadata:
        sample_rate = int(metadata['sample_rate'])
    
    # Convert to stereo int16 (I/Q channels)
    i_samples = np.real(iq_samples) * 32767
    q_samples = np.imag(iq_samples) * 32767
    
    # Clip to int16 range
    i_samples = np.clip(i_samples, -32768, 32767).astype(np.int16)
    q_samples = np.clip(q_samples, -32768, 32767).astype(np.int16)
    
    # Create stereo array
    stereo_data = np.column_stack([i_samples, q_samples])
    
    wavfile.write(filepath, sample_rate, stereo_data)


def _write_hdf5_iq(
    filepath: Path,
    iq_samples: IQSamples,
    metadata: Optional[Dict[str, Any]]
) -> None:
    """Write HDF5 format file."""
    if not HDF5_AVAILABLE:
        raise FileIOError("h5py required for HDF5 support")
    
    with h5py.File(filepath, 'w') as f:
        # Create main dataset
        dset = f.create_dataset('iq_data', data=iq_samples, compression='gzip')
        
        # Add metadata as attributes
        if metadata:
            for key, value in metadata.items():
                try:
                    dset.attrs[key] = value
                except (TypeError, ValueError):
                    # Convert to string if can't store directly
                    dset.attrs[key] = str(value)


def _write_sigmf_data(
    filepath: Path,
    iq_samples: IQSamples,
    metadata: Optional[Dict[str, Any]]
) -> None:
    """Write SigMF format files."""
    # Write data file
    data_file = filepath.with_suffix('.sigmf-data')
    _write_complex64(data_file, iq_samples, CompressionType.NONE, None)
    
    # Create SigMF metadata
    sigmf_meta = {
        "global": {
            "core:datatype": "cf32_le",
            "core:sample_rate": metadata.get('sample_rate', 1.0) if metadata else 1.0,
            "core:version": "1.0.0"
        },
        "captures": [
            {
                "core:sample_start": 0,
                "core:frequency": metadata.get('center_frequency', 0.0) if metadata else 0.0,
                "core:datetime": datetime.now(timezone.utc).isoformat()
            }
        ],
        "annotations": []
    }
    
    # Add additional metadata
    if metadata:
        for key, value in metadata.items():
            if key not in ['sample_rate', 'center_frequency']:
                sigmf_meta['global'][f'user:{key}'] = value
    
    # Write metadata file
    meta_file = filepath.with_suffix('.sigmf-meta')
    _save_metadata_file(meta_file, sigmf_meta)


def _get_metadata_filename(filepath: Path, file_format: FileFormat) -> Optional[Path]:
    """Get appropriate metadata filename for format."""
    if file_format == FileFormat.SIGMF:
        return None  # Handled separately
    elif file_format == FileFormat.HDF5:
        return None  # Metadata embedded
    else:
        return filepath.with_suffix('.json')


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_file_info(filepath: FilePath) -> FileInfo:
    """
    Get comprehensive file information.
    
    Args:
        filepath: Path to file
        
    Returns:
        FileInfo object with file details
    """
    return FileInfo(filepath)


def list_iq_files(
    directory: FilePath,
    recursive: bool = False,
    include_compressed: bool = True
) -> List[FileInfo]:
    """
    List all IQ files in directory.
    
    Args:
        directory: Directory to search
        recursive: Search subdirectories
        include_compressed: Include compressed files
        
    Returns:
        List of FileInfo objects
    """
    directory = Path(directory)
    pattern = "**/*" if recursive else "*"
    
    iq_extensions = ['.iq', '.raw', '.bin', '.cfile', '.cf32', '.cf64', '.cs16']
    if include_compressed:
        iq_extensions.extend(['.iq.gz', '.raw.gz', '.bin.gz'])
    
    iq_files = []
    for filepath in directory.glob(pattern):
        if filepath.is_file() and any(str(filepath).lower().endswith(ext) for ext in iq_extensions):
            iq_files.append(FileInfo(filepath))
    
    return sorted(iq_files, key=lambda x: x.filepath)


def convert_file_format(
    input_filepath: FilePath,
    output_filepath: FilePath,
    output_format: FileFormat,
    compression: CompressionType = CompressionType.NONE,
    progress_callback: ProgressCallback = None
) -> None:
    """
    Convert IQ file between formats.
    
    Args:
        input_filepath: Input file path
        output_filepath: Output file path
        output_format: Target format
        compression: Compression type
        progress_callback: Optional progress callback
    """
    # Read input file
    iq_samples = read_iq_file(input_filepath, progress_callback=progress_callback)
    
    # Get input metadata
    input_info = get_file_info(input_filepath)
    metadata = input_info.metadata.copy() if input_info.metadata else {}
    
    # Add file info to metadata
    if input_info.sample_rate:
        metadata['sample_rate'] = input_info.sample_rate
    if input_info.center_frequency:
        metadata['center_frequency'] = input_info.center_frequency
    
    # Write output file
    write_iq_file(
        output_filepath,
        iq_samples,
        output_format,
        compression,
        metadata,
        progress_callback
    )


def verify_file_integrity(filepath: FilePath, expected_checksum: Optional[str] = None) -> bool:
    """
    Verify file integrity using checksum.
    
    Args:
        filepath: Path to file
        expected_checksum: Expected checksum (or load from metadata)
        
    Returns:
        True if file is valid
    """
    try:
        if expected_checksum is None:
            # Try to load from metadata
            info = get_file_info(filepath)
            expected_checksum = info.metadata.get('checksum')
            
            if expected_checksum is None:
                logger.warning("No expected checksum available")
                return True  # Can't verify without expected value
        
        actual_checksum = _calculate_checksum(Path(filepath))
        return actual_checksum == expected_checksum
        
    except Exception as e:
        logger.error(f"Error verifying file integrity: {e}")
        return False


# =============================================================================
# STREAMING FUNCTIONS
# =============================================================================

def stream_iq_file(
    filepath: FilePath,
    chunk_size: int = 1024*1024,
    format_hint: Optional[FileFormat] = None
) -> Generator[IQSamples, None, None]:
    """
    Stream IQ file in chunks for memory-efficient processing.
    
    Args:
        filepath: Path to IQ file
        chunk_size: Number of samples per chunk
        format_hint: Optional format hint
        
    Yields:
        Chunks of IQ samples
    """
    filepath = Path(filepath)
    file_format = format_hint or _detect_file_format(filepath)
    
    if file_format != FileFormat.COMPLEX64:
        # For non-complex64 formats, read entire file and chunk
        iq_samples = read_iq_file(filepath, format_hint)
        for i in range(0, len(iq_samples), chunk_size):
            yield iq_samples[i:i + chunk_size]
    else:
        # Stream complex64 directly
        with _open_file(filepath, 'rb') as f:
            while True:
                data = f.read(chunk_size * 8)  # 8 bytes per complex64 sample
                if not data:
                    break
                
                chunk = np.frombuffer(data, dtype=np.complex64)
                if len(chunk) > 0:
                    yield chunk


# Backward compatibility functions (maintain original interface)
def read_iq_file_simple(filepath: str) -> np.ndarray:
    """Simple IQ file reader (backward compatible)."""
    return read_iq_file(filepath)


def write_iq_file_simple(filepath: str, data: np.ndarray) -> None:
    """Simple IQ file writer (backward compatible)."""
    write_iq_file(filepath, data)


# Example usage
if __name__ == "__main__":
    print("=== Enhanced File I/O Demo ===")
    
    # Create test data
    import numpy as np
    test_samples = (np.random.randn(10000) + 1j * np.random.randn(10000)).astype(np.complex64)
    
    # Test writing and reading
    test_file = Path("test_iq_file.iq")
    
    print(f"Writing {len(test_samples)} samples to {test_file}")
    write_iq_file(test_file, test_samples, metadata={'sample_rate': 2.048e6, 'center_frequency': 100.1e6})
    
    print(f"Reading back from {test_file}")
    read_samples = read_iq_file(test_file)
    
    print(f"Read {len(read_samples)} samples")
    print(f"Data integrity: {np.allclose(test_samples, read_samples)}")
    
    # Get file info
    info = get_file_info(test_file)
    print(f"File info: {info.format.value}, {info.size_bytes} bytes")
    
    # Clean up
    test_file.unlink()
    metadata_file = test_file.with_suffix('.json')
    if metadata_file.exists():
        metadata_file.unlink()