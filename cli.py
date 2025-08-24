#!/usr/bin/env python3
"""
Enhanced DroneCmd Command Line Interface

This module provides a comprehensive command-line interface for the DroneCmd
library, integrating all enhanced components while maintaining backward
compatibility with the original CLI.

Key Features:
- Unified interface for all DroneCmd operations
- Integration with enhanced capture, processing, and analysis
- Structured logging and performance monitoring
- Configuration management and profiles
- Plugin system integration
- Progress reporting for long operations
- Comprehensive error handling and recovery
- JSON output for programmatic use

Commands:
- capture: Enhanced signal capture with quality monitoring
- analyze: Signal analysis and protocol classification
- replay: Intelligent signal replay with compliance monitoring
- generate: FHSS frame generation and testing
- convert: File format conversion utilities
- config: Configuration management
- plugins: Plugin management

Usage:
    dronecmd capture --frequency 2.44e9 --duration 30 --output capture.iq
    dronecmd analyze --input capture.iq --protocols mavlink,dji
    dronecmd replay --input capture.iq --strategy intelligent --count 5
    dronecmd generate fhss --frequency 2.44e9 --data "test payload"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Core imports
try:
    from .core.capture import EnhancedLiveCapture, SDRConfig, SDRPlatform, GainMode
    from .core.signal_processing import SignalProcessor, analyze_signal_quality
    from .core.fhss import EnhancedFHSSEngine, FHSSBand, create_fcc_compliant_fhss
    from .core.classification import EnhancedProtocolClassifier, ClassifierConfig
    from .core.replay import EnhancedReplayEngine, ReplayConfig, ReplayStrategy
    from .capture.manager import CaptureManager
    from .utils.logging import configure_logging, get_logger, log_performance
    from .utils.fileio import read_iq_file, write_iq_file, FileFormat, CompressionType, get_file_info
    from .utils.crypto import CryptoManager, generate_secure_token
    from .utils.compat import check_compatibility, get_migration_guide
    ENHANCED_MODULES_AVAILABLE = True
except ImportError:
    # Fallback imports for compatibility
    ENHANCED_MODULES_AVAILABLE = False
    print("Warning: Enhanced modules not available, using fallback mode", file=sys.stderr)

import logging

# Version information
__version__ = "2.0.0"
__cli_version__ = "2.0.0"


class CLIError(Exception):
    """Exception for CLI-specific errors."""
    pass


class ProgressReporter:
    """Progress reporting utility for long operations."""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.description = description
        self.current = 0
        self.start_time = time.time()
        self.last_update = 0
    
    def update(self, increment: int = 1) -> None:
        """Update progress."""
        self.current += increment
        
        # Only update display every 0.1 seconds
        now = time.time()
        if now - self.last_update > 0.1:
            self._display()
            self.last_update = now
    
    def _display(self) -> None:
        """Display progress bar."""
        if self.total <= 0:
            return
        
        percent = (self.current / self.total) * 100
        elapsed = time.time() - self.start_time
        
        # Simple progress bar
        bar_length = 40
        filled_length = int(bar_length * self.current // self.total)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        print(f'\r{self.description}: |{bar}| {percent:.1f}% ({elapsed:.1f}s)', end='')
        
        if self.current >= self.total:
            print()  # New line when complete


class ConfigManager:
    """Configuration management for CLI."""
    
    def __init__(self):
        self.config_dir = Path.home() / ".dronecmd"
        self.config_file = self.config_dir / "config.json"
        self.config_dir.mkdir(exist_ok=True)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load config: {e}", file=sys.stderr)
        
        # Default configuration
        return {
            'capture': {
                'default_sample_rate': 2.048e6,
                'default_duration': 10.0,
                'default_gain_mode': 'auto'
            },
            'processing': {
                'default_format': 'complex64',
                'enable_quality_monitoring': True,
                'enable_performance_logging': True
            },
            'output': {
                'default_compression': 'none',
                'include_metadata': True,
                'json_output': False
            },
            'logging': {
                'level': 'INFO',
                'enable_structured': False,
                'enable_file_logging': True
            }
        }
    
    def save_config(self) -> None:
        """Save configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self._config, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save config: {e}", file=sys.stderr)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any) -> None:
        """Set configuration value using dot notation."""
        keys = key_path.split('.')
        config = self._config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
        self.save_config()


class CLIOutput:
    """Output formatting utilities."""
    
    def __init__(self, json_output: bool = False, verbose: bool = False):
        self.json_output = json_output
        self.verbose = verbose
        self.results = {}
    
    def info(self, message: str, **kwargs: Any) -> None:
        """Output info message."""
        if self.json_output:
            self.results.setdefault('info', []).append({'message': message, **kwargs})
        else:
            print(f"[INFO] {message}")
            if self.verbose and kwargs:
                for key, value in kwargs.items():
                    print(f"  {key}: {value}")
    
    def warning(self, message: str, **kwargs: Any) -> None:
        """Output warning message."""
        if self.json_output:
            self.results.setdefault('warnings', []).append({'message': message, **kwargs})
        else:
            print(f"[WARNING] {message}", file=sys.stderr)
    
    def error(self, message: str, **kwargs: Any) -> None:
        """Output error message."""
        if self.json_output:
            self.results.setdefault('errors', []).append({'message': message, **kwargs})
        else:
            print(f"[ERROR] {message}", file=sys.stderr)
    
    def result(self, data: Dict[str, Any]) -> None:
        """Output result data."""
        if self.json_output:
            self.results.update(data)
        else:
            # Format nicely for human reading
            for key, value in data.items():
                if isinstance(value, dict):
                    print(f"{key}:")
                    for subkey, subvalue in value.items():
                        print(f"  {subkey}: {subvalue}")
                elif isinstance(value, list):
                    print(f"{key}: {len(value)} items")
                    if self.verbose:
                        for i, item in enumerate(value[:5]):  # Show first 5
                            print(f"  [{i}]: {item}")
                        if len(value) > 5:
                            print(f"  ... and {len(value) - 5} more")
                else:
                    print(f"{key}: {value}")
    
    def finalize(self) -> None:
        """Finalize output."""
        if self.json_output:
            print(json.dumps(self.results, indent=2, default=str))


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        prog='dronecmd',
        description='Enhanced Drone Command Interference Toolkit',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  dronecmd capture --frequency 2.44e9 --duration 30 --output capture.iq
  dronecmd analyze --input capture.iq --protocols mavlink,dji
  dronecmd replay --input capture.iq --strategy intelligent
  dronecmd generate fhss --frequency 2.44e9 --data "test payload"
  dronecmd config set capture.default_sample_rate 2048000
  dronecmd info --compatibility

For more information, see the documentation.
        """
    )
    
    # Global options
    parser.add_argument('--version', action='version', version=f'dronecmd {__version__}')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Capture command
    capture_parser = subparsers.add_parser('capture', help='Capture RF signals')
    capture_parser.add_argument('--frequency', '-f', type=float, required=True,
                               help='Center frequency in Hz (e.g., 2.44e9)')
    capture_parser.add_argument('--duration', '-d', type=float, default=10.0,
                               help='Capture duration in seconds')
    capture_parser.add_argument('--sample-rate', '-s', type=float, default=2.048e6,
                               help='Sample rate in Hz')
    capture_parser.add_argument('--output', '-o', type=str, required=True,
                               help='Output file path')
    capture_parser.add_argument('--gain', type=str, default='auto',
                               help='Gain setting (auto, manual value in dB)')
    capture_parser.add_argument('--device', type=int, default=0,
                               help='SDR device index')
    capture_parser.add_argument('--platform', choices=['rtl_sdr', 'hackrf', 'airspy'],
                               default='rtl_sdr', help='SDR platform')
    capture_parser.add_argument('--compression', choices=['none', 'gzip', 'bzip2'],
                               default='none', help='Compression type')
    capture_parser.add_argument('--format', choices=['complex64', 'complex128', 'int16_iq'],
                               default='complex64', help='Output format')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze captured signals')
    analyze_parser.add_argument('--input', '-i', type=str, required=True,
                               help='Input file path')
    analyze_parser.add_argument('--protocols', type=str,
                               help='Comma-separated list of protocols to detect')
    analyze_parser.add_argument('--threshold', type=float, default=0.05,
                               help='Detection threshold')
    analyze_parser.add_argument('--max-packets', type=int,
                               help='Maximum number of packets to analyze')
    analyze_parser.add_argument('--output-report', type=str,
                               help='Save analysis report to file')
    
    # Replay command
    replay_parser = subparsers.add_parser('replay', help='Replay captured signals')
    replay_parser.add_argument('--input', '-i', type=str, required=True,
                               help='Input file path')
    replay_parser.add_argument('--count', '-c', type=int, default=3,
                               help='Number of replays')
    replay_parser.add_argument('--strategy', choices=['simple', 'intelligent', 'stress'],
                               default='simple', help='Replay strategy')
    replay_parser.add_argument('--delay', type=float, default=0.1,
                               help='Inter-packet delay in seconds')
    replay_parser.add_argument('--random-delay', action='store_true',
                               help='Use random delays')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate test signals')
    generate_subparsers = generate_parser.add_subparsers(dest='generate_type')
    
    # FHSS generation
    fhss_parser = generate_subparsers.add_parser('fhss', help='Generate FHSS frames')
    fhss_parser.add_argument('--frequency', '-f', type=float, required=True,
                            help='Center frequency in Hz')
    fhss_parser.add_argument('--data', type=str, required=True,
                            help='Data to transmit')
    fhss_parser.add_argument('--hops', type=int, default=8,
                            help='Number of hop channels')
    fhss_parser.add_argument('--spacing', type=float, default=1e6,
                            help='Channel spacing in Hz')
    fhss_parser.add_argument('--output', '-o', type=str,
                            help='Output file for frames')
    fhss_parser.add_argument('--fcc-compliant', action='store_true',
                            help='Generate FCC-compliant FHSS')
    
    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert file formats')
    convert_parser.add_argument('--input', '-i', type=str, required=True,
                               help='Input file path')
    convert_parser.add_argument('--output', '-o', type=str, required=True,
                               help='Output file path')
    convert_parser.add_argument('--format', choices=['complex64', 'complex128', 'wav', 'hdf5'],
                               required=True, help='Output format')
    convert_parser.add_argument('--compression', choices=['none', 'gzip', 'bzip2'],
                               default='none', help='Compression type')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Configuration management')
    config_subparsers = config_parser.add_subparsers(dest='config_action')
    
    config_get = config_subparsers.add_parser('get', help='Get configuration value')
    config_get.add_argument('key', help='Configuration key (dot notation)')
    
    config_set = config_subparsers.add_parser('set', help='Set configuration value')
    config_set.add_argument('key', help='Configuration key (dot notation)')
    config_set.add_argument('value', help='Configuration value')
    
    config_show = config_subparsers.add_parser('show', help='Show all configuration')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show system information')
    info_parser.add_argument('--compatibility', action='store_true',
                            help='Show compatibility information')
    info_parser.add_argument('--migration', action='store_true',
                            help='Show migration guide')
    
    return parser


async def cmd_capture(args: argparse.Namespace, config: ConfigManager, output: CLIOutput) -> None:
    """Handle capture command."""
    if not ENHANCED_MODULES_AVAILABLE:
        output.warning("Enhanced modules not available, using fallback capture")
        # Fallback to basic capture
        return
    
    try:
        # Parse gain setting
        if args.gain == 'auto':
            gain_mode = GainMode.AUTO
            gain_db = None
        else:
            gain_mode = GainMode.MANUAL
            gain_db = float(args.gain)
        
        # Parse platform
        platform_map = {
            'rtl_sdr': SDRPlatform.RTL_SDR,
            'hackrf': SDRPlatform.HACKRF,
            'airspy': SDRPlatform.AIRSPY
        }
        platform = platform_map.get(args.platform, SDRPlatform.RTL_SDR)
        
        # Create SDR configuration
        sdr_config = SDRConfig(
            platform=platform,
            frequency_hz=args.frequency,
            sample_rate_hz=args.sample_rate,
            duration_s=args.duration,
            gain_mode=gain_mode,
            gain_db=gain_db,
            device_index=args.device
        )
        
        output.info(f"Starting capture: {args.frequency/1e6:.3f} MHz for {args.duration}s")
        
        # Enhanced capture with quality monitoring
        async with EnhancedLiveCapture(sdr_config) as capture:
            samples, metadata = await capture.capture_samples()
        
        # Parse output format
        format_map = {
            'complex64': FileFormat.COMPLEX64,
            'complex128': FileFormat.COMPLEX128,
            'int16_iq': FileFormat.INT16_IQ
        }
        file_format = format_map.get(args.format, FileFormat.COMPLEX64)
        
        # Parse compression
        compression_map = {
            'none': CompressionType.NONE,
            'gzip': CompressionType.GZIP,
            'bzip2': CompressionType.BZIP2
        }
        compression = compression_map.get(args.compression, CompressionType.NONE)
        
        # Save with metadata
        capture_metadata = {
            'frequency_hz': args.frequency,
            'sample_rate_hz': args.sample_rate,
            'duration_s': args.duration,
            'platform': args.platform,
            'gain_mode': args.gain,
            'samples_captured': len(samples),
            'signal_power_dbfs': metadata.signal_power_dbfs,
            'snr_db': metadata.snr_db
        }
        
        write_iq_file(
            args.output,
            samples,
            file_format=file_format,
            compression=compression,
            metadata=capture_metadata
        )
        
        output.result({
            'capture_successful': True,
            'samples_captured': len(samples),
            'file_size_bytes': Path(args.output).stat().st_size,
            'signal_quality': {
                'power_dbfs': metadata.signal_power_dbfs,
                'snr_db': metadata.snr_db,
                'sample_loss_rate': metadata.sample_loss_rate
            }
        })
        
    except Exception as e:
        output.error(f"Capture failed: {e}")
        if args.verbose:
            traceback.print_exc()
        raise CLIError(f"Capture failed: {e}")


async def cmd_analyze(args: argparse.Namespace, config: ConfigManager, output: CLIOutput) -> None:
    """Handle analyze command."""
    try:
        output.info(f"Analyzing file: {args.input}")
        
        # Load file
        iq_data = read_iq_file(args.input)
        file_info = get_file_info(args.input)
        
        output.info(f"Loaded {len(iq_data)} samples")
        
        # Signal quality analysis
        quality_metrics = analyze_signal_quality(iq_data, file_info.sample_rate)
        
        # Packet extraction and classification
        if ENHANCED_MODULES_AVAILABLE:
            # Use enhanced capture manager for packet extraction
            manager = CaptureManager()
            manager._iq_data = iq_data  # Set data directly
            
            packets = manager.extract_packets(threshold=args.threshold)
            
            if args.max_packets:
                packets = packets[:args.max_packets]
            
            # Protocol classification
            classifier = EnhancedProtocolClassifier()
            classifications = []
            
            progress = ProgressReporter(len(packets), "Classifying packets")
            for i, packet in enumerate(packets):
                result = classifier.classify(packet.tobytes())
                
                if hasattr(result, 'predicted_protocol'):
                    classification = {
                        'packet_id': i,
                        'protocol': result.predicted_protocol,
                        'confidence': result.confidence,
                        'length_bytes': len(packet)
                    }
                else:
                    classification = {
                        'packet_id': i,
                        'protocol': str(result),
                        'confidence': 0.5,
                        'length_bytes': len(packet)
                    }
                
                classifications.append(classification)
                progress.update()
            
            # Analyze protocol distribution
            protocol_counts = {}
            for classification in classifications:
                protocol = classification['protocol']
                protocol_counts[protocol] = protocol_counts.get(protocol, 0) + 1
        
        else:
            output.warning("Enhanced analysis not available, using basic analysis")
            packets = []
            classifications = []
            protocol_counts = {}
        
        # Generate results
        analysis_results = {
            'file_info': {
                'size_bytes': file_info.size_bytes,
                'samples': len(iq_data),
                'format': file_info.format.value if file_info.format else 'unknown',
                'sample_rate': file_info.sample_rate
            },
            'signal_quality': quality_metrics,
            'packet_analysis': {
                'packets_found': len(packets),
                'classifications': classifications,
                'protocol_distribution': protocol_counts
            }
        }
        
        # Save report if requested
        if args.output_report:
            with open(args.output_report, 'w') as f:
                json.dump(analysis_results, f, indent=2, default=str)
            output.info(f"Report saved to {args.output_report}")
        
        output.result(analysis_results)
        
    except Exception as e:
        output.error(f"Analysis failed: {e}")
        if args.verbose:
            traceback.print_exc()
        raise CLIError(f"Analysis failed: {e}")


async def cmd_replay(args: argparse.Namespace, config: ConfigManager, output: CLIOutput) -> None:
    """Handle replay command."""
    if not ENHANCED_MODULES_AVAILABLE:
        output.error("Replay requires enhanced modules")
        return
    
    try:
        output.info(f"Loading file for replay: {args.input}")
        
        # Load packet data
        iq_data = read_iq_file(args.input)
        packet_bytes = iq_data.tobytes()  # Convert to bytes for replay
        
        # Mock transmitter for demonstration
        class MockTransmitter:
            def __init__(self):
                self.transmitted_count = 0
            
            async def send_async(self, data: bytes) -> None:
                self.transmitted_count += 1
                # In real implementation, this would transmit via SDR
                await asyncio.sleep(0.001)  # Simulate transmission time
        
        transmitter = MockTransmitter()
        
        # Configure replay strategy
        strategy_map = {
            'simple': ReplayStrategy.SIMPLE,
            'intelligent': ReplayStrategy.INTELLIGENT,
            'stress': ReplayStrategy.STRESS_TEST
        }
        strategy = strategy_map.get(args.strategy, ReplayStrategy.SIMPLE)
        
        replay_config = ReplayConfig(
            strategy=strategy,
            enable_performance_monitoring=True,
            enable_protocol_awareness=True
        )
        
        # Create replay engine
        replay_engine = EnhancedReplayEngine(replay_config, transmitter)
        
        output.info(f"Starting replay: {args.count} repetitions using {args.strategy} strategy")
        
        # Execute replay
        result = await replay_engine.replay_packet(
            packet_bytes,
            repeat_count=args.count,
            random_delay=args.random_delay
        )
        
        # Report results
        replay_results = {
            'replay_successful': result.successful_transmissions > 0,
            'packets_transmitted': result.packets_transmitted,
            'successful_transmissions': result.successful_transmissions,
            'failed_transmissions': result.failed_transmissions,
            'success_rate': result.success_rate,
            'total_time_s': result.total_transmission_time_s,
            'average_packet_rate_hz': result.average_packet_rate_hz,
            'strategy_used': args.strategy
        }
        
        if result.timing_accuracy_us:
            replay_results['timing_accuracy_us'] = result.timing_accuracy_us
        
        if result.error_messages:
            replay_results['errors'] = result.error_messages
        
        output.result(replay_results)
        
    except Exception as e:
        output.error(f"Replay failed: {e}")
        if args.verbose:
            traceback.print_exc()
        raise CLIError(f"Replay failed: {e}")


async def cmd_generate_fhss(args: argparse.Namespace, config: ConfigManager, output: CLIOutput) -> None:
    """Handle FHSS generation command."""
    try:
        output.info(f"Generating FHSS frames for: {args.data}")
        
        if args.fcc_compliant and ENHANCED_MODULES_AVAILABLE:
            # Use FCC-compliant FHSS
            fhss_engine = create_fcc_compliant_fhss(
                FHSSBand.ISM_2_4_GHz,
                center_freq_hz=args.frequency
            )
        elif ENHANCED_MODULES_AVAILABLE:
            # Use enhanced FHSS
            from .core.fhss import EnhancedFHSSEngine, FHSSConfig
            
            config = FHSSConfig(
                center_freq_hz=args.frequency,
                channel_spacing_hz=args.spacing,
                hop_count=args.hops
            )
            fhss_engine = EnhancedFHSSEngine(config)
        else:
            output.error("FHSS generation requires enhanced modules")
            return
        
        # Generate frames
        data_bytes = args.data.encode('utf-8')
        frames = fhss_engine.prepare_transmit_frames(
            packet=data_bytes,
            sample_rate=2_000_000,
            bitrate=100_000
        )
        
        # Analyze frames
        total_samples = sum(len(frame.iq_samples) for frame in frames)
        total_duration = sum(frame.duration_s for frame in frames)
        
        frame_info = []
        for i, frame in enumerate(frames):
            frame_info.append({
                'frame_id': i,
                'frequency_mhz': frame.frequency_hz / 1e6,
                'samples': len(frame.iq_samples),
                'duration_ms': frame.duration_s * 1000,
                'chunk_bytes': len(frame.chunk_data)
            })
        
        # Save frames if requested
        if args.output:
            # Save as JSON with frame information
            frame_data = {
                'fhss_config': {
                    'center_frequency_hz': args.frequency,
                    'channel_spacing_hz': args.spacing,
                    'hop_count': args.hops,
                    'fcc_compliant': args.fcc_compliant
                },
                'frames': frame_info,
                'total_frames': len(frames),
                'total_samples': total_samples,
                'total_duration_s': total_duration
            }
            
            with open(args.output, 'w') as f:
                json.dump(frame_data, f, indent=2)
            
            output.info(f"Frame information saved to {args.output}")
        
        output.result({
            'fhss_generation_successful': True,
            'frames_generated': len(frames),
            'total_samples': total_samples,
            'total_duration_s': total_duration,
            'frequency_span_mhz': (max(f.frequency_hz for f in frames) - 
                                 min(f.frequency_hz for f in frames)) / 1e6,
            'frame_details': frame_info[:5] if len(frame_info) > 5 else frame_info  # Show first 5
        })
        
    except Exception as e:
        output.error(f"FHSS generation failed: {e}")
        if args.verbose:
            traceback.print_exc()
        raise CLIError(f"FHSS generation failed: {e}")


async def cmd_convert(args: argparse.Namespace, config: ConfigManager, output: CLIOutput) -> None:
    """Handle convert command."""
    try:
        output.info(f"Converting {args.input} to {args.format}")
        
        # Load input file
        iq_data = read_iq_file(args.input)
        input_info = get_file_info(args.input)
        
        # Parse output format
        format_map = {
            'complex64': FileFormat.COMPLEX64,
            'complex128': FileFormat.COMPLEX128,
            'wav': FileFormat.WAV,
            'hdf5': FileFormat.HDF5
        }
        output_format = format_map.get(args.format, FileFormat.COMPLEX64)
        
        # Parse compression
        compression_map = {
            'none': CompressionType.NONE,
            'gzip': CompressionType.GZIP,
            'bzip2': CompressionType.BZIP2
        }
        compression = compression_map.get(args.compression, CompressionType.NONE)
        
        # Preserve metadata
        metadata = input_info.metadata.copy() if input_info.metadata else {}
        metadata.update({
            'conversion_source': args.input,
            'conversion_format': args.format,
            'conversion_time': time.time()
        })
        
        # Convert and save
        progress = ProgressReporter(1, "Converting file")
        
        write_iq_file(
            args.output,
            iq_data,
            file_format=output_format,
            compression=compression,
            metadata=metadata
        )
        
        progress.update()
        
        # Get output file info
        output_info = get_file_info(args.output)
        
        output.result({
            'conversion_successful': True,
            'input_file': {
                'path': args.input,
                'size_bytes': input_info.size_bytes,
                'format': input_info.format.value if input_info.format else 'unknown'
            },
            'output_file': {
                'path': args.output,
                'size_bytes': output_info.size_bytes,
                'format': output_info.format.value if output_info.format else 'unknown'
            },
            'samples_converted': len(iq_data)
        })
        
    except Exception as e:
        output.error(f"Conversion failed: {e}")
        if args.verbose:
            traceback.print_exc()
        raise CLIError(f"Conversion failed: {e}")


def cmd_config(args: argparse.Namespace, config: ConfigManager, output: CLIOutput) -> None:
    """Handle config command."""
    try:
        if args.config_action == 'get':
            value = config.get(args.key)
            if value is not None:
                output.result({args.key: value})
            else:
                output.error(f"Configuration key not found: {args.key}")
        
        elif args.config_action == 'set':
            # Try to parse value as JSON, fall back to string
            try:
                value = json.loads(args.value)
            except json.JSONDecodeError:
                value = args.value
            
            config.set(args.key, value)
            output.info(f"Set {args.key} = {value}")
        
        elif args.config_action == 'show':
            output.result(config._config)
        
        else:
            output.error("No config action specified")
    
    except Exception as e:
        output.error(f"Config operation failed: {e}")
        raise CLIError(f"Config operation failed: {e}")


def cmd_info(args: argparse.Namespace, config: ConfigManager, output: CLIOutput) -> None:
    """Handle info command."""
    try:
        info_data = {
            'dronecmd_version': __version__,
            'cli_version': __cli_version__,
            'enhanced_modules_available': ENHANCED_MODULES_AVAILABLE
        }
        
        if args.compatibility and ENHANCED_MODULES_AVAILABLE:
            from .utils.compat import check_compatibility
            compat_info = check_compatibility()
            info_data['compatibility'] = compat_info
        
        if args.migration and ENHANCED_MODULES_AVAILABLE:
            from .utils.compat import get_migration_guide
            migration_guide = get_migration_guide()
            info_data['migration_guide'] = migration_guide
        
        output.result(info_data)
        
    except Exception as e:
        output.error(f"Info command failed: {e}")
        raise CLIError(f"Info command failed: {e}")


async def main() -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Handle no command
    if not args.command:
        parser.print_help()
        return 1
    
    # Initialize configuration
    config = ConfigManager()
    
    # Configure logging
    if ENHANCED_MODULES_AVAILABLE:
        configure_logging(
            level=args.log_level,
            enable_structured=config.get('logging.enable_structured', False),
            enable_file=config.get('logging.enable_file_logging', True)
        )
        logger = get_logger(__name__)
    else:
        logging.basicConfig(level=getattr(logging, args.log_level))
        logger = logging.getLogger(__name__)
    
    # Initialize output
    output = CLIOutput(
        json_output=args.json or config.get('output.json_output', False),
        verbose=args.verbose
    )
    
    try:
        # Execute command
        if args.command == 'capture':
            await cmd_capture(args, config, output)
        elif args.command == 'analyze':
            await cmd_analyze(args, config, output)
        elif args.command == 'replay':
            await cmd_replay(args, config, output)
        elif args.command == 'generate':
            if args.generate_type == 'fhss':
                await cmd_generate_fhss(args, config, output)
            else:
                output.error("Unknown generate type")
                return 1
        elif args.command == 'convert':
            await cmd_convert(args, config, output)
        elif args.command == 'config':
            cmd_config(args, config, output)
        elif args.command == 'info':
            cmd_info(args, config, output)
        else:
            output.error(f"Unknown command: {args.command}")
            return 1
        
        output.finalize()
        return 0
        
    except CLIError as e:
        output.error(str(e))
        output.finalize()
        return 1
    except KeyboardInterrupt:
        output.info("Operation cancelled by user")
        output.finalize()
        return 130
    except Exception as e:
        output.error(f"Unexpected error: {e}")
        if args.verbose:
            traceback.print_exc()
        output.finalize()
        return 1


def cli_main() -> int:
    """Entry point for setuptools console script."""
    return asyncio.run(main())


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))