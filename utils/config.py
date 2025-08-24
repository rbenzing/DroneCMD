#!/usr/bin/env python3
"""
Enhanced Configuration Management

This module provides comprehensive configuration management for the dronecmd library,
supporting multiple configuration sources, validation, and environment-specific settings.
It consolidates and enhances the original config.py with advanced features while
maintaining backward compatibility.

Key Features:
- Multiple configuration sources (files, environment, defaults)
- Configuration validation and type checking
- Environment-specific configurations (dev, test, prod)
- Dynamic configuration updates
- Backward compatibility with original config.py
- Integration with enhanced core systems

Usage:
    >>> from dronecmd.utils.config import config, ConfigManager
    >>> 
    >>> # Simple usage (backward compatible)
    >>> sample_rate = config.SAMPLE_RATE
    >>> 
    >>> # Advanced usage
    >>> manager = ConfigManager()
    >>> manager.load_from_file("custom_config.yaml")
    >>> sdr_config = manager.get_sdr_config()
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Type, TypeVar

# Configure module logger
logger = logging.getLogger(__name__)

# Type aliases
ConfigValue = Union[str, int, float, bool, List[Any], Dict[str, Any]]
ConfigDict = Dict[str, ConfigValue]

T = TypeVar('T')


class Environment(Enum):
    """Configuration environments."""
    
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"
    RESEARCH = "research"


class ConfigSource(Enum):
    """Configuration sources in priority order."""
    
    ENVIRONMENT = "environment"    # Highest priority
    FILE = "file"
    DEFAULTS = "defaults"         # Lowest priority


@dataclass
class SDRConfig:
    """SDR-specific configuration."""
    
    # Basic SDR settings
    sample_rate: int = 2_000_000
    default_frequency: float = 100.1e6
    frequency_range: tuple[float, float] = (24e6, 1.75e9)
    
    # Capture settings
    capture_duration: int = 10
    buffer_size: int = 262144
    auto_gain: bool = True
    default_gain: Optional[float] = None
    
    # File paths
    capture_path: str = "captures"
    iq_file_format: str = "complex64"
    
    # Hardware settings
    preferred_platform: str = "rtl_sdr"
    device_index: int = 0
    frequency_correction_ppm: float = 0.0
    
    def validate(self) -> List[str]:
        """Validate SDR configuration."""
        errors = []
        
        if self.sample_rate <= 0:
            errors.append("sample_rate must be positive")
        
        if self.sample_rate > 50e6:
            errors.append("sample_rate too high (>50MHz)")
        
        if not (self.frequency_range[0] <= self.default_frequency <= self.frequency_range[1]):
            errors.append("default_frequency outside frequency_range")
        
        if self.capture_duration <= 0:
            errors.append("capture_duration must be positive")
        
        if self.buffer_size <= 0 or (self.buffer_size & (self.buffer_size - 1)) != 0:
            errors.append("buffer_size must be positive power of 2")
        
        return errors


@dataclass
class AnalysisConfig:
    """Analysis and processing configuration."""
    
    # Protocol analysis
    enable_protocol_detection: bool = True
    enable_classification: bool = True
    classification_confidence_threshold: float = 0.7
    
    # Signal processing
    enable_advanced_demodulation: bool = True
    enable_signal_quality_monitoring: bool = True
    noise_floor_estimation_method: str = "percentile"
    
    # Detection settings
    packet_detection_threshold: float = 0.05
    min_packet_length: int = 1000
    max_packets_per_capture: int = 1000
    
    # Performance settings
    enable_caching: bool = True
    max_cache_size_mb: int = 100
    processing_threads: int = 4
    
    def validate(self) -> List[str]:
        """Validate analysis configuration."""
        errors = []
        
        if not (0.0 <= self.classification_confidence_threshold <= 1.0):
            errors.append("classification_confidence_threshold must be in [0,1]")
        
        if not (0.0 <= self.packet_detection_threshold <= 1.0):
            errors.append("packet_detection_threshold must be in [0,1]")
        
        if self.min_packet_length <= 0:
            errors.append("min_packet_length must be positive")
        
        if self.processing_threads <= 0:
            errors.append("processing_threads must be positive")
        
        return errors


@dataclass
class InjectionConfig:
    """Injection system configuration."""
    
    # Safety limits
    max_transmission_power_dbm: float = 10.0
    max_injection_duration_s: float = 60.0
    enable_safety_monitoring: bool = True
    
    # Timing settings
    default_inter_packet_delay_s: float = 0.1
    enable_timing_jitter: bool = False
    jitter_range_s: tuple[float, float] = (0.05, 0.2)
    
    # Obfuscation settings
    enable_obfuscation: bool = False
    default_stealth_level: str = "medium"
    enable_protocol_mimicry: bool = False
    
    # Compliance
    enable_fcc_compliance_checking: bool = True
    authorized_laboratory_only: bool = True
    
    def validate(self) -> List[str]:
        """Validate injection configuration."""
        errors = []
        
        if self.max_transmission_power_dbm < 0:
            errors.append("max_transmission_power_dbm must be non-negative")
        
        if self.max_transmission_power_dbm > 30.0:
            errors.append("max_transmission_power_dbm >30dBm may violate regulations")
        
        if self.max_injection_duration_s <= 0:
            errors.append("max_injection_duration_s must be positive")
        
        if self.default_inter_packet_delay_s < 0:
            errors.append("default_inter_packet_delay_s must be non-negative")
        
        if self.jitter_range_s[0] >= self.jitter_range_s[1]:
            errors.append("invalid jitter_range_s")
        
        return errors


@dataclass
class PathConfig:
    """File and directory path configuration."""
    
    # Base directories
    project_root: str = "."
    data_dir: str = "data"
    logs_dir: str = "logs"
    cache_dir: str = "cache"
    config_dir: str = "config"
    
    # Specific paths
    capture_path: str = "captures"
    model_path: str = "data/models"
    plugin_dir: str = "plugins"
    
    # File patterns
    iq_file_extension: str = ".iq"
    log_file_pattern: str = "dronecmd_{date}.log"
    config_file_pattern: str = "config_{env}.{ext}"
    
    def get_absolute_path(self, relative_path: str) -> Path:
        """Get absolute path from relative path."""
        return Path(self.project_root).resolve() / relative_path
    
    def ensure_directories(self) -> None:
        """Ensure all configured directories exist."""
        directories = [
            self.data_dir, self.logs_dir, self.cache_dir,
            self.config_dir, self.capture_path, self.model_path, self.plugin_dir
        ]
        
        for directory in directories:
            abs_path = self.get_absolute_path(directory)
            abs_path.mkdir(parents=True, exist_ok=True)


@dataclass
class LoggingConfig:
    """Logging configuration."""
    
    # Log levels
    console_level: str = "INFO"
    file_level: str = "DEBUG"
    library_level: str = "INFO"
    
    # Log format
    console_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    
    # File settings
    enable_file_logging: bool = True
    log_file_max_size_mb: int = 10
    log_file_backup_count: int = 5
    log_rotation: bool = True
    
    # Performance
    enable_async_logging: bool = False
    log_buffer_size: int = 1000
    
    def validate(self) -> List[str]:
        """Validate logging configuration."""
        errors = []
        
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        
        if self.console_level not in valid_levels:
            errors.append(f"invalid console_level: {self.console_level}")
        
        if self.file_level not in valid_levels:
            errors.append(f"invalid file_level: {self.file_level}")
        
        if self.log_file_max_size_mb <= 0:
            errors.append("log_file_max_size_mb must be positive")
        
        if self.log_file_backup_count < 0:
            errors.append("log_file_backup_count must be non-negative")
        
        return errors


@dataclass
class DroneCommandConfig:
    """Complete dronecmd configuration."""
    
    # Environment and metadata
    environment: Environment = Environment.DEVELOPMENT
    version: str = "1.0.0"
    debug_mode: bool = False
    
    # Component configurations
    sdr: SDRConfig = field(default_factory=SDRConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    injection: InjectionConfig = field(default_factory=InjectionConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Feature flags
    enable_enhanced_features: bool = True
    enable_experimental_features: bool = False
    enable_performance_monitoring: bool = True
    
    # Custom settings
    custom: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> List[str]:
        """Validate complete configuration."""
        errors = []
        
        # Validate sub-configurations
        errors.extend([f"sdr.{e}" for e in self.sdr.validate()])
        errors.extend([f"analysis.{e}" for e in self.analysis.validate()])
        errors.extend([f"injection.{e}" for e in self.injection.validate()])
        errors.extend([f"logging.{e}" for e in self.logging.validate()])
        
        # Cross-component validation
        if self.sdr.capture_path != self.paths.capture_path:
            logger.warning("SDR capture_path differs from paths.capture_path")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DroneCommandConfig':
        """Create configuration from dictionary."""
        # Handle nested configurations
        config_data = data.copy()
        
        # Convert sub-configurations
        if 'sdr' in config_data and isinstance(config_data['sdr'], dict):
            config_data['sdr'] = SDRConfig(**config_data['sdr'])
        
        if 'analysis' in config_data and isinstance(config_data['analysis'], dict):
            config_data['analysis'] = AnalysisConfig(**config_data['analysis'])
        
        if 'injection' in config_data and isinstance(config_data['injection'], dict):
            config_data['injection'] = InjectionConfig(**config_data['injection'])
        
        if 'paths' in config_data and isinstance(config_data['paths'], dict):
            config_data['paths'] = PathConfig(**config_data['paths'])
        
        if 'logging' in config_data and isinstance(config_data['logging'], dict):
            config_data['logging'] = LoggingConfig(**config_data['logging'])
        
        # Handle environment enum
        if 'environment' in config_data and isinstance(config_data['environment'], str):
            config_data['environment'] = Environment(config_data['environment'])
        
        return cls(**config_data)


class ConfigManager:
    """
    Advanced configuration manager with multiple sources and validation.
    
    Supports loading configuration from multiple sources with proper
    precedence handling and validation.
    
    Example:
        >>> manager = ConfigManager()
        >>> manager.load_from_file("config.yaml")
        >>> manager.set_environment_overrides()
        >>> config = manager.get_config()
    """
    
    def __init__(
        self,
        environment: Optional[Environment] = None,
        config_dir: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Initialize configuration manager.
        
        Args:
            environment: Target environment
            config_dir: Configuration directory path
        """
        self.environment = environment or self._detect_environment()
        self.config_dir = Path(config_dir) if config_dir else Path("config")
        
        # Configuration sources (in priority order)
        self._config_sources: Dict[ConfigSource, Dict[str, Any]] = {
            ConfigSource.DEFAULTS: {},
            ConfigSource.FILE: {},
            ConfigSource.ENVIRONMENT: {}
        }
        
        # Current merged configuration
        self._current_config: Optional[DroneCommandConfig] = None
        
        # Load default configuration
        self._load_defaults()
        
        logger.info(f"Initialized ConfigManager for {self.environment.value} environment")
    
    def _detect_environment(self) -> Environment:
        """Detect environment from environment variables."""
        env_name = os.getenv('DRONECMD_ENV', 'development').lower()
        
        env_map = {
            'dev': Environment.DEVELOPMENT,
            'development': Environment.DEVELOPMENT,
            'test': Environment.TESTING,
            'testing': Environment.TESTING,
            'prod': Environment.PRODUCTION,
            'production': Environment.PRODUCTION,
            'research': Environment.RESEARCH
        }
        
        return env_map.get(env_name, Environment.DEVELOPMENT)
    
    def _load_defaults(self) -> None:
        """Load default configuration values."""
        default_config = DroneCommandConfig(environment=self.environment)
        self._config_sources[ConfigSource.DEFAULTS] = default_config.to_dict()
    
    def load_from_file(
        self,
        filename: Optional[Union[str, Path]] = None,
        format: str = "auto"
    ) -> bool:
        """
        Load configuration from file.
        
        Args:
            filename: Configuration file path
            format: File format ('json', 'yaml', 'auto')
            
        Returns:
            True if successful, False otherwise
        """
        if filename is None:
            # Auto-detect configuration file
            filename = self._find_config_file()
        
        if filename is None:
            logger.warning("No configuration file found")
            return False
        
        filepath = Path(filename)
        if not filepath.exists():
            logger.warning(f"Configuration file not found: {filepath}")
            return False
        
        try:
            # Determine format
            if format == "auto":
                format = self._detect_file_format(filepath)
            
            # Load configuration
            if format == "json":
                config_data = self._load_json_file(filepath)
            elif format == "yaml":
                config_data = self._load_yaml_file(filepath)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self._config_sources[ConfigSource.FILE] = config_data
            self._current_config = None  # Force rebuild
            
            logger.info(f"Loaded configuration from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {filepath}: {e}")
            return False
    
    def _find_config_file(self) -> Optional[Path]:
        """Find configuration file automatically."""
        # Try environment-specific files first
        candidates = [
            self.config_dir / f"config_{self.environment.value}.yaml",
            self.config_dir / f"config_{self.environment.value}.json",
            self.config_dir / "config.yaml",
            self.config_dir / "config.json",
            Path("config.yaml"),
            Path("config.json")
        ]
        
        for candidate in candidates:
            if candidate.exists():
                return candidate
        
        return None
    
    def _detect_file_format(self, filepath: Path) -> str:
        """Detect file format from extension."""
        suffix = filepath.suffix.lower()
        
        if suffix in ['.yaml', '.yml']:
            return 'yaml'
        elif suffix == '.json':
            return 'json'
        else:
            # Try to guess from content
            try:
                with open(filepath, 'r') as f:
                    content = f.read().strip()
                    if content.startswith('{') and content.endswith('}'):
                        return 'json'
                    else:
                        return 'yaml'
            except Exception:
                return 'json'  # Default fallback
    
    def _load_json_file(self, filepath: Path) -> Dict[str, Any]:
        """Load JSON configuration file."""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def _load_yaml_file(self, filepath: Path) -> Dict[str, Any]:
        """Load YAML configuration file."""
        try:
            import yaml
            with open(filepath, 'r') as f:
                return yaml.safe_load(f) or {}
        except ImportError:
            raise ImportError("PyYAML not available for YAML configuration files")
    
    def set_environment_overrides(self) -> None:
        """Set configuration overrides from environment variables."""
        env_overrides = {}
        
        # Map environment variables to configuration paths
        env_mappings = {
            'DRONECMD_SAMPLE_RATE': 'sdr.sample_rate',
            'DRONECMD_FREQUENCY': 'sdr.default_frequency',
            'DRONECMD_CAPTURE_PATH': 'paths.capture_path',
            'DRONECMD_LOG_LEVEL': 'logging.console_level',
            'DRONECMD_DEBUG': 'debug_mode',
            'DRONECMD_MAX_POWER_DBM': 'injection.max_transmission_power_dbm',
            'DRONECMD_ENABLE_OBFUSCATION': 'injection.enable_obfuscation'
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                self._set_nested_value(env_overrides, config_path, self._convert_env_value(value))
        
        self._config_sources[ConfigSource.ENVIRONMENT] = env_overrides
        self._current_config = None  # Force rebuild
        
        if env_overrides:
            logger.info(f"Applied environment overrides: {list(env_overrides.keys())}")
    
    def _set_nested_value(self, data: Dict[str, Any], path: str, value: Any) -> None:
        """Set nested dictionary value using dot notation."""
        keys = path.split('.')
        current = data
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type."""
        # Boolean conversion
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Numeric conversion
        try:
            if '.' in value or 'e' in value.lower():
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # String value
        return value
    
    def get_config(self) -> DroneCommandConfig:
        """
        Get merged configuration from all sources.
        
        Returns:
            Complete merged configuration
        """
        if self._current_config is None:
            self._current_config = self._build_merged_config()
        
        return self._current_config
    
    def _build_merged_config(self) -> DroneCommandConfig:
        """Build merged configuration from all sources."""
        # Merge configurations in priority order
        merged_data = {}
        
        # Start with defaults
        self._deep_merge(merged_data, self._config_sources[ConfigSource.DEFAULTS])
        
        # Override with file configuration
        self._deep_merge(merged_data, self._config_sources[ConfigSource.FILE])
        
        # Override with environment variables
        self._deep_merge(merged_data, self._config_sources[ConfigSource.ENVIRONMENT])
        
        # Create configuration object
        config = DroneCommandConfig.from_dict(merged_data)
        
        # Validate configuration
        errors = config.validate()
        if errors:
            logger.warning(f"Configuration validation warnings: {errors}")
        
        return config
    
    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """Deep merge source dictionary into target dictionary."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value
    
    def save_config(
        self,
        filepath: Union[str, Path],
        format: str = "yaml",
        include_defaults: bool = False
    ) -> bool:
        """
        Save current configuration to file.
        
        Args:
            filepath: Output file path
            format: Output format ('json', 'yaml')
            include_defaults: Include default values
            
        Returns:
            True if successful, False otherwise
        """
        try:
            config = self.get_config()
            config_data = config.to_dict()
            
            # Remove defaults if requested
            if not include_defaults:
                defaults = self._config_sources[ConfigSource.DEFAULTS]
                config_data = self._remove_default_values(config_data, defaults)
            
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            if format == "json":
                with open(filepath, 'w') as f:
                    json.dump(config_data, f, indent=2, default=str)
            elif format == "yaml":
                import yaml
                with open(filepath, 'w') as f:
                    yaml.dump(config_data, f, indent=2, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Saved configuration to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def _remove_default_values(self, data: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
        """Remove default values from configuration data."""
        filtered = {}
        
        for key, value in data.items():
            if key not in defaults:
                filtered[key] = value
            elif isinstance(value, dict) and isinstance(defaults[key], dict):
                nested_filtered = self._remove_default_values(value, defaults[key])
                if nested_filtered:  # Only include if not empty
                    filtered[key] = nested_filtered
            elif value != defaults[key]:
                filtered[key] = value
        
        return filtered
    
    def get_sdr_config(self) -> SDRConfig:
        """Get SDR-specific configuration."""
        return self.get_config().sdr
    
    def get_analysis_config(self) -> AnalysisConfig:
        """Get analysis-specific configuration."""
        return self.get_config().analysis
    
    def get_injection_config(self) -> InjectionConfig:
        """Get injection-specific configuration."""
        return self.get_config().injection
    
    def get_paths_config(self) -> PathConfig:
        """Get paths configuration."""
        return self.get_config().paths
    
    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration."""
        return self.get_config().logging
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
        """
        current_overrides = self._config_sources[ConfigSource.ENVIRONMENT].copy()
        self._deep_merge(current_overrides, updates)
        self._config_sources[ConfigSource.ENVIRONMENT] = current_overrides
        self._current_config = None  # Force rebuild
        
        logger.info(f"Updated configuration: {list(updates.keys())}")


# =============================================================================
# BACKWARD COMPATIBILITY (Original config.py constants)
# =============================================================================

# Global configuration manager instance
_config_manager = ConfigManager()

# Load configuration automatically
_config_manager.set_environment_overrides()
try:
    _config_manager.load_from_file()
except Exception as e:
    logger.debug(f"Could not load config file: {e}")

# Get current configuration
_current_config = _config_manager.get_config()

# Backward compatible constants (from original config.py)
SAMPLE_RATE = _current_config.sdr.sample_rate
FREQ_RANGE = _current_config.sdr.frequency_range
CAPTURE_DURATION = _current_config.sdr.capture_duration

# Paths (from original config.py)
PLUGIN_DIR = _current_config.paths.get_absolute_path(_current_config.paths.plugin_dir)
LOG_DIR = _current_config.paths.get_absolute_path(_current_config.paths.logs_dir)
IQ_CAPTURE_PATH = _current_config.paths.get_absolute_path(_current_config.paths.capture_path)
MODEL_PATH = _current_config.paths.get_absolute_path(_current_config.paths.model_path)

# Ensure directories exist
_current_config.paths.ensure_directories()

# Export the global config object for advanced usage
config = _current_config


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_config() -> DroneCommandConfig:
    """Get current global configuration."""
    return config


def get_config_manager() -> ConfigManager:
    """Get global configuration manager."""
    return _config_manager


def reload_config() -> None:
    """Reload configuration from all sources."""
    global config, _current_config
    global SAMPLE_RATE, FREQ_RANGE, CAPTURE_DURATION
    global PLUGIN_DIR, LOG_DIR, IQ_CAPTURE_PATH, MODEL_PATH
    
    _config_manager.set_environment_overrides()
    _config_manager.load_from_file()
    
    _current_config = _config_manager.get_config()
    config = _current_config
    
    # Update backward compatible constants
    SAMPLE_RATE = config.sdr.sample_rate
    FREQ_RANGE = config.sdr.frequency_range
    CAPTURE_DURATION = config.sdr.capture_duration
    
    PLUGIN_DIR = config.paths.get_absolute_path(config.paths.plugin_dir)
    LOG_DIR = config.paths.get_absolute_path(config.paths.logs_dir)
    IQ_CAPTURE_PATH = config.paths.get_absolute_path(config.paths.capture_path)
    MODEL_PATH = config.paths.get_absolute_path(config.paths.model_path)
    
    config.paths.ensure_directories()
    
    logger.info("Configuration reloaded")


def set_config_value(key: str, value: Any) -> None:
    """
    Set configuration value using dot notation.
    
    Args:
        key: Configuration key in dot notation (e.g., 'sdr.sample_rate')
        value: New value
    """
    updates = {}
    _config_manager._set_nested_value(updates, key, value)
    _config_manager.update_config(updates)
    
    # Reload to update global constants
    reload_config()


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    print("=== Configuration Management Demo ===")
    
    try:
        # Test backward compatibility
        print(f"\n=== Backward Compatibility ===")
        print(f"SAMPLE_RATE: {SAMPLE_RATE:,} Hz")
        print(f"FREQ_RANGE: {FREQ_RANGE[0]/1e6:.1f} - {FREQ_RANGE[1]/1e6:.1f} MHz")
        print(f"CAPTURE_DURATION: {CAPTURE_DURATION} seconds")
        print(f"IQ_CAPTURE_PATH: {IQ_CAPTURE_PATH}")
        
        # Test advanced configuration
        print(f"\n=== Advanced Configuration ===")
        manager = ConfigManager()
        
        # Show current configuration
        current_config = manager.get_config()
        print(f"Environment: {current_config.environment.value}")
        print(f"Debug mode: {current_config.debug_mode}")
        print(f"Enhanced features: {current_config.enable_enhanced_features}")
        
        # Test component configurations
        sdr_config = manager.get_sdr_config()
        print(f"\nSDR Configuration:")
        print(f"  Sample rate: {sdr_config.sample_rate:,} Hz")
        print(f"  Default frequency: {sdr_config.default_frequency/1e6:.3f} MHz")
        print(f"  Preferred platform: {sdr_config.preferred_platform}")
        
        analysis_config = manager.get_analysis_config()
        print(f"\nAnalysis Configuration:")
        print(f"  Protocol detection: {analysis_config.enable_protocol_detection}")
        print(f"  Classification threshold: {analysis_config.classification_confidence_threshold}")
        print(f"  Processing threads: {analysis_config.processing_threads}")
        
        injection_config = manager.get_injection_config()
        print(f"\nInjection Configuration:")
        print(f"  Max power: {injection_config.max_transmission_power_dbm} dBm")
        print(f"  Safety monitoring: {injection_config.enable_safety_monitoring}")
        print(f"  FCC compliance: {injection_config.enable_fcc_compliance_checking}")
        
        # Test configuration updates
        print(f"\n=== Configuration Updates ===")
        print(f"Original sample rate: {sdr_config.sample_rate:,}")
        
        # Update through manager
        manager.update_config({'sdr': {'sample_rate': 2_400_000}})
        updated_config = manager.get_sdr_config()
        print(f"Updated sample rate: {updated_config.sample_rate:,}")
        
        # Test validation
        print(f"\n=== Configuration Validation ===")
        test_config = DroneCommandConfig()
        test_config.sdr.sample_rate = -1  # Invalid
        errors = test_config.validate()
        print(f"Validation errors: {errors}")
        
        # Test environment variable override
        print(f"\n=== Environment Variable Override ===")
        os.environ['DRONECMD_SAMPLE_RATE'] = '1000000'
        manager.set_environment_overrides()
        env_config = manager.get_sdr_config()
        print(f"Environment override sample rate: {env_config.sample_rate:,}")
        
        # Clean up
        if 'DRONECMD_SAMPLE_RATE' in os.environ:
            del os.environ['DRONECMD_SAMPLE_RATE']
        
        # Test configuration saving
        print(f"\n=== Configuration Persistence ===")
        test_file = Path("test_config.json")
        success = manager.save_config(test_file, format="json")
        print(f"Save config success: {success}")
        
        if test_file.exists():
            print(f"Config file size: {test_file.stat().st_size} bytes")
            test_file.unlink()  # Clean up
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()