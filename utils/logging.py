#!/usr/bin/env python3
"""
Enhanced Logging System

This module provides a comprehensive logging system for the dronecmd library
with structured logging, performance monitoring, and advanced configuration
capabilities.

Key Features:
- Structured logging with JSON output
- Performance monitoring and metrics
- Configurable log levels and formats
- File rotation and compression
- Context managers for operation logging
- Integration with monitoring systems
- Thread-safe operation
- Memory-efficient buffering
- Custom formatters and filters

Usage:
    # Basic setup
    configure_logging(level='INFO', enable_structured=True)
    logger = get_logger(__name__)
    logger.info("Application started")
    
    # Performance monitoring
    with log_performance("signal_processing"):
        process_signal()
    
    # Structured logging
    logger.info("Processed signal", extra={
        'samples': 10000,
        'frequency': 2.4e9,
        'processing_time': 0.5
    })
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TextIO, Callable
import uuid
import warnings

# Configure basic logging first
logging.basicConfig(level=logging.WARNING)

# Module constants
DEFAULT_LOG_DIR = Path.home() / ".dronecmd" / "logs"
DEFAULT_LOG_FILE = "dronecmd.log"
DEFAULT_FORMAT = '[%(asctime)s] %(levelname)s %(name)s: %(message)s'
JSON_FORMAT = '%(message)s'

# Thread-local storage for context
_context_storage = threading.local()


class LogLevel(Enum):
    """Log levels with numeric values."""
    
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


class LogFormat(Enum):
    """Available log formats."""
    
    STANDARD = "standard"
    DETAILED = "detailed"  
    JSON = "json"
    COMPACT = "compact"


@dataclass
class LogEntry:
    """Structured log entry for JSON logging."""
    
    timestamp: str
    level: str
    logger: str
    message: str
    module: Optional[str] = None
    function: Optional[str] = None
    line: Optional[int] = None
    thread_id: Optional[int] = None
    process_id: Optional[int] = None
    context: Optional[Dict[str, Any]] = None
    performance: Optional[Dict[str, Any]] = None
    exception: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def __init__(self, include_context: bool = True, include_performance: bool = True):
        super().__init__()
        self.include_context = include_context
        self.include_performance = include_performance
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Create base log entry
        entry = LogEntry(
            timestamp=datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            level=record.levelname,
            logger=record.name,
            message=record.getMessage(),
            module=record.module,
            function=record.funcName,
            line=record.lineno,
            thread_id=record.thread,
            process_id=record.process
        )
        
        # Add context if available
        if self.include_context and hasattr(record, 'context'):
            entry.context = record.context
        
        # Add performance data if available
        if self.include_performance and hasattr(record, 'performance'):
            entry.performance = record.performance
        
        # Add exception information
        if record.exc_info:
            entry.exception = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields from record
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'exc_info',
                          'exc_text', 'stack_info', 'context', 'performance']:
                extra_fields[key] = value
        
        if extra_fields:
            if entry.context is None:
                entry.context = {}
            entry.context.update(extra_fields)
        
        return json.dumps(entry.to_dict(), ensure_ascii=False, default=str)


class DetailedFormatter(logging.Formatter):
    """Detailed formatter with more information."""
    
    def __init__(self):
        super().__init__(
            fmt='[%(asctime)s] %(levelname)-8s %(name)-25s %(funcName)-20s:%(lineno)-4d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )


class CompactFormatter(logging.Formatter):
    """Compact formatter for minimal output."""
    
    def __init__(self):
        super().__init__(
            fmt='%(asctime)s %(levelname)s %(name)s: %(message)s',
            datefmt='%H:%M:%S'
        )


class PerformanceFilter(logging.Filter):
    """Filter to add performance metrics to log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add performance context to records."""
        # Add performance data if available in context
        context = getattr(_context_storage, 'context', {})
        performance_data = context.get('performance')
        
        if performance_data:
            record.performance = performance_data
        
        return True


class ContextFilter(logging.Filter):
    """Filter to add context information to log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add context information to records."""
        context = getattr(_context_storage, 'context', {})
        
        # Remove performance data (handled by PerformanceFilter)
        filtered_context = {k: v for k, v in context.items() if k != 'performance'}
        
        if filtered_context:
            record.context = filtered_context
        
        return True


class LoggingConfig:
    """Configuration for the logging system."""
    
    def __init__(
        self,
        level: Union[str, int] = 'INFO',
        format_type: LogFormat = LogFormat.STANDARD,
        log_dir: Optional[Path] = None,
        log_file: str = DEFAULT_LOG_FILE,
        enable_console: bool = True,
        enable_file: bool = True,
        enable_rotation: bool = True,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        enable_compression: bool = True,
        enable_structured: bool = False,
        enable_performance: bool = True,
        buffer_size: int = 0,  # 0 = unbuffered
        custom_formatters: Optional[Dict[str, logging.Formatter]] = None
    ):
        """
        Initialize logging configuration.
        
        Args:
            level: Log level (string or numeric)
            format_type: Log format type
            log_dir: Directory for log files (None for default)
            log_file: Log file name
            enable_console: Enable console output
            enable_file: Enable file output
            enable_rotation: Enable log file rotation
            max_file_size: Maximum size per log file
            backup_count: Number of backup files to keep
            enable_compression: Compress rotated files
            enable_structured: Use structured (JSON) logging
            enable_performance: Enable performance monitoring
            buffer_size: Buffer size for file output
            custom_formatters: Custom formatters by handler name
        """
        self.level = self._parse_level(level)
        self.format_type = format_type
        self.log_dir = log_dir or DEFAULT_LOG_DIR
        self.log_file = log_file
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.enable_rotation = enable_rotation
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.enable_compression = enable_compression
        self.enable_structured = enable_structured
        self.enable_performance = enable_performance
        self.buffer_size = buffer_size
        self.custom_formatters = custom_formatters or {}
        
        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def _parse_level(self, level: Union[str, int]) -> int:
        """Parse log level to numeric value."""
        if isinstance(level, int):
            return level
        
        level_map = {
            'TRACE': LogLevel.TRACE.value,
            'DEBUG': LogLevel.DEBUG.value,
            'INFO': LogLevel.INFO.value,
            'WARNING': LogLevel.WARNING.value,
            'ERROR': LogLevel.ERROR.value,
            'CRITICAL': LogLevel.CRITICAL.value
        }
        
        return level_map.get(level.upper(), LogLevel.INFO.value)


class PerformanceMonitor:
    """Performance monitoring for logging operations."""
    
    def __init__(self):
        self.operations = {}
        self.lock = threading.Lock()
    
    def start_operation(self, operation_id: str, operation_name: str) -> None:
        """Start monitoring an operation."""
        with self.lock:
            self.operations[operation_id] = {
                'name': operation_name,
                'start_time': time.perf_counter(),
                'thread_id': threading.get_ident()
            }
    
    def end_operation(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """End monitoring an operation and return metrics."""
        with self.lock:
            if operation_id not in self.operations:
                return None
            
            operation = self.operations.pop(operation_id)
            end_time = time.perf_counter()
            
            return {
                'operation': operation['name'],
                'duration_s': end_time - operation['start_time'],
                'thread_id': operation['thread_id'],
                'operation_id': operation_id
            }
    
    def get_active_operations(self) -> List[Dict[str, Any]]:
        """Get list of currently active operations."""
        with self.lock:
            current_time = time.perf_counter()
            return [
                {
                    'operation_id': op_id,
                    'name': op_data['name'],
                    'elapsed_s': current_time - op_data['start_time'],
                    'thread_id': op_data['thread_id']
                }
                for op_id, op_data in self.operations.items()
            ]


# Global performance monitor
_performance_monitor = PerformanceMonitor()


def configure_logging(
    level: Union[str, int] = 'INFO',
    format_type: Union[str, LogFormat] = LogFormat.STANDARD,
    log_dir: Optional[Union[str, Path]] = None,
    log_file: str = DEFAULT_LOG_FILE,
    enable_console: bool = True,
    enable_file: bool = True,
    enable_structured: bool = False,
    enable_performance: bool = True,
    **kwargs: Any
) -> None:
    """
    Configure the logging system with enhanced options.
    
    Args:
        level: Log level
        format_type: Format type
        log_dir: Log directory
        log_file: Log file name
        enable_console: Enable console logging
        enable_file: Enable file logging
        enable_structured: Enable structured JSON logging
        enable_performance: Enable performance monitoring
        **kwargs: Additional configuration options
    """
    # Parse format type
    if isinstance(format_type, str):
        format_type = LogFormat(format_type.lower())
    
    # Create configuration
    config = LoggingConfig(
        level=level,
        format_type=format_type,
        log_dir=Path(log_dir) if log_dir else None,
        log_file=log_file,
        enable_console=enable_console,
        enable_file=enable_file,
        enable_structured=enable_structured,
        enable_performance=enable_performance,
        **kwargs
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(config.level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatters
    formatters = _create_formatters(config)
    
    # Add console handler
    if config.enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(config.level)
        
        formatter_name = 'json' if config.enable_structured else config.format_type.value
        console_handler.setFormatter(formatters[formatter_name])
        
        # Add filters
        if config.enable_performance:
            console_handler.addFilter(PerformanceFilter())
        console_handler.addFilter(ContextFilter())
        
        root_logger.addHandler(console_handler)
    
    # Add file handler
    if config.enable_file:
        log_file_path = config.log_dir / config.log_file
        
        if config.enable_rotation:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file_path,
                maxBytes=config.max_file_size,
                backupCount=config.backup_count
            )
            
            # Enable compression for rotated files
            if config.enable_compression:
                file_handler.rotator = _compress_rotated_file
        else:
            file_handler = logging.FileHandler(log_file_path)
        
        file_handler.setLevel(config.level)
        
        formatter_name = 'json' if config.enable_structured else config.format_type.value
        file_handler.setFormatter(formatters[formatter_name])
        
        # Add filters
        if config.enable_performance:
            file_handler.addFilter(PerformanceFilter())
        file_handler.addFilter(ContextFilter())
        
        root_logger.addHandler(file_handler)
    
    # Set up TRACE level if needed
    if config.level <= LogLevel.TRACE.value:
        logging.addLevelName(LogLevel.TRACE.value, 'TRACE')
    
    # Log configuration
    logger = logging.getLogger(__name__)
    logger.info("Logging configured", extra={
        'level': logging.getLevelName(config.level),
        'format': config.format_type.value,
        'structured': config.enable_structured,
        'performance': config.enable_performance,
        'log_dir': str(config.log_dir)
    })


def _create_formatters(config: LoggingConfig) -> Dict[str, logging.Formatter]:
    """Create formatters based on configuration."""
    formatters = {}
    
    # Standard formatters
    formatters['standard'] = logging.Formatter(DEFAULT_FORMAT)
    formatters['detailed'] = DetailedFormatter()
    formatters['compact'] = CompactFormatter()
    formatters['json'] = JSONFormatter(
        include_context=True,
        include_performance=config.enable_performance
    )
    
    # Add custom formatters
    formatters.update(config.custom_formatters)
    
    return formatters


def _compress_rotated_file(source: str, dest: str) -> None:
    """Compress rotated log file."""
    import gzip
    import shutil
    
    try:
        with open(source, 'rb') as f_in:
            with gzip.open(dest + '.gz', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(source)
    except Exception as e:
        warnings.warn(f"Failed to compress log file {source}: {e}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # Add TRACE method if not present
    if not hasattr(logger, 'trace'):
        def trace(message, *args, **kwargs):
            if logger.isEnabledFor(LogLevel.TRACE.value):
                logger._log(LogLevel.TRACE.value, message, args, **kwargs)
        
        logger.trace = trace
    
    return logger


def set_context(context: Dict[str, Any]) -> None:
    """
    Set logging context for the current thread.
    
    Args:
        context: Context dictionary
    """
    if not hasattr(_context_storage, 'context'):
        _context_storage.context = {}
    
    _context_storage.context.update(context)


def clear_context() -> None:
    """Clear logging context for the current thread."""
    if hasattr(_context_storage, 'context'):
        _context_storage.context.clear()


def get_context() -> Dict[str, Any]:
    """
    Get current logging context.
    
    Returns:
        Current context dictionary
    """
    return getattr(_context_storage, 'context', {}).copy()


@contextmanager
def log_context(context: Dict[str, Any]):
    """
    Context manager for temporary logging context.
    
    Args:
        context: Temporary context to add
    """
    original_context = get_context()
    
    try:
        set_context(context)
        yield
    finally:
        clear_context()
        set_context(original_context)


@contextmanager
def log_performance(operation_name: str, logger: Optional[logging.Logger] = None):
    """
    Context manager for performance logging.
    
    Args:
        operation_name: Name of the operation being monitored
        logger: Optional logger instance
    """
    if logger is None:
        logger = get_logger(__name__)
    
    operation_id = str(uuid.uuid4())
    
    try:
        # Start monitoring
        _performance_monitor.start_operation(operation_id, operation_name)
        
        logger.debug(f"Started operation: {operation_name}", extra={
            'operation': operation_name,
            'operation_id': operation_id
        })
        
        yield
        
    except Exception as e:
        # Log error
        logger.error(f"Operation failed: {operation_name}", extra={
            'operation': operation_name,
            'operation_id': operation_id,
            'error': str(e)
        }, exc_info=True)
        raise
        
    finally:
        # End monitoring and log results
        performance_data = _performance_monitor.end_operation(operation_id)
        
        if performance_data:
            # Add to context for other log messages
            if not hasattr(_context_storage, 'context'):
                _context_storage.context = {}
            _context_storage.context['performance'] = performance_data
            
            logger.info(f"Completed operation: {operation_name}", extra={
                'operation': operation_name,
                'operation_id': operation_id,
                'duration_s': performance_data['duration_s']
            })


def log_function_call(
    logger: Optional[logging.Logger] = None,
    level: Union[str, int] = 'DEBUG',
    include_args: bool = False,
    include_result: bool = False
):
    """
    Decorator for automatic function call logging.
    
    Args:
        logger: Logger instance
        level: Log level for function calls
        include_args: Include function arguments in log
        include_result: Include function result in log
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)
            
            func_name = f"{func.__module__}.{func.__name__}"
            
            # Log function entry
            entry_extra = {'function': func_name}
            if include_args:
                entry_extra['args'] = args
                entry_extra['kwargs'] = kwargs
            
            logger.log(level, f"Entering function: {func_name}", extra=entry_extra)
            
            # Execute function with performance monitoring
            with log_performance(func_name, logger):
                try:
                    result = func(*args, **kwargs)
                    
                    # Log successful exit
                    exit_extra = {'function': func_name}
                    if include_result:
                        exit_extra['result'] = result
                    
                    logger.log(level, f"Exiting function: {func_name}", extra=exit_extra)
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"Function failed: {func_name}", extra={
                        'function': func_name,
                        'error': str(e)
                    }, exc_info=True)
                    raise
        
        return wrapper
    return decorator


def create_operation_logger(operation_name: str) -> logging.Logger:
    """
    Create a logger specifically for an operation with context.
    
    Args:
        operation_name: Name of the operation
        
    Returns:
        Logger with operation context
    """
    logger = get_logger(f"operation.{operation_name}")
    
    # Set operation context
    operation_id = str(uuid.uuid4())
    set_context({
        'operation': operation_name,
        'operation_id': operation_id
    })
    
    return logger


def get_log_statistics() -> Dict[str, Any]:
    """
    Get logging system statistics.
    
    Returns:
        Dictionary with logging statistics
    """
    root_logger = logging.getLogger()
    
    stats = {
        'root_level': logging.getLevelName(root_logger.level),
        'handler_count': len(root_logger.handlers),
        'handlers': []
    }
    
    for handler in root_logger.handlers:
        handler_info = {
            'type': type(handler).__name__,
            'level': logging.getLevelName(handler.level),
            'formatter': type(handler.formatter).__name__ if handler.formatter else None
        }
        
        if hasattr(handler, 'baseFilename'):
            handler_info['file'] = handler.baseFilename
        
        stats['handlers'].append(handler_info)
    
    # Add performance monitoring stats
    active_operations = _performance_monitor.get_active_operations()
    stats['active_operations'] = len(active_operations)
    stats['performance_operations'] = active_operations
    
    return stats


def enable_debug_logging() -> None:
    """Enable debug logging for troubleshooting."""
    root_logger = logging.getLogger()
    root_logger.setLevel(LogLevel.DEBUG.value)
    
    for handler in root_logger.handlers:
        handler.setLevel(LogLevel.DEBUG.value)
    
    logger = get_logger(__name__)
    logger.debug("Debug logging enabled")


def disable_debug_logging() -> None:
    """Disable debug logging."""
    root_logger = logging.getLogger()
    root_logger.setLevel(LogLevel.INFO.value)
    
    for handler in root_logger.handlers:
        handler.setLevel(LogLevel.INFO.value)
    
    logger = get_logger(__name__)
    logger.info("Debug logging disabled")


# Initialize with basic configuration
configure_logging()


# Example usage and testing
if __name__ == "__main__":
    # Test basic logging
    print("=== Enhanced Logging System Demo ===")
    
    # Configure with structured logging
    configure_logging(
        level='DEBUG',
        format_type=LogFormat.DETAILED,
        enable_structured=False,
        enable_performance=True
    )
    
    logger = get_logger(__name__)
    
    # Test basic logging
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    
    # Test structured logging with context
    with log_context({'component': 'demo', 'version': '1.0'}):
        logger.info("Structured message", extra={
            'user_id': 12345,
            'action': 'login',
            'success': True
        })
    
    # Test performance logging
    with log_performance("test_operation"):
        time.sleep(0.1)  # Simulate work
        logger.info("Work completed")
    
    # Test function decoration
    @log_function_call(include_args=True, include_result=True)
    def test_function(x: int, y: int) -> int:
        """Test function for logging."""
        return x + y
    
    result = test_function(5, 3)
    logger.info(f"Function result: {result}")
    
    # Test exception logging
    try:
        raise ValueError("Test exception")
    except Exception:
        logger.error("Exception occurred", exc_info=True)
    
    # Show statistics
    stats = get_log_statistics()
    logger.info("Logging statistics", extra=stats)
    
    print("Demo completed - check log files for output")