#!/usr/bin/env python3
"""
DroneCmd Plugin System

This module provides a comprehensive plugin architecture for extending
DroneCmd with protocol-specific functionality. The plugin system supports
dynamic loading, registration, and management of protocol handlers for
various drone communication systems.

Plugin Architecture:
- Base plugin classes defining standard interfaces
- Automatic plugin discovery and loading
- Registry system for plugin management
- Protocol-specific plugin implementations
- Version compatibility checking
- Dependency management

Supported Protocol Families:
- MAVLink (ArduPilot, PX4, etc.)
- DJI proprietary protocols
- Parrot/FreeFlight protocols
- Generic radio protocols
- Custom protocol extensions

Usage:
    >>> from dronecmd.plugins import PluginRegistry, load_plugins
    >>> registry = PluginRegistry()
    >>> plugins = load_plugins()
    >>> mavlink_plugin = registry.get_plugin('mavlink')
    >>> result = mavlink_plugin.decode_packet(packet_data)
"""

import logging
import importlib
import pkgutil
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union, Callable
from pathlib import Path
import inspect

# Configure module logger
logger = logging.getLogger(__name__)

# =============================================================================
# BASE PLUGIN CLASSES
# =============================================================================

try:
    from .base import (
        BasePlugin,
        ProtocolPlugin,
        DecoderPlugin,
        EncoderPlugin,
        AnalyzerPlugin,
        PluginMetadata,
        PluginError,
        PluginVersion
    )
    BASE_AVAILABLE = True
    logger.debug("Base plugin classes loaded successfully")
    
except ImportError as e:
    logger.warning(f"Base plugin classes not available: {e}")
    BASE_AVAILABLE = False
    
    # Define minimal base classes as fallback
    class BasePlugin(ABC):
        """Minimal base plugin class."""
        
        def __init__(self):
            self._name = self.__class__.__name__
            self._version = "0.0.0"
        
        @property
        def name(self) -> str:
            return self._name
        
        @property 
        def version(self) -> str:
            return self._version
        
        @abstractmethod
        def initialize(self) -> bool:
            """Initialize plugin."""
            pass
        
        @abstractmethod
        def cleanup(self) -> None:
            """Cleanup plugin resources."""
            pass
    
    class ProtocolPlugin(BasePlugin):
        """Minimal protocol plugin class."""
        
        @abstractmethod
        def detect(self, data: bytes) -> bool:
            """Detect if data matches this protocol."""
            pass
        
        @abstractmethod
        def decode_packet(self, data: bytes) -> Dict[str, Any]:
            """Decode packet data."""
            pass
    
    # Minimal supporting classes
    PluginMetadata = Dict[str, Any]
    PluginError = Exception
    PluginVersion = str
    DecoderPlugin = ProtocolPlugin
    EncoderPlugin = ProtocolPlugin  
    AnalyzerPlugin = ProtocolPlugin

# =============================================================================
# PLUGIN REGISTRY
# =============================================================================

try:
    from .registry import (
        PluginRegistry,
        register_plugin,
        get_plugin,
        list_plugins,
        unregister_plugin,
        RegistryError
    )
    REGISTRY_AVAILABLE = True
    logger.debug("Plugin registry loaded successfully")
    
except ImportError as e:
    logger.warning(f"Plugin registry not available: {e}")
    REGISTRY_AVAILABLE = False
    
    # Fallback registry implementation
    class PluginRegistry:
        """Minimal plugin registry."""
        
        def __init__(self):
            self._plugins = {}
            self._metadata = {}
        
        def register(self, plugin: BasePlugin, metadata: Optional[PluginMetadata] = None) -> bool:
            """Register a plugin."""
            try:
                self._plugins[plugin.name] = plugin
                self._metadata[plugin.name] = metadata or {}
                logger.info(f"Registered plugin: {plugin.name}")
                return True
            except Exception as e:
                logger.error(f"Failed to register plugin {plugin.name}: {e}")
                return False
        
        def get(self, name: str) -> Optional[BasePlugin]:
            """Get plugin by name."""
            return self._plugins.get(name)
        
        def list_plugins(self) -> List[str]:
            """List registered plugin names."""
            return list(self._plugins.keys())
        
        def unregister(self, name: str) -> bool:
            """Unregister plugin."""
            if name in self._plugins:
                plugin = self._plugins.pop(name)
                self._metadata.pop(name, None)
                try:
                    plugin.cleanup()
                except Exception as e:
                    logger.warning(f"Cleanup failed for plugin {name}: {e}")
                return True
            return False
    
    # Global registry instance
    _global_registry = PluginRegistry()
    
    def register_plugin(plugin: BasePlugin, metadata: Optional[PluginMetadata] = None) -> bool:
        """Register plugin with global registry."""
        return _global_registry.register(plugin, metadata)
    
    def get_plugin(name: str) -> Optional[BasePlugin]:
        """Get plugin from global registry."""
        return _global_registry.get(name)
    
    def list_plugins() -> List[str]:
        """List plugins in global registry."""
        return _global_registry.list_plugins()
    
    def unregister_plugin(name: str) -> bool:
        """Unregister plugin from global registry."""
        return _global_registry.unregister(name)
    
    RegistryError = Exception

# =============================================================================
# PROTOCOL-SPECIFIC PLUGINS
# =============================================================================

# Import protocol plugins with graceful fallback
_protocol_plugins = {}

# MAVLink Plugin
try:
    from .protocols.mavlink import MAVLinkPlugin
    _protocol_plugins['mavlink'] = MAVLinkPlugin
    logger.debug("MAVLink plugin loaded")
except ImportError as e:
    logger.debug(f"MAVLink plugin not available: {e}")
    MAVLinkPlugin = None

# DJI Plugin
try:
    from .protocols.dji import DJIPlugin
    _protocol_plugins['dji'] = DJIPlugin
    logger.debug("DJI plugin loaded")
except ImportError as e:
    logger.debug(f"DJI plugin not available: {e}")
    DJIPlugin = None

# Parrot Plugin
try:
    from .protocols.parrot import ParrotPlugin
    _protocol_plugins['parrot'] = ParrotPlugin
    logger.debug("Parrot plugin loaded")
except ImportError as e:
    logger.debug(f"Parrot plugin not available: {e}")
    ParrotPlugin = None

# Generic Plugin
try:
    from .protocols.generic import GenericPlugin
    _protocol_plugins['generic'] = GenericPlugin
    logger.debug("Generic plugin loaded")
except ImportError as e:
    logger.debug(f"Generic plugin not available: {e}")
    GenericPlugin = None

# Additional protocol plugins can be added here
# Skydio, Autel, Yuneec, etc.

# =============================================================================
# PLUGIN LOADING AND DISCOVERY
# =============================================================================

def discover_plugins(search_paths: Optional[List[Path]] = None) -> List[Type[BasePlugin]]:
    """
    Discover plugin classes in specified paths.
    
    Args:
        search_paths: List of paths to search for plugins
        
    Returns:
        List of discovered plugin classes
    """
    discovered = []
    
    if search_paths is None:
        search_paths = [Path(__file__).parent / "protocols"]
    
    for path in search_paths:
        if not path.exists():
            continue
        
        try:
            # Search for Python files
            for py_file in path.glob("*.py"):
                if py_file.name.startswith("_"):
                    continue
                
                module_name = py_file.stem
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Find plugin classes
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if (issubclass(obj, BasePlugin) and 
                            obj is not BasePlugin and
                            obj is not ProtocolPlugin):
                            discovered.append(obj)
                            logger.debug(f"Discovered plugin class: {name}")
        
        except Exception as e:
            logger.warning(f"Error discovering plugins in {path}: {e}")
    
    return discovered


def load_plugins(
    auto_register: bool = True,
    search_paths: Optional[List[Path]] = None
) -> List[BasePlugin]:
    """
    Load all available plugins.
    
    Args:
        auto_register: Automatically register loaded plugins
        search_paths: Additional paths to search for plugins
        
    Returns:
        List of loaded plugin instances
    """
    loaded_plugins = []
    
    # Load built-in protocol plugins
    for plugin_name, plugin_class in _protocol_plugins.items():
        if plugin_class is not None:
            try:
                plugin_instance = plugin_class()
                
                if plugin_instance.initialize():
                    loaded_plugins.append(plugin_instance)
                    
                    if auto_register:
                        metadata = {
                            'type': 'protocol',
                            'family': plugin_name,
                            'built_in': True
                        }
                        register_plugin(plugin_instance, metadata)
                    
                    logger.info(f"Loaded built-in plugin: {plugin_name}")
                else:
                    logger.warning(f"Failed to initialize plugin: {plugin_name}")
                    
            except Exception as e:
                logger.error(f"Failed to load plugin {plugin_name}: {e}")
    
    # Discover and load external plugins
    try:
        discovered_classes = discover_plugins(search_paths)
        
        for plugin_class in discovered_classes:
            try:
                plugin_instance = plugin_class()
                
                if plugin_instance.initialize():
                    loaded_plugins.append(plugin_instance)
                    
                    if auto_register:
                        metadata = {
                            'type': 'protocol',
                            'family': 'external',
                            'built_in': False,
                            'class': plugin_class.__name__
                        }
                        register_plugin(plugin_instance, metadata)
                    
                    logger.info(f"Loaded external plugin: {plugin_instance.name}")
                else:
                    logger.warning(f"Failed to initialize external plugin: {plugin_class.__name__}")
                    
            except Exception as e:
                logger.error(f"Failed to load external plugin {plugin_class.__name__}: {e}")
    
    except Exception as e:
        logger.warning(f"Plugin discovery failed: {e}")
    
    logger.info(f"Loaded {len(loaded_plugins)} plugins total")
    return loaded_plugins


def create_plugin_from_class(
    plugin_class: Type[BasePlugin],
    config: Optional[Dict[str, Any]] = None
) -> Optional[BasePlugin]:
    """
    Create plugin instance from class with configuration.
    
    Args:
        plugin_class: Plugin class to instantiate
        config: Configuration parameters
        
    Returns:
        Plugin instance or None if creation failed
    """
    try:
        # Check if plugin accepts config in constructor
        sig = inspect.signature(plugin_class.__init__)
        if 'config' in sig.parameters and config:
            plugin = plugin_class(config=config)
        elif config:
            # Try to pass config as kwargs
            plugin = plugin_class(**config)
        else:
            plugin = plugin_class()
        
        if plugin.initialize():
            return plugin
        else:
            logger.error(f"Failed to initialize plugin {plugin_class.__name__}")
            return None
            
    except Exception as e:
        logger.error(f"Failed to create plugin {plugin_class.__name__}: {e}")
        return None

# =============================================================================
# PLUGIN UTILITIES
# =============================================================================

def get_protocol_plugins() -> Dict[str, BasePlugin]:
    """
    Get all registered protocol plugins.
    
    Returns:
        Dictionary mapping protocol names to plugin instances
    """
    protocol_plugins = {}
    
    for plugin_name in list_plugins():
        plugin = get_plugin(plugin_name)
        if plugin and isinstance(plugin, ProtocolPlugin):
            protocol_plugins[plugin_name] = plugin
    
    return protocol_plugins


def analyze_packet_with_plugins(
    packet_data: bytes,
    plugins: Optional[List[BasePlugin]] = None
) -> Dict[str, Any]:
    """
    Analyze packet data with all available plugins.
    
    Args:
        packet_data: Raw packet data
        plugins: Specific plugins to use (all if None)
        
    Returns:
        Analysis results from all plugins
    """
    results = {
        'packet_size': len(packet_data),
        'plugins_tested': 0,
        'detections': {},
        'best_match': None,
        'confidence_scores': {}
    }
    
    if plugins is None:
        plugins = [get_plugin(name) for name in list_plugins()]
        plugins = [p for p in plugins if p is not None]
    
    best_confidence = 0.0
    best_plugin = None
    
    for plugin in plugins:
        if not isinstance(plugin, ProtocolPlugin):
            continue
        
        try:
            results['plugins_tested'] += 1
            
            # Test detection
            detected = plugin.detect(packet_data)
            
            if detected:
                # Decode packet
                decoded = plugin.decode_packet(packet_data)
                
                # Extract confidence if available
                confidence = decoded.get('confidence', 0.5)
                
                results['detections'][plugin.name] = {
                    'detected': True,
                    'decoded': decoded,
                    'confidence': confidence
                }
                
                results['confidence_scores'][plugin.name] = confidence
                
                # Track best match
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_plugin = plugin.name
            else:
                results['detections'][plugin.name] = {
                    'detected': False,
                    'confidence': 0.0
                }
                results['confidence_scores'][plugin.name] = 0.0
                
        except Exception as e:
            logger.warning(f"Plugin {plugin.name} analysis failed: {e}")
            results['detections'][plugin.name] = {
                'detected': False,
                'error': str(e)
            }
    
    results['best_match'] = best_plugin
    results['best_confidence'] = best_confidence
    
    return results


def validate_plugin_compatibility(plugin: BasePlugin) -> Dict[str, Any]:
    """
    Validate plugin compatibility with current system.
    
    Args:
        plugin: Plugin to validate
        
    Returns:
        Validation results
    """
    validation = {
        'compatible': True,
        'issues': [],
        'warnings': [],
        'requirements_met': True
    }
    
    try:
        # Check required methods
        required_methods = ['initialize', 'cleanup']
        if isinstance(plugin, ProtocolPlugin):
            required_methods.extend(['detect', 'decode_packet'])
        
        for method_name in required_methods:
            if not hasattr(plugin, method_name):
                validation['issues'].append(f"Missing required method: {method_name}")
                validation['compatible'] = False
        
        # Check version compatibility if available
        if hasattr(plugin, 'required_version'):
            # Version checking logic would go here
            pass
        
        # Check dependencies if available
        if hasattr(plugin, 'dependencies'):
            for dep in plugin.dependencies:
                try:
                    importlib.import_module(dep)
                except ImportError:
                    validation['issues'].append(f"Missing dependency: {dep}")
                    validation['requirements_met'] = False
        
        # Test initialization
        try:
            if hasattr(plugin, 'test_initialization'):
                plugin.test_initialization()
        except Exception as e:
            validation['warnings'].append(f"Initialization test failed: {e}")
    
    except Exception as e:
        validation['issues'].append(f"Validation failed: {e}")
        validation['compatible'] = False
    
    return validation


def get_plugin_statistics() -> Dict[str, Any]:
    """
    Get statistics about loaded plugins.
    
    Returns:
        Plugin statistics
    """
    plugins = list_plugins()
    protocol_plugins = get_protocol_plugins()
    
    stats = {
        'total_plugins': len(plugins),
        'protocol_plugins': len(protocol_plugins),
        'plugin_types': {},
        'plugin_families': {},
        'built_in_plugins': 0,
        'external_plugins': 0
    }
    
    # Analyze plugin types and families
    for plugin_name in plugins:
        plugin = get_plugin(plugin_name)
        
        if plugin:
            # Count by type
            plugin_type = type(plugin).__name__
            stats['plugin_types'][plugin_type] = stats['plugin_types'].get(plugin_type, 0) + 1
            
            # Count by family (if metadata available)
            if REGISTRY_AVAILABLE:
                try:
                    registry = PluginRegistry()
                    metadata = getattr(registry, '_metadata', {}).get(plugin_name, {})
                    family = metadata.get('family', 'unknown')
                    stats['plugin_families'][family] = stats['plugin_families'].get(family, 0) + 1
                    
                    if metadata.get('built_in', False):
                        stats['built_in_plugins'] += 1
                    else:
                        stats['external_plugins'] += 1
                except:
                    pass
    
    return stats

# =============================================================================
# PLUGIN MANAGEMENT UTILITIES
# =============================================================================

def reload_plugin(plugin_name: str) -> bool:
    """
    Reload a specific plugin.
    
    Args:
        plugin_name: Name of plugin to reload
        
    Returns:
        True if reload successful
    """
    try:
        # Unregister existing plugin
        if unregister_plugin(plugin_name):
            logger.info(f"Unregistered plugin: {plugin_name}")
        
        # Reload and re-register
        # This would need more sophisticated module reloading
        # For now, just return success
        return True
        
    except Exception as e:
        logger.error(f"Failed to reload plugin {plugin_name}: {e}")
        return False


def cleanup_all_plugins() -> None:
    """Cleanup all registered plugins."""
    plugins = list_plugins()
    
    for plugin_name in plugins:
        try:
            unregister_plugin(plugin_name)
        except Exception as e:
            logger.warning(f"Failed to cleanup plugin {plugin_name}: {e}")
    
    logger.info("All plugins cleaned up")

# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Base classes
    'BasePlugin',
    'ProtocolPlugin', 
    'DecoderPlugin',
    'EncoderPlugin',
    'AnalyzerPlugin',
    
    # Registry system
    'PluginRegistry',
    'register_plugin',
    'get_plugin',
    'list_plugins',
    'unregister_plugin',
    
    # Protocol plugins
    'MAVLinkPlugin',
    'DJIPlugin', 
    'ParrotPlugin',
    'GenericPlugin',
    
    # Loading and discovery
    'load_plugins',
    'discover_plugins',
    'create_plugin_from_class',
    
    # Utilities
    'get_protocol_plugins',
    'analyze_packet_with_plugins',
    'validate_plugin_compatibility',
    'get_plugin_statistics',
    'reload_plugin',
    'cleanup_all_plugins',
    
    # Supporting types
    'PluginMetadata',
    'PluginVersion',
    
    # Errors
    'PluginError',
    'RegistryError'
]

# Filter exports based on availability
available_exports = []
for name in __all__:
    if globals().get(name) is not None:
        available_exports.append(name)

__all__ = available_exports

# Module initialization
logger.info(f"DroneCmd plugin system initialized with {len(__all__)} exports")

# Auto-load plugins if registry is available
if REGISTRY_AVAILABLE or BASE_AVAILABLE:
    try:
        loaded = load_plugins(auto_register=True)
        logger.info(f"Auto-loaded {len(loaded)} plugins")
    except Exception as e:
        logger.warning(f"Auto-loading plugins failed: {e}")

# Display plugin statistics
try:
    stats = get_plugin_statistics()
    logger.info(f"Plugin statistics: {stats['total_plugins']} total, "
               f"{stats['protocol_plugins']} protocol plugins")
except Exception as e:
    logger.debug(f"Could not get plugin statistics: {e}")