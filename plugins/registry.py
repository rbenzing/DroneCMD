#!/usr/bin/env python3
"""
Plugin Registry System

This module provides comprehensive plugin discovery, loading, and management
capabilities for the DroneCmd framework. It handles automatic plugin discovery,
dependency resolution, and provides a unified interface for accessing plugins.

Key Features:
- Automatic plugin discovery from multiple sources
- Plugin validation and dependency checking
- Dynamic loading and unloading
- Plugin lifecycle management
- Performance monitoring and statistics
- Plugin configuration management
- Error handling and recovery

Integration Points:
- Base plugin architecture
- Core signal processing systems
- Enhanced classification systems
- Protocol parsing systems
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
import pkgutil
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type, Union

from .base import (
    BasePlugin, PluginType, PluginCapability, PluginMetadata,
    ProtocolPlugin, InjectionPlugin, AnalysisPlugin, CompositePlugin,
    PluginError, PluginNotFoundError, PluginValidationError,
    validate_plugin_metadata, check_plugin_compatibility
)

# Configure module logger
logger = logging.getLogger(__name__)


class PluginLoadError(PluginError):
    """Exception raised when plugin loading fails."""
    pass


class PluginRegistryError(Exception):
    """Exception raised by plugin registry operations."""
    pass


class PluginInfo:
    """Information about a registered plugin."""
    
    def __init__(
        self,
        plugin_class: Type[BasePlugin],
        metadata: PluginMetadata,
        module_path: str,
        file_path: Optional[Path] = None
    ) -> None:
        """Initialize plugin info."""
        self.plugin_class = plugin_class
        self.metadata = metadata
        self.module_path = module_path
        self.file_path = file_path
        self.instance: Optional[BasePlugin] = None
        self.load_time: Optional[float] = None
        self.is_loaded = False
        self.is_initialized = False
        self.load_errors: List[str] = []
        self.validation_errors: List[str] = []
        
        # Performance tracking
        self.usage_count = 0
        self.total_processing_time = 0.0
        self.average_confidence = 0.0
        
    def __str__(self) -> str:
        """String representation."""
        status = "loaded" if self.is_loaded else "unloaded"
        return f"{self.metadata.name} v{self.metadata.version} ({status})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return (f"PluginInfo(name='{self.metadata.name}', "
                f"version='{self.metadata.version}', "
                f"type={self.metadata.plugin_type.value}, "
                f"loaded={self.is_loaded})")


class PluginLoader:
    """
    Plugin loader responsible for discovering and loading plugins.
    
    Supports loading plugins from:
    - Package modules
    - Individual Python files
    - Plugin directories
    - Entry points
    """
    
    def __init__(self) -> None:
        """Initialize plugin loader."""
        self._loaded_modules = set()
        self._load_lock = threading.Lock()
    
    def discover_plugins_in_package(
        self,
        package_name: str,
        recursive: bool = True
    ) -> List[Type[BasePlugin]]:
        """
        Discover plugins in a Python package.
        
        Args:
            package_name: Name of package to search
            recursive: Search recursively in subpackages
            
        Returns:
            List of discovered plugin classes
        """
        plugins = []
        
        try:
            package = importlib.import_module(package_name)
            
            if hasattr(package, '__path__'):
                # Search package modules
                for module_info in pkgutil.iter_modules(package.__path__, package_name + '.'):
                    try:
                        plugin_classes = self._load_module_plugins(module_info.name)
                        plugins.extend(plugin_classes)
                    except Exception as e:
                        logger.debug(f"Failed to load plugins from {module_info.name}: {e}")
                
                # Recursive search if requested
                if recursive:
                    for module_info in pkgutil.walk_packages(package.__path__, package_name + '.'):
                        try:
                            plugin_classes = self._load_module_plugins(module_info.name)
                            plugins.extend(plugin_classes)
                        except Exception as e:
                            logger.debug(f"Failed to load plugins from {module_info.name}: {e}")
            
        except ImportError as e:
            logger.warning(f"Failed to import package {package_name}: {e}")
        
        return plugins
    
    def discover_plugins_in_directory(
        self,
        directory: Union[str, Path],
        pattern: str = "*.py"
    ) -> List[Type[BasePlugin]]:
        """
        Discover plugins in a directory.
        
        Args:
            directory: Directory to search
            pattern: File pattern to match
            
        Returns:
            List of discovered plugin classes
        """
        plugins = []
        dir_path = Path(directory)
        
        if not dir_path.exists():
            logger.warning(f"Plugin directory not found: {directory}")
            return plugins
        
        for file_path in dir_path.glob(pattern):
            if file_path.name.startswith('_'):
                continue  # Skip private modules
            
            try:
                plugin_classes = self._load_file_plugins(file_path)
                plugins.extend(plugin_classes)
            except Exception as e:
                logger.debug(f"Failed to load plugins from {file_path}: {e}")
        
        return plugins
    
    def load_plugin_from_file(self, file_path: Union[str, Path]) -> List[Type[BasePlugin]]:
        """
        Load plugins from a specific file.
        
        Args:
            file_path: Path to Python file
            
        Returns:
            List of plugin classes found in file
        """
        return self._load_file_plugins(Path(file_path))
    
    def _load_module_plugins(self, module_name: str) -> List[Type[BasePlugin]]:
        """Load plugins from a module."""
        with self._load_lock:
            if module_name in self._loaded_modules:
                return []  # Already loaded
            
            try:
                module = importlib.import_module(module_name)
                self._loaded_modules.add(module_name)
                return self._extract_plugins_from_module(module)
                
            except Exception as e:
                logger.debug(f"Failed to load module {module_name}: {e}")
                return []
    
    def _load_file_plugins(self, file_path: Path) -> List[Type[BasePlugin]]:
        """Load plugins from a file."""
        with self._load_lock:
            file_key = str(file_path.resolve())
            if file_key in self._loaded_modules:
                return []  # Already loaded
            
            try:
                spec = importlib.util.spec_from_file_location(
                    f"plugin_{file_path.stem}",
                    file_path
                )
                
                if spec is None or spec.loader is None:
                    return []
                
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                self._loaded_modules.add(file_key)
                return self._extract_plugins_from_module(module)
                
            except Exception as e:
                logger.debug(f"Failed to load file {file_path}: {e}")
                return []
    
    def _extract_plugins_from_module(self, module) -> List[Type[BasePlugin]]:
        """Extract plugin classes from a module."""
        plugins = []
        
        for name, obj in inspect.getmembers(module, inspect.isclass):
            # Skip imported classes that aren't defined in this module
            if obj.__module__ != module.__name__:
                continue
            
            # Check if it's a plugin class
            if (issubclass(obj, BasePlugin) and 
                obj is not BasePlugin and
                not obj.__name__.startswith('Base')):
                
                try:
                    # Quick validation - can we get metadata?
                    if hasattr(obj, 'metadata'):
                        plugins.append(obj)
                        logger.debug(f"Found plugin class: {obj.__name__}")
                except Exception as e:
                    logger.debug(f"Invalid plugin class {obj.__name__}: {e}")
        
        return plugins


class PluginRegistry:
    """
    Central registry for managing all plugins in the system.
    
    Provides a unified interface for plugin discovery, loading, configuration,
    and lifecycle management.
    """
    
    def __init__(self) -> None:
        """Initialize plugin registry."""
        self._plugins: Dict[str, PluginInfo] = {}
        self._plugins_by_type: Dict[PluginType, List[str]] = defaultdict(list)
        self._plugins_by_capability: Dict[PluginCapability, List[str]] = defaultdict(list)
        self._plugins_by_protocol: Dict[str, List[str]] = defaultdict(list)
        
        self._loader = PluginLoader()
        self._registry_lock = threading.RLock()
        
        # Statistics
        self._stats = {
            'plugins_registered': 0,
            'plugins_loaded': 0,
            'plugins_initialized': 0,
            'discovery_runs': 0,
            'load_errors': 0,
            'validation_errors': 0
        }
        
        logger.info("Initialized plugin registry")
    
    def discover_and_register_plugins(
        self,
        sources: Optional[List[Union[str, Path]]] = None,
        auto_load: bool = False
    ) -> int:
        """
        Discover and register plugins from various sources.
        
        Args:
            sources: List of package names or directory paths to search
            auto_load: Automatically load discovered plugins
            
        Returns:
            Number of plugins registered
        """
        with self._registry_lock:
            self._stats['discovery_runs'] += 1
            initial_count = len(self._plugins)
            
            # Default sources if none provided
            if sources is None:
                sources = [
                    'dronecmd.plugins',  # Package plugins
                    'plugins',           # Local plugins directory
                ]
            
            # Discover from each source
            for source in sources:
                try:
                    if isinstance(source, (str, Path)) and Path(source).exists():
                        # Directory source
                        plugin_classes = self._loader.discover_plugins_in_directory(source)
                    else:
                        # Package source
                        plugin_classes = self._loader.discover_plugins_in_package(str(source))
                    
                    # Register discovered plugins
                    for plugin_class in plugin_classes:
                        self._register_plugin_class(plugin_class, auto_load)
                        
                except Exception as e:
                    logger.warning(f"Failed to discover plugins from {source}: {e}")
            
            registered_count = len(self._plugins) - initial_count
            logger.info(f"Discovered and registered {registered_count} plugins")
            return registered_count
    
    def register_plugin(
        self,
        plugin: Union[Type[BasePlugin], BasePlugin],
        auto_load: bool = True
    ) -> bool:
        """
        Register a single plugin.
        
        Args:
            plugin: Plugin class or instance to register
            auto_load: Automatically load the plugin
            
        Returns:
            True if registration successful
        """
        with self._registry_lock:
            try:
                if isinstance(plugin, type):
                    return self._register_plugin_class(plugin, auto_load)
                else:
                    return self._register_plugin_instance(plugin)
                    
            except Exception as e:
                logger.error(f"Failed to register plugin: {e}")
                self._stats['load_errors'] += 1
                return False
    
    def _register_plugin_class(
        self,
        plugin_class: Type[BasePlugin],
        auto_load: bool = False
    ) -> bool:
        """Register a plugin class."""
        try:
            # Create temporary instance to get metadata
            temp_instance = plugin_class()
            metadata = temp_instance.metadata
            
            # Validate metadata
            validation_errors = validate_plugin_metadata(metadata)
            if validation_errors:
                logger.warning(f"Plugin {metadata.name} validation failed: {validation_errors}")
                self._stats['validation_errors'] += 1
                return False
            
            # Check for name conflicts
            if metadata.name in self._plugins:
                existing = self._plugins[metadata.name]
                logger.warning(f"Plugin name conflict: {metadata.name} already registered "
                             f"(existing: {existing.metadata.version}, "
                             f"new: {metadata.version})")
                return False
            
            # Create plugin info
            plugin_info = PluginInfo(
                plugin_class=plugin_class,
                metadata=metadata,
                module_path=plugin_class.__module__
            )
            
            # Store plugin info
            self._plugins[metadata.name] = plugin_info
            self._plugins_by_type[metadata.plugin_type].append(metadata.name)
            
            for capability in metadata.capabilities:
                self._plugins_by_capability[capability].append(metadata.name)
            
            for protocol in metadata.supported_protocols:
                self._plugins_by_protocol[protocol].append(metadata.name)
            
            self._stats['plugins_registered'] += 1
            
            # Auto-load if requested
            if auto_load:
                self.load_plugin(metadata.name)
            
            logger.info(f"Registered plugin: {metadata.name} v{metadata.version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register plugin class {plugin_class.__name__}: {e}")
            self._stats['load_errors'] += 1
            return False
    
    def _register_plugin_instance(self, plugin: BasePlugin) -> bool:
        """Register a plugin instance."""
        try:
            metadata = plugin.metadata
            
            # Validate metadata
            validation_errors = validate_plugin_metadata(metadata)
            if validation_errors:
                logger.warning(f"Plugin {metadata.name} validation failed: {validation_errors}")
                self._stats['validation_errors'] += 1
                return False
            
            # Create plugin info with existing instance
            plugin_info = PluginInfo(
                plugin_class=type(plugin),
                metadata=metadata,
                module_path=type(plugin).__module__
            )
            plugin_info.instance = plugin
            plugin_info.is_loaded = True
            plugin_info.load_time = time.time()
            
            # Store plugin info
            self._plugins[metadata.name] = plugin_info
            self._plugins_by_type[metadata.plugin_type].append(metadata.name)
            
            for capability in metadata.capabilities:
                self._plugins_by_capability[capability].append(metadata.name)
            
            for protocol in metadata.supported_protocols:
                self._plugins_by_protocol[protocol].append(metadata.name)
            
            self._stats['plugins_registered'] += 1
            self._stats['plugins_loaded'] += 1
            
            logger.info(f"Registered plugin instance: {metadata.name} v{metadata.version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register plugin instance: {e}")
            self._stats['load_errors'] += 1
            return False
    
    def load_plugin(self, plugin_name: str, **kwargs: Any) -> bool:
        """
        Load a registered plugin.
        
        Args:
            plugin_name: Name of plugin to load
            **kwargs: Configuration parameters for plugin
            
        Returns:
            True if loading successful
        """
        with self._registry_lock:
            if plugin_name not in self._plugins:
                raise PluginNotFoundError(f"Plugin '{plugin_name}' not registered")
            
            plugin_info = self._plugins[plugin_name]
            
            if plugin_info.is_loaded:
                logger.debug(f"Plugin {plugin_name} already loaded")
                return True
            
            try:
                # Create plugin instance
                plugin_info.instance = plugin_info.plugin_class()
                
                # Configure plugin
                if kwargs:
                    plugin_info.instance.configure(**kwargs)
                
                # Initialize plugin
                if plugin_info.instance.initialize(**kwargs):
                    plugin_info.is_loaded = True
                    plugin_info.is_initialized = True
                    plugin_info.load_time = time.time()
                    self._stats['plugins_loaded'] += 1
                    self._stats['plugins_initialized'] += 1
                    
                    logger.info(f"Loaded plugin: {plugin_name}")
                    return True
                else:
                    plugin_info.load_errors.append("Plugin initialization failed")
                    logger.error(f"Failed to initialize plugin: {plugin_name}")
                    return False
                    
            except Exception as e:
                error_msg = f"Failed to load plugin {plugin_name}: {e}"
                plugin_info.load_errors.append(error_msg)
                logger.error(error_msg)
                self._stats['load_errors'] += 1
                return False
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """
        Unload a plugin.
        
        Args:
            plugin_name: Name of plugin to unload
            
        Returns:
            True if unloading successful
        """
        with self._registry_lock:
            if plugin_name not in self._plugins:
                raise PluginNotFoundError(f"Plugin '{plugin_name}' not registered")
            
            plugin_info = self._plugins[plugin_name]
            
            if not plugin_info.is_loaded:
                return True
            
            try:
                if plugin_info.instance:
                    plugin_info.instance.cleanup()
                    plugin_info.instance = None
                
                plugin_info.is_loaded = False
                plugin_info.is_initialized = False
                self._stats['plugins_loaded'] -= 1
                
                logger.info(f"Unloaded plugin: {plugin_name}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to unload plugin {plugin_name}: {e}")
                return False
    
    def get_plugin(self, plugin_name: str, auto_load: bool = True) -> Optional[BasePlugin]:
        """
        Get a plugin instance.
        
        Args:
            plugin_name: Name of plugin to get
            auto_load: Automatically load plugin if not loaded
            
        Returns:
            Plugin instance or None if not available
        """
        with self._registry_lock:
            if plugin_name not in self._plugins:
                return None
            
            plugin_info = self._plugins[plugin_name]
            
            if not plugin_info.is_loaded and auto_load:
                if not self.load_plugin(plugin_name):
                    return None
            
            return plugin_info.instance
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> List[BasePlugin]:
        """Get all loaded plugins of specified type."""
        plugins = []
        
        for plugin_name in self._plugins_by_type[plugin_type]:
            plugin = self.get_plugin(plugin_name)
            if plugin:
                plugins.append(plugin)
        
        return plugins
    
    def get_plugins_by_capability(self, capability: PluginCapability) -> List[BasePlugin]:
        """Get all loaded plugins with specified capability."""
        plugins = []
        
        for plugin_name in self._plugins_by_capability[capability]:
            plugin = self.get_plugin(plugin_name)
            if plugin:
                plugins.append(plugin)
        
        return plugins
    
    def get_plugins_for_protocol(self, protocol: str) -> List[BasePlugin]:
        """Get all loaded plugins supporting specified protocol."""
        plugins = []
        
        for plugin_name in self._plugins_by_protocol[protocol]:
            plugin = self.get_plugin(plugin_name)
            if plugin:
                plugins.append(plugin)
        
        return plugins
    
    def find_best_plugin(
        self,
        plugin_type: Optional[PluginType] = None,
        capability: Optional[PluginCapability] = None,
        protocol: Optional[str] = None,
        **criteria: Any
    ) -> Optional[BasePlugin]:
        """
        Find the best plugin matching criteria.
        
        Args:
            plugin_type: Required plugin type
            capability: Required capability
            protocol: Required protocol support
            **criteria: Additional matching criteria
            
        Returns:
            Best matching plugin or None
        """
        candidates = []
        
        # Get candidate plugins
        if protocol:
            candidates.extend(self.get_plugins_for_protocol(protocol))
        elif capability:
            candidates.extend(self.get_plugins_by_capability(capability))
        elif plugin_type:
            candidates.extend(self.get_plugins_by_type(plugin_type))
        else:
            # All loaded plugins
            candidates = [info.instance for info in self._plugins.values() 
                         if info.instance is not None]
        
        # Filter by additional criteria
        filtered_candidates = []
        for plugin in candidates:
            if self._matches_criteria(plugin, plugin_type, capability, protocol, **criteria):
                filtered_candidates.append(plugin)
        
        # Return best match (for now, just return first)
        # TODO: Implement scoring system based on performance, confidence, etc.
        return filtered_candidates[0] if filtered_candidates else None
    
    def _matches_criteria(
        self,
        plugin: BasePlugin,
        plugin_type: Optional[PluginType],
        capability: Optional[PluginCapability],
        protocol: Optional[str],
        **criteria: Any
    ) -> bool:
        """Check if plugin matches criteria."""
        metadata = plugin.metadata
        
        # Check plugin type
        if plugin_type and metadata.plugin_type != plugin_type:
            return False
        
        # Check capability
        if capability and capability not in metadata.capabilities:
            return False
        
        # Check protocol
        if protocol and protocol not in metadata.supported_protocols:
            return False
        
        # Check additional criteria
        for key, value in criteria.items():
            if hasattr(metadata, key):
                if getattr(metadata, key) != value:
                    return False
        
        return True
    
    def list_plugins(
        self,
        loaded_only: bool = False,
        plugin_type: Optional[PluginType] = None
    ) -> List[PluginInfo]:
        """
        List registered plugins.
        
        Args:
            loaded_only: Only return loaded plugins
            plugin_type: Filter by plugin type
            
        Returns:
            List of plugin information
        """
        plugins = []
        
        for plugin_info in self._plugins.values():
            if loaded_only and not plugin_info.is_loaded:
                continue
            
            if plugin_type and plugin_info.metadata.plugin_type != plugin_type:
                continue
            
            plugins.append(plugin_info)
        
        return sorted(plugins, key=lambda p: p.metadata.name)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        stats = self._stats.copy()
        
        # Add current state info
        stats.update({
            'total_plugins_registered': len(self._plugins),
            'plugins_by_type': {
                ptype.value: len(plugins) 
                for ptype, plugins in self._plugins_by_type.items()
            },
            'plugins_by_capability': {
                cap.value: len(plugins)
                for cap, plugins in self._plugins_by_capability.items()
            },
            'protocols_supported': list(self._plugins_by_protocol.keys()),
            'currently_loaded': sum(1 for p in self._plugins.values() if p.is_loaded),
            'load_success_rate': (
                self._stats['plugins_loaded'] / max(1, self._stats['plugins_registered'])
            ) * 100
        })
        
        return stats
    
    def cleanup(self) -> None:
        """Cleanup all plugins and registry resources."""
        with self._registry_lock:
            for plugin_name in list(self._plugins.keys()):
                try:
                    self.unload_plugin(plugin_name)
                except Exception as e:
                    logger.warning(f"Error unloading plugin {plugin_name}: {e}")
            
            self._plugins.clear()
            self._plugins_by_type.clear()
            self._plugins_by_capability.clear()
            self._plugins_by_protocol.clear()
            
            logger.info("Plugin registry cleaned up")


# Global plugin registry instance
_global_registry: Optional[PluginRegistry] = None
_registry_lock = threading.Lock()


def get_plugin_registry() -> PluginRegistry:
    """Get the global plugin registry instance."""
    global _global_registry
    
    with _registry_lock:
        if _global_registry is None:
            _global_registry = PluginRegistry()
            
            # Auto-discover plugins on first access
            try:
                _global_registry.discover_and_register_plugins()
            except Exception as e:
                logger.warning(f"Failed to auto-discover plugins: {e}")
        
        return _global_registry


def reset_plugin_registry() -> None:
    """Reset the global plugin registry."""
    global _global_registry
    
    with _registry_lock:
        if _global_registry:
            _global_registry.cleanup()
        _global_registry = None


# Convenience functions for common operations

def register_plugin(plugin: Union[Type[BasePlugin], BasePlugin]) -> bool:
    """Register a plugin with the global registry."""
    return get_plugin_registry().register_plugin(plugin)


def get_plugin(plugin_name: str, auto_load: bool = True) -> Optional[BasePlugin]:
    """Get a plugin from the global registry."""
    return get_plugin_registry().get_plugin(plugin_name, auto_load)


def get_protocol_plugins() -> List[ProtocolPlugin]:
    """Get all protocol plugins."""
    plugins = get_plugin_registry().get_plugins_by_type(PluginType.PROTOCOL)
    return [p for p in plugins if isinstance(p, ProtocolPlugin)]


def get_injection_plugins() -> List[InjectionPlugin]:
    """Get all injection plugins."""
    plugins = get_plugin_registry().get_plugins_by_type(PluginType.INJECTION)
    return [p for p in plugins if isinstance(p, InjectionPlugin)]


def find_plugins_for_protocol(protocol: str) -> List[BasePlugin]:
    """Find plugins supporting a specific protocol."""
    return get_plugin_registry().get_plugins_for_protocol(protocol)


if __name__ == "__main__":
    # Test plugin registry
    logging.basicConfig(level=logging.INFO)
    
    print("=== Plugin Registry Demo ===")
    
    # Get registry and discover plugins
    registry = get_plugin_registry()
    
    # Show statistics
    stats = registry.get_statistics()
    print(f"Registry stats: {stats}")
    
    # List registered plugins
    plugins = registry.list_plugins()
    print(f"\nRegistered plugins ({len(plugins)}):")
    for plugin_info in plugins:
        print(f"  - {plugin_info}")
    
    # Test protocol plugins
    protocol_plugins = get_protocol_plugins()
    print(f"\nProtocol plugins: {len(protocol_plugins)}")
    
    # Cleanup
    registry.cleanup()