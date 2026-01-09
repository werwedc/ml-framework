# Spec: Model Zoo Extensibility and Plugin Architecture

## Overview
Implement a plugin architecture to allow custom model registries and extensions to the Model Zoo.

## Requirements

### 1. IModelRegistryPlugin Interface
Base interface for registry plugins:
- `string RegistryName`: Registry identifier
- `int Priority`: Plugin priority (higher = checked first)
- `Task<ModelMetadata> GetModelMetadataAsync(string modelName, string version = null)`: Get model metadata
- `Task<Stream> DownloadModelAsync(ModelMetadata metadata, IProgress<double> progress = null)`: Download model
- `Task<bool> ModelExistsAsync(string modelName, string version = null)`: Check if model exists
- `Task<string[]> ListModelsAsync()`: List all models in registry
- `bool CanHandle(string modelName)`: Check if registry can handle model name

### 2. PluginManager Class
Manage model registry plugins:
- `RegisterPlugin(IModelRegistryPlugin plugin)`: Register a plugin
- `UnregisterPlugin(string registryName)`: Unregister a plugin
- `GetPlugin(string registryName)`: Get registered plugin
- `ListPlugins()`: List all registered plugins
- `FindPlugin(string modelName)`: Find plugin that can handle model name
- `GetAllModelMetadata(string modelName, string version = null)`: Query all plugins for model metadata
- `GetBestSource(string modelName, string version = null)`: Find best source (based on priority)

### 3. CustomModelRegistry Base Class
Abstract base class for custom registries:
- Implements IModelRegistryPlugin
- Provides common functionality (HTTP client, caching, logging)
- Subclasses override specific methods as needed
- Built-in retry logic
- Built-in caching support

### 4. RegistryPluginAttribute
Attribute to mark plugins for auto-discovery:
- `[RegistryPlugin(name = "my-registry", priority = 100)]`
- Automatic registration on startup
- Optional: Load from specific assembly

### 5. PluginConfiguration
Configuration for plugins:
- `[RegistryPluginConfig]`: Attribute to mark config classes
- Properties for URL, authentication, cache settings, etc.
- Load from appsettings.json or environment variables
- Support for encrypted credentials

### 6. PluginDiscovery
Discover plugins automatically:
- Scan assemblies for IModelRegistryPlugin implementations
- Scan for RegistryPluginAttribute
- Register discovered plugins on startup
- Support for plugin directory (load external DLLs)
- Support for hot-reloading plugins

### 7. Authentication Support for Plugins
Base authentication interface:
- `IRegistryAuthentication`: Base for authentication implementations
- `ApiKeyAuthentication`: API key authentication
- `TokenAuthentication`: OAuth token authentication
- `BasicAuthentication`: Username/password authentication
- `CustomAuthentication`: Custom auth logic

### 8. Plugin Extensions
Allow plugins to extend ModelZoo functionality:
- `IModelZooExtension`: Base interface for extensions
- Extension points: Pre-download, Post-download, Pre-load, Post-load
- Example extensions: Model conversion, format validation, encryption/decryption

### 9. Built-in Plugins
Include standard plugins:
- `LocalFileRegistry`: Load models from local file system
- `MemoryRegistry`: In-memory registry for testing
- `RemoteRegistry`: HTTP-based remote registry
- `CloudStorageRegistry`: AWS S3, Azure Blob, GCS

### 10. Plugin Marketplace (Optional)
Future-proofing for plugin distribution:
- Plugin manifest format (JSON)
- Plugin metadata (name, version, description, dependencies)
- Plugin installation mechanism
- Update checking

### 11. Unit Tests
Test cases for:
- Register and unregister plugins
- Plugin priority handling
- Find plugin for model name
- Custom registry plugin implementation
- Plugin auto-discovery
- Authentication configuration
- Plugin extensions lifecycle
- Multiple plugins handling same model name
- Edge cases (no plugins, plugin throws exception)

### 12. Example Plugin
Create example plugin for demonstration:
- `ExampleRegistry`: Simple HTTP-based registry
- Demonstrates plugin API usage
- Included in tests

## Files to Create
- `src/ModelZoo/Plugins/IModelRegistryPlugin.cs`
- `src/ModelZoo/Plugins/PluginManager.cs`
- `src/ModelZoo/Plugins/CustomModelRegistry.cs`
- `src/ModelZoo/Plugins/RegistryPluginAttribute.cs`
- `src/ModelZoo/Plugins/PluginConfiguration.cs`
- `src/ModelZoo/Plugins/PluginDiscovery.cs`
- `src/ModelZoo/Plugins/IRegistryAuthentication.cs`
- `src/ModelZoo/Plugins/IModelZooExtension.cs`
- `src/ModelZoo/Plugins/BuiltIn/LocalFileRegistry.cs`
- `src/ModelZoo/Plugins/BuiltIn/MemoryRegistry.cs`
- `src/ModelZoo/Plugins/BuiltIn/RemoteRegistry.cs`
- `tests/ModelZooTests/Plugins/PluginTests.cs`
- `tests/ModelZooTests/Plugins/ExampleRegistry.cs`

## Dependencies
- `ModelMetadata` (from spec_model_metadata.md)
- `ModelZoo` (from spec_model_zoo_load_api.md)
- System.Reflection (for plugin discovery)
- System.IO (for loading plugin DLLs)

## Success Criteria
- Can register and use custom registries
- Plugin auto-discovery works
- Plugin priority is respected
- Multiple plugins coexist without conflicts
- Test coverage > 85%
- Example plugin demonstrates usage clearly
