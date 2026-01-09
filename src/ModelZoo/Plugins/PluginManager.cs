using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace MLFramework.ModelZoo.Plugins
{
    /// <summary>
    /// Manages model registry plugins and extensions.
    /// Provides methods to register, unregister, and query plugins.
    /// </summary>
    public class PluginManager
    {
        private readonly Dictionary<string, IModelRegistryPlugin> _plugins;
        private readonly List<IModelZooExtension> _extensions;
        private readonly object _lock = new object();

        /// <summary>
        /// Initializes a new instance of the PluginManager.
        /// </summary>
        public PluginManager()
        {
            _plugins = new Dictionary<string, IModelRegistryPlugin>(StringComparer.OrdinalIgnoreCase);
            _extensions = new List<IModelZooExtension>();
        }

        /// <summary>
        /// Registers a plugin.
        /// </summary>
        /// <param name="plugin">The plugin to register.</param>
        /// <exception cref="ArgumentNullException">Thrown when plugin is null.</exception>
        /// <exception cref="InvalidOperationException">Thrown when a plugin with the same name is already registered.</exception>
        public void RegisterPlugin(IModelRegistryPlugin plugin)
        {
            if (plugin == null)
            {
                throw new ArgumentNullException(nameof(plugin));
            }

            if (string.IsNullOrEmpty(plugin.RegistryName))
            {
                throw new ArgumentException("Plugin registry name cannot be null or empty", nameof(plugin));
            }

            lock (_lock)
            {
                if (_plugins.ContainsKey(plugin.RegistryName))
                {
                    throw new InvalidOperationException(
                        $"Plugin with registry name '{plugin.RegistryName}' is already registered");
                }

                _plugins[plugin.RegistryName] = plugin;
                Console.WriteLine($"[PluginManager] Registered plugin: {plugin.RegistryName} (Priority: {plugin.Priority})");
            }
        }

        /// <summary>
        /// Unregisters a plugin by registry name.
        /// </summary>
        /// <param name="registryName">The registry name of the plugin to unregister.</param>
        /// <returns>True if the plugin was unregistered, false if not found.</returns>
        public bool UnregisterPlugin(string registryName)
        {
            if (string.IsNullOrEmpty(registryName))
            {
                throw new ArgumentException("Registry name cannot be null or empty", nameof(registryName));
            }

            lock (_lock)
            {
                if (_plugins.Remove(registryName))
                {
                    Console.WriteLine($"[PluginManager] Unregistered plugin: {registryName}");
                    return true;
                }
                return false;
            }
        }

        /// <summary>
        /// Gets a registered plugin by registry name.
        /// </summary>
        /// <param name="registryName">The registry name of the plugin.</param>
        /// <returns>The plugin, or null if not found.</returns>
        public IModelRegistryPlugin GetPlugin(string registryName)
        {
            if (string.IsNullOrEmpty(registryName))
            {
                throw new ArgumentException("Registry name cannot be null or empty", nameof(registryName));
            }

            lock (_lock)
            {
                _plugins.TryGetValue(registryName, out var plugin);
                return plugin;
            }
        }

        /// <summary>
        /// Lists all registered plugins.
        /// </summary>
        /// <returns>A list of registry names.</returns>
        public IReadOnlyList<string> ListPlugins()
        {
            lock (_lock)
            {
                return _plugins.Keys.ToList().AsReadOnly();
            }
        }

        /// <summary>
        /// Gets all registered plugins sorted by priority (highest first).
        /// </summary>
        /// <returns>A list of plugins sorted by priority.</returns>
        public IReadOnlyList<IModelRegistryPlugin> GetAllPlugins()
        {
            lock (_lock)
            {
                return _plugins.Values.OrderByDescending(p => p.Priority).ToList().AsReadOnly();
            }
        }

        /// <summary>
        /// Finds the plugin that can handle the specified model name.
        /// Returns the plugin with highest priority among those that can handle it.
        /// </summary>
        /// <param name="modelName">The name of the model.</param>
        /// <returns>The plugin that can handle the model, or null if none found.</returns>
        public IModelRegistryPlugin FindPlugin(string modelName)
        {
            if (string.IsNullOrEmpty(modelName))
            {
                throw new ArgumentException("Model name cannot be null or empty", nameof(modelName));
            }

            lock (_lock)
            {
                return _plugins.Values
                    .Where(p => p.CanHandle(modelName))
                    .OrderByDescending(p => p.Priority)
                    .FirstOrDefault();
            }
        }

        /// <summary>
        /// Queries all plugins for model metadata.
        /// Returns metadata from the first plugin that has the model.
        /// </summary>
        /// <param name="modelName">The name of the model.</param>
        /// <param name="version">Optional version.</param>
        /// <returns>Model metadata, or null if not found in any plugin.</returns>
        public async Task<ModelVersioning.ModelMetadata> GetAllModelMetadata(string modelName, string version = null)
        {
            if (string.IsNullOrEmpty(modelName))
            {
                throw new ArgumentException("Model name cannot be null or empty", nameof(modelName));
            }

            var plugins = GetAllPlugins();

            foreach (var plugin in plugins)
            {
                try
                {
                    if (await plugin.ModelExistsAsync(modelName, version))
                    {
                        Console.WriteLine($"[PluginManager] Found model {modelName} in registry {plugin.RegistryName}");
                        return await plugin.GetModelMetadataAsync(modelName, version);
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[PluginManager] Error checking registry {plugin.RegistryName}: {ex.Message}");
                }
            }

            return null;
        }

        /// <summary>
        /// Finds the best source (plugin with highest priority) for a model.
        /// </summary>
        /// <param name="modelName">The name of the model.</param>
        /// <param name="version">Optional version.</param>
        /// <returns>The best plugin, or null if not found.</returns>
        public async Task<IModelRegistryPlugin> GetBestSource(string modelName, string version = null)
        {
            if (string.IsNullOrEmpty(modelName))
            {
                throw new ArgumentException("Model name cannot be null or empty", nameof(modelName));
            }

            var plugins = GetAllPlugins();

            foreach (var plugin in plugins)
            {
                try
                {
                    if (await plugin.ModelExistsAsync(modelName, version))
                    {
                        Console.WriteLine($"[PluginManager] Best source for {modelName}: {plugin.RegistryName}");
                        return plugin;
                    }
                }
                catch
                {
                    // Skip this plugin and try the next one
                    continue;
                }
            }

            return null;
        }

        /// <summary>
        /// Registers an extension.
        /// </summary>
        /// <param name="extension">The extension to register.</param>
        public void RegisterExtension(IModelZooExtension extension)
        {
            if (extension == null)
            {
                throw new ArgumentNullException(nameof(extension));
            }

            lock (_lock)
            {
                _extensions.Add(extension);
                _extensions.Sort((e1, e2) => e2.Priority.CompareTo(e1.Priority));
                Console.WriteLine($"[PluginManager] Registered extension: {extension.ExtensionName} (Priority: {extension.Priority})");
            }
        }

        /// <summary>
        /// Unregisters an extension by name.
        /// </summary>
        /// <param name="extensionName">The name of the extension to unregister.</param>
        /// <returns>True if the extension was unregistered, false if not found.</returns>
        public bool UnregisterExtension(string extensionName)
        {
            if (string.IsNullOrEmpty(extensionName))
            {
                throw new ArgumentException("Extension name cannot be null or empty", nameof(extensionName));
            }

            lock (_lock)
            {
                var extension = _extensions.FirstOrDefault(e => e.ExtensionName == extensionName);
                if (extension != null)
                {
                    _extensions.Remove(extension);
                    Console.WriteLine($"[PluginManager] Unregistered extension: {extensionName}");
                    return true;
                }
                return false;
            }
        }

        /// <summary>
        /// Gets all registered extensions sorted by priority.
        /// </summary>
        /// <returns>A read-only list of extensions.</returns>
        public IReadOnlyList<IModelZooExtension> GetAllExtensions()
        {
            lock (_lock)
            {
                return _extensions.ToList().AsReadOnly();
            }
        }

        /// <summary>
        /// Executes pre-download extensions.
        /// </summary>
        /// <param name="metadata">Model metadata.</param>
        public async Task ExecutePreDownloadExtensions(ModelVersioning.ModelMetadata metadata)
        {
            var extensions = GetAllExtensions();

            foreach (var extension in extensions)
            {
                try
                {
                    await extension.PreDownloadAsync(metadata);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[PluginManager] Error in extension {extension.ExtensionName}: {ex.Message}");
                }
            }
        }

        /// <summary>
        /// Executes post-download extensions.
        /// </summary>
        /// <param name="metadata">Model metadata.</param>
        /// <param name="stream">Model stream.</param>
        /// <returns>The potentially modified stream.</returns>
        public async Task<System.IO.Stream> ExecutePostDownloadExtensions(ModelVersioning.ModelMetadata metadata, System.IO.Stream stream)
        {
            var extensions = GetAllExtensions();
            var currentStream = stream;

            foreach (var extension in extensions)
            {
                try
                {
                    var newStream = await extension.PostDownloadAsync(metadata, currentStream);
                    if (newStream != null)
                    {
                        currentStream = newStream;
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[PluginManager] Error in extension {extension.ExtensionName}: {ex.Message}");
                }
            }

            return currentStream;
        }

        /// <summary>
        /// Executes pre-load extensions.
        /// </summary>
        /// <param name="metadata">Model metadata.</param>
        public async Task ExecutePreLoadExtensions(ModelVersioning.ModelMetadata metadata)
        {
            var extensions = GetAllExtensions();

            foreach (var extension in extensions)
            {
                try
                {
                    await extension.PreLoadAsync(metadata);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[PluginManager] Error in extension {extension.ExtensionName}: {ex.Message}");
                }
            }
        }

        /// <summary>
        /// Executes post-load extensions.
        /// </summary>
        /// <param name="metadata">Model metadata.</param>
        public async Task ExecutePostLoadExtensions(ModelVersioning.ModelMetadata metadata)
        {
            var extensions = GetAllExtensions();

            foreach (var extension in extensions)
            {
                try
                {
                    await extension.PostLoadAsync(metadata);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[PluginManager] Error in extension {extension.ExtensionName}: {ex.Message}");
                }
            }
        }

        /// <summary>
        /// Gets the number of registered plugins.
        /// </summary>
        public int PluginCount
        {
            get
            {
                lock (_lock)
                {
                    return _plugins.Count;
                }
            }
        }

        /// <summary>
        /// Gets the number of registered extensions.
        /// </summary>
        public int ExtensionCount
        {
            get
            {
                lock (_lock)
                {
                    return _extensions.Count;
                }
            }
        }

        /// <summary>
        /// Clears all registered plugins and extensions.
        /// </summary>
        public void ClearAll()
        {
            lock (_lock)
            {
                _plugins.Clear();
                _extensions.Clear();
                Console.WriteLine("[PluginManager] Cleared all plugins and extensions");
            }
        }
    }
}
