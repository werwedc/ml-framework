using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using MLFramework.Core;
using MLFramework.NN;

namespace MLFramework.ModelZoo
{
    /// <summary>
    /// Primary API for loading pre-trained models from the Model Zoo with caching support.
    /// </summary>
    public static class ModelZoo
    {
        private static readonly ModelRegistry _registry;
        private static readonly ModelDownloadService _downloadService;
        private static readonly ModelCacheManager _cacheManager;
        private static readonly ModelDeserializer _deserializer;
        private static readonly ModelZooConfiguration _configuration;
        private static readonly LoadFlow _loadFlow;
        private static readonly object _lock = new object();

        static ModelZoo()
        {
            _registry = new ModelRegistry();
            _downloadService = new ModelDownloadService();
            _cacheManager = new ModelCacheManager();
            _deserializer = new ModelDeserializer();
            _configuration = new ModelZooConfiguration();
            _loadFlow = new LoadFlow(_registry, _downloadService, _cacheManager, _deserializer, _configuration);

            // Load default models
            LoadDefaultModels();
        }

        /// <summary>
        /// Gets the global model registry.
        /// </summary>
        public static ModelRegistry Registry => _registry;

        /// <summary>
        /// Gets the global cache manager.
        /// </summary>
        public static ModelCacheManager CacheManager => _cacheManager;

        /// <summary>
        /// Gets or sets the global ModelZoo configuration.
        /// </summary>
        public static ModelZooConfiguration Configuration
        {
            get => _configuration;
            set
            {
                if (value == null)
                    throw new ArgumentNullException(nameof(value));

                lock (_lock)
                {
                    _configuration.DefaultDevice = value.DefaultDevice;
                    _configuration.CacheEnabled = value.CacheEnabled;
                    _configuration.AutoDownloadEnabled = value.AutoDownloadEnabled;
                    _configuration.DefaultDownloadTimeoutMs = value.DefaultDownloadTimeoutMs;
                }
            }
        }

        /// <summary>
        /// Loads a model from the Model Zoo.
        /// </summary>
        /// <typeparam name="T">The type of the model (must inherit from Module).</typeparam>
        /// <param name="modelName">The model name (e.g., "resnet50").</param>
        /// <param name="version">The model version (null for latest).</param>
        /// <param name="variant">The model variant (optional).</param>
        /// <param name="pretrained">Whether to load pre-trained weights (default: true).</param>
        /// <param name="device">The target device (null for default).</param>
        /// <returns>The loaded model.</returns>
        /// <exception cref="ModelNotFoundException">Thrown when the model is not found in the registry.</exception>
        /// <exception cref="VersionNotFoundException">Thrown when the specified version is not found.</exception>
        /// <exception cref="DownloadFailedException">Thrown when the download fails.</exception>
        /// <exception cref="DeserializationException">Thrown when model deserialization fails.</exception>
        /// <exception cref="IncompatibleModelException">Thrown when the model architecture doesn't match.</exception>
        public static async Task<T> LoadAsync<T>(
            string modelName,
            string? version = null,
            string? variant = null,
            bool pretrained = true,
            Device? device = null) where T : Module
        {
            return await _loadFlow.ExecuteAsync<T>(modelName, version, variant, pretrained, device);
        }

        /// <summary>
        /// Loads a model from the Model Zoo (synchronous version).
        /// </summary>
        /// <typeparam name="T">The type of the model (must inherit from Module).</typeparam>
        /// <param name="modelName">The model name (e.g., "resnet50").</param>
        /// <param name="version">The model version (null for latest).</param>
        /// <param name="variant">The model variant (optional).</param>
        /// <param name="pretrained">Whether to load pre-trained weights (default: true).</param>
        /// <param name="device">The target device (null for default).</param>
        /// <returns>The loaded model.</returns>
        public static T Load<T>(
            string modelName,
            string? version = null,
            string? variant = null,
            bool pretrained = true,
            Device? device = null) where T : Module
        {
            return LoadAsync<T>(modelName, version, variant, pretrained, device)
                .GetAwaiter()
                .GetResult();
        }

        /// <summary>
        /// Loads a model from the Model Zoo without type specification.
        /// </summary>
        /// <param name="modelName">The model name (e.g., "resnet50").</param>
        /// <param name="version">The model version (null for latest).</param>
        /// <param name="variant">The model variant (optional).</param>
        /// <param name="pretrained">Whether to load pre-trained weights (default: true).</param>
        /// <param name="device">The target device (null for default).</param>
        /// <returns>The loaded model as a Module.</returns>
        public static Module Load(
            string modelName,
            string? version = null,
            string? variant = null,
            bool pretrained = true,
            Device? device = null)
        {
            return Load<Module>(modelName, version, variant, pretrained, device);
        }

        /// <summary>
        /// Loads a model from a local file path.
        /// </summary>
        /// <typeparam name="T">The type of the model (must inherit from Module).</typeparam>
        /// <param name="path">The path to the local model file.</param>
        /// <param name="device">The target device (null for default).</param>
        /// <returns>The loaded model.</returns>
        public static T LoadFromPath<T>(string path, Device? device = null) where T : Module
        {
            return _loadFlow.ExecuteFromPath<T>(path, pretrained: true, device);
        }

        /// <summary>
        /// Loads a model from a local file path without type specification.
        /// </summary>
        /// <param name="path">The path to the local model file.</param>
        /// <param name="device">The target device (null for default).</param>
        /// <returns>The loaded model as a Module.</returns>
        public static Module LoadFromPath(string path, Device? device = null)
        {
            return LoadFromPath<Module>(path, device);
        }

        /// <summary>
        /// Checks if a model is available in the registry.
        /// </summary>
        /// <param name="modelName">The model name.</param>
        /// <param name="version">The model version (null for any version).</param>
        /// <returns>True if the model is available, false otherwise.</returns>
        public static bool IsAvailable(string modelName, string? version = null)
        {
            return _registry.Exists(modelName, version);
        }

        /// <summary>
        /// Lists all available models in the registry.
        /// </summary>
        /// <returns>A list of all available model metadata.</returns>
        public static IReadOnlyList<ModelMetadata> ListModels()
        {
            return _registry.ListAll();
        }

        /// <summary>
        /// Lists available models by architecture type.
        /// </summary>
        /// <param name="architecture">The architecture type (e.g., "ResNet", "BERT").</param>
        /// <returns>A list of models matching the architecture.</returns>
        public static IReadOnlyList<ModelMetadata> ListModelsByArchitecture(string architecture)
        {
            return _registry.ListByArchitecture(architecture);
        }

        /// <summary>
        /// Lists available models by task type.
        /// </summary>
        /// <param name="task">The task type.</param>
        /// <returns>A list of models matching the task type.</returns>
        public static IReadOnlyList<ModelMetadata> ListModelsByTask(TaskType task)
        {
            return _registry.ListByTask(task);
        }

        /// <summary>
        /// Gets the latest version of a model.
        /// </summary>
        /// <param name="modelName">The model name.</param>
        /// <returns>The model metadata for the latest version, or null if not found.</returns>
        public static ModelMetadata? GetLatestVersion(string modelName)
        {
            return _registry.GetLatestVersion(modelName);
        }

        /// <summary>
        /// Clears the model cache.
        /// </summary>
        public static void ClearCache()
        {
            _cacheManager.ClearCache();
        }

        /// <summary>
        /// Gets the cache statistics.
        /// </summary>
        /// <returns>The cache statistics.</returns>
        public static CacheStatistics GetCacheStatistics()
        {
            return _cacheManager.Statistics;
        }

        /// <summary>
        /// Lists all cached models.
        /// </summary>
        /// <returns>A list of cached models with their metadata.</returns>
        public static List<CacheMetadata> ListCachedModels()
        {
            return _cacheManager.ListCachedModels();
        }

        /// <summary>
        /// Registers a custom model in the registry.
        /// </summary>
        /// <param name="metadata">The model metadata to register.</param>
        public static void RegisterModel(ModelMetadata metadata)
        {
            _registry.Register(metadata);
        }

        /// <summary>
        /// Registers a custom model from a JSON metadata file.
        /// </summary>
        /// <param name="jsonFilePath">The path to the JSON metadata file.</param>
        public static void RegisterModelFromJson(string jsonFilePath)
        {
            var metadata = ModelMetadata.LoadFromJsonFile(jsonFilePath);
            _registry.Register(metadata);
        }

        /// <summary>
        /// Loads model registry from a JSON file.
        /// </summary>
        /// <param name="jsonFilePath">The path to the JSON registry file.</param>
        public static void LoadRegistryFromJson(string jsonFilePath)
        {
            _registry.LoadFromJson(jsonFilePath);
        }

        /// <summary>
        /// Saves the model registry to a JSON file.
        /// </summary>
        /// <param name="jsonFilePath">The path to save the JSON registry file.</param>
        public static void SaveRegistryToJson(string jsonFilePath)
        {
            _registry.SaveToJson(jsonFilePath);
        }

        /// <summary>
        /// Loads default models into the registry.
        /// </summary>
        private static void LoadDefaultModels()
        {
            // Load built-in models from embedded resources or default locations
            // This is a placeholder - in a real implementation, this would load
            // a curated set of popular models
        }

        /// <summary>
        /// Sets the default device for model operations.
        /// </summary>
        /// <param name="device">The default device.</param>
        public static void SetDefaultDevice(Device device)
        {
            if (device == null)
                throw new ArgumentNullException(nameof(device));

            _configuration.DefaultDevice = device;
        }

        /// <summary>
        /// Gets the default device for model operations.
        /// </summary>
        /// <returns>The default device.</returns>
        public static Device GetDefaultDevice()
        {
            return _configuration.DefaultDevice;
        }

        /// <summary>
        /// Enables or disables caching.
        /// </summary>
        /// <param name="enabled">Whether to enable caching.</param>
        public static void SetCacheEnabled(bool enabled)
        {
            _configuration.CacheEnabled = enabled;
        }

        /// <summary>
        /// Enables or disables auto-download.
        /// </summary>
        /// <param name="enabled">Whether to enable auto-download.</param>
        public static void SetAutoDownloadEnabled(bool enabled)
        {
            _configuration.AutoDownloadEnabled = enabled;
        }

        /// <summary>
        /// Sets the default download timeout.
        /// </summary>
        /// <param name="timeoutMs">The timeout in milliseconds.</param>
        public static void SetDownloadTimeout(int timeoutMs)
        {
            if (timeoutMs <= 0)
                throw new ArgumentException("Timeout must be positive", nameof(timeoutMs));

            _configuration.DefaultDownloadTimeoutMs = timeoutMs;
        }
    }
}
