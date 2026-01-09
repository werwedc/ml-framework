using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using MLFramework.Core;
using MLFramework.NN;

namespace MLFramework.ModelZoo
{
    /// <summary>
    /// Internal class to orchestrate the model loading process.
    /// </summary>
    internal class LoadFlow
    {
        private readonly ModelRegistry _registry;
        private readonly ModelDownloadService _downloadService;
        private readonly ModelCacheManager _cacheManager;
        private readonly ModelDeserializer _deserializer;
        private readonly ModelZooConfiguration _configuration;

        /// <summary>
        /// Creates a new LoadFlow instance.
        /// </summary>
        /// <param name="registry">The model registry.</param>
        /// <param name="downloadService">The download service.</param>
        /// <param name="cacheManager">The cache manager.</param>
        /// <param name="deserializer">The model deserializer.</param>
        /// <param name="configuration">The ModelZoo configuration.</param>
        public LoadFlow(
            ModelRegistry registry,
            ModelDownloadService downloadService,
            ModelCacheManager cacheManager,
            ModelDeserializer deserializer,
            ModelZooConfiguration configuration)
        {
            _registry = registry ?? throw new ArgumentNullException(nameof(registry));
            _downloadService = downloadService ?? throw new ArgumentNullException(nameof(downloadService));
            _cacheManager = cacheManager ?? throw new ArgumentNullException(nameof(cacheManager));
            _deserializer = deserializer ?? throw new ArgumentNullException(nameof(deserializer));
            _configuration = configuration ?? throw new ArgumentNullException(nameof(configuration));
        }

        /// <summary>
        /// Executes the full load flow for a model.
        /// </summary>
        /// <typeparam name="T">The type of the model.</typeparam>
        /// <param name="modelName">The model name.</param>
        /// <param name="version">The model version (null for latest).</param>
        /// <param name="variant">The model variant (optional).</param>
        /// <param name="pretrained">Whether to load pre-trained weights.</param>
        /// <param name="device">The target device (null for default).</param>
        /// <param name="cancellationToken">Optional cancellation token.</param>
        /// <returns>The loaded model.</returns>
        public async Task<T> ExecuteAsync<T>(
            string modelName,
            string? version,
            string? variant,
            bool pretrained,
            Device? device,
            CancellationToken cancellationToken = default) where T : Module
        {
            // Step 1: Check registry for model metadata
            ModelMetadata metadata = GetMetadataFromRegistry(modelName, version);

            // Determine actual version (in case version was null)
            string actualVersion = metadata.Version;

            // Step 2: Determine the file path
            string modelPath;
            bool useCache = _configuration.CacheEnabled;

            if (useCache && _cacheManager.CacheExists(modelName, actualVersion))
            {
                // Step 3a: Load from cache
                modelPath = _cacheManager.GetModelPath(modelName, actualVersion);
                _cacheManager.RecordAccess(modelName, actualVersion);
            }
            else if (_configuration.AutoDownloadEnabled)
            {
                // Step 3b: Download model
                modelPath = await DownloadModelAsync(metadata, cancellationToken);
            }
            else
            {
                throw new ModelNotFoundException(modelName);
            }

            // Step 4: Create model instance
            T model = CreateModelInstance<T>(metadata, variant);

            // Step 5: Load pretrained weights if requested
            if (pretrained)
            {
                LoadPretrainedWeights(model, modelPath, metadata.Architecture);
            }
            else
            {
                InitializeRandomWeights(model);
            }

            // Step 6: Move model to target device
            Device targetDevice = device ?? _configuration.DefaultDevice;
            MoveToDevice(model, targetDevice);

            return model;
        }

        /// <summary>
        /// Executes the load flow for a model loaded from a local file path.
        /// </summary>
        /// <typeparam name="T">The type of the model.</typeparam>
        /// <param name="path">The path to the local model file.</param>
        /// <param name="pretrained">Whether to load pre-trained weights.</param>
        /// <param name="device">The target device (null for default).</param>
        /// <returns>The loaded model.</returns>
        public T ExecuteFromPath<T>(string path, bool pretrained, Device? device) where T : Module
        {
            if (string.IsNullOrWhiteSpace(path))
                throw new ArgumentException("Path cannot be null or empty", nameof(path));

            if (!File.Exists(path))
                throw new FileNotFoundException($"Model file not found: {path}");

            // Step 1: Create model instance (we need to know the architecture)
            // For path-based loading, we'll need to infer the architecture from the file
            string architecture = InferArchitectureFromPath(path);

            // Step 2: Create model instance
            T model = CreateModelInstance<T>(architecture);

            // Step 3: Load pretrained weights if requested
            if (pretrained)
            {
                LoadPretrainedWeights(model, path, architecture);
            }
            else
            {
                InitializeRandomWeights(model);
            }

            // Step 4: Move model to target device
            Device targetDevice = device ?? _configuration.DefaultDevice;
            MoveToDevice(model, targetDevice);

            return model;
        }

        /// <summary>
        /// Gets metadata from the registry.
        /// </summary>
        private ModelMetadata GetMetadataFromRegistry(string modelName, string? version)
        {
            if (!_registry.Exists(modelName))
            {
                throw new ModelNotFoundException(modelName);
            }

            var metadata = _registry.Get(modelName, version);
            if (metadata == null)
            {
                if (!string.IsNullOrWhiteSpace(version))
                {
                    throw new VersionNotFoundException(modelName, version);
                }
                else
                {
                    throw new ModelNotFoundException(modelName);
                }
            }

            return metadata;
        }

        /// <summary>
        /// Downloads a model from the registry.
        /// </summary>
        private async Task<string> DownloadModelAsync(ModelMetadata metadata, CancellationToken cancellationToken)
        {
            string tempPath = Path.GetTempFileName();
            string cachePath = _cacheManager.GetModelPath(metadata.Name, metadata.Version);
            string cacheDirectory = Path.GetDirectoryName(cachePath);

            // Ensure cache directory exists
            if (!Directory.Exists(cacheDirectory))
            {
                Directory.CreateDirectory(cacheDirectory);
            }

            try
            {
                // Download to temporary location
                await _downloadService.DownloadModelAsync(
                    metadata.DownloadUrl,
                    tempPath,
                    metadata.Sha256Checksum,
                    progress: null,
                    cancellationToken);

                // Move to cache location
                File.Move(tempPath, cachePath, overwrite: true);

                // Add to cache metadata
                using (var fileStream = File.OpenRead(cachePath))
                {
                    _cacheManager.AddToCache(metadata.Name, metadata.Version, fileStream);
                }

                return cachePath;
            }
            catch (Exception)
            {
                // Clean up temporary file on failure
                if (File.Exists(tempPath))
                {
                    try
                    {
                        File.Delete(tempPath);
                    }
                    catch
                    {
                        // Ignore cleanup errors
                    }
                }
                throw;
            }
        }

        /// <summary>
        /// Creates a model instance from metadata.
        /// </summary>
        private T CreateModelInstance<T>(ModelMetadata metadata, string? variant) where T : Module
        {
            // In a real implementation, this would use a factory to create the appropriate model
            // based on the architecture and variant
            try
            {
                // This is a simplified version - real implementation would use reflection or a factory
                var model = Activator.CreateInstance<T>();
                return model;
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException(
                    $"Failed to create model instance for architecture '{metadata.Architecture}' with variant '{variant ?? "default"}'",
                    ex);
            }
        }

        /// <summary>
        /// Creates a model instance from architecture name.
        /// </summary>
        private T CreateModelInstance<T>(string architecture) where T : Module
        {
            try
            {
                var model = Activator.CreateInstance<T>();
                return model;
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException(
                    $"Failed to create model instance for architecture '{architecture}'",
                    ex);
            }
        }

        /// <summary>
        /// Loads pre-trained weights into the model.
        /// </summary>
        private void LoadPretrainedWeights(Module model, string modelPath, string architecture)
        {
            try
            {
                _deserializer.LoadWeights(model, modelPath, architecture);
            }
            catch (DeserializationException ex)
            {
                throw new InvalidOperationException($"Failed to load pre-trained weights from '{modelPath}': {ex.Message}", ex);
            }
            catch (IncompatibleModelException ex)
            {
                throw new InvalidOperationException($"Model architecture mismatch: {ex.Message}", ex);
            }
        }

        /// <summary>
        /// Initializes model weights with random values.
        /// </summary>
        private void InitializeRandomWeights(Module model)
        {
            // In a real implementation, this would initialize parameters with random values
            // For now, we assume parameters are already initialized by the model constructor
            // or we use a default initialization strategy
        }

        /// <summary>
        /// Moves the model to the target device.
        /// </summary>
        private void MoveToDevice(Module model, Device device)
        {
            if (device == null)
                return;

            // In a real implementation, this would move all parameters to the specified device
            // For now, this is a placeholder
        }

        /// <summary>
        /// Infers the model architecture from a file path.
        /// </summary>
        private string InferArchitectureFromPath(string path)
        {
            // Try to infer from filename
            string fileName = Path.GetFileNameWithoutExtension(path);

            if (fileName.Contains("ResNet", StringComparison.OrdinalIgnoreCase))
                return "ResNet";
            if (fileName.Contains("BERT", StringComparison.OrdinalIgnoreCase))
                return "BERT";
            if (fileName.Contains("GPT", StringComparison.OrdinalIgnoreCase))
                return "GPT";

            // Default to unknown
            return "Unknown";
        }
    }
}
