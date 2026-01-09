using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace MLFramework.ModelZoo.Plugins.BuiltIn
{
    /// <summary>
    /// In-memory registry plugin for testing and development.
    /// Stores models and metadata in memory.
    /// </summary>
    [RegistryPlugin("memory", priority: 200)]
    public class MemoryRegistry : IModelRegistryPlugin
    {
        private readonly Dictionary<string, (ModelVersioning.ModelMetadata Metadata, byte[] Data)> _models;
        private readonly object _lock = new object();

        public string RegistryName => "memory";
        public int Priority => 200;

        public MemoryRegistry()
        {
            _models = new Dictionary<string, (ModelVersioning.ModelMetadata, byte[])>(StringComparer.OrdinalIgnoreCase);
        }

        /// <summary>
        /// Adds a model to the in-memory registry.
        /// </summary>
        /// <param name="modelName">The model name.</param>
        /// <param name="metadata">The model metadata.</param>
        /// <param name="data">The model data.</param>
        public void AddModel(string modelName, ModelVersioning.ModelMetadata metadata, byte[] data)
        {
            if (string.IsNullOrEmpty(modelName))
            {
                throw new ArgumentException("Model name cannot be null or empty", nameof(modelName));
            }

            if (metadata == null)
            {
                throw new ArgumentNullException(nameof(metadata));
            }

            if (data == null || data.Length == 0)
            {
                throw new ArgumentException("Model data cannot be null or empty", nameof(data));
            }

            lock (_lock)
            {
                _models[modelName] = (metadata, data);
                Console.WriteLine($"[MemoryRegistry] Added model: {modelName}");
            }
        }

        /// <summary>
        /// Removes a model from the in-memory registry.
        /// </summary>
        /// <param name="modelName">The model name.</param>
        /// <returns>True if the model was removed, false if not found.</returns>
        public bool RemoveModel(string modelName)
        {
            if (string.IsNullOrEmpty(modelName))
            {
                throw new ArgumentException("Model name cannot be null or empty", nameof(modelName));
            }

            lock (_lock)
            {
                var removed = _models.Remove(modelName);
                if (removed)
                {
                    Console.WriteLine($"[MemoryRegistry] Removed model: {modelName}");
                }
                return removed;
            }
        }

        /// <summary>
        /// Clears all models from the registry.
        /// </summary>
        public void Clear()
        {
            lock (_lock)
            {
                var count = _models.Count;
                _models.Clear();
                Console.WriteLine($"[MemoryRegistry] Cleared {count} models");
            }
        }

        public Task<ModelVersioning.ModelMetadata> GetModelMetadataAsync(string modelName, string version = null)
        {
            if (string.IsNullOrEmpty(modelName))
            {
                throw new ArgumentException("Model name cannot be null or empty", nameof(modelName));
            }

            lock (_lock)
            {
                if (_models.TryGetValue(modelName, out var modelData))
                {
                    return Task.FromResult(modelData.Metadata);
                }

                throw new KeyNotFoundException($"Model not found in memory registry: {modelName}");
            }
        }

        public async Task<Stream> DownloadModelAsync(ModelVersioning.ModelMetadata metadata, IProgress<double> progress = null)
        {
            if (metadata == null)
            {
                throw new ArgumentNullException(nameof(metadata));
            }

            string modelName = metadata.ModelName;
            byte[] data;

            lock (_lock)
            {
                if (_models.TryGetValue(modelName, out var modelData))
                {
                    data = modelData.Data;
                }
                else
                {
                    throw new KeyNotFoundException($"Model not found in memory registry: {modelName}");
                }
            }

            // Report progress
            progress?.Report(0.5);
            await Task.Delay(10); // Simulate async operation
            progress?.Report(1.0);

            Console.WriteLine($"[MemoryRegistry] Downloaded model: {modelName} ({data.Length} bytes)");
            return new MemoryStream(data);
        }

        public Task<bool> ModelExistsAsync(string modelName, string version = null)
        {
            if (string.IsNullOrEmpty(modelName))
            {
                throw new ArgumentException("Model name cannot be null or empty", nameof(modelName));
            }

            lock (_lock)
            {
                return Task.FromResult(_models.ContainsKey(modelName));
            }
        }

        public Task<string[]> ListModelsAsync()
        {
            lock (_lock)
            {
                return Task.FromResult(_models.Keys.ToArray());
            }
        }

        public bool CanHandle(string modelName)
        {
            if (string.IsNullOrEmpty(modelName))
            {
                return false;
            }

            // Handle models starting with "memory:"
            if (modelName.StartsWith("memory:", StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }

            // Check if model exists in registry
            lock (_lock)
            {
                return _models.ContainsKey(modelName);
            }
        }

        /// <summary>
        /// Gets the number of models in the registry.
        /// </summary>
        public int Count
        {
            get
            {
                lock (_lock)
                {
                    return _models.Count;
                }
            }
        }

        /// <summary>
        /// Gets the total size of all model data in bytes.
        /// </summary>
        public long TotalSize
        {
            get
            {
                lock (_lock)
                {
                    return _models.Values.Sum(m => m.Data.Length);
                }
            }
        }
    }
}
