using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;

namespace MLFramework.Serving.Deployment
{
    /// <summary>
    /// Thread-safe implementation of IModelRegistry using ConcurrentDictionary.
    /// Supports high-frequency queries (1000+ RPS) with minimal overhead.
    /// </summary>
    public class ModelRegistry : IModelRegistry
    {
        // Nested thread-safe storage: modelName -> {version -> ModelMetadata}
        private readonly ConcurrentDictionary<string, ConcurrentDictionary<string, ModelMetadata>> _registry;

        public ModelRegistry()
        {
            _registry = new ConcurrentDictionary<string, ConcurrentDictionary<string, ModelMetadata>>();
        }

        /// <inheritdoc />
        public void RegisterModel(string name, string version, ModelMetadata metadata)
        {
            if (string.IsNullOrWhiteSpace(name))
                throw new ArgumentException("Model name cannot be null or whitespace.", nameof(name));

            if (string.IsNullOrWhiteSpace(version))
                throw new ArgumentException("Version cannot be null or whitespace.", nameof(version));

            if (metadata == null)
                throw new ArgumentNullException(nameof(metadata));

            // Get or create the version dictionary for this model name
            var versions = _registry.GetOrAdd(name, _ => new ConcurrentDictionary<string, ModelMetadata>());

            // Try to add the version - if it already exists, throw
            if (!versions.TryAdd(version, metadata))
            {
                throw new InvalidOperationException(
                    $"Model '{name}' with version '{version}' is already registered.");
            }
        }

        /// <inheritdoc />
        public void UnregisterModel(string name, string version)
        {
            if (string.IsNullOrWhiteSpace(name))
                throw new ArgumentException("Model name cannot be null or whitespace.", nameof(name));

            if (string.IsNullOrWhiteSpace(version))
                throw new ArgumentException("Version cannot be null or whitespace.", nameof(version));

            if (!_registry.TryGetValue(name, out var versions))
            {
                throw new KeyNotFoundException(
                    $"Model '{name}' is not registered.");
            }

            if (!versions.TryRemove(version, out _))
            {
                throw new KeyNotFoundException(
                    $"Version '{version}' for model '{name}' is not found.");
            }

            // Clean up empty model entries
            if (versions.IsEmpty)
            {
                _registry.TryRemove(name, out _);
            }
        }

        /// <inheritdoc />
        public bool HasVersion(string name, string version)
        {
            if (string.IsNullOrWhiteSpace(name))
                return false;

            if (string.IsNullOrWhiteSpace(version))
                return false;

            return _registry.TryGetValue(name, out var versions) &&
                   versions.ContainsKey(version);
        }

        /// <inheritdoc />
        public IEnumerable<string> GetVersions(string name)
        {
            if (string.IsNullOrWhiteSpace(name))
                return Enumerable.Empty<string>();

            return _registry.TryGetValue(name, out var versions)
                ? versions.Keys.OrderBy(v => v).ToList() // Return sorted list for consistency
                : Enumerable.Empty<string>();
        }

        /// <inheritdoc />
        public ModelMetadata GetMetadata(string name, string version)
        {
            if (string.IsNullOrWhiteSpace(name))
                throw new ArgumentException("Model name cannot be null or whitespace.", nameof(name));

            if (string.IsNullOrWhiteSpace(version))
                throw new ArgumentException("Version cannot be null or whitespace.", nameof(version));

            if (!_registry.TryGetValue(name, out var versions))
            {
                throw new KeyNotFoundException(
                    $"Model '{name}' is not registered.");
            }

            if (!versions.TryGetValue(version, out var metadata))
            {
                throw new KeyNotFoundException(
                    $"Version '{version}' for model '{name}' is not found.");
            }

            return metadata;
        }

        /// <inheritdoc />
        public IEnumerable<string> GetAllModelNames()
        {
            return _registry.Keys.OrderBy(n => n).ToList();
        }
    }
}
