using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.Text.Json;
using System.Threading;

namespace MLFramework.ModelZoo
{
    /// <summary>
    /// Thread-safe in-memory registry to manage ModelMetadata objects and provide lookup functionality.
    /// </summary>
    public class ModelRegistry
    {
        private readonly ConcurrentDictionary<string, ModelMetadata> _models;
        private readonly ReaderWriterLockSlim _versionLock;
        private readonly ConcurrentDictionary<string, HashSet<string>> _modelVersions;

        /// <summary>
        /// Gets the number of registered models.
        /// </summary>
        public int Count => _models.Count;

        /// <summary>
        /// Initializes a new instance of the ModelRegistry class.
        /// </summary>
        public ModelRegistry()
        {
            _models = new ConcurrentDictionary<string, ModelMetadata>();
            _modelVersions = new ConcurrentDictionary<string, HashSet<string>>();
            _versionLock = new ReaderWriterLockSlim();
        }

        /// <summary>
        /// Registers a model in the registry.
        /// </summary>
        /// <param name="metadata">The model metadata to register.</param>
        /// <exception cref="ArgumentNullException">Thrown when metadata is null.</exception>
        public void Register(ModelMetadata metadata)
        {
            if (metadata == null)
                throw new ArgumentNullException(nameof(metadata));

            if (string.IsNullOrWhiteSpace(metadata.Name))
                throw new ArgumentException("Model name cannot be empty", nameof(metadata));

            if (string.IsNullOrWhiteSpace(metadata.Version))
                throw new ArgumentException("Model version cannot be empty", nameof(metadata));

            string key = GetKey(metadata.Name, metadata.Version);

            // Update version tracking
            _versionLock.EnterWriteLock();
            try
            {
                if (!_modelVersions.TryGetValue(metadata.Name, out var versions))
                {
                    versions = new HashSet<string>();
                    _modelVersions[metadata.Name] = versions;
                }
                versions.Add(metadata.Version);
            }
            finally
            {
                _versionLock.ExitWriteLock();
            }

            _models[key] = metadata;
        }

        /// <summary>
        /// Retrieves metadata for a specific model by name and version.
        /// </summary>
        /// <param name="name">The model name.</param>
        /// <param name="version">The model version (null for latest).</param>
        /// <returns>The model metadata, or null if not found.</returns>
        /// <exception cref="ArgumentNullException">Thrown when name is null or empty.</exception>
        public ModelMetadata? Get(string name, string? version = null)
        {
            if (string.IsNullOrWhiteSpace(name))
                throw new ArgumentNullException(nameof(name));

            // If version not specified, get the latest version
            if (string.IsNullOrWhiteSpace(version))
            {
                version = GetLatestVersionName(name);
                if (version == null)
                    return null;
            }

            string key = GetKey(name, version);
            _models.TryGetValue(key, out var metadata);
            return metadata;
        }

        /// <summary>
        /// Gets the latest version of a model by name.
        /// </summary>
        /// <param name="name">The model name.</param>
        /// <returns>The latest version metadata, or null if not found.</returns>
        /// <exception cref="ArgumentNullException">Thrown when name is null or empty.</exception>
        public ModelMetadata? GetLatestVersion(string name)
        {
            if (string.IsNullOrWhiteSpace(name))
                throw new ArgumentNullException(nameof(name));

            string latestVersion = GetLatestVersionName(name);
            if (latestVersion == null)
                return null;

            return Get(name, latestVersion);
        }

        /// <summary>
        /// Returns all registered models.
        /// </summary>
        /// <returns>A list of all model metadata.</returns>
        public IReadOnlyList<ModelMetadata> ListAll()
        {
            return _models.Values.ToList().AsReadOnly();
        }

        /// <summary>
        /// Filters models by architecture type.
        /// </summary>
        /// <param name="architecture">The architecture type.</param>
        /// <returns>A list of models matching the architecture.</returns>
        /// <exception cref="ArgumentNullException">Thrown when architecture is null or empty.</exception>
        public IReadOnlyList<ModelMetadata> ListByArchitecture(string architecture)
        {
            if (string.IsNullOrWhiteSpace(architecture))
                throw new ArgumentNullException(nameof(architecture));

            return _models.Values
                .Where(m => m.Architecture.Equals(architecture, StringComparison.OrdinalIgnoreCase))
                .ToList()
                .AsReadOnly();
        }

        /// <summary>
        /// Filters models by task type.
        /// </summary>
        /// <param name="task">The task type.</param>
        /// <returns>A list of models matching the task type.</returns>
        public IReadOnlyList<ModelMetadata> ListByTask(TaskType task)
        {
            return _models.Values
                .Where(m => m.Task == task)
                .ToList()
                .AsReadOnly();
        }

        /// <summary>
        /// Checks if a model is registered.
        /// </summary>
        /// <param name="name">The model name.</param>
        /// <param name="version">The model version (null for any version).</param>
        /// <returns>True if the model is registered, false otherwise.</returns>
        /// <exception cref="ArgumentNullException">Thrown when name is null or empty.</exception>
        public bool Exists(string name, string? version = null)
        {
            if (string.IsNullOrWhiteSpace(name))
                throw new ArgumentNullException(nameof(name));

            if (string.IsNullOrWhiteSpace(version))
            {
                _versionLock.EnterReadLock();
                try
                {
                    return _modelVersions.ContainsKey(name);
                }
                finally
                {
                    _versionLock.ExitReadLock();
                }
            }

            string key = GetKey(name, version);
            return _models.ContainsKey(key);
        }

        /// <summary>
        /// Removes a model from the registry.
        /// </summary>
        /// <param name="name">The model name.</param>
        /// <param name="version">The model version (null for all versions).</param>
        /// <returns>True if the model was removed, false otherwise.</returns>
        /// <exception cref="ArgumentNullException">Thrown when name is null or empty.</exception>
        public bool Remove(string name, string? version = null)
        {
            if (string.IsNullOrWhiteSpace(name))
                throw new ArgumentNullException(nameof(name));

            if (string.IsNullOrWhiteSpace(version))
            {
                // Remove all versions
                _versionLock.EnterWriteLock();
                try
                {
                    if (!_modelVersions.TryRemove(name, out var versions))
                        return false;

                    bool removedAny = false;
                    foreach (var v in versions)
                    {
                        string key = GetKey(name, v);
                        removedAny |= _models.TryRemove(key, out _);
                    }
                    return removedAny;
                }
                finally
                {
                    _versionLock.ExitWriteLock();
                }
            }

            string key = GetKey(name, version);
            if (_models.TryRemove(key, out _))
            {
                _versionLock.EnterWriteLock();
                try
                {
                    if (_modelVersions.TryGetValue(name, out var versions))
                    {
                        versions.Remove(version);
                        if (versions.Count == 0)
                        {
                            _modelVersions.TryRemove(name, out _);
                        }
                    }
                }
                finally
                {
                    _versionLock.ExitWriteLock();
                }
                return true;
            }

            return false;
        }

        /// <summary>
        /// Saves the current registry to a JSON file.
        /// </summary>
        /// <param name="filePath">The file path to save to.</param>
        /// <exception cref="ArgumentNullException">Thrown when filePath is null or empty.</exception>
        public void SaveToJson(string filePath)
        {
            if (string.IsNullOrWhiteSpace(filePath))
                throw new ArgumentNullException(nameof(filePath));

            var options = new JsonSerializerOptions
            {
                WriteIndented = true,
                PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower
            };

            string json = JsonSerializer.Serialize(_models.Values.ToList(), options);
            File.WriteAllText(filePath, json);
        }

        /// <summary>
        /// Loads models from a JSON file and adds them to the registry.
        /// </summary>
        /// <param name="filePath">The file path to load from.</param>
        /// <exception cref="ArgumentNullException">Thrown when filePath is null or empty.</exception>
        /// <exception cref="FileNotFoundException">Thrown when the file does not exist.</exception>
        public void LoadFromJson(string filePath)
        {
            if (string.IsNullOrWhiteSpace(filePath))
                throw new ArgumentNullException(nameof(filePath));

            if (!File.Exists(filePath))
                throw new FileNotFoundException($"Registry file not found: {filePath}");

            string json = File.ReadAllText(filePath);
            var models = JsonSerializer.Deserialize<List<ModelMetadata>>(json, new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            });

            if (models != null)
            {
                foreach (var model in models)
                {
                    Register(model);
                }
            }
        }

        /// <summary>
        /// Clears all models from the registry.
        /// </summary>
        public void Clear()
        {
            _models.Clear();
            _modelVersions.Clear();
        }

        private static string GetKey(string name, string version)
        {
            return $"{name}_{version}";
        }

        private string? GetLatestVersionName(string name)
        {
            _versionLock.EnterReadLock();
            try
            {
                if (!_modelVersions.TryGetValue(name, out var versions))
                    return null;

                return versions
                    .OrderByDescending(v => SemanticVersion.Parse(v))
                    .FirstOrDefault();
            }
            finally
            {
                _versionLock.ExitReadLock();
            }
        }
    }

    /// <summary>
    /// Helper class for comparing semantic versions.
    /// </summary>
    internal static class SemanticVersion
    {
        public static int Parse(string version)
        {
            if (string.IsNullOrWhiteSpace(version))
                return 0;

            var parts = version.Split('.');
            int major = parts.Length > 0 ? int.TryParse(parts[0], out var m) ? m : 0 : 0;
            int minor = parts.Length > 1 ? int.TryParse(parts[1], out var min) ? min : 0 : 0;
            int patch = parts.Length > 2 ? int.TryParse(parts[2], out var p) ? p : 0 : 0;

            return major * 10000 + minor * 100 + patch;
        }
    }
}
