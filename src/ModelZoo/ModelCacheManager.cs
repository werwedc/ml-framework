using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;

namespace MLFramework.ModelZoo
{
    /// <summary>
    /// Manages a local cache system for storing downloaded models with efficient management capabilities.
    /// </summary>
    public class ModelCacheManager
    {
        private readonly CacheConfiguration _configuration;
        private readonly CacheStatistics _statistics;
        private readonly string _modelsDirectory;
        private readonly string _registryDirectory;
        private readonly string _registryFilePath;
        private readonly object _lock = new object();

        /// <summary>
        /// Gets the configuration for this cache manager.
        /// </summary>
        public CacheConfiguration Configuration => _configuration;

        /// <summary>
        /// Gets the cache statistics.
        /// </summary>
        public CacheStatistics Statistics => _statistics;

        /// <summary>
        /// Initializes a new instance of the ModelCacheManager class with default configuration.
        /// </summary>
        public ModelCacheManager() : this(new CacheConfiguration())
        {
        }

        /// <summary>
        /// Initializes a new instance of the ModelCacheManager class with specified configuration.
        /// </summary>
        /// <param name="configuration">The cache configuration.</param>
        public ModelCacheManager(CacheConfiguration configuration)
        {
            _configuration = configuration ?? throw new ArgumentNullException(nameof(configuration));
            _statistics = new CacheStatistics();

            _modelsDirectory = Path.Combine(_configuration.CacheRootPath, "models");
            _registryDirectory = Path.Combine(_configuration.CacheRootPath, "registry");
            _registryFilePath = Path.Combine(_registryDirectory, "models.json");

            InitializeCacheDirectories();

            if (_configuration.CleanupOnStartup)
            {
                EnforceMaxCacheSize(_configuration.MaxCacheSizeBytes);
                PruneOldModels(_configuration.MaxFileAge);
            }
        }

        /// <summary>
        /// Initializes the cache directory structure.
        /// </summary>
        private void InitializeCacheDirectories()
        {
            try
            {
                if (!Directory.Exists(_modelsDirectory))
                {
                    Directory.CreateDirectory(_modelsDirectory);
                }

                if (!Directory.Exists(_registryDirectory))
                {
                    Directory.CreateDirectory(_registryDirectory);
                }

                if (!File.Exists(_registryFilePath))
                {
                    File.WriteAllText(_registryFilePath, "{}");
                }
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Failed to initialize cache directories: {ex.Message}", ex);
            }
        }

        /// <summary>
        /// Gets the cache root directory path.
        /// </summary>
        /// <returns>The cache root directory path.</returns>
        public string GetCachePath()
        {
            return _configuration.CacheRootPath;
        }

        /// <summary>
        /// Gets the full path to a cached model file.
        /// </summary>
        /// <param name="modelName">The model name.</param>
        /// <param name="version">The model version.</param>
        /// <returns>The full path to the model file.</returns>
        public string GetModelPath(string modelName, string version)
        {
            return Path.Combine(_modelsDirectory, modelName, version, "model.bin");
        }

        /// <summary>
        /// Gets the metadata file path for a cached model.
        /// </summary>
        /// <param name="modelName">The model name.</param>
        /// <param name="version">The model version.</param>
        /// <returns>The full path to the metadata file.</returns>
        public string GetMetadataPath(string modelName, string version)
        {
            return Path.Combine(_modelsDirectory, modelName, version, "metadata.json");
        }

        /// <summary>
        /// Checks if a model is cached.
        /// </summary>
        /// <param name="modelName">The model name.</param>
        /// <param name="version">The model version.</param>
        /// <returns>True if the model is cached, false otherwise.</returns>
        public bool CacheExists(string modelName, string version)
        {
            string modelPath = GetModelPath(modelName, version);
            string metadataPath = GetMetadataPath(modelName, version);

            return File.Exists(modelPath) && File.Exists(metadataPath);
        }

        /// <summary>
        /// Gets the total size of the cache in bytes.
        /// </summary>
        /// <returns>The total cache size in bytes.</returns>
        public long GetCacheSize()
        {
            lock (_lock)
            {
                long totalSize = 0;

                if (!Directory.Exists(_modelsDirectory))
                {
                    return totalSize;
                }

                foreach (string modelDir in Directory.GetDirectories(_modelsDirectory))
                {
                    foreach (string versionDir in Directory.GetDirectories(modelDir))
                    {
                        foreach (string file in Directory.GetFiles(versionDir, "*", SearchOption.AllDirectories))
                        {
                            try
                            {
                                totalSize += new FileInfo(file).Length;
                            }
                            catch
                            {
                                // Ignore files that can't be accessed
                            }
                        }
                    }
                }

                _statistics.TotalCacheSize = totalSize;
                return totalSize;
            }
        }

        /// <summary>
        /// Returns the total cache size in human-readable format (e.g., "2.3 GB").
        /// </summary>
        /// <returns>Human-readable cache size.</returns>
        public string GetCacheSizeFormatted()
        {
            long size = GetCacheSize();
            return CacheStatistics.FormatBytes(size);
        }

        /// <summary>
        /// Gets metadata for a cached model.
        /// </summary>
        /// <param name="modelName">The model name.</param>
        /// <param name="version">The model version.</param>
        /// <returns>The cache metadata, or null if not found.</returns>
        public CacheMetadata? GetMetadata(string modelName, string version)
        {
            string metadataPath = GetMetadataPath(modelName, version);

            if (!File.Exists(metadataPath))
            {
                return null;
            }

            try
            {
                string json = File.ReadAllText(metadataPath);
                return CacheMetadata.FromJson(json);
            }
            catch
            {
                return null;
            }
        }

        /// <summary>
        /// Saves metadata for a cached model.
        /// </summary>
        /// <param name="metadata">The metadata to save.</param>
        private void SaveMetadata(CacheMetadata metadata)
        {
            string versionDir = Path.Combine(_modelsDirectory, metadata.ModelName, metadata.Version);
            string metadataPath = Path.Combine(versionDir, "metadata.json");

            if (!Directory.Exists(versionDir))
            {
                Directory.CreateDirectory(versionDir);
            }

            File.WriteAllText(metadataPath, metadata.ToJson());
        }

        /// <summary>
        /// Lists all cached models with their metadata.
        /// </summary>
        /// <returns>A list of cache metadata for all cached models.</returns>
        public List<CacheMetadata> ListCachedModels()
        {
            lock (_lock)
            {
                var models = new List<CacheMetadata>();

                if (!Directory.Exists(_modelsDirectory))
                {
                    return models;
                }

                foreach (string modelDir in Directory.GetDirectories(_modelsDirectory))
                {
                    foreach (string versionDir in Directory.GetDirectories(modelDir))
                    {
                        string modelName = Path.GetFileName(modelDir);
                        string version = Path.GetFileName(versionDir);
                        var metadata = GetMetadata(modelName, version);

                        if (metadata != null)
                        {
                            models.Add(metadata);
                        }
                    }
                }

                _statistics.TotalModels = models.Count;
                return models;
            }
        }

        /// <summary>
        /// Records access to a cached model.
        /// </summary>
        /// <param name="modelName">The model name.</param>
        /// <param name="version">The model version.</param>
        public void RecordAccess(string modelName, string version)
        {
            lock (_lock)
            {
                var metadata = GetMetadata(modelName, version);

                if (metadata != null)
                {
                    metadata.RecordAccess();
                    SaveMetadata(metadata);
                    _statistics.RecordHit();
                }
                else
                {
                    _statistics.RecordMiss();
                }

                UpdateAccessTracking();
            }
        }

        /// <summary>
        /// Updates the LRU and MRU tracking lists.
        /// </summary>
        private void UpdateAccessTracking()
        {
            var models = ListCachedModels();
            var sortedByLastAccess = models.OrderByDescending(m => m.LastAccessed).ToList();

            _statistics.MostRecentlyUsed = sortedByLastAccess.Take(10)
                .Select(m => new ModelAccessInfo
                {
                    ModelName = m.ModelName,
                    Version = m.Version,
                    LastAccessed = m.LastAccessed,
                    AccessCount = m.AccessCount,
                    FileSize = m.FileSize
                }).ToList();

            _statistics.LeastRecentlyUsed = sortedByLastAccess.TakeLast(10).Reverse()
                .Select(m => new ModelAccessInfo
                {
                    ModelName = m.ModelName,
                    Version = m.Version,
                    LastAccessed = m.LastAccessed,
                    AccessCount = m.AccessCount,
                    FileSize = m.FileSize
                }).ToList();
        }

        /// <summary>
        /// Removes a specific model from the cache.
        /// </summary>
        /// <param name="modelName">The model name.</param>
        /// <param name="version">The model version.</param>
        /// <returns>True if the model was removed, false if it didn't exist.</returns>
        public bool RemoveFromCache(string modelName, string version)
        {
            lock (_lock)
            {
                string modelVersionDir = Path.Combine(_modelsDirectory, modelName, version);

                if (!Directory.Exists(modelVersionDir))
                {
                    return false;
                }

                try
                {
                    Directory.Delete(modelVersionDir, true);

                    // Try to remove the model directory if it's empty
                    string modelDir = Path.Combine(_modelsDirectory, modelName);
                    if (Directory.Exists(modelDir) && !Directory.GetFiles(modelDir, "*", SearchOption.AllDirectories).Any())
                    {
                        Directory.Delete(modelDir, true);
                    }

                    UpdateAccessTracking();
                    return true;
                }
                catch (Exception ex)
                {
                    throw new InvalidOperationException($"Failed to remove model from cache: {ex.Message}", ex);
                }
            }
        }

        /// <summary>
        /// Clears all cached models.
        /// </summary>
        public void ClearCache()
        {
            lock (_lock)
            {
                if (!Directory.Exists(_modelsDirectory))
                {
                    return;
                }

                try
                {
                    foreach (string modelDir in Directory.GetDirectories(_modelsDirectory))
                    {
                        Directory.Delete(modelDir, true);
                    }

                    _statistics.TotalModels = 0;
                    _statistics.TotalCacheSize = 0;
                    _statistics.MostRecentlyUsed.Clear();
                    _statistics.LeastRecentlyUsed.Clear();
                }
                catch (Exception ex)
                {
                    throw new InvalidOperationException($"Failed to clear cache: {ex.Message}", ex);
                }
            }
        }

        /// <summary>
        /// Removes models older than the specified age.
        /// </summary>
        /// <param name="maxAge">The maximum age of models to keep.</param>
        /// <returns>The number of models removed.</returns>
        public int PruneOldModels(TimeSpan maxAge)
        {
            lock (_lock)
            {
                int removedCount = 0;
                var cutoffDate = DateTime.UtcNow.Subtract(maxAge);
                var models = ListCachedModels();

                foreach (var model in models.Where(m => m.DownloadDate < cutoffDate))
                {
                    if (RemoveFromCache(model.ModelName, model.Version))
                    {
                        removedCount++;
                    }
                }

                return removedCount;
            }
        }

        /// <summary>
        /// Removes least recently used models to fit within the size limit.
        /// </summary>
        /// <param name="maxBytes">The maximum cache size in bytes.</param>
        /// <returns>The number of models removed.</returns>
        public int EnforceMaxCacheSize(long maxBytes)
        {
            lock (_lock)
            {
                int removedCount = 0;
                long currentSize = GetCacheSize();

                if (currentSize <= maxBytes)
                {
                    return removedCount;
                }

                var models = ListCachedModels();
                var sortedByLRU = models.OrderBy(m => m.LastAccessed).ToList();

                foreach (var model in sortedByLRU)
                {
                    if (currentSize <= maxBytes)
                    {
                        break;
                    }

                    if (RemoveFromCache(model.ModelName, model.Version))
                    {
                        currentSize -= model.FileSize;
                        removedCount++;
                    }
                }

                _statistics.TotalCacheSize = GetCacheSize();
                return removedCount;
            }
        }

        /// <summary>
        /// Adds a model to the cache with the given file stream.
        /// </summary>
        /// <param name="modelName">The model name.</param>
        /// <param name="version">The model version.</param>
        /// <param name="fileStream">The file stream containing the model data.</param>
        /// <returns>The path where the model was cached.</returns>
        public string AddToCache(string modelName, string version, Stream fileStream)
        {
            lock (_lock)
            {
                string versionDir = Path.Combine(_modelsDirectory, modelName, version);
                string modelPath = Path.Combine(versionDir, "model.bin");

                if (!Directory.Exists(versionDir))
                {
                    Directory.CreateDirectory(versionDir);
                }

                using (var fileStreamCopy = new FileStream(modelPath, FileMode.Create, FileAccess.Write))
                {
                    fileStream.CopyTo(fileStreamCopy);
                }

                long fileSize = new FileInfo(modelPath).Length;
                var metadata = new CacheMetadata
                {
                    ModelName = modelName,
                    Version = version,
                    FileSize = fileSize,
                    DownloadDate = DateTime.UtcNow,
                    LastAccessed = DateTime.UtcNow,
                    AccessCount = 1
                };

                SaveMetadata(metadata);
                UpdateAccessTracking();

                // Enforce cache size limit
                EnforceMaxCacheSize(_configuration.MaxCacheSizeBytes);

                return modelPath;
            }
        }

        /// <summary>
        /// Acquires a file lock with timeout.
        /// </summary>
        /// <param name="filePath">The file path to lock.</param>
        /// <returns>A disposable file lock.</returns>
        public IDisposable AcquireFileLock(string filePath)
        {
            string lockPath = filePath + ".lock";
            int attempts = 0;
            int maxAttempts = _configuration.LockTimeoutMs / 100;

            while (attempts < maxAttempts)
            {
                try
                {
                    var stream = new FileStream(lockPath, FileMode.CreateNew, FileAccess.Write, FileShare.None);
                    return new FileLock(stream, lockPath);
                }
                catch (IOException)
                {
                    attempts++;
                    if (attempts < maxAttempts)
                    {
                        Thread.Sleep(100);
                    }
                    else
                    {
                        throw new TimeoutException($"Failed to acquire lock on {filePath} within timeout period");
                    }
                }
            }

            throw new TimeoutException($"Failed to acquire lock on {filePath} within timeout period");
        }

        /// <summary>
        /// Disposes of resources used by the cache manager.
        /// </summary>
        public void Dispose()
        {
            lock (_lock)
            {
                // Clean up any resources
            }
        }
    }

    /// <summary>
    /// Represents a file lock that is disposed when no longer needed.
    /// </summary>
    internal class FileLock : IDisposable
    {
        private readonly FileStream _lockStream;
        private readonly string _lockPath;

        public FileLock(FileStream lockStream, string lockPath)
        {
            _lockStream = lockStream;
            _lockPath = lockPath;
        }

        public void Dispose()
        {
            _lockStream.Dispose();
            try
            {
                if (File.Exists(_lockPath))
                {
                    File.Delete(_lockPath);
                }
            }
            catch
            {
                // Ignore errors deleting lock file
            }
        }
    }
}
