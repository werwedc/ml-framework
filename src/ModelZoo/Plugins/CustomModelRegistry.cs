using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace MLFramework.ModelZoo.Plugins
{
    /// <summary>
    /// Abstract base class for custom model registries.
    /// Provides common functionality: HTTP client, caching, logging, and retry logic.
    /// Subclasses override specific methods as needed.
    /// </summary>
    public abstract class CustomModelRegistry : IModelRegistryPlugin, IDisposable
    {
        protected readonly HttpClient _httpClient;
        protected readonly PluginConfigurationBase _configuration;
        protected readonly IRegistryAuthentication _authentication;
        protected readonly Dictionary<string, (ModelVersioning.ModelMetadata Metadata, DateTime Timestamp)> _cache;
        protected readonly object _cacheLock = new object();
        protected bool _disposed;

        public abstract string RegistryName { get; }
        public abstract int Priority { get; }

        protected CustomModelRegistry(PluginConfigurationBase configuration, IRegistryAuthentication authentication = null)
        {
            _configuration = configuration ?? throw new ArgumentNullException(nameof(configuration));
            _authentication = authentication;
            _cache = new Dictionary<string, (ModelVersioning.ModelMetadata, DateTime)>();

            _httpClient = new HttpClient
            {
                Timeout = TimeSpan.FromSeconds(configuration.HttpTimeoutSeconds)
            };

            if (authentication != null)
            {
                _httpClient.DefaultRequestHeaders.Clear();
            }
        }

        /// <summary>
        /// Gets model metadata with caching support.
        /// </summary>
        public virtual async Task<ModelVersioning.ModelMetadata> GetModelMetadataAsync(string modelName, string version = null)
        {
            if (string.IsNullOrEmpty(modelName))
            {
                throw new ArgumentException("Model name cannot be null or empty", nameof(modelName));
            }

            var cacheKey = BuildCacheKey(modelName, version);

            // Check cache
            if (_configuration.EnableCaching)
            {
                lock (_cacheLock)
                {
                    if (_cache.TryGetValue(cacheKey, out var cached))
                    {
                        var age = DateTime.UtcNow - cached.Timestamp;
                        if (age.TotalHours < _configuration.CacheExpirationHours)
                        {
                            LogInfo($"Retrieved model metadata from cache: {modelName}");
                            return cached.Metadata;
                        }
                    }
                }
            }

            // Fetch from registry
            var metadata = await FetchModelMetadataAsync(modelName, version);

            // Update cache
            if (_configuration.EnableCaching)
            {
                lock (_cacheLock)
                {
                    _cache[cacheKey] = (metadata, DateTime.UtcNow);
                }
            }

            LogInfo($"Retrieved model metadata: {modelName}");
            return metadata;
        }

        /// <summary>
        /// Fetches model metadata from the registry (to be implemented by subclasses).
        /// </summary>
        protected abstract Task<ModelVersioning.ModelMetadata> FetchModelMetadataAsync(string modelName, string version = null);

        /// <summary>
        /// Downloads model with retry logic and progress reporting.
        /// </summary>
        public virtual async Task<Stream> DownloadModelAsync(ModelVersioning.ModelMetadata metadata, IProgress<double> progress = null)
        {
            if (metadata == null)
            {
                throw new ArgumentNullException(nameof(metadata));
            }

            int attempt = 0;
            while (attempt <= _configuration.MaxRetries)
            {
                try
                {
                    LogInfo($"Downloading model: {metadata.ModelName} (attempt {attempt + 1})");
                    var stream = await FetchModelAsync(metadata, progress);
                    LogInfo($"Successfully downloaded model: {metadata.ModelName}");
                    return stream;
                }
                catch (Exception ex) when (attempt < _configuration.MaxRetries)
                {
                    LogWarning($"Download failed for {metadata.ModelName}: {ex.Message}");
                    attempt++;
                    await Task.Delay(_configuration.RetryDelayMilliseconds);
                }
            }

            throw new InvalidOperationException(
                $"Failed to download model {metadata.ModelName} after {_configuration.MaxRetries} attempts");
        }

        /// <summary>
        /// Fetches model from the registry (to be implemented by subclasses).
        /// </summary>
        protected abstract Task<Stream> FetchModelAsync(ModelVersioning.ModelMetadata metadata, IProgress<double> progress = null);

        /// <summary>
        /// Checks if model exists.
        /// </summary>
        public virtual async Task<bool> ModelExistsAsync(string modelName, string version = null)
        {
            try
            {
                await GetModelMetadataAsync(modelName, version);
                return true;
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Lists all models in the registry (to be implemented by subclasses).
        /// </summary>
        public abstract Task<string[]> ListModelsAsync();

        /// <summary>
        /// Determines if this registry can handle the model.
        /// Default implementation checks if model name starts with registry name prefix.
        /// </summary>
        public virtual bool CanHandle(string modelName)
        {
            if (string.IsNullOrEmpty(modelName))
            {
                return false;
            }

            // Default behavior: check if model name contains registry name or uses specific prefix
            var prefix = $"{RegistryName.ToLower()}/";
            return modelName.StartsWith(prefix, StringComparison.OrdinalIgnoreCase) ||
                   modelName.Contains(RegistryName, StringComparison.OrdinalIgnoreCase);
        }

        /// <summary>
        /// Builds a cache key for the model.
        /// </summary>
        protected virtual string BuildCacheKey(string modelName, string version = null)
        {
            return string.IsNullOrEmpty(version) ? modelName : $"{modelName}:{version}";
        }

        /// <summary>
        /// Clears the metadata cache.
        /// </summary>
        public virtual void ClearCache()
        {
            lock (_cacheLock)
            {
                _cache.Clear();
                LogInfo("Cache cleared");
            }
        }

        /// <summary>
        /// Removes expired entries from the cache.
        /// </summary>
        public virtual void CleanExpiredCache()
        {
            lock (_cacheLock)
            {
                var expiredKeys = _cache
                    .Where(kvp => (DateTime.UtcNow - kvp.Value.Timestamp).TotalHours >= _configuration.CacheExpirationHours)
                    .Select(kvp => kvp.Key)
                    .ToList();

                foreach (var key in expiredKeys)
                {
                    _cache.Remove(key);
                }

                LogInfo($"Removed {expiredKeys.Count} expired cache entries");
            }
        }

        /// <summary>
        /// Logs an informational message.
        /// </summary>
        protected virtual void LogInfo(string message)
        {
            Console.WriteLine($"[{RegistryName}] INFO: {message}");
        }

        /// <summary>
        /// Logs a warning message.
        /// </summary>
        protected virtual void LogWarning(string message)
        {
            Console.WriteLine($"[{RegistryName}] WARNING: {message}");
        }

        /// <summary>
        /// Logs an error message.
        /// </summary>
        protected virtual void LogError(string message)
        {
            Console.WriteLine($"[{RegistryName}] ERROR: {message}");
        }

        /// <summary>
        /// Creates an authenticated HTTP request.
        /// </summary>
        protected virtual HttpRequestMessage CreateAuthenticatedRequest(HttpMethod method, string url)
        {
            var request = new HttpRequestMessage(method, url);

            if (_authentication != null)
            {
                _authentication.Authenticate(request);
            }

            return request;
        }

        /// <summary>
        /// Executes an HTTP request with retry logic.
        /// </summary>
        protected virtual async Task<HttpResponseMessage> ExecuteRequestAsync(HttpRequestMessage request)
        {
            int attempt = 0;
            while (attempt <= _configuration.MaxRetries)
            {
                try
                {
                    var response = await _httpClient.SendAsync(request);
                    response.EnsureSuccessStatusCode();
                    return response;
                }
                catch (HttpRequestException ex) when (attempt < _configuration.MaxRetries)
                {
                    LogWarning($"HTTP request failed: {ex.Message}");
                    attempt++;
                    await Task.Delay(_configuration.RetryDelayMilliseconds);
                }
            }

            throw new HttpRequestException($"Request failed after {_configuration.MaxRetries} attempts");
        }

        public virtual void Dispose()
        {
            if (!_disposed)
            {
                _httpClient?.Dispose();
                _disposed = true;
            }
        }
    }
}
