using System;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Text.Json;
using System.Threading.Tasks;

namespace MLFramework.ModelZoo.Plugins.BuiltIn
{
    /// <summary>
    /// HTTP-based remote registry plugin.
    /// Loads models from a remote HTTP server.
    /// </summary>
    [RegistryPlugin("remote", priority: 50)]
    public class RemoteRegistry : CustomModelRegistry
    {
        private readonly string _baseUrl;
        private readonly string _modelsEndpoint;
        private readonly string _metadataEndpoint;

        public override string RegistryName => "remote";
        public override int Priority => 50;

        public RemoteRegistry(string baseUrl, AuthenticatedPluginConfiguration configuration = null)
            : base(configuration ?? new AuthenticatedPluginConfiguration(),
                  configuration != null ? CreateAuthentication(configuration) : null)
        {
            if (string.IsNullOrEmpty(baseUrl))
            {
                throw new ArgumentException("Base URL cannot be null or empty", nameof(baseUrl));
            }

            // Ensure URL ends with /
            _baseUrl = baseUrl.EndsWith("/") ? baseUrl : baseUrl + "/";
            _modelsEndpoint = "models/";
            _metadataEndpoint = "metadata/";

            LogInfo($"Initialized with base URL: {_baseUrl}");
        }

        private static IRegistryAuthentication CreateAuthentication(AuthenticatedPluginConfiguration config)
        {
            if (string.IsNullOrEmpty(config.AuthenticationType))
            {
                return null;
            }

            switch (config.AuthenticationType.ToLowerInvariant())
            {
                case "apikey":
                    return new ApiKeyAuthentication(config.ApiKey, config.ApiKeyHeaderName);

                case "token":
                case "bearer":
                    return new TokenAuthentication(config.Token, config.TokenType);

                case "basic":
                    return new BasicAuthentication(config.Username, config.Password);

                default:
                    throw new ArgumentException($"Unknown authentication type: {config.AuthenticationType}");
            }
        }

        protected override async Task<ModelVersioning.ModelMetadata> FetchModelMetadataAsync(string modelName, string version = null)
        {
            var url = BuildMetadataUrl(modelName, version);

            try
            {
                var request = CreateAuthenticatedRequest(HttpMethod.Get, url);
                var response = await ExecuteRequestAsync(request);
                var content = await response.Content.ReadAsStringAsync();

                var metadata = JsonSerializer.Deserialize<ModelVersioning.ModelMetadata>(content, new JsonSerializerOptions
                {
                    PropertyNameCaseInsensitive = true
                });

                if (metadata == null)
                {
                    throw new InvalidOperationException("Failed to deserialize model metadata");
                }

                // Add custom metadata with download URL
                metadata.CustomMetadata = metadata.CustomMetadata ?? new System.Collections.Generic.Dictionary<string, string>();
                metadata.CustomMetadata["DownloadUrl"] = BuildDownloadUrl(modelName, version);

                return metadata;
            }
            catch (HttpRequestException ex)
            {
                LogError($"Failed to fetch metadata for {modelName}: {ex.Message}");
                throw;
            }
        }

        protected override async Task<Stream> FetchModelAsync(ModelVersioning.ModelMetadata metadata, IProgress<double> progress = null)
        {
            var downloadUrl = metadata.CustomMetadata?.GetValueOrDefault("DownloadUrl");

            if (string.IsNullOrEmpty(downloadUrl))
            {
                throw new InvalidOperationException("Download URL not found in metadata");
            }

            try
            {
                var request = CreateAuthenticatedRequest(HttpMethod.Get, downloadUrl);
                var response = await ExecuteRequestAsync(request);
                var stream = await response.Content.ReadAsStreamAsync();

                // Wrap in progress stream if progress reporter is provided
                if (progress != null)
                {
                    var contentLength = response.Content.Headers.ContentLength;
                    if (contentLength.HasValue && contentLength.Value > 0)
                    {
                        return new ProgressStream(stream, contentLength.Value, progress);
                    }
                }

                return stream;
            }
            catch (HttpRequestException ex)
            {
                LogError($"Failed to download model: {ex.Message}");
                throw;
            }
        }

        public override async Task<string[]> ListModelsAsync()
        {
            var url = $"{_baseUrl}{_modelsEndpoint}";

            try
            {
                var request = CreateAuthenticatedRequest(HttpMethod.Get, url);
                var response = await ExecuteRequestAsync(request);
                var content = await response.Content.ReadAsStringAsync();

                var models = JsonSerializer.Deserialize<string[]>(content);

                return models ?? Array.Empty<string>();
            }
            catch (HttpRequestException ex)
            {
                LogError($"Failed to list models: {ex.Message}");
                throw;
            }
        }

        public override bool CanHandle(string modelName)
        {
            // Handle models starting with "http://" or "https://"
            if (modelName.StartsWith("http://", StringComparison.OrdinalIgnoreCase) ||
                modelName.StartsWith("https://", StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }

            // Handle models starting with "remote:"
            if (modelName.StartsWith("remote:", StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }

            return false;
        }

        private string BuildMetadataUrl(string modelName, string version = null)
        {
            // Remove "remote:" prefix if present
            if (modelName.StartsWith("remote:", StringComparison.OrdinalIgnoreCase))
            {
                modelName = modelName.Substring(7);
            }

            var url = $"{_baseUrl}{_metadataEndpoint}{Uri.EscapeDataString(modelName)}";

            if (!string.IsNullOrEmpty(version))
            {
                url += $"?version={Uri.EscapeDataString(version)}";
            }

            return url;
        }

        private string BuildDownloadUrl(string modelName, string version = null)
        {
            // Remove "remote:" prefix if present
            if (modelName.StartsWith("remote:", StringComparison.OrdinalIgnoreCase))
            {
                modelName = modelName.Substring(7);
            }

            var url = $"{_baseUrl}{_modelsEndpoint}{Uri.EscapeDataString(modelName)}/download";

            if (!string.IsNullOrEmpty(version))
            {
                url += $"?version={Uri.EscapeDataString(version)}";
            }

            return url;
        }

        /// <summary>
        /// Helper class for progress reporting during file download.
        /// </summary>
        private class ProgressStream : Stream
        {
            private readonly Stream _innerStream;
            private readonly long _totalBytes;
            private readonly IProgress<double> _progress;
            private long _bytesRead = 0;

            public ProgressStream(Stream innerStream, long totalBytes, IProgress<double> progress)
            {
                _innerStream = innerStream ?? throw new ArgumentNullException(nameof(innerStream));
                _totalBytes = totalBytes;
                _progress = progress;
            }

            public override bool CanRead => _innerStream.CanRead;
            public override bool CanSeek => _innerStream.CanSeek;
            public override bool CanWrite => false;
            public override long Length => _innerStream.Length;
            public override long Position { get => _innerStream.Position; set => _innerStream.Position = value; }

            public override void Flush() => _innerStream.Flush();

            public override int Read(byte[] buffer, int offset, int count)
            {
                int bytesRead = _innerStream.Read(buffer, offset, count);
                _bytesRead += bytesRead;

                if (_totalBytes > 0)
                {
                    _progress?.Report((double)_bytesRead / _totalBytes);
                }

                return bytesRead;
            }

            public override long Seek(long offset, SeekOrigin origin) => _innerStream.Seek(offset, origin);
            public override void SetLength(long value) => throw new NotSupportedException();
            public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();

            protected override void Dispose(bool disposing)
            {
                if (disposing)
                {
                    _innerStream.Dispose();
                }
                base.Dispose(disposing);
            }
        }
    }
}
