using System.Text.Json;
using MLFramework.ModelVersioning;

namespace MLFramework.ModelZoo.ExternalHubs;

/// <summary>
/// TensorFlow Hub implementation for downloading models and metadata.
/// </summary>
public class TensorFlowHub : IModelHub
{
    private readonly HttpClient _httpClient;
    private readonly JsonSerializerOptions _jsonOptions;

    /// <summary>
    /// Gets the hub identifier.
    /// </summary>
    public string HubName => "tensorflow";

    /// <summary>
    /// Gets the authentication method used by this hub.
    /// </summary>
    public IHubAuthentication? Authentication { get; }

    /// <summary>
    /// Gets the hub configuration.
    /// </summary>
    public HubConfiguration Configuration { get; }

    /// <summary>
    /// Gets the TensorFlow-specific configuration.
    /// </summary>
    public TensorFlowHubConfiguration TfConfig => (TensorFlowHubConfiguration)Configuration;

    /// <summary>
    /// Initializes a new instance of the TensorFlowHub class.
    /// </summary>
    /// <param name="configuration">The hub configuration.</param>
    /// <param name="authentication">Optional authentication method.</param>
    public TensorFlowHub(TensorFlowHubConfiguration? configuration = null, IHubAuthentication? authentication = null)
    {
        Configuration = configuration ?? new TensorFlowHubConfiguration();
        Authentication = authentication ?? new AnonymousAuth();

        _httpClient = new HttpClient
        {
            Timeout = TimeSpan.FromSeconds(Configuration.TimeoutSeconds)
        };

        _jsonOptions = new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true,
            PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower
        };
    }

    /// <summary>
    /// Gets model metadata from TensorFlow Hub.
    /// </summary>
    /// <param name="modelId">The model identifier (e.g., "google/imagenet/resnet_v2_50").</param>
    /// <returns>A task that returns the model metadata.</returns>
    public async Task<ModelMetadata> GetModelMetadataAsync(string modelId)
    {
        if (string.IsNullOrWhiteSpace(modelId))
        {
            throw new ArgumentException("Model ID cannot be null or empty.", nameof(modelId));
        }

        var baseUrl = TfConfig.BaseUrl;
        var normalizedModelId = NormalizeModelId(modelId);
        var metadataUrl = $"{baseUrl}/{normalizedModelId}?tfhub-format=compressed";

        var request = CreateRequest(HttpMethod.Get, metadataUrl);
        var response = await _httpClient.SendAsync(request);

        if (!response.IsSuccessStatusCode)
        {
            throw new HttpRequestException(
                $"Failed to get metadata for model '{modelId}': {response.StatusCode}");
        }

        // Parse metadata from response headers
        var metadata = new ModelMetadata
        {
            ModelName = modelId,
            Framework = "tensorflow",
            Architecture = ExtractArchitecture(modelId),
            CustomMetadata = new Dictionary<string, string>
            {
                { "hub", HubName },
                { "format", TfConfig.PreferredFormat }
            }
        };

        // Try to get additional metadata from the hub
        if (response.Content.Headers.ContentType != null)
        {
            metadata.CustomMetadata["contentType"] = response.Content.Headers.ContentType.ToString();
        }

        return metadata;
    }

    /// <summary>
    /// Downloads model files from TensorFlow Hub.
    /// </summary>
    /// <param name="modelId">The model identifier.</param>
    /// <param name="progress">Optional progress reporter for download progress.</param>
    /// <returns>A task that returns a stream containing the model data.</returns>
    public async Task<Stream> DownloadModelAsync(string modelId, IProgress<double>? progress = null)
    {
        if (string.IsNullOrWhiteSpace(modelId))
        {
            throw new ArgumentException("Model ID cannot be null or empty.", nameof(modelId));
        }

        var baseUrl = TfConfig.BaseUrl;
        var normalizedModelId = NormalizeModelId(modelId);
        var downloadUrl = $"{baseUrl}/{normalizedModelId}?tfhub-format=compressed";

        return await DownloadWithProgressAsync(downloadUrl, progress);
    }

    /// <summary>
    /// Checks if a model exists in TensorFlow Hub.
    /// </summary>
    /// <param name="modelId">The model identifier.</param>
    /// <returns>A task that returns true if the model exists, false otherwise.</returns>
    public async Task<bool> ModelExistsAsync(string modelId)
    {
        if (string.IsNullOrWhiteSpace(modelId))
        {
            return false;
        }

        var baseUrl = TfConfig.BaseUrl;
        var normalizedModelId = NormalizeModelId(modelId);
        var checkUrl = $"{baseUrl}/{normalizedModelId}?tfhub-format=compressed";

        try
        {
            var request = CreateRequest(HttpMethod.Head, checkUrl);
            var response = await _httpClient.SendAsync(request);
            return response.IsSuccessStatusCode;
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Lists available models from TensorFlow Hub.
    /// </summary>
    /// <param name="filter">Optional filter to narrow down the list of models.</param>
    /// <returns>A task that returns an array of available model identifiers.</returns>
    public async Task<string[]> ListModelsAsync(string? filter = null)
    {
        // TensorFlow Hub doesn't have a public API to list all models
        // This is a placeholder implementation
        // In practice, users would need to browse the TF Hub website or use a predefined list

        var popularModels = new[]
        {
            "google/imagenet/resnet_v2_50",
            "google/imagenet/mobilenet_v2_100_224",
            "google/imagenet/efficientnet_v2_imagenet1k_b0",
            "google/nnlm-en-dim50",
            "google/universal-sentence-encoder"
        };

        if (string.IsNullOrEmpty(filter))
        {
            return popularModels;
        }

        return popularModels
            .Where(m => m.Contains(filter, StringComparison.OrdinalIgnoreCase))
            .ToArray();
    }

    /// <summary>
    /// Checks if this hub can handle the given model ID.
    /// </summary>
    /// <param name="modelId">The model identifier.</param>
    /// <returns>True if this hub can handle the model ID, false otherwise.</returns>
    public bool CanHandle(string modelId)
    {
        var components = ModelIdParser.Parse(modelId);
        return components.HubName == HubName || components.HubName == "tf";
    }

    private string NormalizeModelId(string modelId)
    {
        // Convert slashes to URL-friendly format
        return modelId.Replace("/", "/v1/");
    }

    private string ExtractArchitecture(string modelId)
    {
        // Try to extract architecture from model ID
        var parts = modelId.Split('/');
        if (parts.Length >= 2)
        {
            var lastPart = parts[^1];
            return lastPart.Contains("_") ? lastPart.Split('_')[0] : lastPart;
        }
        return "unknown";
    }

    private HttpRequestMessage CreateRequest(HttpMethod method, string url)
    {
        var request = new HttpRequestMessage(method, url);

        // Add authentication if configured
        if (Authentication != null && Authentication.IsValid() && Authentication.AuthType != "anonymous")
        {
            var (headerName, headerValue) = Authentication.GetAuthHeader();
            if (!string.IsNullOrEmpty(headerName))
            {
                request.Headers.Add(headerName, headerValue);
            }
        }

        // Add custom headers
        foreach (var header in Configuration.CustomHeaders)
        {
            request.Headers.Add(header.Key, header.Value);
        }

        // Accept compressed responses
        if (TfConfig.UseCompression)
        {
            request.Headers.AcceptEncoding.ParseAdd("gzip");
        }

        return request;
    }

    private async Task<Stream> DownloadWithProgressAsync(string url, IProgress<double>? progress)
    {
        var request = CreateRequest(HttpMethod.Get, url);
        var response = await _httpClient.SendAsync(request, HttpCompletionOption.ResponseHeadersRead);

        if (!response.IsSuccessStatusCode)
        {
            throw new HttpRequestException($"Failed to download model: {response.StatusCode}");
        }

        var contentLength = response.Content.Headers.ContentLength ?? 0;
        var memoryStream = new MemoryStream();

        using (var stream = await response.Content.ReadAsStreamAsync())
        {
            var buffer = new byte[8192];
            int bytesRead;
            long totalBytesRead = 0;

            while ((bytesRead = await stream.ReadAsync(buffer, 0, buffer.Length)) > 0)
            {
                await memoryStream.WriteAsync(buffer, 0, bytesRead);
                totalBytesRead += bytesRead;

                if (contentLength > 0 && progress != null)
                {
                    var percentage = (double)totalBytesRead / contentLength * 100;
                    progress.Report(percentage);
                }
            }
        }

        memoryStream.Position = 0;
        return memoryStream;
    }

    /// <summary>
    /// Releases the HTTP client resources.
    /// </summary>
    public void Dispose()
    {
        _httpClient?.Dispose();
    }
}
