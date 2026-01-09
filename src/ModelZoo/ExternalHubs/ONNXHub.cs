using System.Text.Json;
using MLFramework.ModelVersioning;

namespace MLFramework.ModelZoo.ExternalHubs;

/// <summary>
/// ONNX Model Zoo implementation for downloading models and metadata.
/// </summary>
public class ONNXHub : IModelHub
{
    private readonly HttpClient _httpClient;
    private readonly JsonSerializerOptions _jsonOptions;

    /// <summary>
    /// Gets the hub identifier.
    /// </summary>
    public string HubName => "onnx";

    /// <summary>
    /// Gets the authentication method used by this hub.
    /// </summary>
    public IHubAuthentication? Authentication { get; }

    /// <summary>
    /// Gets the hub configuration.
    /// </summary>
    public HubConfiguration Configuration { get; }

    /// <summary>
    /// Gets the ONNX-specific configuration.
    /// </summary>
    public ONNXHubConfiguration OnnxConfig => (ONNXHubConfiguration)Configuration;

    /// <summary>
    /// Initializes a new instance of the ONNXHub class.
    /// </summary>
    /// <param name="configuration">The hub configuration.</param>
    /// <param name="authentication">Optional authentication method.</param>
    public ONNXHub(ONNXHubConfiguration? configuration = null, IHubAuthentication? authentication = null)
    {
        Configuration = configuration ?? new ONNXHubConfiguration();
        Authentication = authentication ?? new AnonymousAuth();

        _httpClient = new HttpClient
        {
            Timeout = TimeSpan.FromSeconds(Configuration.TimeoutSeconds)
        };

        _jsonOptions = new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        };
    }

    /// <summary>
    /// Gets model metadata from ONNX Model Zoo.
    /// </summary>
    /// <param name="modelId">The model identifier (e.g., "resnet50").</param>
    /// <returns>A task that returns the model metadata.</returns>
    public async Task<ModelMetadata> GetModelMetadataAsync(string modelId)
    {
        if (string.IsNullOrWhiteSpace(modelId))
        {
            throw new ArgumentException("Model ID cannot be null or empty.", nameof(modelId));
        }

        // Parse model ID to extract model name and version
        var modelName = ExtractModelName(modelId);
        var version = ExtractVersion(modelId) ?? OnnxConfig.DefaultVersion;

        // Try to fetch metadata from the ONNX Model Zoo
        var rawUrl = OnnxConfig.RawContentUrl;
        var metadataUrl = $"{rawUrl}/validated/vision/{version}/{modelName}/data/config.json";

        var request = CreateRequest(HttpMethod.Get, metadataUrl);
        var response = await _httpClient.SendAsync(request);

        var metadata = new ModelMetadata
        {
            ModelName = modelName,
            Framework = "onnx",
            Architecture = ExtractArchitecture(modelName),
            CustomMetadata = new Dictionary<string, string>
            {
                { "hub", HubName },
                { "version", version }
            }
        };

        if (response.IsSuccessStatusCode)
        {
            var jsonResponse = await response.Content.ReadAsStringAsync();
            try
            {
                var modelConfig = JsonSerializer.Deserialize<OnnxModelConfig>(jsonResponse, _jsonOptions);
                if (modelConfig != null)
                {
                    metadata.Description = modelConfig.Description;
                    if (modelConfig.InputShape != null && modelConfig.InputShape.Length > 0)
                    {
                        metadata.InputShape = modelConfig.InputShape;
                    }
                    if (modelConfig.OutputShape != null && modelConfig.OutputShape.Length > 0)
                    {
                        metadata.OutputShape = modelConfig.OutputShape;
                    }
                    metadata.CustomMetadata["categories"] = string.Join(",", modelConfig.Categories ?? Array.Empty<string>());
                }
            }
            catch (JsonException)
            {
                // If parsing fails, use default metadata
            }
        }

        return metadata;
    }

    /// <summary>
    /// Downloads model files from ONNX Model Zoo.
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

        var modelName = ExtractModelName(modelId);
        var version = ExtractVersion(modelId) ?? OnnxConfig.DefaultVersion;

        // Construct download URL for the .onnx file
        var baseUrl = OnnxConfig.PreferMirrors && OnnxConfig.MirrorUrls.Count > 0
            ? OnnxConfig.MirrorUrls[0]
            : OnnxConfig.RawContentUrl;

        var downloadUrl = $"{baseUrl}/validated/vision/{version}/{modelName}/model/{modelName}.onnx";

        return await DownloadWithProgressAsync(downloadUrl, progress);
    }

    /// <summary>
    /// Checks if a model exists in ONNX Model Zoo.
    /// </summary>
    /// <param name="modelId">The model identifier.</param>
    /// <returns>A task that returns true if the model exists, false otherwise.</returns>
    public async Task<bool> ModelExistsAsync(string modelId)
    {
        if (string.IsNullOrWhiteSpace(modelId))
        {
            return false;
        }

        var modelName = ExtractModelName(modelId);
        var version = ExtractVersion(modelId) ?? OnnxConfig.DefaultVersion;
        var rawUrl = OnnxConfig.RawContentUrl;
        var checkUrl = $"{rawUrl}/validated/vision/{version}/{modelName}/model/{modelName}.onnx";

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
    /// Lists available models from ONNX Model Zoo.
    /// </summary>
    /// <param name="filter">Optional filter to narrow down the list of models.</param>
    /// <returns>A task that returns an array of available model identifiers.</returns>
    public async Task<string[]> ListModelsAsync(string? filter = null)
    {
        // ONNX Model Zoo doesn't have a public API to list all models
        // This is a placeholder implementation with popular models
        var popularModels = new[]
        {
            "resnet50",
            "resnet101",
            "vgg16",
            "vgg19",
            "mobilenet",
            "inception_v1",
            "inception_v2",
            "densenet121",
            "shufflenet",
            "squeezenet"
        };

        // Try to fetch the list from the repository
        try
        {
            var rawUrl = OnnxConfig.RawContentUrl;
            var listUrl = $"{rawUrl}/validated/vision/master";

            var request = CreateRequest(HttpMethod.Get, listUrl);
            var response = await _httpClient.SendAsync(request);

            if (response.IsSuccessStatusCode)
            {
                var content = await response.Content.ReadAsStringAsync();
                // Parse the HTML to extract model names
                // This is a simplified implementation
                var models = ParseModelListFromHtml(content);
                if (models.Length > 0)
                {
                    popularModels = models;
                }
            }
        }
        catch
        {
            // If fetching fails, use the default list
        }

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
        return components.HubName == HubName;
    }

    private string ExtractModelName(string modelId)
    {
        // Extract the model name from the ID
        // e.g., "resnet50" or "resnet50-7" -> "resnet50"
        var dashIndex = modelId.IndexOf('-');
        return dashIndex >= 0 ? modelId[..dashIndex] : modelId;
    }

    private string? ExtractVersion(string modelId)
    {
        // Extract the version from the ID
        // e.g., "resnet50-7" -> "7"
        var dashIndex = modelId.IndexOf('-');
        return dashIndex >= 0 && dashIndex < modelId.Length - 1 ? modelId[(dashIndex + 1)..] : null;
    }

    private string ExtractArchitecture(string modelName)
    {
        // Extract architecture from model name
        var lowerName = modelName.ToLowerInvariant();
        if (lowerName.Contains("resnet")) return "resnet";
        if (lowerName.Contains("vgg")) return "vgg";
        if (lowerName.Contains("inception")) return "inception";
        if (lowerName.Contains("mobilenet")) return "mobilenet";
        if (lowerName.Contains("densenet")) return "densenet";
        return "unknown";
    }

    private string[] ParseModelListFromHtml(string html)
    {
        // Simplified HTML parsing to extract model names
        // In a production implementation, you would use a proper HTML parser
        var models = new List<string>();
        var startTag = "<td><a href=\"";
        var endTag = "\">";

        var startIndex = 0;
        while (true)
        {
            var idx = html.IndexOf(startTag, startIndex);
            if (idx < 0) break;

            var nameStart = idx + startTag.Length;
            var nameEnd = html.IndexOf(endTag, nameStart);
            if (nameEnd < 0) break;

            var modelName = html[nameStart..nameEnd];
            if (!string.IsNullOrEmpty(modelName) && !modelName.StartsWith("."))
            {
                models.Add(modelName);
            }

            startIndex = nameEnd + endTag.Length;
        }

        return models.ToArray();
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

        // Set User-Agent to avoid being blocked
        request.Headers.UserAgent.ParseAdd("MLFramework/1.0");

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

#region Internal DTOs for ONNX Model Zoo

internal class OnnxModelConfig
{
    public string? Description { get; set; }
    public int[]? InputShape { get; set; }
    public int[]? OutputShape { get; set; }
    public string[]? Categories { get; set; }
}

#endregion
