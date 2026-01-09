using System.Net.Http.Json;
using System.Text.Json;
using MLFramework.ModelVersioning;

namespace MLFramework.ModelZoo.ExternalHubs;

/// <summary>
/// Hugging Face Hub implementation for downloading models and metadata.
/// </summary>
public class HuggingFaceHub : IModelHub
{
    private readonly HttpClient _httpClient;
    private readonly JsonSerializerOptions _jsonOptions;

    /// <summary>
    /// Gets the hub identifier.
    /// </summary>
    public string HubName => "huggingface";

    /// <summary>
    /// Gets the authentication method used by this hub.
    /// </summary>
    public IHubAuthentication? Authentication { get; }

    /// <summary>
    /// Gets the hub configuration.
    /// </summary>
    public HubConfiguration Configuration { get; }

    /// <summary>
    /// Gets the Hugging Face-specific configuration.
    /// </summary>
    public HuggingFaceHubConfiguration HfConfig => (HuggingFaceHubConfiguration)Configuration;

    /// <summary>
    /// Initializes a new instance of the HuggingFaceHub class.
    /// </summary>
    /// <param name="configuration">The hub configuration.</param>
    /// <param name="authentication">Optional authentication method.</param>
    public HuggingFaceHub(HuggingFaceHubConfiguration? configuration = null, IHubAuthentication? authentication = null)
    {
        Configuration = configuration ?? new HuggingFaceHubConfiguration();
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
    /// Gets model metadata from Hugging Face Hub.
    /// </summary>
    /// <param name="modelId">The model identifier (e.g., "bert-base-uncased").</param>
    /// <returns>A task that returns the model metadata.</returns>
    public async Task<ModelMetadata> GetModelMetadataAsync(string modelId)
    {
        if (string.IsNullOrWhiteSpace(modelId))
        {
            throw new ArgumentException("Model ID cannot be null or empty.", nameof(modelId));
        }

        var baseUrl = HfConfig.EffectiveBaseUrl;
        var apiUrl = $"{baseUrl}/api/models/{modelId}";

        var request = CreateRequest(HttpMethod.Get, apiUrl);
        var response = await _httpClient.SendAsync(request);

        if (!response.IsSuccessStatusCode)
        {
            throw new HttpRequestException(
                $"Failed to get metadata for model '{modelId}': {response.StatusCode}");
        }

        var jsonResponse = await response.Content.ReadAsStringAsync();
        var modelInfo = JsonSerializer.Deserialize<HfModelInfo>(jsonResponse, _jsonOptions)
            ?? throw new JsonException("Failed to deserialize model info.");

        return new ModelMetadata
        {
            ModelName = modelInfo.ModelId ?? modelId,
            Description = modelInfo.CardData?.Description ?? modelInfo.Description,
            Framework = modelInfo.LibraryName ?? "pytorch",
            Architecture = modelInfo.ModelType ?? "unknown",
            CustomMetadata = new Dictionary<string, string>
            {
                { "hub", HubName },
                { "author", modelInfo.Author ?? "unknown" },
                { "downloads", modelInfo.Downloads.ToString() },
                { "likes", modelInfo.Likes.ToString() },
                { "lastModified", modelInfo.LastModified ?? "" }
            }
        };
    }

    /// <summary>
    /// Downloads model files from Hugging Face Hub.
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

        // First, get the list of files to find the main model file
        var baseUrl = HfConfig.EffectiveBaseUrl;
        var filesUrl = $"{baseUrl}/api/models/{modelId}";

        var request = CreateRequest(HttpMethod.Get, filesUrl);
        var response = await _httpClient.SendAsync(request);

        if (!response.IsSuccessStatusCode)
        {
            throw new HttpRequestException(
                $"Failed to get file list for model '{modelId}': {response.StatusCode}");
        }

        var jsonResponse = await response.Content.ReadAsStringAsync();
        var modelInfo = JsonSerializer.Deserialize<HfModelInfo>(jsonResponse, _jsonOptions)
            ?? throw new JsonException("Failed to deserialize model info.");

        // Find the main model file (safetensors, pytorch_model.bin, etc.)
        var modelFile = modelInfo.Siblings?
            .FirstOrDefault(f => f.RfFilename?.EndsWith(".safetensors") == true ||
                                 f.RfFilename?.EndsWith(".bin") == true ||
                                 f.RfFilename?.EndsWith(".pt") == true);

        if (modelFile == null || modelFile.RfFilename == null)
        {
            throw new InvalidOperationException($"No model file found for '{modelId}'.");
        }

        // Download the model file
        var downloadUrl = $"{baseUrl}/{modelId}/resolve/main/{modelFile.RfFilename}";
        return await DownloadWithProgressAsync(downloadUrl, progress);
    }

    /// <summary>
    /// Checks if a model exists in the Hugging Face Hub.
    /// </summary>
    /// <param name="modelId">The model identifier.</param>
    /// <returns>A task that returns true if the model exists, false otherwise.</returns>
    public async Task<bool> ModelExistsAsync(string modelId)
    {
        if (string.IsNullOrWhiteSpace(modelId))
        {
            return false;
        }

        var baseUrl = HfConfig.EffectiveBaseUrl;
        var apiUrl = $"{baseUrl}/api/models/{modelId}";

        try
        {
            var request = CreateRequest(HttpMethod.Head, apiUrl);
            var response = await _httpClient.SendAsync(request);
            return response.IsSuccessStatusCode;
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Lists available models from Hugging Face Hub.
    /// </summary>
    /// <param name="filter">Optional filter to narrow down the list of models.</param>
    /// <returns>A task that returns an array of available model identifiers.</returns>
    public async Task<string[]> ListModelsAsync(string? filter = null)
    {
        var baseUrl = HfConfig.EffectiveBaseUrl;
        var apiUrl = $"{baseUrl}/api/models";

        if (!string.IsNullOrEmpty(filter))
        {
            apiUrl += $"?filter={Uri.EscapeDataString(filter)}";
        }

        var request = CreateRequest(HttpMethod.Get, apiUrl);
        var response = await _httpClient.SendAsync(request);

        if (!response.IsSuccessStatusCode)
        {
            throw new HttpRequestException(
                $"Failed to list models: {response.StatusCode}");
        }

        var jsonResponse = await response.Content.ReadAsStringAsync();
        var models = JsonSerializer.Deserialize<List<HfModelInfo>>(jsonResponse, _jsonOptions)
            ?? new List<HfModelInfo>();

        return models.Select(m => m.ModelId ?? "").Where(id => !string.IsNullOrEmpty(id)).ToArray()!;
    }

    /// <summary>
    /// Checks if this hub can handle the given model ID.
    /// </summary>
    /// <param name="modelId">The model identifier.</param>
    /// <returns>True if this hub can handle the model ID, false otherwise.</returns>
    public bool CanHandle(string modelId)
    {
        var components = ModelIdParser.Parse(modelId);
        return components.HubName == HubName || components.HubName == "hf";
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

#region Internal DTOs for Hugging Face API

internal class HfModelInfo
{
    public string? ModelId { get; set; }
    public string? Author { get; set; }
    public string? Description { get; set; }
    public string? LibraryName { get; set; }
    public string? ModelType { get; set; }
    public int Downloads { get; set; }
    public int Likes { get; set; }
    public string? LastModified { get; set; }
    public HfCardData? CardData { get; set; }
    public List<HfSibling>? Siblings { get; set; }
}

internal class HfCardData
{
    public string? Description { get; set; }
}

internal class HfSibling
{
    public string? RfFilename { get; set; }
}

#endregion
