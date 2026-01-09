namespace MLFramework.ModelZoo.ExternalHubs;

/// <summary>
/// Base class for hub configuration.
/// </summary>
public abstract class HubConfiguration
{
    /// <summary>
    /// Gets or sets the request timeout in seconds.
    /// </summary>
    public int TimeoutSeconds { get; set; } = 300;

    /// <summary>
    /// Gets or sets the number of retry attempts for failed requests.
    /// </summary>
    public int MaxRetries { get; set; } = 3;

    /// <summary>
    /// Gets or sets the delay between retries in seconds.
    /// </summary>
    public int RetryDelaySeconds { get; set; } = 5;

    /// <summary>
    /// Gets or sets whether to verify SSL certificates.
    /// </summary>
    public bool VerifySsl { get; set; } = true;

    /// <summary>
    /// Gets or sets custom headers to include in requests.
    /// </summary>
    public Dictionary<string, string> CustomHeaders { get; set; } = new();

    /// <summary>
    /// Validates the configuration.
    /// </summary>
    /// <returns>True if the configuration is valid, false otherwise.</returns>
    public virtual bool IsValid()
    {
        return TimeoutSeconds > 0 && MaxRetries >= 0 && RetryDelaySeconds >= 0;
    }
}

/// <summary>
/// Configuration for Hugging Face Hub.
/// </summary>
public class HuggingFaceHubConfiguration : HubConfiguration
{
    /// <summary>
    /// Gets or sets the Hugging Face API token.
    /// </summary>
    public string? ApiToken { get; set; }

    /// <summary>
    /// Gets or sets the default repository (e.g., "models").
    /// </summary>
    public string DefaultRepository { get; set; } = "models";

    /// <summary>
    /// Gets or sets the base URL for the Hugging Face Hub.
    /// </summary>
    public string BaseUrl { get; set; } = "https://huggingface.co";

    /// <summary>
    /// Gets or sets whether to use the mirror URL.
    /// </summary>
    public bool UseMirror { get; set; }

    /// <summary>
    /// Gets or sets the mirror URL if UseMirror is true.
    /// </summary>
    public string? MirrorUrl { get; set; }

    /// <summary>
    /// Gets the effective base URL (mirror or original).
    /// </summary>
    public string EffectiveBaseUrl => UseMirror && !string.IsNullOrEmpty(MirrorUrl)
        ? MirrorUrl
        : BaseUrl;

    /// <summary>
    /// Validates the configuration.
    /// </summary>
    /// <returns>True if the configuration is valid, false otherwise.</returns>
    public override bool IsValid()
    {
        return base.IsValid() && !string.IsNullOrEmpty(BaseUrl);
    }
}

/// <summary>
/// Configuration for TensorFlow Hub.
/// </summary>
public class TensorFlowHubConfiguration : HubConfiguration
{
    /// <summary>
    /// Gets or sets the base URL for TensorFlow Hub.
    /// </summary>
    public string BaseUrl { get; set; } = "https://tfhub.dev";

    /// <summary>
    /// Gets or sets whether to use compression for downloads.
    /// </summary>
    public bool UseCompression { get; set; } = true;

    /// <summary>
    /// Gets or sets the compression format (e.g., "gzip", "bz2").
    /// </summary>
    public string CompressionFormat { get; set; } = "gzip";

    /// <summary>
    /// Gets or sets the preferred model format (e.g., "saved_model", "hdf5").
    /// </summary>
    public string PreferredFormat { get; set; } = "saved_model";

    /// <summary>
    /// Validates the configuration.
    /// </summary>
    /// <returns>True if the configuration is valid, false otherwise.</returns>
    public override bool IsValid()
    {
        return base.IsValid() && !string.IsNullOrEmpty(BaseUrl);
    }
}

/// <summary>
/// Configuration for ONNX Model Zoo.
/// </summary>
public class ONNXHubConfiguration : HubConfiguration
{
    /// <summary>
    /// Gets or sets the base URL for the ONNX Model Zoo.
    /// </summary>
    public string BaseUrl { get; set; } = "https://github.com/onnx/models";

    /// <summary>
    /// Gets or sets the raw content URL for direct file downloads.
    /// </summary>
    public string RawContentUrl { get; set; } = "https://raw.githubusercontent.com/onnx/models/main";

    /// <summary>
    /// Gets or sets mirror URLs for faster downloads.
    /// </summary>
    public List<string> MirrorUrls { get; set; } = new();

    /// <summary>
    /// Gets or sets whether to prefer mirror URLs.
    /// </summary>
    public bool PreferMirrors { get; set; }

    /// <summary>
    /// Gets or sets the default version to use if not specified.
    /// </summary>
    public string DefaultVersion { get; set; } = "latest";

    /// <summary>
    /// Validates the configuration.
    /// </summary>
    /// <returns>True if the configuration is valid, false otherwise.</returns>
    public override bool IsValid()
    {
        return base.IsValid() && !string.IsNullOrEmpty(BaseUrl) && !string.IsNullOrEmpty(RawContentUrl);
    }
}
