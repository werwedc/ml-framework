namespace MachineLearning.Checkpointing;

/// <summary>
/// Configuration options for checkpoint storage backends
/// </summary>
public class StorageOptions
{
    /// <summary>
    /// Storage provider type: "local", "s3", "gcs", "azure"
    /// </summary>
    public string Provider { get; set; } = "local";

    /// <summary>
    /// Connection settings specific to the provider
    /// For "local": basePath (directory path)
    /// For "s3": bucket, accessKey, secretKey, region, endpoint
    /// For "gcs": bucket, credentialsPath
    /// For "azure": containerName, connectionString
    /// </summary>
    public Dictionary<string, string> ConnectionSettings { get; set; } = new();
}

/// <summary>
/// Factory for creating checkpoint storage instances
/// </summary>
public static class StorageFactory
{
    /// <summary>
    /// Creates a checkpoint storage instance based on the provided options
    /// </summary>
    /// <param name="options">Storage configuration options</param>
    /// <returns>A configured ICheckpointStorage implementation</returns>
    /// <exception cref="ArgumentNullException">Thrown when options is null</exception>
    /// <exception cref="ArgumentException">Thrown when provider is unknown or settings are invalid</exception>
    public static ICheckpointStorage Create(StorageOptions options)
    {
        if (options == null)
            throw new ArgumentNullException(nameof(options));

        if (string.IsNullOrWhiteSpace(options.Provider))
            throw new ArgumentException("Provider cannot be empty", nameof(options));

        var provider = options.Provider.ToLowerInvariant();

        return provider switch
        {
            "local" => CreateLocalStorage(options.ConnectionSettings),
            "s3" => throw new NotSupportedException("S3 storage not yet implemented"),
            "gcs" => throw new NotSupportedException("GCS storage not yet implemented"),
            "azure" => throw new NotSupportedException("Azure Blob storage not yet implemented"),
            _ => throw new ArgumentException($"Unknown storage provider: {options.Provider}", nameof(options.Provider))
        };
    }

    private static ICheckpointStorage CreateLocalStorage(Dictionary<string, string> settings)
    {
        if (!settings.TryGetValue("basePath", out var basePath))
            throw new ArgumentException("Local storage requires 'basePath' in ConnectionSettings");

        if (string.IsNullOrWhiteSpace(basePath))
            throw new ArgumentException("basePath cannot be empty");

        return new LocalFileSystemStorage(basePath);
    }
}
