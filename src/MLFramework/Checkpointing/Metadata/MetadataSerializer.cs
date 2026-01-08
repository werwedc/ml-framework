namespace MachineLearning.Checkpointing;

using System.Text.Json;

/// <summary>
/// Serializer for checkpoint metadata
/// </summary>
public static class MetadataSerializer
{
    private static readonly JsonSerializerOptions Options = new JsonSerializerOptions
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase
    };

    /// <summary>
    /// Serialize checkpoint metadata to JSON
    /// </summary>
    public static string Serialize(CheckpointMetadata metadata)
    {
        if (metadata == null)
            throw new ArgumentNullException(nameof(metadata));

        return JsonSerializer.Serialize(metadata, Options);
    }

    /// <summary>
    /// Deserialize checkpoint metadata from JSON
    /// </summary>
    public static CheckpointMetadata Deserialize(string json)
    {
        if (string.IsNullOrWhiteSpace(json))
            throw new ArgumentException("JSON cannot be empty", nameof(json));

        return JsonSerializer.Deserialize<CheckpointMetadata>(json, Options)
            ?? throw new InvalidOperationException("Failed to deserialize checkpoint metadata");
    }

    /// <summary>
    /// Write checkpoint metadata to a file
    /// </summary>
    public static async Task WriteAsync(
        CheckpointMetadata metadata,
        string path,
        CancellationToken cancellationToken = default)
    {
        if (metadata == null)
            throw new ArgumentNullException(nameof(metadata));

        if (string.IsNullOrWhiteSpace(path))
            throw new ArgumentException("Path cannot be empty", nameof(path));

        var json = Serialize(metadata);
        await File.WriteAllTextAsync(path, json, cancellationToken);
    }

    /// <summary>
    /// Read checkpoint metadata from a file
    /// </summary>
    public static async Task<CheckpointMetadata> ReadAsync(
        string path,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(path))
            throw new ArgumentException("Path cannot be empty", nameof(path));

        if (!File.Exists(path))
            throw new FileNotFoundException($"Metadata file not found: {path}");

        var json = await File.ReadAllTextAsync(path, cancellationToken);
        return Deserialize(json);
    }
}
