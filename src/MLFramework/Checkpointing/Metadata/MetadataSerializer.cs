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
}
