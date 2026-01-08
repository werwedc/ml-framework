namespace MLFramework.Serving.Deployment;

/// <summary>
/// Interface for serializing and deserializing model metadata to/from JSON.
/// </summary>
public interface IMetadataSerializer
{
    /// <summary>
    /// Serializes model metadata to JSON string.
    /// </summary>
    /// <param name="metadata">The metadata to serialize.</param>
    /// <returns>JSON string representation of the metadata.</returns>
    string Serialize(ModelMetadata metadata);

    /// <summary>
    /// Deserializes a JSON string to model metadata.
    /// </summary>
    /// <param name="json">The JSON string to deserialize.</param>
    /// <returns>The deserialized model metadata.</returns>
    ModelMetadata Deserialize(string json);

    /// <summary>
    /// Serializes model metadata and saves it to a file.
    /// </summary>
    /// <param name="filePath">The file path to save to.</param>
    /// <param name="metadata">The metadata to serialize and save.</param>
    void SaveToFile(string filePath, ModelMetadata metadata);

    /// <summary>
    /// Loads and deserializes model metadata from a file.
    /// </summary>
    /// <param name="filePath">The file path to load from.</param>
    /// <returns>The deserialized model metadata.</returns>
    ModelMetadata LoadFromFile(string filePath);

    /// <summary>
    /// Asynchronously serializes model metadata to JSON string.
    /// </summary>
    /// <param name="metadata">The metadata to serialize.</param>
    /// <returns>A task that returns the JSON string representation of the metadata.</returns>
    Task<string> SerializeAsync(ModelMetadata metadata);

    /// <summary>
    /// Asynchronously deserializes a JSON string to model metadata.
    /// </summary>
    /// <param name="json">The JSON string to deserialize.</param>
    /// <returns>A task that returns the deserialized model metadata.</returns>
    Task<ModelMetadata> DeserializeAsync(string json);
}
