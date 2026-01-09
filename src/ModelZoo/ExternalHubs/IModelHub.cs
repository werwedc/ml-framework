using MLFramework.ModelVersioning;

namespace MLFramework.ModelZoo.ExternalHubs;

/// <summary>
/// Base interface for model hubs (Hugging Face, TensorFlow Hub, ONNX Model Zoo, etc.).
/// </summary>
public interface IModelHub
{
    /// <summary>
    /// Gets the hub identifier (e.g., "huggingface", "tensorflow", "onnx").
    /// </summary>
    string HubName { get; }

    /// <summary>
    /// Gets the authentication method used by this hub.
    /// </summary>
    IHubAuthentication? Authentication { get; }

    /// <summary>
    /// Gets the hub configuration.
    /// </summary>
    HubConfiguration Configuration { get; }

    /// <summary>
    /// Gets model metadata from the hub.
    /// </summary>
    /// <param name="modelId">The model identifier (e.g., "bert-base-uncased").</param>
    /// <returns>A task that returns the model metadata.</returns>
    Task<ModelMetadata> GetModelMetadataAsync(string modelId);

    /// <summary>
    /// Downloads model files from the hub.
    /// </summary>
    /// <param name="modelId">The model identifier.</param>
    /// <param name="progress">Optional progress reporter for download progress.</param>
    /// <returns>A task that returns a stream containing the model data.</returns>
    Task<Stream> DownloadModelAsync(string modelId, IProgress<double>? progress = null);

    /// <summary>
    /// Checks if a model exists in the hub.
    /// </summary>
    /// <param name="modelId">The model identifier.</param>
    /// <returns>A task that returns true if the model exists, false otherwise.</returns>
    Task<bool> ModelExistsAsync(string modelId);

    /// <summary>
    /// Lists available models in the hub.
    /// </summary>
    /// <param name="filter">Optional filter to narrow down the list of models.</param>
    /// <returns>A task that returns an array of available model identifiers.</returns>
    Task<string[]> ListModelsAsync(string? filter = null);

    /// <summary>
    /// Checks if this hub can handle the given model ID.
    /// </summary>
    /// <param name="modelId">The model identifier.</param>
    /// <returns>True if this hub can handle the model ID, false otherwise.</returns>
    bool CanHandle(string modelId);
}
