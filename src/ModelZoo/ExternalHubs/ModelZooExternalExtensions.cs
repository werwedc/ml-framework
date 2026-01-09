using MLFramework.Core;
using MLFramework.ModelVersioning;

namespace MLFramework.ModelZoo.ExternalHubs;

/// <summary>
/// Extension methods for adding external hub support to ModelZoo.
/// </summary>
public static class ModelZooExternalExtensions
{
    private static HubRegistry? _hubRegistry;

    /// <summary>
    /// Gets the hub registry instance.
    /// </summary>
    public static HubRegistry HubRegistry
    {
        get
        {
            _hubRegistry ??= new HubRegistry();
            return _hubRegistry;
        }
    }

    /// <summary>
    /// Initializes the ModelZoo with default external hubs.
    /// </summary>
    public static void InitializeExternalHubs()
    {
        HubRegistry.RegisterDefaultHubs();
    }

    /// <summary>
    /// Loads a model from an external hub.
    /// </summary>
    /// <param name="modelZoo">The ModelZoo instance.</param>
    /// <param name="modelId">The model identifier (e.g., "hub:huggingface/bert-base-uncased" or "bert-base-uncased").</param>
    /// <param name="pretrained">Whether to load pretrained weights (default: true).</param>
    /// <param name="device">The target device (optional).</param>
    /// <returns>A task that returns the loaded model.</returns>
    /// <exception cref="ArgumentException">Thrown when the model ID is null or empty.</exception>
    /// <exception cref="InvalidOperationException">Thrown when no hub can handle the model ID.</exception>
    public static async Task<Module> LoadFromHubAsync(
        this ModelZoo modelZoo,
        string modelId,
        bool pretrained = true,
        Device? device = null)
    {
        if (string.IsNullOrWhiteSpace(modelId))
        {
            throw new ArgumentException("Model ID cannot be null or empty.", nameof(modelId));
        }

        // Parse the model ID
        var components = ModelIdParser.Parse(modelId);

        // Get the appropriate hub
        var hub = components.IsLocal
            ? HubRegistry.GetDefaultHub()
            : HubRegistry.GetHubForModel(modelId);

        if (hub == null)
        {
            throw new InvalidOperationException($"No hub registered that can handle model ID: {modelId}");
        }

        // Get model metadata
        var metadata = await hub.GetModelMetadataAsync(components.ModelName);

        // TODO: Implement actual model loading from stream
        // This is a placeholder that would need to be implemented based on
        // the specific model format and the framework's model loading capabilities
        throw new NotImplementedException("Model loading from hub streams is not yet implemented.");
    }

    /// <summary>
    /// Gets metadata for a model from an external hub.
    /// </summary>
    /// <param name="modelZoo">The ModelZoo instance.</param>
    /// <param name="modelId">The model identifier.</param>
    /// <returns>A task that returns the model metadata.</returns>
    /// <exception cref="ArgumentException">Thrown when the model ID is null or empty.</exception>
    /// <exception cref="InvalidOperationException">Thrown when no hub can handle the model ID.</exception>
    public static async Task<ModelMetadata> GetModelMetadataFromHubAsync(
        this ModelZoo modelZoo,
        string modelId)
    {
        if (string.IsNullOrWhiteSpace(modelId))
        {
            throw new ArgumentException("Model ID cannot be null or empty.", nameof(modelId));
        }

        // Parse the model ID
        var components = ModelIdParser.Parse(modelId);

        // Get the appropriate hub
        var hub = components.IsLocal
            ? HubRegistry.GetDefaultHub()
            : HubRegistry.GetHubForModel(modelId);

        if (hub == null)
        {
            throw new InvalidOperationException($"No hub registered that can handle model ID: {modelId}");
        }

        return await hub.GetModelMetadataAsync(components.ModelName);
    }

    /// <summary>
    /// Lists models from a specific external hub.
    /// </summary>
    /// <param name="modelZoo">The ModelZoo instance.</param>
    /// <param name="hubName">The name of the hub (e.g., "huggingface", "tensorflow", "onnx").</param>
    /// <param name="filter">Optional filter to narrow down the list of models.</param>
    /// <returns>A task that returns an array of available model identifiers.</returns>
    public static async Task<string[]> ListHubModelsAsync(
        this ModelZoo modelZoo,
        string hubName,
        string? filter = null)
    {
        if (string.IsNullOrWhiteSpace(hubName))
        {
            throw new ArgumentException("Hub name cannot be null or empty.", nameof(hubName));
        }

        var hub = HubRegistry.GetHub(hubName);
        if (hub == null)
        {
            throw new InvalidOperationException($"Hub '{hubName}' is not registered.");
        }

        return await hub.ListModelsAsync(filter);
    }

    /// <summary>
    /// Downloads a model from an external hub without loading it.
    /// </summary>
    /// <param name="modelZoo">The ModelZoo instance.</param>
    /// <param name="modelId">The model identifier.</param>
    /// <param name="progress">Optional progress reporter for download progress.</param>
    /// <returns>A task that returns a stream containing the model data.</returns>
    public static async Task<Stream> DownloadModelFromHubAsync(
        this ModelZoo modelZoo,
        string modelId,
        IProgress<double>? progress = null)
    {
        if (string.IsNullOrWhiteSpace(modelId))
        {
            throw new ArgumentException("Model ID cannot be null or empty.", nameof(modelId));
        }

        // Parse the model ID
        var components = ModelIdParser.Parse(modelId);

        // Get the appropriate hub
        var hub = components.IsLocal
            ? HubRegistry.GetDefaultHub()
            : HubRegistry.GetHubForModel(modelId);

        if (hub == null)
        {
            throw new InvalidOperationException($"No hub registered that can handle model ID: {modelId}");
        }

        return await hub.DownloadModelAsync(components.ModelName, progress);
    }

    /// <summary>
    /// Checks if a model exists in an external hub.
    /// </summary>
    /// <param name="modelZoo">The ModelZoo instance.</param>
    /// <param name="modelId">The model identifier.</param>
    /// <returns>A task that returns true if the model exists, false otherwise.</returns>
    public static async Task<bool> ModelExistsInHubAsync(
        this ModelZoo modelZoo,
        string modelId)
    {
        if (string.IsNullOrWhiteSpace(modelId))
        {
            return false;
        }

        // Parse the model ID
        var components = ModelIdParser.Parse(modelId);

        // Get the appropriate hub
        var hub = components.IsLocal
            ? HubRegistry.GetDefaultHub()
            : HubRegistry.GetHubForModel(modelId);

        if (hub == null)
        {
            return false;
        }

        return await hub.ModelExistsAsync(components.ModelName);
    }

    /// <summary>
    /// Registers a custom hub.
    /// </summary>
    /// <param name="modelZoo">The ModelZoo instance.</param>
    /// <param name="hub">The hub to register.</param>
    public static void RegisterHub(this ModelZoo modelZoo, IModelHub hub)
    {
        if (hub == null)
        {
            throw new ArgumentNullException(nameof(hub));
        }

        HubRegistry.RegisterHub(hub);
    }

    /// <summary>
    /// Unregisters a hub.
    /// </summary>
    /// <param name="modelZoo">The ModelZoo instance.</param>
    /// <param name="hubName">The name of the hub to unregister.</param>
    /// <returns>True if the hub was successfully unregistered; false otherwise.</returns>
    public static bool UnregisterHub(this ModelZoo modelZoo, string hubName)
    {
        return HubRegistry.UnregisterHub(hubName);
    }

    /// <summary>
    /// Gets a registered hub.
    /// </summary>
    /// <param name="modelZoo">The ModelZoo instance.</param>
    /// <param name="hubName">The name of the hub.</param>
    /// <returns>The hub if found; otherwise, null.</returns>
    public static IModelHub? GetHub(this ModelZoo modelZoo, string hubName)
    {
        return HubRegistry.GetHub(hubName);
    }

    /// <summary>
    /// Lists all registered hubs.
    /// </summary>
    /// <param name="modelZoo">The ModelZoo instance.</param>
    /// <returns>An array of registered hub names.</returns>
    public static string[] ListHubs(this ModelZoo modelZoo)
    {
        return HubRegistry.ListHubs();
    }

    /// <summary>
    /// Sets the default hub.
    /// </summary>
    /// <param name="modelZoo">The ModelZoo instance.</param>
    /// <param name="hubName">The name of the hub to set as default.</param>
    /// <returns>True if the default hub was successfully set; false otherwise.</returns>
    public static bool SetDefaultHub(this ModelZoo modelZoo, string hubName)
    {
        return HubRegistry.SetDefaultHub(hubName);
    }

    /// <summary>
    /// Gets the default hub.
    /// </summary>
    /// <param name="modelZoo">The ModelZoo instance.</param>
    /// <returns>The default hub; otherwise, null.</returns>
    public static IModelHub? GetDefaultHub(this ModelZoo modelZoo)
    {
        return HubRegistry.GetDefaultHub();
    }
}
