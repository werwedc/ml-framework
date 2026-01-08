namespace MLFramework.Serving.Deployment;

/// <summary>
/// Interface for loading and unloading model artifacts
/// </summary>
public interface IModelLoader
{
    /// <summary>
    /// Synchronously load a model from the given path
    /// </summary>
    /// <param name="modelPath">Path to the model file/directory</param>
    /// <param name="version">Version identifier for the model</param>
    /// <returns>The loaded model instance</returns>
    IModel Load(string modelPath, string version);

    /// <summary>
    /// Asynchronously load a model from the given path
    /// </summary>
    /// <param name="modelPath">Path to the model file/directory</param>
    /// <param name="version">Version identifier for the model</param>
    /// <param name="ct">Cancellation token to abort the load operation</param>
    /// <returns>The loaded model instance</returns>
    Task<IModel> LoadAsync(string modelPath, string version, CancellationToken ct = default);

    /// <summary>
    /// Unload a model and release its resources
    /// </summary>
    /// <param name="model">The model to unload</param>
    void Unload(IModel model);

    /// <summary>
    /// Check if a model with the given name and version is loaded
    /// </summary>
    /// <param name="name">The model name</param>
    /// <param name="version">The model version</param>
    /// <returns>True if loaded, false otherwise</returns>
    bool IsLoaded(string name, string version);

    /// <summary>
    /// Get all currently loaded models
    /// </summary>
    /// <returns>Enumeration of loaded models</returns>
    IEnumerable<IModel> GetLoadedModels();
}
