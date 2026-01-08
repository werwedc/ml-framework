namespace MLFramework.Serving.Deployment;

/// <summary>
/// Interface representing a loaded model that can perform inference
/// </summary>
public interface IModel : IDisposable
{
    /// <summary>
    /// The model name
    /// </summary>
    string Name { get; }

    /// <summary>
    /// The model version
    /// </summary>
    string Version { get; }

    /// <summary>
    /// When the model was loaded
    /// </summary>
    DateTime LoadTime { get; }

    /// <summary>
    /// Whether the model is currently active and serving requests
    /// </summary>
    bool IsActive { get; set; }

    /// <summary>
    /// Perform inference on the given input
    /// </summary>
    /// <param name="input">The inference input</param>
    /// <returns>The inference result</returns>
    Task<InferenceResult> InferAsync(InferenceInput input);
}
