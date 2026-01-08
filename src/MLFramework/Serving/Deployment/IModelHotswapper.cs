namespace MLFramework.Serving.Deployment;

/// <summary>
/// Interface for model hotswapping functionality
/// </summary>
public interface IModelHotswapper
{
    /// <summary>
    /// Hotswap from one model version to another
    /// </summary>
    Task HotswapAsync(string modelName, string fromVersion, string toVersion);

    /// <summary>
    /// Check if a hotswap is currently in progress
    /// </summary>
    bool IsHotswapInProgress(string modelName);
}
