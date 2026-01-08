namespace MLFramework.Serving.Deployment;

/// <summary>
/// Interface for model hotswapping functionality
/// </summary>
public interface IModelHotswapper
{
    /// <summary>
    /// Swap from one version to another without dropping requests
    /// </summary>
    Task<SwapOperation> SwapVersionAsync(string modelName, string fromVersion, string toVersion);

    /// <summary>
    /// Get the status of a swap operation
    /// </summary>
    SwapOperation GetSwapStatus(string operationId);

    /// <summary>
    /// Wait for the current version to drain (complete in-flight requests)
    /// </summary>
    void WaitForDrainage(string modelName, string version, TimeSpan timeout);

    /// <summary>
    /// Check if a version is currently active
    /// </summary>
    bool IsVersionActive(string modelName, string version);

    /// <summary>
    /// Rollback to the previous version
    /// </summary>
    Task RollbackAsync(string operationId);
}
