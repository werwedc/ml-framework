namespace MLFramework.Serving.Deployment;

/// <summary>
/// Interface for version routing core functionality
/// </summary>
public interface IVersionRouterCore
{
    /// <summary>
    /// Update the routing to point to a specific model version
    /// </summary>
    Task UpdateRoutingAsync(string modelName, string version);

    /// <summary>
    /// Wait for the current version to drain (complete in-flight requests)
    /// </summary>
    Task WaitForDrainAsync(string modelName, TimeSpan timeout);
}
