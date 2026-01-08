using MLFramework.Serving.Deployment;

namespace MLFramework.Serving.Routing;

/// <summary>
/// Interface for routing inference requests to specific model versions
/// </summary>
public interface IVersionRouter
{
    /// <summary>
    /// Get a model based on routing context
    /// </summary>
    /// <param name="modelName">Name of the model</param>
    /// <param name="context">Routing context containing version preferences and metadata</param>
    /// <returns>The appropriate model instance</returns>
    /// <exception cref="RoutingException">Thrown when routing fails</exception>
    IModel GetModel(string modelName, RoutingContext context);

    /// <summary>
    /// Get a model by specific version
    /// </summary>
    /// <param name="modelName">Name of the model</param>
    /// <param name="version">Specific version to retrieve</param>
    /// <returns>The model instance for the specified version</returns>
    /// <exception cref="RoutingException">Thrown when version doesn't exist</exception>
    IModel GetModel(string modelName, string version);

    /// <summary>
    /// Set the default version for a model
    /// </summary>
    /// <param name="modelName">Name of the model</param>
    /// <param name="version">Version to set as default</param>
    void SetDefaultVersion(string modelName, string version);

    /// <summary>
    /// Get the default version for a model
    /// </summary>
    /// <param name="modelName">Name of the model</param>
    /// <returns>The default version, or null if not set</returns>
    string? GetDefaultVersion(string modelName);
}
