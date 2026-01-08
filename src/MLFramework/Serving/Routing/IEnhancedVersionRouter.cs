namespace MLFramework.Serving.Routing;

/// <summary>
/// Enhanced version router with header-based routing and traffic splitting capabilities
/// </summary>
public interface IEnhancedVersionRouter : IVersionRouter
{
    /// <summary>
    /// Register a routing rule for a model
    /// </summary>
    /// <param name="modelName">Name of the model</param>
    /// <param name="rule">Routing rule to register</param>
    void RegisterRoutingRule(string modelName, RoutingRule rule);

    /// <summary>
    /// Set traffic split percentages for different versions
    /// </summary>
    /// <param name="modelName">Name of the model</param>
    /// <param name="versionPercentages">Dictionary mapping version names to percentage (0-100)</param>
    void SetTrafficSplit(string modelName, Dictionary<string, float> versionPercentages);
}
