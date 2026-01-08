namespace MLFramework.Serving.Routing;

/// <summary>
/// Interface for header-based routing
/// </summary>
public interface IHeaderRouter
{
    /// <summary>
    /// Register a routing rule for a model
    /// </summary>
    /// <param name="modelName">Name of the model</param>
    /// <param name="rule">Routing rule to register</param>
    /// <exception cref="ArgumentException">Thrown if rule is invalid or target version doesn't exist</exception>
    void RegisterRoutingRule(string modelName, RoutingRule rule);

    /// <summary>
    /// Unregister a routing rule
    /// </summary>
    /// <param name="modelName">Name of the model</param>
    /// <param name="ruleId">ID of the rule to unregister</param>
    void UnregisterRoutingRule(string modelName, string ruleId);

    /// <summary>
    /// Route a request based on headers
    /// </summary>
    /// <param name="modelName">Name of the model</param>
    /// <param name="context">Routing context containing headers</param>
    /// <returns>Target version name, or null if no rule matches</returns>
    string? RouteByHeaders(string modelName, RoutingContext context);

    /// <summary>
    /// Get all routing rules for a model
    /// </summary>
    /// <param name="modelName">Name of the model</param>
    /// <returns>List of routing rules sorted by priority</returns>
    IEnumerable<RoutingRule> GetRules(string modelName);

    /// <summary>
    /// Clear all routing rules for a model
    /// </summary>
    /// <param name="modelName">Name of the model</param>
    void ClearRules(string modelName);
}
