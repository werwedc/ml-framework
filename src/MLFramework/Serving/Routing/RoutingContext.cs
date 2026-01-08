namespace MLFramework.Serving.Routing;

/// <summary>
/// Context for routing decisions
/// </summary>
public class RoutingContext
{
    /// <summary>
    /// Preferred version (if specified by client)
    /// </summary>
    public string? PreferredVersion { get; set; }

    /// <summary>
    /// HTTP headers for header-based routing
    /// </summary>
    public Dictionary<string, string>? Headers { get; set; }

    /// <summary>
    /// Experiment identifier for A/B testing
    /// </summary>
    public string? ExperimentId { get; set; }

    /// <summary>
    /// User identifier for user-based routing
    /// </summary>
    public string? UserId { get; set; }
}
