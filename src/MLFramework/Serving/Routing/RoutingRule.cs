namespace MLFramework.Serving.Routing;

/// <summary>
/// Defines a header-based routing rule
/// </summary>
public class RoutingRule
{
    private static int _nextId = 0;

    /// <summary>
    /// Unique identifier for this rule
    /// </summary>
    public string Id { get; }

    /// <summary>
    /// HTTP header name to match
    /// </summary>
    public string HeaderName { get; set; } = string.Empty;

    /// <summary>
    /// Header value pattern to match
    /// </summary>
    public string HeaderValue { get; set; } = string.Empty;

    /// <summary>
    /// Type of matching to perform
    /// </summary>
    public MatchType MatchType { get; set; }

    /// <summary>
    /// Target model version to route to
    /// </summary>
    public string TargetVersion { get; set; } = string.Empty;

    /// <summary>
    /// Priority for rule evaluation (higher values evaluated first)
    /// </summary>
    public int Priority { get; set; }

    /// <summary>
    /// Optional description of this rule
    /// </summary>
    public string? Description { get; set; }

    /// <summary>
    /// Model name this rule applies to (set during registration)
    /// </summary>
    public string? ModelName { get; internal set; }

    public RoutingRule()
    {
        Id = System.Threading.Interlocked.Increment(ref _nextId).ToString();
    }
}
