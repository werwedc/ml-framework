namespace MLFramework.Serving.Routing;

/// <summary>
/// Type of matching for header-based routing rules
/// </summary>
public enum MatchType
{
    /// <summary>
    /// Exact match of header value
    /// </summary>
    Exact,

    /// <summary>
    /// Prefix match (header value starts with pattern)
    /// </summary>
    Prefix,

    /// <summary>
    /// Regular expression match
    /// </summary>
    Regex,

    /// <summary>
    /// Contains match (header value contains pattern)
    /// </summary>
    Contains
}
