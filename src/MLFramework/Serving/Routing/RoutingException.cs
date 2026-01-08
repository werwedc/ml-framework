namespace MLFramework.Serving.Routing;

/// <summary>
/// Exception thrown when routing fails
/// </summary>
public class RoutingException : Exception
{
    /// <summary>
    /// Name of the model being routed
    /// </summary>
    public string ModelName { get; }

    /// <summary>
    /// Requested version (if applicable)
    /// </summary>
    public string? RequestedVersion { get; }

    public RoutingException(string modelName, string? requestedVersion, string message)
        : base(message)
    {
        ModelName = modelName;
        RequestedVersion = requestedVersion;
    }

    public RoutingException(string modelName, string? requestedVersion, string message, Exception innerException)
        : base(message, innerException)
    {
        ModelName = modelName;
        RequestedVersion = requestedVersion;
    }
}
