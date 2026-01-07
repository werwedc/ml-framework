namespace MLFramework.Inference.ContinuousBatching;

/// <summary>
/// Unique identifier for each inference request.
/// Lightweight struct to minimize memory overhead.
/// </summary>
public record struct RequestId(Guid Id)
{
    /// <summary>
    /// Creates a new unique request ID.
    /// </summary>
    public static RequestId New() => new(Guid.NewGuid());

    /// <summary>
    /// Returns an empty request ID (all zeros).
    /// </summary>
    public static RequestId Empty => new(Guid.Empty);
}
