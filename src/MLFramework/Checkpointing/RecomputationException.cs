namespace MLFramework.Checkpointing;

/// <summary>
/// Exception thrown when recomputation fails
/// </summary>
public class RecomputationException : Exception
{
    public RecomputationException(string message) : base(message) { }

    public RecomputationException(string message, Exception innerException)
        : base(message, innerException) { }

    /// <summary>
    /// Creates a recomputation exception for a specific layer
    /// </summary>
    public static RecomputationException ForLayer(string layerId, Exception innerException)
    {
        return new RecomputationException($"Recomputation failed for layer '{layerId}'", innerException)
        {
            LayerId = layerId
        };
    }

    public string? LayerId { get; set; }
}
