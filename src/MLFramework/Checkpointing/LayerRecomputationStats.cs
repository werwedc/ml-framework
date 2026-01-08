namespace MLFramework.Checkpointing;

/// <summary>
/// Recomputation statistics for a specific layer
/// </summary>
public class LayerRecomputationStats
{
    /// <summary>
    /// Unique identifier for the layer
    /// </summary>
    public string LayerId { get; set; } = string.Empty;

    /// <summary>
    /// Number of times this layer was recomputed
    /// </summary>
    public int CallCount { get; set; }

    /// <summary>
    /// Total time spent recomputing this layer (in milliseconds)
    /// </summary>
    public long TotalComputationTimeMs { get; set; }

    /// <summary>
    /// Average time per recomputation for this layer (in milliseconds)
    /// </summary>
    public double AverageComputationTimeMs =>
        CallCount > 0 ? (double)TotalComputationTimeMs / CallCount : 0.0;

    /// <summary>
    /// Timestamp of last recomputation
    /// </summary>
    public DateTime LastCalledAt { get; set; }
}
