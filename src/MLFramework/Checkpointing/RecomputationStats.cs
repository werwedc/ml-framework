namespace MLFramework.Checkpointing;

/// <summary>
/// Statistics about recomputation operations
/// </summary>
public class RecomputationStats
{
    /// <summary>
    /// Total number of recomputation calls
    /// </summary>
    public int TotalRecomputations { get; set; }

    /// <summary>
    /// Total time spent on recomputation (in milliseconds)
    /// </summary>
    public long TotalRecomputationTimeMs { get; set; }

    /// <summary>
    /// Average time per recomputation (in milliseconds)
    /// </summary>
    public double AverageRecomputationTimeMs =>
        TotalRecomputations > 0 ? (double)TotalRecomputationTimeMs / TotalRecomputations : 0.0;

    /// <summary>
    /// Number of layers with registered recompute functions
    /// </summary>
    public int RegisteredLayerCount { get; set; }

    /// <summary>
    /// Per-layer recomputation statistics
    /// </summary>
    public Dictionary<string, LayerRecomputationStats> LayerStats { get; set; } = new();

    /// <summary>
    /// Timestamp when stats were collected
    /// </summary>
    public DateTime Timestamp { get; set; }
}
