namespace MLFramework.Checkpointing;

/// <summary>
/// Statistics for recomputation operations
/// </summary>
public class RecomputationStats
{
    /// <summary>
    /// Total number of recomputations performed
    /// </summary>
    public int TotalRecomputations { get; set; }

    /// <summary>
    /// Total time spent recomputing in milliseconds
    /// </summary>
    public long TotalRecomputationTimeMs { get; set; }

    /// <summary>
    /// Average recomputation time in milliseconds
    /// </summary>
    public double AverageRecomputationTimeMs =>
        TotalRecomputations > 0 ? (double)TotalRecomputationTimeMs / TotalRecomputations : 0.0;

    /// <summary>
    /// Number of registered recompute functions
    /// </summary>
    public int RegisteredFunctions { get; set; }
}
