namespace MLFramework.Checkpointing.Profiling;

/// <summary>
/// Type of recommendation
/// </summary>
public enum RecommendationType
{
    ReduceRecomputation,
    ImproveCacheHitRate,
    ImproveEfficiency,
    AdjustCheckpointStrategy,
    EnableAsyncRecomputation,
    IncreaseCheckpointFrequency
}

/// <summary>
/// Priority of recommendation
/// </summary>
public enum RecommendationPriority
{
    High,
    Medium,
    Low
}
