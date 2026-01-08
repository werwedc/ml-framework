namespace MLFramework.Checkpointing.Profiling;

/// <summary>
/// Profile for a specific layer
/// </summary>
public class LayerProfile
{
    /// <summary>
    /// Layer ID
    /// </summary>
    public string LayerId { get; set; } = string.Empty;

    /// <summary>
    /// Number of checkpoints
    /// </summary>
    public int CheckpointCount { get; set; }

    /// <summary>
    /// Total checkpoint time in milliseconds
    /// </summary>
    public long TotalCheckpointTimeMs { get; set; }

    /// <summary>
    /// Average checkpoint time in milliseconds
    /// </summary>
    public double AverageCheckpointTimeMs =>
        CheckpointCount > 0 ? (double)TotalCheckpointTimeMs / CheckpointCount : 0.0;

    /// <summary>
    /// Number of recomputations
    /// </summary>
    public int RecomputeCount { get; set; }

    /// <summary>
    /// Total recomputation time in milliseconds
    /// </summary>
    public long TotalRecomputeTimeMs { get; set; }

    /// <summary>
    /// Average recomputation time in milliseconds
    /// </summary>
    public double AverageRecomputeTimeMs =>
        RecomputeCount > 0 ? (double)TotalRecomputeTimeMs / RecomputeCount : 0.0;

    /// <summary>
    /// Number of cache hits
    /// </summary>
    public int CacheHitCount { get; set; }

    /// <summary>
    /// Cache hit rate
    /// </summary>
    public double CacheHitRate =>
        CheckpointCount > 0 ? (double)CacheHitCount / CheckpointCount : 0.0;

    /// <summary>
    /// Total memory saved in bytes
    /// </summary>
    public long TotalMemorySaved { get; set; }

    /// <summary>
    /// Records an event
    /// </summary>
    public void RecordEvent(CheckpointEvent @event)
    {
        switch (@event.EventType)
        {
            case CheckpointEventType.Checkpoint:
                CheckpointCount++;
                TotalCheckpointTimeMs += @event.DurationMs;
                TotalMemorySaved += @event.MemoryBytes;
                break;
            case CheckpointEventType.Recompute:
                RecomputeCount++;
                TotalRecomputeTimeMs += @event.DurationMs;
                break;
            case CheckpointEventType.Retrieve:
                CacheHitCount++;
                break;
        }
    }
}
