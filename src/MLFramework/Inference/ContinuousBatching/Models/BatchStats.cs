namespace MLFramework.Inference.ContinuousBatching;

/// <summary>
/// Statistics for monitoring batch performance.
/// </summary>
public record class BatchStats(
    int BatchId,
    int RequestCount,
    long MemoryBytesUsed,
    double UtilizationPercentage,
    TimeSpan ProcessingTime
);
