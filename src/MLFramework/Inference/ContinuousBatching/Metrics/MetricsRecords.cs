namespace MLFramework.Inference.ContinuousBatching;

/// <summary>
/// Metrics for a single request.
/// </summary>
public record class RequestMetrics(
    RequestId RequestId,
    int TokensGenerated,
    CompletionReason Reason,
    TimeSpan QueueTime,
    TimeSpan ProcessingTime,
    TimeSpan TotalTime,
    DateTime CompletedTime
);

/// <summary>
/// Metrics for a single iteration.
/// </summary>
public record class IterationMetrics(
    int IterationNumber,
    int RequestCount,
    int TokensGenerated,
    int RequestsCompleted,
    TimeSpan ProcessingTime,
    long MemoryBytesUsed,
    DateTime Timestamp
);

/// <summary>
/// Metrics for a single batch.
/// </summary>
public record class BatchMetrics(
    int BatchId,
    int RequestCount,
    double Utilization,
    long MemoryBytesUsed,
    DateTime Timestamp
);
