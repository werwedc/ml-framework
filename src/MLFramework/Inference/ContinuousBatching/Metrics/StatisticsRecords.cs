namespace MLFramework.Inference.ContinuousBatching;

/// <summary>
/// Aggregated request statistics.
/// </summary>
public record class RequestStatistics(
    long TotalRequests,
    long CompletedRequests,
    long FailedRequests,
    long CancelledRequests,
    double AverageTokensPerRequest,
    double P50Latency,
    double P95Latency,
    double P99Latency,
    double RequestsPerSecond,
    double TokensPerSecond,
    TimeSpan AverageQueueTime,
    TimeSpan AverageProcessingTime
);

/// <summary>
/// Aggregated iteration statistics.
/// </summary>
public record class IterationStatistics(
    long TotalIterations,
    int AverageRequestsPerIteration,
    int AverageTokensPerIteration,
    double AverageMemoryBytesPerIteration,
    double AverageProcessingTimeMs,
    double IterationsPerSecond,
    double AverageUtilization
);

/// <summary>
/// Aggregated batch statistics.
/// </summary>
public record class BatchStatistics(
    long TotalBatches,
    double AverageBatchSize,
    double AverageUtilization,
    double AverageMemoryBytesPerBatch,
    int MaxBatchSize,
    int MinBatchSize,
    double MaxBatchSizeRaw
);

/// <summary>
/// Error statistics.
/// </summary>
public record class ErrorStatistics(
    long TotalErrors,
    Dictionary<string, long> ErrorsByType,
    double ErrorRate,
    DateTime LastErrorTime
);

/// <summary>
/// Complete metrics snapshot.
/// </summary>
public record class MetricsSnapshot(
    RequestStatistics RequestStats,
    IterationStatistics IterationStats,
    BatchStatistics BatchStats,
    ErrorStatistics ErrorStats,
    DateTime SnapshotTime
);
