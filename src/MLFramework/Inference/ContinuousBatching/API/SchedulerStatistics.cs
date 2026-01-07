namespace MLFramework.Inference.ContinuousBatching;

/// <summary>
/// Historical scheduler statistics.
/// </summary>
public record class SchedulerStatistics(
    int TotalRequests,
    int TotalCompletedRequests,
    int TotalFailedRequests,
    int TotalCancelledRequests,
    TimeSpan TotalProcessingTime,
    double AverageRequestsPerSecond,
    double AverageTokensPerSecond,
    double P50Latency,
    double P95Latency,
    double P99Latency,
    double AverageBatchUtilization,
    DateTime StartTime
);
