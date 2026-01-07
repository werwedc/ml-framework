namespace MLFramework.Inference.ContinuousBatching;

/// <summary>
/// Current scheduler status.
/// </summary>
public record class SchedulerStatus(
    bool IsRunning,
    int ActiveRequests,
    int QueuedRequests,
    int CompletedRequests,
    double GpuUtilization,
    double MemoryUtilization,
    DateTime LastUpdateTime
);
