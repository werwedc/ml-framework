namespace MLFramework.Inference.ContinuousBatching.Prefill;

/// <summary>
/// Result of batch prefill operation.
/// </summary>
public record class BatchPrefillResult(
    int TotalRequests,
    int SuccessfulRequests,
    int FailedRequests,
    List<PrefillResult> RequestResults,
    TimeSpan TotalProcessingTime,
    long TotalMemoryUsed
);
