namespace MLFramework.Inference.ContinuousBatching;

/// <summary>
/// Configuration for metrics collection.
/// </summary>
public record class MetricsConfiguration(
    int MaxRequestSamples,              // Max request metrics to keep
    int MaxIterationSamples,            // Max iteration metrics to keep
    int MaxBatchSamples,                // Max batch metrics to keep
    int CounterWindowSeconds,          // Window for rate counters
    int PercentilePrecision,            // Precision for percentile calculation
    bool EnableDetailedLogging,         // Log detailed metrics
    int DetailedLogIntervalSeconds      // Interval for detailed logs
)
{
    /// <summary>
    /// Default metrics configuration.
    /// </summary>
    public static readonly MetricsConfiguration Default = new(
        MaxRequestSamples: 10000,
        MaxIterationSamples: 10000,
        MaxBatchSamples: 10000,
        CounterWindowSeconds: 60,
        PercentilePrecision: 2,
        EnableDetailedLogging: false,
        DetailedLogIntervalSeconds: 30
    );
}
