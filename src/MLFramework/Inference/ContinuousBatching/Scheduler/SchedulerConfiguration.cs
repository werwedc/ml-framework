namespace MLFramework.Inference.ContinuousBatching;

/// <summary>
/// Configuration for the continuous batch scheduler.
/// </summary>
public record class SchedulerConfiguration(
    int IterationTimeoutMs,              // Timeout per iteration
    int MaxIdleIterations,               // Max idle iterations before throttle
    int MinIterationsPerSecond,          // Target iteration rate
    bool EnableAdaptiveBatching,         // Adjust batch size based on load
    int WarmupIterations,                // Number of warmup iterations
    double TargetUtilization,            // Target GPU utilization (0-1)
    int MaxBatchSize                     // Maximum batch size
)
{
    /// <summary>
    /// Default scheduler configuration.
    /// </summary>
    public static readonly SchedulerConfiguration Default = new(
        IterationTimeoutMs: 1000,
        MaxIdleIterations: 10,
        MinIterationsPerSecond: 30,
        EnableAdaptiveBatching: true,
        WarmupIterations: 5,
        TargetUtilization: 0.85,
        MaxBatchSize: 32
    );
}
