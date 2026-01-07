namespace MachineLearning.Distributed.Configuration;

using MachineLearning.Distributed.Enums;

/// <summary>
/// Configuration for elastic training behavior
/// </summary>
public class ElasticTrainingConfig
{
    /// <summary>
    /// Minimum and maximum worker count
    /// </summary>
    public int MinWorkers { get; set; } = 1;
    public int MaxWorkers { get; set; } = int.MaxValue;

    /// <summary>
    /// How long to wait for new workers before proceeding (milliseconds)
    /// </summary>
    public int RescaleTimeoutMs { get; set; } = 30000;

    /// <summary>
    /// Maximum number of consecutive failures before aborting
    /// </summary>
    public int MaxConsecutiveFailures { get; set; } = 3;

    /// <summary>
    /// Learning rate adaptation strategy
    /// </summary>
    public AdaptationStrategy LRAdaptationStrategy { get; set; } = AdaptationStrategy.Linear;

    /// <summary>
    /// Data redistribution method
    /// </summary>
    public RedistributionType RedistributionType { get; set; } = RedistributionType.FullReshuffle;

    /// <summary>
    /// Whether to use synchronous or asynchronous rescaling
    /// </summary>
    public bool UseSynchronousRescaling { get; set; } = true;

    /// <summary>
    /// Stability window before triggering rescaling (milliseconds)
    /// </summary>
    public int StabilityWindowMs { get; set; } = 30000;

    /// <summary>
    /// Timeout for worker heartbeats before considering it failed (milliseconds)
    /// </summary>
    public int WorkerHeartbeatTimeoutMs { get; set; } = 10000;

    /// <summary>
    /// Number of worker failures to tolerate before aborting (as percentage)
    /// </summary>
    public int FailureTolerancePercentage { get; set; } = 20;

    /// <summary>
    /// Whether to use parallel data transfer during redistribution
    /// </summary>
    public bool UseParallelDataTransfer { get; set; } = true;

    public void Validate()
    {
        if (MinWorkers < 1)
            throw new ArgumentException("MinWorkers must be at least 1");

        if (MaxWorkers < MinWorkers)
            throw new ArgumentException("MaxWorkers must be >= MinWorkers");

        if (RescaleTimeoutMs < 0)
            throw new ArgumentException("RescaleTimeoutMs cannot be negative");

        if (MaxConsecutiveFailures < 0)
            throw new ArgumentException("MaxConsecutiveFailures cannot be negative");

        if (StabilityWindowMs < 0)
            throw new ArgumentException("StabilityWindowMs cannot be negative");

        if (WorkerHeartbeatTimeoutMs <= 0)
            throw new ArgumentException("WorkerHeartbeatTimeoutMs must be positive");

        if (FailureTolerancePercentage < 0 || FailureTolerancePercentage > 100)
            throw new ArgumentException("FailureTolerancePercentage must be between 0 and 100");
    }
}
