namespace MLFramework.Inference.ContinuousBatching;

/// <summary>
/// Configuration for the scheduler API client.
/// </summary>
public record class SchedulerApiClientConfiguration(
    int DefaultMaxTokens,
    double TimeoutMultiplier,
    bool EnableRequestLogging,
    bool EnableStatisticsCollection,
    int MaxConcurrentEnqueue
)
{
    /// <summary>
    /// Default API client configuration.
    /// </summary>
    public static readonly SchedulerApiClientConfiguration Default = new(
        DefaultMaxTokens: 256,
        TimeoutMultiplier: 1.5,
        EnableRequestLogging: true,
        EnableStatisticsCollection: true,
        MaxConcurrentEnqueue: 100
    );
}
