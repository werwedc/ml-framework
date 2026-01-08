namespace MachineLearning.Distributed.Enums;

/// <summary>
/// Learning rate adaptation strategies when worker count changes
/// </summary>
public enum AdaptationStrategy
{
    /// <summary>
    /// Scale LR proportionally to new worker count
    /// </summary>
    Linear,

    /// <summary>
    /// LR scales with sqrt(worker_count) for more stability
    /// </summary>
    SquareRoot,

    /// <summary>
    /// Keep global LR constant, only change throughput
    /// </summary>
    None
}
