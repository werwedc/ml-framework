namespace MachineLearning.Distributed.Enums;

/// <summary>
/// Data redistribution strategies when topology changes
/// </summary>
public enum RedistributionType
{
    /// <summary>
    /// Redistribute all data across new worker set (better load balance, more data movement)
    /// </summary>
    FullReshuffle,

    /// <summary>
    /// Keep existing workers' data, only redistribute from lost/new workers (faster, temporary imbalance)
    /// </summary>
    Incremental
}
