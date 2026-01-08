namespace MachineLearning.Distributed.Enums;

/// <summary>
/// Status of a worker in the elastic cluster
/// </summary>
public enum WorkerStatus
{
    /// <summary>
    /// Worker has just joined and is initializing
    /// </summary>
    Joining,

    /// <summary>
    /// Worker is actively training
    /// </summary>
    Active,

    /// <summary>
    /// Worker is in rescaling process
    /// </summary>
    Rescaling,

    /// <summary>
    /// Worker is gracefully shutting down
    /// </summary>
    Leaving,

    /// <summary>
    /// Worker has failed or become unresponsive
    /// </summary>
    Failed
}
