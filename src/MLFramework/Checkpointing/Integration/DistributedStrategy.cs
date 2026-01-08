namespace MachineLearning.Checkpointing;

/// <summary>
/// Distributed training strategy enumeration
/// </summary>
public enum DistributedStrategy
{
    /// <summary>
    /// DDP - full model on each rank
    /// </summary>
    DataParallel,

    /// <summary>
    /// FSDP - model sharded across ranks
    /// </summary>
    FullyShardedDataParallel,

    /// <summary>
    /// TP - model split along tensor dimensions
    /// </summary>
    TensorParallel,

    /// <summary>
    /// PP - model split across layers
    /// </summary>
    PipelineParallel,

    /// <summary>
    /// Combination of strategies
    /// </summary>
    Hybrid
}
