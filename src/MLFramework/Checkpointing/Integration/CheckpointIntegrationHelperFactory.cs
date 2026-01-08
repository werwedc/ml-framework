namespace MachineLearning.Checkpointing;

using Microsoft.Extensions.Logging;

/// <summary>
/// Factory for creating checkpoint integration helpers based on distributed strategy
/// </summary>
public static class CheckpointIntegrationHelperFactory
{
    /// <summary>
    /// Create a checkpoint helper based on the model's distributed strategy
    /// </summary>
    public static object CreateHelper(
        IDistributedModel model,
        IDistributedCoordinator coordinator,
        ILogger? logger = null)
    {
        return model.Strategy switch
        {
            DistributedStrategy.FullyShardedDataParallel =>
                new FSDPCheckpointHelper(model, coordinator, logger as ILogger<FSDPCheckpointHelper>),
            DistributedStrategy.DataParallel =>
                new DDPCheckpointHelper(model, coordinator, logger as ILogger<DDPCheckpointHelper>),
            DistributedStrategy.TensorParallel =>
                new TensorParallelCheckpointHelper(model, coordinator, logger as ILogger<TensorParallelCheckpointHelper>),
            _ => throw new ArgumentException($"Unsupported distributed strategy: {model.Strategy}")
        };
    }
}
