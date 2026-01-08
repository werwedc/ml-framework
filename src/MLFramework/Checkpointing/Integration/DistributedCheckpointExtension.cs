namespace MachineLearning.Checkpointing;

using MachineLearning.Distributed.Checkpointing;

/// <summary>
/// Extension methods for distributed checkpointing with strategy detection
/// </summary>
public static class DistributedCheckpointExtension
{
    /// <summary>
    /// Save checkpoint with automatic strategy detection
    /// </summary>
    public static async Task<string> SaveDistributedAsync(
        this DistributedCheckpoint checkpoint,
        IDistributedModel model,
        IStateful optimizer,
        SaveOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        // Create appropriate helper
        var helper = CheckpointIntegrationHelperFactory.CreateHelper(
            model,
            checkpoint.GetCoordinator());

        // Use helper-specific save logic
        return model.Strategy switch
        {
            DistributedStrategy.FullyShardedDataParallel =>
                await SaveFSDPAsync(checkpoint, model, optimizer, helper as FSDPCheckpointHelper, options, cancellationToken),
            DistributedStrategy.DataParallel =>
                await SaveDDPAsync(checkpoint, model, optimizer, helper as DDPCheckpointHelper, options, cancellationToken),
            DistributedStrategy.TensorParallel =>
                await SaveTPAsync(checkpoint, model, optimizer, helper as TensorParallelCheckpointHelper, options, cancellationToken),
            _ => throw new NotSupportedException($"Unsupported strategy: {model.Strategy}")
        };
    }

    private static async Task<string> SaveFSDPAsync(
        DistributedCheckpoint checkpoint,
        IDistributedModel model,
        IStateful optimizer,
        FSDPCheckpointHelper? helper,
        SaveOptions? options,
        CancellationToken cancellationToken)
    {
        // FSDP-specific save logic
        if (helper == null)
        {
            throw new ArgumentNullException(nameof(helper));
        }

        // Collect local shard
        var shardData = helper.CollectLocalShard();
        var metadata = helper.GetShardingMetadata();

        // Create checkpoint with sharded data
        var trainingCheckpoint = new TrainingCheckpoint
        {
            Id = $"{options?.CheckpointPrefix ?? "checkpoint"}_{Guid.NewGuid():N}",
            Epoch = 0,  // Would be populated from training state
            Step = 0,
            LearningRate = 0.001f,
            ModelState = shardData.Data,
            OptimizerState = null,  // Would be populated from optimizer
            WorkerCount = model.WorldSize,
            IsRescalingCheckpoint = false,
            Timestamp = DateTime.UtcNow
        };

        // Add metadata
        trainingCheckpoint.Metadata["strategy"] = metadata.Strategy;
        trainingCheckpoint.Metadata["shard_count"] = metadata.ShardCount.ToString();
        trainingCheckpoint.Metadata["precision"] = metadata.Precision;

        await checkpoint.GetCheckpointManager().SaveCheckpointAsync(trainingCheckpoint);

        return trainingCheckpoint.Id;
    }

    private static async Task<string> SaveDDPAsync(
        DistributedCheckpoint checkpoint,
        IDistributedModel model,
        IStateful optimizer,
        DDPCheckpointHelper? helper,
        SaveOptions? options,
        CancellationToken cancellationToken)
    {
        // DDP-specific save logic
        if (helper == null)
        {
            throw new ArgumentNullException(nameof(helper));
        }

        // Collect full state (same on all ranks)
        var shardData = helper.CollectFullState();
        var metadata = helper.GetShardingMetadata();

        // Create checkpoint with full data
        var trainingCheckpoint = new TrainingCheckpoint
        {
            Id = $"{options?.CheckpointPrefix ?? "checkpoint"}_{Guid.NewGuid():N}",
            Epoch = 0,  // Would be populated from training state
            Step = 0,
            LearningRate = 0.001f,
            ModelState = shardData.Data,
            OptimizerState = null,  // Would be populated from optimizer
            WorkerCount = model.WorldSize,
            IsRescalingCheckpoint = false,
            Timestamp = DateTime.UtcNow
        };

        // Add metadata
        trainingCheckpoint.Metadata["strategy"] = metadata.Strategy;
        trainingCheckpoint.Metadata["shard_count"] = metadata.ShardCount.ToString();
        trainingCheckpoint.Metadata["precision"] = metadata.Precision;

        await checkpoint.GetCheckpointManager().SaveCheckpointAsync(trainingCheckpoint);

        return trainingCheckpoint.Id;
    }

    private static async Task<string> SaveTPAsync(
        DistributedCheckpoint checkpoint,
        IDistributedModel model,
        IStateful optimizer,
        TensorParallelCheckpointHelper? helper,
        SaveOptions? options,
        CancellationToken cancellationToken)
    {
        // TP-specific save logic
        if (helper == null)
        {
            throw new ArgumentNullException(nameof(helper));
        }

        // Collect TP shard
        var shardData = helper.CollectTPShard();
        var metadata = helper.GetShardingMetadata();

        // Create checkpoint with TP data
        var trainingCheckpoint = new TrainingCheckpoint
        {
            Id = $"{options?.CheckpointPrefix ?? "checkpoint"}_{Guid.NewGuid():N}",
            Epoch = 0,  // Would be populated from training state
            Step = 0,
            LearningRate = 0.001f,
            ModelState = shardData.Data,
            OptimizerState = null,  // Would be populated from optimizer
            WorkerCount = model.WorldSize,
            IsRescalingCheckpoint = false,
            Timestamp = DateTime.UtcNow
        };

        // Add metadata
        trainingCheckpoint.Metadata["strategy"] = metadata.Strategy;
        trainingCheckpoint.Metadata["shard_count"] = metadata.ShardCount.ToString();
        trainingCheckpoint.Metadata["precision"] = metadata.Precision;

        await checkpoint.GetCheckpointManager().SaveCheckpointAsync(trainingCheckpoint);

        return trainingCheckpoint.Id;
    }
}
