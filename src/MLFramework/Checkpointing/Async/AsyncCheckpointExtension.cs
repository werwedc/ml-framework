namespace MachineLearning.Checkpointing.Async;

using Microsoft.Extensions.Logging;

/// <summary>
/// Extension methods for creating async checkpoint managers
/// </summary>
public static class AsyncCheckpointExtension
{
    /// <summary>
    /// Create an async checkpoint manager from a DistributedCheckpoint instance
    /// </summary>
    /// <param name="checkpoint">The distributed checkpoint instance.</param>
    /// <param name="maxQueueSize">The maximum queue size.</param>
    /// <param name="logger">Optional logger instance.</param>
    /// <returns>A new async checkpoint manager.</returns>
    public static AsyncCheckpointManager CreateAsyncManager(
        this DistributedCheckpoint checkpoint,
        int maxQueueSize = 10,
        ILogger<AsyncCheckpointManager>? logger = null)
    {
        var coordinator = checkpoint.GetCoordinator();
        return new AsyncCheckpointManager(coordinator, checkpoint, maxQueueSize, logger);
    }

    /// <summary>
    /// Queue a checkpoint save operation using the async manager
    /// </summary>
    /// <param name="checkpoint">The distributed checkpoint instance.</param>
    /// <param name="model">The model state to save.</param>
    /// <param name="optimizer">The optimizer state to save.</param>
    /// <param name="options">The save options.</param>
    /// <param name="maxQueueSize">The maximum queue size for the manager.</param>
    /// <param name="logger">Optional logger instance.</param>
    /// <returns>A tuple containing the async manager and checkpoint ID.</returns>
    public static (AsyncCheckpointManager manager, string checkpointId) QueueSaveAsync(
        this DistributedCheckpoint checkpoint,
        IStateful model,
        IStateful optimizer,
        SaveOptions? options = null,
        int maxQueueSize = 10,
        ILogger<AsyncCheckpointManager>? logger = null)
    {
        var manager = checkpoint.CreateAsyncManager(maxQueueSize, logger);
        var checkpointId = manager.QueueSaveAsync(model, optimizer, options);
        return (manager, checkpointId);
    }
}
