namespace MachineLearning.Checkpointing;

using MachineLearning.Distributed.Checkpointing;

/// <summary>
/// Base class for distributed checkpoint operations
/// </summary>
public class DistributedCheckpoint
{
    private readonly IDistributedCoordinator _coordinator;
    private readonly ElasticCheckpointManager _checkpointManager;

    public DistributedCheckpoint(
        IDistributedCoordinator coordinator,
        ElasticCheckpointManager checkpointManager)
    {
        _coordinator = coordinator;
        _checkpointManager = checkpointManager;
    }

    /// <summary>
    /// Gets the distributed coordinator
    /// </summary>
    public IDistributedCoordinator GetCoordinator()
    {
        return _coordinator;
    }

    /// <summary>
    /// Gets the checkpoint manager
    /// </summary>
    public ElasticCheckpointManager GetCheckpointManager()
    {
        return _checkpointManager;
    }

    /// <summary>
    /// Save checkpoint
    /// </summary>
    public async Task<string> SaveAsync(
        IStateful model,
        IStateful optimizer,
        SaveOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        // Implementation will be added later
        await Task.CompletedTask;
        return "checkpoint_id";
    }
}

/// <summary>
/// Options for saving checkpoints
/// </summary>
public class SaveOptions
{
    /// <summary>
    /// Prefix for checkpoint files
    /// </summary>
    public string CheckpointPrefix { get; set; } = string.Empty;
}
