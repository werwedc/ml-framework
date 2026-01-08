namespace MachineLearning.Checkpointing;

/// <summary>
/// Factory for creating DistributedCheckpoint instances
/// </summary>
public static class DistributedCheckpointFactory
{
    /// <summary>
    /// Create a new DistributedCheckpoint with default options
    /// </summary>
    public static DistributedCheckpoint Create(
        IDistributedCoordinator coordinator,
        CheckpointOptions? options = null)
    {
        if (coordinator == null)
            throw new ArgumentNullException(nameof(coordinator));

        options ??= new CheckpointOptions();

        // Create storage
        var storage = StorageFactory.Create(options.Storage);

        // Create fault handler with RetryPolicy
        var faultHandler = new FaultToleranceHandler(
            storage,
            options.RetryPolicy);

        // Create validator
        var validator = new CheckpointValidator(
            storage,
            options.IntegrityCheckers,
            options.CompatibilityCheckers);

        return new DistributedCheckpoint(
            coordinator,
            storage,
            faultHandler,
            validator);
    }
}
