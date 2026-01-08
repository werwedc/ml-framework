namespace MLFramework.Checkpointing.Strategies;

/// <summary>
/// Factory for creating checkpoint strategies
/// </summary>
public static class CheckpointStrategyFactory
{
    /// <summary>
    /// Creates a strategy from a configuration
    /// </summary>
    /// <param name="config">Checkpoint configuration</param>
    /// <returns>Checkpoint strategy instance</returns>
    public static ICheckpointStrategy CreateStrategy(CheckpointConfig config)
    {
        return config.Strategy switch
        {
            CheckpointStrategy.Interval => new IntervalCheckpointStrategy(config.Interval),
            CheckpointStrategy.Selective => new SelectiveCheckpointStrategy(
                config.CheckpointLayers,
                config.ExcludeLayers),
            CheckpointStrategy.SizeBased => new SizeBasedCheckpointStrategy(
                config.MinActivationSizeBytes,
                config.ExcludeLayers),
            CheckpointStrategy.MemoryAware => new MemoryAwareCheckpointStrategy(
                config.MaxMemoryPercentage),
            CheckpointStrategy.Smart => new SmartCheckpointStrategy(
                config.ExcludeLayers),
            _ => throw new ArgumentException($"Unknown strategy: {config.Strategy}")
        };
    }

    /// <summary>
    /// Creates an interval strategy
    /// </summary>
    public static ICheckpointStrategy CreateInterval(int interval = 2)
    {
        return new IntervalCheckpointStrategy(interval);
    }

    /// <summary>
    /// Creates a selective strategy
    /// </summary>
    public static ICheckpointStrategy CreateSelective(
        IEnumerable<string>? checkpointLayers = null,
        IEnumerable<string>? excludeLayers = null)
    {
        return new SelectiveCheckpointStrategy(checkpointLayers, excludeLayers);
    }

    /// <summary>
    /// Creates a size-based strategy
    /// </summary>
    public static ICheckpointStrategy CreateSizeBased(
        long minActivationSizeBytes = 1024 * 1024,
        IEnumerable<string>? excludeLayers = null)
    {
        return new SizeBasedCheckpointStrategy(minActivationSizeBytes, excludeLayers);
    }

    /// <summary>
    /// Creates a memory-aware strategy
    /// </summary>
    public static ICheckpointStrategy CreateMemoryAware(
        float maxMemoryPercentage = 0.8f,
        MemoryTracker? memoryTracker = null)
    {
        return new MemoryAwareCheckpointStrategy(maxMemoryPercentage, memoryTracker);
    }

    /// <summary>
    /// Creates a smart strategy
    /// </summary>
    public static ICheckpointStrategy CreateSmart(
        IEnumerable<string>? excludeLayers = null)
    {
        return new SmartCheckpointStrategy(excludeLayers);
    }
}
