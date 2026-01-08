namespace MLFramework.Checkpointing.Strategies;

/// <summary>
/// Checkpoints layers at fixed intervals (every N layers)
/// </summary>
public class IntervalCheckpointStrategy : ICheckpointStrategy
{
    private readonly int _interval;
    private int _checkpointedCount;

    /// <summary>
    /// Initializes a new instance of IntervalCheckpointStrategy
    /// </summary>
    /// <param name="interval">Number of layers between checkpoints</param>
    public IntervalCheckpointStrategy(int interval = 2)
    {
        if (interval <= 0)
            throw new ArgumentException("Interval must be greater than 0", nameof(interval));

        _interval = interval;
        _checkpointedCount = 0;
    }

    /// <summary>
    /// Gets the interval value
    /// </summary>
    public int Interval => _interval;

    /// <summary>
    /// Gets the strategy name
    /// </summary>
    public string Name => $"Interval({_interval})";

    /// <summary>
    /// Determines whether to checkpoint a layer
    /// </summary>
    public bool ShouldCheckpoint(string layerId, Tensor activation, int layerIndex)
    {
        // Checkpoint every N layers
        var shouldCheckpoint = (layerIndex % _interval) == 0;

        if (shouldCheckpoint)
        {
            _checkpointedCount++;
        }

        return shouldCheckpoint;
    }

    /// <summary>
    /// Resets the strategy state
    /// </summary>
    public void Reset()
    {
        _checkpointedCount = 0;
    }
}
