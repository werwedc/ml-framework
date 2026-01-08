namespace MLFramework.Checkpointing.Strategies;

/// <summary>
/// Dynamically adjusts checkpointing based on available memory
/// </summary>
public class MemoryAwareCheckpointStrategy : ICheckpointStrategy
{
    private readonly float _maxMemoryPercentage;
    private readonly MemoryTracker _memoryTracker;
    private readonly long _totalSystemMemory;
    private readonly int _initialInterval;
    private int _currentInterval;
    private long _lastAdjustmentTime;
    private int _consecutiveHighMemoryPressure;

    /// <summary>
    /// Initializes a new instance of MemoryAwareCheckpointStrategy
    /// </summary>
    /// <param name="maxMemoryPercentage">Maximum memory percentage to use (0.0 to 1.0)</param>
    /// <param name="memoryTracker">Memory tracker instance</param>
    /// <param name="totalSystemMemory">Total system memory available</param>
    public MemoryAwareCheckpointStrategy(
        float maxMemoryPercentage = 0.8f,
        MemoryTracker? memoryTracker = null,
        long? totalSystemMemory = null)
    {
        if (maxMemoryPercentage <= 0 || maxMemoryPercentage > 1.0f)
            throw new ArgumentException("MaxMemoryPercentage must be between 0 and 1");

        _maxMemoryPercentage = maxMemoryPercentage;
        _memoryTracker = memoryTracker ?? new MemoryTracker();
        _totalSystemMemory = totalSystemMemory ?? EstimateTotalSystemMemory();
        _initialInterval = 2;
        _currentInterval = _initialInterval;
        _lastAdjustmentTime = DateTime.UtcNow.Ticks;
        _consecutiveHighMemoryPressure = 0;
    }

    /// <summary>
    /// Gets the strategy name
    /// </summary>
    public string Name => $"MemoryAware({_maxMemoryPercentage:P0})";

    /// <summary>
    /// Determines whether to checkpoint a layer
    /// </summary>
    public bool ShouldCheckpoint(string layerId, Tensor activation, int layerIndex)
    {
        // Check current memory pressure
        var memoryPressure = CalculateMemoryPressure();

        // Adjust interval based on memory pressure
        AdjustInterval(memoryPressure);

        // Checkpoint based on current interval
        var shouldCheckpoint = (layerIndex % _currentInterval) == 0;

        return shouldCheckpoint;
    }

    private float CalculateMemoryPressure()
    {
        var stats = _memoryTracker.GetStats();
        return (float)stats.CurrentMemoryUsed / _totalSystemMemory;
    }

    private void AdjustInterval(float memoryPressure)
    {
        var now = DateTime.UtcNow.Ticks;
        var timeSinceLastAdjustment = TimeSpan.FromTicks(now - _lastAdjustmentTime).TotalSeconds;

        // Only adjust every 10 seconds to avoid oscillation
        if (timeSinceLastAdjustment < 10)
            return;

        var threshold = _maxMemoryPercentage;

        if (memoryPressure > threshold)
        {
            // High memory pressure - increase checkpoint frequency (decrease interval)
            _currentInterval = Math.Max(1, _currentInterval - 1);
            _consecutiveHighMemoryPressure++;
        }
        else if (memoryPressure < threshold * 0.8f)
        {
            // Low memory pressure - decrease checkpoint frequency (increase interval)
            if (_consecutiveHighMemoryPressure == 0 || timeSinceLastAdjustment > 30)
            {
                _currentInterval = Math.Min(10, _currentInterval + 1);
            }
        }
        else
        {
            _consecutiveHighMemoryPressure = 0;
        }

        _lastAdjustmentTime = now;
    }

    private long EstimateTotalSystemMemory()
    {
        // This would use system APIs to get available memory
        // For now, return a reasonable default (16GB)
        return 16L * 1024 * 1024 * 1024;
    }

    /// <summary>
    /// Resets the strategy state
    /// </summary>
    public void Reset()
    {
        _currentInterval = _initialInterval;
        _lastAdjustmentTime = DateTime.UtcNow.Ticks;
        _consecutiveHighMemoryPressure = 0;
    }
}
