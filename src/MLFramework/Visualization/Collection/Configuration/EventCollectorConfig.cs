namespace MachineLearning.Visualization.Collection.Configuration;

/// <summary>
/// Flush strategy for event collection
/// </summary>
public enum FlushStrategy
{
    /// <summary>
    /// Flush on timer interval
    /// </summary>
    TimeBased,

    /// <summary>
    /// Flush when buffer reaches size
    /// </summary>
    SizeBased,

    /// <summary>
    /// Flush only when explicitly requested
    /// </summary>
    Manual,

    /// <summary>
    /// Combination of time and size
    /// </summary>
    Hybrid
}

/// <summary>
/// Configuration for event collector
/// </summary>
public class EventCollectorConfig
{
    /// <summary>
    /// Maximum number of events in the buffer
    /// </summary>
    public int BufferCapacity { get; set; } = 1000;

    /// <summary>
    /// Time interval between automatic flushes
    /// </summary>
    public TimeSpan FlushInterval { get; set; } = TimeSpan.FromSeconds(1);

    /// <summary>
    /// Number of events to process in a single batch
    /// </summary>
    public int BatchSize { get; set; } = 100;

    /// <summary>
    /// Strategy for flushing events
    /// </summary>
    public FlushStrategy Strategy { get; set; } = FlushStrategy.Hybrid;

    /// <summary>
    /// Enable backpressure handling when system is overloaded
    /// </summary>
    public bool EnableBackpressure { get; set; } = true;

    /// <summary>
    /// Maximum length of the event queue (for backpressure)
    /// </summary>
    public int MaxQueueLength { get; set; } = 10000;

    /// <summary>
    /// Action to take when queue is full and backpressure is enabled
    /// </summary>
    public BackpressureAction BackpressureAction { get; set; } = BackpressureAction.DropOldest;

    /// <summary>
    /// Maximum time to wait for queue space before dropping events
    /// </summary>
    public TimeSpan BackpressureTimeout { get; set; } = TimeSpan.FromMilliseconds(100);

    /// <summary>
    /// Enable object pooling for event instances
    /// </summary>
    public bool EnableObjectPooling { get; set; } = false;

    /// <summary>
    /// Pool size for object pooling
    /// </summary>
    public int PoolSize { get; set; } = 100;

    /// <summary>
    /// Validates the configuration and throws if invalid
    /// </summary>
    public void Validate()
    {
        if (BufferCapacity <= 0)
            throw new InvalidOperationException("BufferCapacity must be positive");

        if (BatchSize <= 0)
            throw new InvalidOperationException("BatchSize must be positive");

        if (BatchSize > BufferCapacity)
            throw new InvalidOperationException("BatchSize cannot exceed BufferCapacity");

        if (MaxQueueLength <= 0)
            throw new InvalidOperationException("MaxQueueLength must be positive");

        if (FlushInterval <= TimeSpan.Zero)
            throw new InvalidOperationException("FlushInterval must be positive");

        if (EnableObjectPooling && PoolSize <= 0)
            throw new InvalidOperationException("PoolSize must be positive when EnableObjectPooling is true");

        if (BackpressureTimeout < TimeSpan.Zero)
            throw new InvalidOperationException("BackpressureTimeout cannot be negative");
    }

    /// <summary>
    /// Creates a copy of this configuration
    /// </summary>
    public EventCollectorConfig Clone()
    {
        return new EventCollectorConfig
        {
            BufferCapacity = BufferCapacity,
            FlushInterval = FlushInterval,
            BatchSize = BatchSize,
            Strategy = Strategy,
            EnableBackpressure = EnableBackpressure,
            MaxQueueLength = MaxQueueLength,
            BackpressureAction = BackpressureAction,
            BackpressureTimeout = BackpressureTimeout,
            EnableObjectPooling = EnableObjectPooling,
            PoolSize = PoolSize
        };
    }
}

/// <summary>
/// Action to take when backpressure is triggered
/// </summary>
public enum BackpressureAction
{
    /// <summary>
    /// Drop the oldest events
    /// </summary>
    DropOldest,

    /// <summary>
    /// Drop the newest events
    /// </summary>
    DropNewest,

    /// <summary>
    /// Block until space is available
    /// </summary>
    Block,

    /// <summary>
    /// Throw an exception
    /// </summary>
    Throw
}
