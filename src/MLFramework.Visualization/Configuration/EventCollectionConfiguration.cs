namespace MLFramework.Visualization.Configuration;

/// <summary>
/// Configuration for event collection
/// </summary>
public class EventCollectionConfiguration
{
    /// <summary>
    /// Enable asynchronous event collection
    /// </summary>
    public bool EnableAsync { get; set; } = true;

    /// <summary>
    /// Buffer capacity for events
    /// </summary>
    public int BufferCapacity { get; set; } = 1000;

    /// <summary>
    /// Batch size for writing events
    /// </summary>
    public int BatchSize { get; set; } = 100;

    /// <summary>
    /// Enable backpressure handling
    /// </summary>
    public bool EnableBackpressure { get; set; } = true;

    /// <summary>
    /// Maximum queue length for events
    /// </summary>
    public int MaxQueueLength { get; set; } = 10000;
}
