using MachineLearning.Visualization.Events;
using MachineLearning.Visualization.Collection.Configuration;

namespace MachineLearning.Visualization.Collection;

/// <summary>
/// Interface for collecting and managing events
/// </summary>
public interface IEventCollector : IDisposable
{
    /// <summary>
    /// Gets or sets the collector configuration
    /// </summary>
    EventCollectorConfig Config { get; set; }

    /// <summary>
    /// Collects an event synchronously
    /// </summary>
    /// <typeparam name="T">Event type</typeparam>
    /// <param name="eventData">Event to collect</param>
    void Collect<T>(T eventData) where T : Event;

    /// <summary>
    /// Collects an event asynchronously
    /// </summary>
    /// <typeparam name="T">Event type</typeparam>
    /// <param name="eventData">Event to collect</param>
    Task CollectAsync<T>(T eventData) where T : Event;

    /// <summary>
    /// Flushes all pending events synchronously
    /// </summary>
    void Flush();

    /// <summary>
    /// Flushes all pending events asynchronously
    /// </summary>
    Task FlushAsync();

    /// <summary>
    /// Starts the event collector
    /// </summary>
    void Start();

    /// <summary>
    /// Stops the event collector
    /// </summary>
    void Stop();

    /// <summary>
    /// Gets whether the collector is currently running
    /// </summary>
    bool IsRunning { get; }

    /// <summary>
    /// Gets the number of events waiting to be processed
    /// </summary>
    int PendingEventCount { get; }

    /// <summary>
    /// Gets statistics about the event collector
    /// </summary>
    EventCollectorStatistics GetStatistics();
}

/// <summary>
/// Statistics for event collection
/// </summary>
public class EventCollectorStatistics
{
    /// <summary>
    /// Number of events collected
    /// </summary>
    public long EventsCollected { get; set; }

    /// <summary>
    /// Number of events processed
    /// </summary>
    public long EventsProcessed { get; set; }

    /// <summary>
    /// Number of events dropped
    /// </summary>
    public long EventsDropped { get; set; }

    /// <summary>
    /// Number of pending events
    /// </summary>
    public int PendingEvents { get; set; }

    /// <summary>
    /// Whether backpressure is currently active
    /// </summary>
    public bool IsBackpressureActive { get; set; }

    /// <summary>
    /// Peak buffer size
    /// </summary>
    public int PeakBufferSize { get; set; }

    /// <summary>
    /// Current buffer size
    /// </summary>
    public int CurrentBufferSize { get; set; }

    /// <summary>
    /// Total number of flushes performed
    /// </summary>
    public long TotalFlushes { get; set; }

    /// <summary>
    /// Average events per flush
    /// </summary>
    public double AverageEventsPerFlush { get; set; }

    /// <summary>
    /// Last flush time
    /// </summary>
    public DateTime LastFlushTime { get; set; }

    /// <summary>
    /// Uptime in seconds
    /// </summary>
    public double UptimeSeconds { get; set; }
}
