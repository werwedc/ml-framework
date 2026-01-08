using MachineLearning.Visualization.Events;

namespace MachineLearning.Visualization.Storage;

/// <summary>
/// Interface for storing visualization events with lifecycle management
/// </summary>
public interface IStorageBackend : IDisposable
{
    /// <summary>
    /// Initializes the storage backend with a connection string
    /// </summary>
    /// <param name="connectionString">Connection string for the backend</param>
    void Initialize(string connectionString);

    /// <summary>
    /// Shuts down the storage backend and releases resources
    /// </summary>
    void Shutdown();

    /// <summary>
    /// Gets whether the backend has been initialized
    /// </summary>
    bool IsInitialized { get; }

    /// <summary>
    /// Stores a single event synchronously
    /// </summary>
    /// <param name="eventData">Event to store</param>
    void StoreEvent(Event eventData);

    /// <summary>
    /// Stores a single event asynchronously
    /// </summary>
    /// <param name="eventData">Event to store</param>
    Task StoreEventAsync(Event eventData);

    /// <summary>
    /// Stores multiple events synchronously
    /// </summary>
    /// <param name="events">Events to store</param>
    void StoreEvents(IEnumerable<Event> events);

    /// <summary>
    /// Stores multiple events asynchronously
    /// </summary>
    /// <param name="events">Events to store</param>
    Task StoreEventsAsync(IEnumerable<Event> events);

    /// <summary>
    /// Retrieves events within a step range synchronously
    /// </summary>
    /// <param name="startStep">Starting step (inclusive)</param>
    /// <param name="endStep">Ending step (inclusive)</param>
    IEnumerable<Event> GetEvents(long startStep, long endStep);

    /// <summary>
    /// Retrieves events within a step range asynchronously
    /// </summary>
    /// <param name="startStep">Starting step (inclusive)</param>
    /// <param name="endStep">Ending step (inclusive)</param>
    Task<IEnumerable<Event>> GetEventsAsync(long startStep, long endStep);

    /// <summary>
    /// Flushes any buffered data to storage synchronously
    /// </summary>
    void Flush();

    /// <summary>
    /// Flushes any buffered data to storage asynchronously
    /// </summary>
    Task FlushAsync();

    /// <summary>
    /// Gets the total number of events stored
    /// </summary>
    long EventCount { get; }

    /// <summary>
    /// Clears all stored events
    /// </summary>
    void Clear();
}
