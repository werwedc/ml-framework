using MachineLearning.Visualization.Events;

namespace MachineLearning.Visualization.Storage;

/// <summary>
/// Interface for storing visualization events
/// </summary>
public interface IStorageBackend
{
    /// <summary>
    /// Stores a single event asynchronously
    /// </summary>
    /// <param name="eventData">Event to store</param>
    /// <param name="cancellationToken">Cancellation token</param>
    Task StoreAsync(Event eventData, CancellationToken cancellationToken = default);

    /// <summary>
    /// Stores multiple events in a batch asynchronously
    /// </summary>
    /// <param name="events">Events to store</param>
    /// <param name="cancellationToken">Cancellation token</param>
    Task StoreBatchAsync(System.Collections.Immutable.ImmutableArray<Event> events, CancellationToken cancellationToken = default);

    /// <summary>
    /// Disposes of the storage backend and releases resources
    /// </summary>
    void Dispose();
}
