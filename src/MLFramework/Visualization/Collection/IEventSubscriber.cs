using MachineLearning.Visualization.Events;

namespace MachineLearning.Visualization.Collection;

/// <summary>
/// Interface for event subscribers that receive events from the collector
/// </summary>
public interface IEventSubscriber
{
    /// <summary>
    /// Processes a single event asynchronously
    /// </summary>
    /// <param name="eventData">Event to process</param>
    /// <param name="cancellationToken">Cancellation token</param>
    Task ProcessAsync(Event eventData, CancellationToken cancellationToken = default);

    /// <summary>
    /// Processes a batch of events asynchronously
    /// </summary>
    /// <param name="events">Events to process</param>
    /// <param name="cancellationToken">Cancellation token</param>
    Task ProcessBatchAsync(System.Collections.Immutable.ImmutableArray<Event> events, CancellationToken cancellationToken = default);
}
