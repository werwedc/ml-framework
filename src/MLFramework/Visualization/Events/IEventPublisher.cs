namespace MachineLearning.Visualization.Events;

/// <summary>
/// Interface for publishing events to the event system
/// </summary>
public interface IEventPublisher
{
    /// <summary>
    /// Publishes an event synchronously to all subscribers
    /// </summary>
    /// <typeparam name="T">Event type</typeparam>
    /// <param name="eventData">Event to publish</param>
    void Publish<T>(T eventData) where T : Event;

    /// <summary>
    /// Publishes an event asynchronously to all subscribers
    /// </summary>
    /// <typeparam name="T">Event type</typeparam>
    /// <param name="eventData">Event to publish</param>
    /// <returns>Task that completes when all subscribers have been notified</returns>
    Task PublishAsync<T>(T eventData) where T : Event;
}
