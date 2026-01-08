namespace MachineLearning.Visualization.Events;

/// <summary>
/// Interface for subscribing to events from the event system
/// </summary>
public interface IEventSubscriber
{
    /// <summary>
    /// Subscribes to events of a specific type
    /// </summary>
    /// <typeparam name="T">Event type to subscribe to</typeparam>
    /// <param name="handler">Handler for events of this type</param>
    void Subscribe<T>(Action<T> handler) where T : Event;

    /// <summary>
    /// Unsubscribes from events of a specific type
    /// </summary>
    /// <typeparam name="T">Event type to unsubscribe from</typeparam>
    /// <param name="handler">Handler to remove</param>
    void Unsubscribe<T>(Action<T> handler) where T : Event;

    /// <summary>
    /// Subscribes to all events regardless of type
    /// </summary>
    /// <param name="handler">Handler for all events</param>
    void SubscribeAll(Action<Event> handler);
}
