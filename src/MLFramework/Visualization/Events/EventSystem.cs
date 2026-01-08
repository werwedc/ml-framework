using System.Collections.Concurrent;

namespace MachineLearning.Visualization.Events;

/// <summary>
/// Core event system implementing publish-subscribe pattern for visualization events
/// </summary>
public class EventSystem : IEventPublisher, IEventSubscriber, IDisposable
{
    // Thread-safe storage for subscribers
    private readonly ConcurrentDictionary<Type, ConcurrentDictionary<Delegate, bool>> _typedSubscribers;
    private readonly ConcurrentDictionary<Delegate, bool> _allEventSubscribers;
    private readonly bool _enableAsync;
    private readonly SemaphoreSlim _lock;
    private bool _isDisposed;

    /// <summary>
    /// Gets whether the event system is currently running
    /// </summary>
    public bool IsRunning => !_isDisposed;

    /// <summary>
    /// Creates a new event system
    /// </summary>
    /// <param name="enableAsync">Whether to enable async publishing by default</param>
    public EventSystem(bool enableAsync = true)
    {
        _typedSubscribers = new ConcurrentDictionary<Type, ConcurrentDictionary<Delegate, bool>>();
        _allEventSubscribers = new ConcurrentDictionary<Delegate, bool>();
        _enableAsync = enableAsync;
        _lock = new SemaphoreSlim(1, 1);
    }

    /// <summary>
    /// Publishes an event synchronously to all subscribers
    /// </summary>
    /// <typeparam name="T">Event type</typeparam>
    /// <param name="eventData">Event to publish</param>
    public void Publish<T>(T eventData) where T : Event
    {
        if (eventData == null)
            throw new ArgumentNullException(nameof(eventData));

        if (_isDisposed)
            return;

        var eventType = typeof(T);

        // Notify typed subscribers
        if (_typedSubscribers.TryGetValue(eventType, out var subscribers))
        {
            foreach (var subscriber in subscribers.Keys)
            {
                try
                {
                    var handler = (Action<T>)subscriber;
                    handler(eventData);
                }
                catch (Exception ex)
                {
                    // Log error but continue processing other subscribers
                    System.Console.Error.WriteLine($"Error in event subscriber: {ex.Message}");
                }
            }
        }

        // Notify subscribers that want all events
        foreach (var subscriber in _allEventSubscribers.Keys)
        {
            try
            {
                var handler = (Action<Event>)subscriber;
                handler(eventData);
            }
            catch (Exception ex)
            {
                // Log error but continue processing other subscribers
                System.Console.Error.WriteLine($"Error in event subscriber: {ex.Message}");
            }
        }
    }

    /// <summary>
    /// Publishes an event asynchronously to all subscribers
    /// </summary>
    /// <typeparam name="T">Event type</typeparam>
    /// <param name="eventData">Event to publish</param>
    /// <returns>Task that completes when all subscribers have been notified</returns>
    public async Task PublishAsync<T>(T eventData) where T : Event
    {
        if (eventData == null)
            throw new ArgumentNullException(nameof(eventData));

        if (_isDisposed)
            return;

        var eventType = typeof(T);
        var tasks = new List<Task>();

        // Notify typed subscribers
        if (_typedSubscribers.TryGetValue(eventType, out var typedSubscribers))
        {
            foreach (var subscriber in typedSubscribers.Keys)
            {
                tasks.Add(Task.Run(() =>
                {
                    try
                    {
                        var handler = (Action<T>)subscriber;
                        handler(eventData);
                    }
                    catch (Exception ex)
                    {
                        System.Console.Error.WriteLine($"Error in event subscriber: {ex.Message}");
                    }
                }));
            }
        }

        // Notify subscribers that want all events
        foreach (var subscriber in _allEventSubscribers.Keys)
        {
            tasks.Add(Task.Run(() =>
            {
                try
                {
                    var handler = (Action<Event>)subscriber;
                    handler(eventData);
                }
                catch (Exception ex)
                {
                    System.Console.Error.WriteLine($"Error in event subscriber: {ex.Message}");
                }
            }));
        }

        // Wait for all subscribers to complete
        await Task.WhenAll(tasks);
    }

    /// <summary>
    /// Subscribes to events of a specific type
    /// </summary>
    /// <typeparam name="T">Event type to subscribe to</typeparam>
    /// <param name="handler">Handler for events of this type</param>
    public void Subscribe<T>(Action<T> handler) where T : Event
    {
        if (handler == null)
            throw new ArgumentNullException(nameof(handler));

        if (_isDisposed)
            throw new ObjectDisposedException(nameof(EventSystem));

        var eventType = typeof(T);
        var subscribers = _typedSubscribers.GetOrAdd(eventType, _ => new ConcurrentDictionary<Delegate, bool>());
        subscribers[handler] = true;
    }

    /// <summary>
    /// Unsubscribes from events of a specific type
    /// </summary>
    /// <typeparam name="T">Event type to unsubscribe from</typeparam>
    /// <param name="handler">Handler to remove</param>
    public void Unsubscribe<T>(Action<T> handler) where T : Event
    {
        if (handler == null)
            throw new ArgumentNullException(nameof(handler));

        if (_isDisposed)
            return;

        var eventType = typeof(T);
        if (_typedSubscribers.TryGetValue(eventType, out var subscribers))
        {
            subscribers.TryRemove(handler, out _);
        }
    }

    /// <summary>
    /// Subscribes to all events regardless of type
    /// </summary>
    /// <param name="handler">Handler for all events</param>
    public void SubscribeAll(Action<Event> handler)
    {
        if (handler == null)
            throw new ArgumentNullException(nameof(handler));

        if (_isDisposed)
            throw new ObjectDisposedException(nameof(EventSystem));

        _allEventSubscribers[handler] = true;
    }

    /// <summary>
    /// Shuts down the event system and releases resources
    /// </summary>
    public void Shutdown()
    {
        Dispose();
    }

    /// <summary>
    /// Disposes of the event system
    /// </summary>
    public void Dispose()
    {
        if (_isDisposed)
            return;

        _isDisposed = true;

        // Clear all subscribers
        _typedSubscribers.Clear();
        _allEventSubscribers.Clear();

        _lock.Dispose();
        GC.SuppressFinalize(this);
    }
}
