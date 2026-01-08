using System.Collections.Immutable;
using MachineLearning.Visualization.Events;
using MachineLearning.Visualization.Storage;
using MachineLearning.Visualization.Collection.Configuration;

namespace MachineLearning.Visualization.Collection;

/// <summary>
/// Asynchronous event collector with support for storage backends and event subscribers
/// </summary>
public class AsyncEventCollector : EventCollector
{
    private readonly List<IEventSubscriber> _subscribers;
    private readonly Dictionary<Type, List<IEventSubscriber>> _typedSubscribers;

    /// <summary>
    /// Creates a new async event collector with a storage backend
    /// </summary>
    /// <param name="storageBackend">Storage backend for persisting events</param>
    /// <param name="config">Collector configuration</param>
    public AsyncEventCollector(IStorageBackend storageBackend, EventCollectorConfig? config = null)
        : base(storageBackend, config)
    {
        _subscribers = new List<IEventSubscriber>();
        _typedSubscribers = new Dictionary<Type, List<IEventSubscriber>>();
    }

    /// <summary>
    /// Creates a new async event collector with event subscribers
    /// </summary>
    /// <param name="subscribers">Event subscribers to notify</param>
    /// <param name="config">Collector configuration</param>
    public AsyncEventCollector(IEnumerable<IEventSubscriber> subscribers, EventCollectorConfig? config = null)
        : base(null, config)
    {
        if (subscribers == null)
            throw new ArgumentNullException(nameof(subscribers));

        _subscribers = new List<IEventSubscriber>(subscribers);
        _typedSubscribers = new Dictionary<Type, List<IEventSubscriber>>();

        // Build typed subscriber lookup
        foreach (var subscriber in _subscribers)
        {
            // For simplicity, we'll assume subscribers handle all events
            // In a more sophisticated implementation, we could use reflection to determine
            // which event types each subscriber handles
            foreach (var eventType in GetSupportedEventTypes(subscriber))
            {
                if (!_typedSubscribers.ContainsKey(eventType))
                {
                    _typedSubscribers[eventType] = new List<IEventSubscriber>();
                }
                _typedSubscribers[eventType].Add(subscriber);
            }
        }
    }

    /// <summary>
    /// Adds an event subscriber to the collector
    /// </summary>
    /// <param name="subscriber">Subscriber to add</param>
    public void AddSubscriber(IEventSubscriber subscriber)
    {
        if (subscriber == null)
            throw new ArgumentNullException(nameof(subscriber));

        lock (_subscribers)
        {
            _subscribers.Add(subscriber);

            foreach (var eventType in GetSupportedEventTypes(subscriber))
            {
                if (!_typedSubscribers.ContainsKey(eventType))
                {
                    _typedSubscribers[eventType] = new List<IEventSubscriber>();
                }
                _typedSubscribers[eventType].Add(subscriber);
            }
        }
    }

    /// <summary>
    /// Removes an event subscriber from the collector
    /// </summary>
    /// <param name="subscriber">Subscriber to remove</param>
    public bool RemoveSubscriber(IEventSubscriber subscriber)
    {
        if (subscriber == null)
            throw new ArgumentNullException(nameof(subscriber));

        lock (_subscribers)
        {
            if (!_subscribers.Remove(subscriber))
                return false;

            foreach (var key in _typedSubscribers.Keys.ToList())
            {
                _typedSubscribers[key].Remove(subscriber);
                if (_typedSubscribers[key].Count == 0)
                {
                    _typedSubscribers.Remove(key);
                }
            }

            return true;
        }
    }

    /// <summary>
    /// Gets all registered subscribers
    /// </summary>
    public IReadOnlyList<IEventSubscriber> GetSubscribers()
    {
        lock (_subscribers)
        {
            return _subscribers.ToList().AsReadOnly();
        }
    }

    /// <summary>
    /// Processes events from the buffer by sending them to storage and/or subscribers
    /// </summary>
    protected override async Task ProcessEventsAsync()
    {
        var events = _buffer.DequeueBatch(Config.BatchSize);

        if (events.IsDefaultOrEmpty)
            return;

        // Update processed count
        Interlocked.Add(ref _eventsProcessed, events.Length);

        // Send to storage backend if configured
        if (_storageBackend != null)
        {
            try
            {
                await _storageBackend.StoreEventsAsync(events);
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Error storing events: {ex.Message}");
                throw;
            }
        }

        // Send to subscribers if any are registered
        if (_subscribers.Count > 0)
        {
            await SendToSubscribersAsync(events);
        }
    }

    /// <summary>
    /// Sends events to appropriate subscribers based on event type
    /// </summary>
    private async Task SendToSubscribersAsync(ImmutableArray<Event> events)
    {
        // Group events by type for efficient batch processing
        var groupedEvents = events
            .GroupBy(e => e.GetType())
            .ToDictionary(g => g.Key, g => g.ToImmutableArray());

        foreach (var kvp in groupedEvents)
        {
            Type eventType = kvp.Key;
            var eventList = kvp.Value;

            // Get typed subscribers
            List<IEventSubscriber> typedSubscribersCopy;
            lock (_typedSubscribers)
            {
                if (_typedSubscribers.TryGetValue(eventType, out var subscribers))
                {
                    typedSubscribersCopy = subscribers.ToList();
                }
                else
                {
                    typedSubscribersCopy = new List<IEventSubscriber>();
                }
            }

            // Notify typed subscribers
            foreach (var subscriber in typedSubscribersCopy)
            {
                try
                {
                    await subscriber.ProcessBatchAsync(eventList, _cancellationTokenSource.Token);
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"Error processing events in subscriber: {ex.Message}");
                }
            }

            // Get generic subscribers
            List<IEventSubscriber> genericSubscribersCopy;
            lock (_subscribers)
            {
                genericSubscribersCopy = _subscribers
                    .Where(s => !_typedSubscribers.ContainsKey(eventType) ||
                                !_typedSubscribers[eventType].Contains(s))
                    .ToList();
            }

            // Notify generic subscribers
            foreach (var subscriber in genericSubscribersCopy)
            {
                try
                {
                    await subscriber.ProcessBatchAsync(eventList, _cancellationTokenSource.Token);
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"Error processing events in subscriber: {ex.Message}");
                }
            }
        }
    }

    /// <summary>
    /// Gets the list of event types supported by a subscriber
    /// This is a simple implementation - in production you might use reflection or attributes
    /// </summary>
    private List<Type> GetSupportedEventTypes(IEventSubscriber subscriber)
    {
        // For simplicity, we'll assume subscribers handle the Event base class
        // This can be extended to support more sophisticated type detection
        var types = new List<Type> { typeof(Event) };

        // You could add logic here to inspect the subscriber's implementation
        // and determine which specific event types it handles

        return types;
    }
}
