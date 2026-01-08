using System.Collections.Concurrent;
using System.Collections.Immutable;
using MachineLearning.Visualization.Events;

namespace MachineLearning.Visualization.Collection;

/// <summary>
/// Thread-safe buffer for storing events with bounded capacity and batch processing support
/// </summary>
public class EventBuffer : IDisposable
{
    private readonly ConcurrentQueue<Event> _events;
    private readonly int _capacity;
    private readonly object _lock = new();
    private long _droppedEvents;
    private int _peakSize;

    /// <summary>
    /// Gets the current number of events in the buffer
    /// </summary>
    public int Count
    {
        get
        {
            lock (_lock)
            {
                return _events.Count;
            }
        }
    }

    /// <summary>
    /// Gets the maximum number of events the buffer has held
    /// </summary>
    public int PeakSize
    {
        get
        {
            lock (_lock)
            {
                return _peakSize;
            }
        }
    }

    /// <summary>
    /// Gets the total number of events that have been dropped
    /// </summary>
    public long DroppedEvents
    {
        get
        {
            lock (_lock)
            {
                return _droppedEvents;
            }
        }
    }

    /// <summary>
    /// Gets the buffer capacity
    /// </summary>
    public int Capacity => _capacity;

    /// <summary>
    /// Gets whether the buffer is currently full
    /// </summary>
    public bool IsFull
    {
        get
        {
            lock (_lock)
            {
                return Count >= _capacity;
            }
        }
    }

    /// <summary>
    /// Creates a new event buffer with the specified capacity
    /// </summary>
    /// <param name="capacity">Maximum number of events the buffer can hold</param>
    public EventBuffer(int capacity = 1000)
    {
        if (capacity <= 0)
            throw new ArgumentOutOfRangeException(nameof(capacity), "Capacity must be positive");

        _capacity = capacity;
        _events = new ConcurrentQueue<Event>();
    }

    /// <summary>
    /// Adds an event to the buffer, dropping oldest events if full
    /// </summary>
    /// <param name="eventData">Event to add</param>
    /// <returns>True if the event was added, false if dropped</returns>
    public bool Enqueue(Event eventData)
    {
        if (eventData == null)
            throw new ArgumentNullException(nameof(eventData));

        lock (_lock)
        {
            // Drop oldest events if we're at capacity
            while (Count >= _capacity && _events.TryDequeue(out _))
            {
                _droppedEvents++;
            }

            _events.Enqueue(eventData);

            // Update peak size
            if (Count > _peakSize)
            {
                _peakSize = Count;
            }

            return true;
        }
    }

    /// <summary>
    /// Tries to dequeue a single event from the buffer
    /// </summary>
    /// <param name="eventData">Dequeued event, or null if buffer is empty</param>
    /// <returns>True if an event was dequeued, false otherwise</returns>
    public bool TryDequeue(out Event? eventData)
    {
        lock (_lock)
        {
            return _events.TryDequeue(out eventData);
        }
    }

    /// <summary>
    /// Dequeues up to the specified number of events from the buffer
    /// </summary>
    /// <param name="maxCount">Maximum number of events to dequeue</param>
    /// <returns>List of dequeued events (may be empty)</returns>
    public ImmutableArray<Event> DequeueBatch(int maxCount)
    {
        if (maxCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxCount), "Max count must be positive");

        var events = new List<Event>();

        lock (_lock)
        {
            for (int i = 0; i < maxCount && _events.TryDequeue(out var eventData); i++)
            {
                if (eventData != null)
                {
                    events.Add(eventData);
                }
            }
        }

        return events.ToImmutableArray();
    }

    /// <summary>
    /// Tries to dequeue a batch of events, blocking until events are available or timeout occurs
    /// </summary>
    /// <param name="maxCount">Maximum number of events to dequeue</param>
    /// <param name="timeout">Maximum time to wait for events</param>
    /// <returns>List of dequeued events (may be empty if timeout)</returns>
    public async Task<ImmutableArray<Event>> DequeueBatchAsync(int maxCount, TimeSpan timeout)
    {
        var events = ImmutableArray<Event>.Empty;
        var startTime = DateTime.UtcNow;
        var delay = TimeSpan.FromMilliseconds(10);

        while (events.IsEmpty && (DateTime.UtcNow - startTime) < timeout)
        {
            events = DequeueBatch(maxCount);

            if (events.IsEmpty)
            {
                await Task.Delay(delay);
            }
        }

        return events;
    }

    /// <summary>
    /// Clears all events from the buffer
    /// </summary>
    /// <returns>Number of events cleared</returns>
    public int Clear()
    {
        int count = 0;

        lock (_lock)
        {
            while (_events.TryDequeue(out _))
            {
                count++;
            }
        }

        return count;
    }

    /// <summary>
    /// Resets buffer statistics (dropped events, peak size)
    /// </summary>
    public void ResetStatistics()
    {
        lock (_lock)
        {
            _droppedEvents = 0;
            _peakSize = Count;
        }
    }

    /// <summary>
    /// Disposes of the event buffer and clears all events
    /// </summary>
    public void Dispose()
    {
        Clear();
        GC.SuppressFinalize(this);
    }
}
