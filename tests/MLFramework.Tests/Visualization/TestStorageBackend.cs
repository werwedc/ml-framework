using System.Collections.Immutable;
using MachineLearning.Visualization.Events;
using MachineLearning.Visualization.Storage;

namespace MachineLearning.Visualization.Tests;

/// <summary>
/// Simple in-memory storage backend for testing
/// </summary>
public class TestStorageBackend : IStorageBackend
{
    private readonly List<Event> _storedEvents = new();
    private readonly List<ImmutableArray<Event>> _storedBatches = new();

    /// <summary>
    /// Gets all events that were stored
    /// </summary>
    public IReadOnlyList<Event> StoredEvents => _storedEvents.AsReadOnly();

    /// <summary>
    /// Gets all batches that were stored
    /// </summary>
    public IReadOnlyList<ImmutableArray<Event>> StoredBatches => _storedBatches.AsReadOnly();

    /// <summary>
    /// Gets the total number of events stored
    /// </summary>
    public int TotalEventsStored => _storedEvents.Count;

    /// <summary>
    /// Gets the total number of batches stored
    /// </summary>
    public int TotalBatchesStored => _storedBatches.Count;

    /// <summary>
    /// Stores a single event
    /// </summary>
    public Task StoreAsync(Event eventData, CancellationToken cancellationToken = default)
    {
        _storedEvents.Add(eventData);
        return Task.CompletedTask;
    }

    /// <summary>
    /// Stores a batch of events
    /// </summary>
    public Task StoreBatchAsync(ImmutableArray<Event> events, CancellationToken cancellationToken = default)
    {
        _storedBatches.Add(events);
        foreach (var evt in events)
        {
            _storedEvents.Add(evt);
        }
        return Task.CompletedTask;
    }

    /// <summary>
    /// Clears all stored events
    /// </summary>
    public void Clear()
    {
        _storedEvents.Clear();
        _storedBatches.Clear();
    }

    public void Dispose()
    {
        Clear();
    }
}
