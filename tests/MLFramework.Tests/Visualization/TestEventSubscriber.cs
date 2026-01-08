using System.Collections.Immutable;
using MachineLearning.Visualization.Events;
using MachineLearning.Visualization.Collection;

namespace MachineLearning.Visualization.Tests;

/// <summary>
/// Test event subscriber for testing
/// </summary>
public class TestEventSubscriber : IEventSubscriber
{
    private readonly List<Event> _processedEvents = new();
    private readonly List<ImmutableArray<Event>> _processedBatches = new();
    private readonly Action<Event>? _onEventProcessed;
    private readonly Action<ImmutableArray<Event>>? _onBatchProcessed;

    /// <summary>
    /// Gets all events that were processed
    /// </summary>
    public IReadOnlyList<Event> ProcessedEvents => _processedEvents.AsReadOnly();

    /// <summary>
    /// Gets all batches that were processed
    /// </summary>
    public IReadOnlyList<ImmutableArray<Event>> ProcessedBatches => _processedBatches.AsReadOnly();

    /// <summary>
    /// Gets the total number of events processed
    /// </summary>
    public int TotalEventsProcessed => _processedEvents.Count;

    /// <summary>
    /// Gets the total number of batches processed
    /// </summary>
    public int TotalBatchesProcessed => _processedBatches.Count;

    /// <summary>
    /// Gets or sets a delay to simulate slow processing
    /// </summary>
    public TimeSpan ProcessingDelay { get; set; } = TimeSpan.Zero;

    public TestEventSubscriber()
    {
    }

    public TestEventSubscriber(Action<Event> onEventProcessed)
    {
        _onEventProcessed = onEventProcessed;
    }

    public TestEventSubscriber(Action<ImmutableArray<Event>> onBatchProcessed)
    {
        _onBatchProcessed = onBatchProcessed;
    }

    /// <summary>
    /// Processes a single event
    /// </summary>
    public Task ProcessAsync(Event eventData, CancellationToken cancellationToken = default)
    {
        if (ProcessingDelay > TimeSpan.Zero)
        {
            Thread.Sleep(ProcessingDelay);
        }

        _processedEvents.Add(eventData);
        _onEventProcessed?.Invoke(eventData);

        return Task.CompletedTask;
    }

    /// <summary>
    /// Processes a batch of events
    /// </summary>
    public Task ProcessBatchAsync(ImmutableArray<Event> events, CancellationToken cancellationToken = default)
    {
        if (ProcessingDelay > TimeSpan.Zero)
        {
            Thread.Sleep(ProcessingDelay);
        }

        _processedBatches.Add(events);
        foreach (var evt in events)
        {
            _processedEvents.Add(evt);
        }
        _onBatchProcessed?.Invoke(events);

        return Task.CompletedTask;
    }

    /// <summary>
    /// Clears all processed events
    /// </summary>
    public void Clear()
    {
        _processedEvents.Clear();
        _processedBatches.Clear();
    }
}
