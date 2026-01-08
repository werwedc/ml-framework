using System.Collections.Immutable;
using MachineLearning.Visualization.Events;

namespace MachineLearning.Visualization.Collection.Performance;

/// <summary>
/// Delegate for processing a batch of events
/// </summary>
/// <param name="events">Batch of events to process</param>
/// <param name="cancellationToken">Cancellation token for async operations</param>
public delegate Task ProcessEventBatchAsync(ImmutableArray<Event> events, CancellationToken cancellationToken);

/// <summary>
/// Handles batch processing of events with configurable strategies
/// </summary>
public class BatchProcessor
{
    private readonly int _batchSize;
    private readonly TimeSpan _maxWaitTime;
    private readonly List<Event> _currentBatch;
    private readonly SemaphoreSlim _processingLock;
    private DateTime _lastFlushTime;
    private long _totalEventsProcessed;
    private long _totalBatchesProcessed;

    /// <summary>
    /// Gets the current batch size
    /// </summary>
    public int BatchSize => _batchSize;

    /// <summary>
    /// Gets the number of events currently in the batch
    /// </summary>
    public int CurrentBatchSize => _currentBatch.Count;

    /// <summary>
    /// Gets the total number of events processed
    /// </summary>
    public long TotalEventsProcessed => Interlocked.Read(ref _totalEventsProcessed);

    /// <summary>
    /// Gets the total number of batches processed
    /// </summary>
    public long TotalBatchesProcessed => Interlocked.Read(ref _totalBatchesProcessed);

    /// <summary>
    /// Creates a new batch processor
    /// </summary>
    /// <param name="batchSize">Maximum number of events per batch</param>
    /// <param name="maxWaitTime">Maximum time to wait before flushing a partial batch</param>
    public BatchProcessor(int batchSize = 100, TimeSpan? maxWaitTime = null)
    {
        if (batchSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(batchSize), "Batch size must be positive");

        _batchSize = batchSize;
        _maxWaitTime = maxWaitTime ?? TimeSpan.FromSeconds(1);
        _currentBatch = new List<Event>(batchSize);
        _processingLock = new SemaphoreSlim(1, 1);
        _lastFlushTime = DateTime.UtcNow;
    }

    /// <summary>
    /// Adds an event to the current batch, flushing if necessary
    /// </summary>
    /// <param name="eventData">Event to add</param>
    /// <param name="processor">Async processor delegate</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Task representing the async operation</returns>
    public async Task AddEventAsync(
        Event eventData,
        ProcessEventBatchAsync processor,
        CancellationToken cancellationToken = default)
    {
        if (eventData == null)
            throw new ArgumentNullException(nameof(eventData));

        if (processor == null)
            throw new ArgumentNullException(nameof(processor));

        await _processingLock.WaitAsync(cancellationToken);

        try
        {
            _currentBatch.Add(eventData);

            // Check if we should flush based on batch size or time
            if (_currentBatch.Count >= _batchSize ||
                (DateTime.UtcNow - _lastFlushTime) >= _maxWaitTime)
            {
                await FlushAsync(processor, cancellationToken);
            }
        }
        finally
        {
            _processingLock.Release();
        }
    }

    /// <summary>
    /// Flushes the current batch even if it's not full
    /// </summary>
    /// <param name="processor">Async processor delegate</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Task representing the async operation</returns>
    public async Task FlushAsync(
        ProcessEventBatchAsync processor,
        CancellationToken cancellationToken = default)
    {
        if (processor == null)
            throw new ArgumentNullException(nameof(processor));

        await _processingLock.WaitAsync(cancellationToken);

        try
        {
            if (_currentBatch.Count > 0)
            {
                var batch = _currentBatch.ToImmutableArray();
                _currentBatch.Clear();

                await processor(batch, cancellationToken);

                Interlocked.Add(ref _totalEventsProcessed, batch.Length);
                Interlocked.Increment(ref _totalBatchesProcessed);
                _lastFlushTime = DateTime.UtcNow;
            }
        }
        finally
        {
            _processingLock.Release();
        }
    }

    /// <summary>
    /// Processes multiple events as a single batch
    /// </summary>
    /// <param name="events">Events to process</param>
    /// <param name="processor">Async processor delegate</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Task representing the async operation</returns>
    public async Task ProcessBatchAsync(
        ImmutableArray<Event> events,
        ProcessEventBatchAsync processor,
        CancellationToken cancellationToken = default)
    {
        if (events.IsDefaultOrEmpty)
            return;

        if (processor == null)
            throw new ArgumentNullException(nameof(processor));

        await _processingLock.WaitAsync(cancellationToken);

        try
        {
            // Flush current batch first
            if (_currentBatch.Count > 0)
            {
                var currentBatch = _currentBatch.ToImmutableArray();
                _currentBatch.Clear();
                await processor(currentBatch, cancellationToken);
                Interlocked.Add(ref _totalEventsProcessed, currentBatch.Length);
                Interlocked.Increment(ref _totalBatchesProcessed);
            }

            // Process the new events
            await processor(events, cancellationToken);
            Interlocked.Add(ref _totalEventsProcessed, events.Length);
            Interlocked.Increment(ref _totalBatchesProcessed);
            _lastFlushTime = DateTime.UtcNow;
        }
        finally
        {
            _processingLock.Release();
        }
    }

    /// <summary>
    /// Gets statistics about the batch processor
    /// </summary>
    public BatchProcessorStatistics GetStatistics()
    {
        long eventsProcessed = TotalEventsProcessed;
        long batchesProcessed = TotalBatchesProcessed;
        double avgBatchSize = batchesProcessed > 0 ? (double)eventsProcessed / batchesProcessed : 0;

        return new BatchProcessorStatistics
        {
            TotalEventsProcessed = eventsProcessed,
            TotalBatchesProcessed = batchesProcessed,
            AverageBatchSize = avgBatchSize,
            CurrentBatchSize = CurrentBatchSize,
            LastFlushTime = _lastFlushTime
        };
    }

    /// <summary>
    /// Resets the statistics
    /// </summary>
    public void ResetStatistics()
    {
        Interlocked.Exchange(ref _totalEventsProcessed, 0);
        Interlocked.Exchange(ref _totalBatchesProcessed, 0);
        _lastFlushTime = DateTime.UtcNow;
    }
}

/// <summary>
/// Statistics for batch processing
/// </summary>
public class BatchProcessorStatistics
{
    /// <summary>
    /// Total number of events processed
    /// </summary>
    public long TotalEventsProcessed { get; set; }

    /// <summary>
    /// Total number of batches processed
    /// </summary>
    public long TotalBatchesProcessed { get; set; }

    /// <summary>
    /// Average batch size
    /// </summary>
    public double AverageBatchSize { get; set; }

    /// <summary>
    /// Current batch size
    /// </summary>
    public int CurrentBatchSize { get; set; }

    /// <summary>
    /// Last flush time
    /// </summary>
    public DateTime LastFlushTime { get; set; }
}
