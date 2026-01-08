using MachineLearning.Visualization.Events;
using MachineLearning.Visualization.Collection.Configuration;
using MachineLearning.Visualization.Storage;

namespace MachineLearning.Visualization.Collection;

/// <summary>
/// Base implementation for event collectors
/// </summary>
public abstract class EventCollector : IEventCollector
{
    protected readonly EventBuffer _buffer;
    protected readonly IStorageBackend? _storageBackend;
    protected readonly CancellationTokenSource _cancellationTokenSource;
    protected readonly SemaphoreSlim _flushLock;
    protected readonly object _statisticsLock = new();

    private Task? _processingTask;
    private DateTime _startTime;
    private DateTime _lastFlushTime;
    private long _eventsCollected;
    protected long _eventsProcessed;
    private long _eventsDropped;
    private long _totalFlushes;
    private bool _isRunning;

    /// <summary>
    /// Gets or sets the collector configuration
    /// </summary>
    public EventCollectorConfig Config { get; set; }

    /// <summary>
    /// Gets whether the collector is currently running
    /// </summary>
    public bool IsRunning => _isRunning;

    /// <summary>
    /// Gets the number of events waiting to be processed
    /// </summary>
    public int PendingEventCount => _buffer.Count;

    /// <summary>
    /// Gets the internal event buffer (for testing)
    /// </summary>
    protected internal EventBuffer Buffer => _buffer;

    /// <summary>
    /// Creates a new event collector with the specified storage backend and configuration
    /// </summary>
    /// <param name="storageBackend">Storage backend for persisting events</param>
    /// <param name="config">Collector configuration</param>
    protected EventCollector(IStorageBackend? storageBackend, EventCollectorConfig? config = null)
    {
        Config = config ?? new EventCollectorConfig();
        Config.Validate();

        _storageBackend = storageBackend;
        _buffer = new EventBuffer(Config.BufferCapacity);
        _cancellationTokenSource = new CancellationTokenSource();
        _flushLock = new SemaphoreSlim(1, 1);
        _startTime = DateTime.UtcNow;
        _lastFlushTime = DateTime.UtcNow;
    }

    /// <summary>
    /// Collects an event synchronously
    /// </summary>
    public virtual void Collect<T>(T eventData) where T : Event
    {
        if (eventData == null)
            throw new ArgumentNullException(nameof(eventData));

        if (!_isRunning)
            throw new InvalidOperationException("Collector is not running");

        // Handle backpressure if enabled
        if (Config.EnableBackpressure && _buffer.IsFull)
        {
            HandleBackpressure(eventData);
        }
        else
        {
            _buffer.Enqueue(eventData);
            Interlocked.Increment(ref _eventsCollected);
        }
    }

    /// <summary>
    /// Collects an event asynchronously
    /// </summary>
    public virtual async Task CollectAsync<T>(T eventData) where T : Event
    {
        if (eventData == null)
            throw new ArgumentNullException(nameof(eventData));

        if (!_isRunning)
            throw new InvalidOperationException("Collector is not running");

        // Handle backpressure if enabled
        if (Config.EnableBackpressure && _buffer.IsFull)
        {
            await HandleBackpressureAsync(eventData);
        }
        else
        {
            _buffer.Enqueue(eventData);
            Interlocked.Increment(ref _eventsCollected);
        }

        // Trigger flush based on strategy
        if (Config.Strategy == FlushStrategy.SizeBased || Config.Strategy == FlushStrategy.Hybrid)
        {
            if (_buffer.Count >= Config.BatchSize)
            {
                await FlushAsync();
            }
        }
    }

    /// <summary>
    /// Flushes all pending events synchronously
    /// </summary>
    public virtual void Flush()
    {
        FlushAsync().GetAwaiter().GetResult();
    }

    /// <summary>
    /// Flushes all pending events asynchronously
    /// </summary>
    public virtual async Task FlushAsync()
    {
        await _flushLock.WaitAsync(_cancellationTokenSource.Token);
        try
        {
            await ProcessEventsAsync();
            UpdateFlushStats();
        }
        finally
        {
            _flushLock.Release();
        }
    }

    /// <summary>
    /// Starts the event collector
    /// </summary>
    public virtual void Start()
    {
        if (_isRunning)
            return;

        _isRunning = true;
        _startTime = DateTime.UtcNow;
        _processingTask = RunProcessingLoopAsync(_cancellationTokenSource.Token);
    }

    /// <summary>
    /// Stops the event collector
    /// </summary>
    public virtual void Stop()
    {
        if (!_isRunning)
            return;

        _isRunning = false;
        _cancellationTokenSource.Cancel();

        try
        {
            // Flush remaining events
            Flush();
        }
        catch
        {
            // Ignore errors during shutdown
        }

        _processingTask?.Wait(TimeSpan.FromMilliseconds(500));
        _processingTask?.Dispose();
        _processingTask = null;
    }

    /// <summary>
    /// Gets statistics about the event collector
    /// </summary>
    public virtual EventCollectorStatistics GetStatistics()
    {
        lock (_statisticsLock)
        {
            long totalFlushes = Interlocked.Read(ref _totalFlushes);
            double avgEventsPerFlush = totalFlushes > 0
                ? (double)Interlocked.Read(ref _eventsProcessed) / totalFlushes
                : 0;

            return new EventCollectorStatistics
            {
                EventsCollected = Interlocked.Read(ref _eventsCollected),
                EventsProcessed = Interlocked.Read(ref _eventsProcessed),
                EventsDropped = Interlocked.Read(ref _eventsDropped),
                PendingEvents = _buffer.Count,
                IsBackpressureActive = Config.EnableBackpressure && _buffer.IsFull,
                PeakBufferSize = _buffer.PeakSize,
                CurrentBufferSize = _buffer.Count,
                TotalFlushes = totalFlushes,
                AverageEventsPerFlush = avgEventsPerFlush,
                LastFlushTime = _lastFlushTime,
                UptimeSeconds = _isRunning ? (DateTime.UtcNow - _startTime).TotalSeconds : 0
            };
        }
    }

    /// <summary>
    /// Processes events from the buffer
    /// </summary>
    protected abstract Task ProcessEventsAsync();

    /// <summary>
    /// Handles backpressure when the buffer is full
    /// </summary>
    protected virtual void HandleBackpressure(Event eventData)
    {
        switch (Config.BackpressureAction)
        {
            case BackpressureAction.DropOldest:
                // The buffer already drops oldest events when full
                Interlocked.Increment(ref _eventsDropped);
                _buffer.Enqueue(eventData);
                Interlocked.Increment(ref _eventsCollected);
                break;

            case BackpressureAction.DropNewest:
                Interlocked.Increment(ref _eventsDropped);
                break;

            case BackpressureAction.Throw:
                throw new InvalidOperationException("Event buffer is full and backpressure is set to throw");

            case BackpressureAction.Block:
                while (_buffer.IsFull)
                {
                    Thread.Sleep(10);
                }
                _buffer.Enqueue(eventData);
                Interlocked.Increment(ref _eventsCollected);
                break;
        }
    }

    /// <summary>
    /// Handles backpressure asynchronously when the buffer is full
    /// </summary>
    protected virtual async Task HandleBackpressureAsync(Event eventData)
    {
        switch (Config.BackpressureAction)
        {
            case BackpressureAction.DropOldest:
                Interlocked.Increment(ref _eventsDropped);
                _buffer.Enqueue(eventData);
                Interlocked.Increment(ref _eventsCollected);
                break;

            case BackpressureAction.DropNewest:
                Interlocked.Increment(ref _eventsDropped);
                break;

            case BackpressureAction.Throw:
                throw new InvalidOperationException("Event buffer is full and backpressure is set to throw");

            case BackpressureAction.Block:
                var startTime = DateTime.UtcNow;
                while (_buffer.IsFull && (DateTime.UtcNow - startTime) < Config.BackpressureTimeout)
                {
                    await Task.Delay(10, _cancellationTokenSource.Token);
                }

                if (_buffer.IsFull)
                {
                    Interlocked.Increment(ref _eventsDropped);
                }
                else
                {
                    _buffer.Enqueue(eventData);
                    Interlocked.Increment(ref _eventsCollected);
                }
                break;
        }
    }

    /// <summary>
    /// Runs the main processing loop for time-based flush strategies
    /// </summary>
    protected virtual async Task RunProcessingLoopAsync(CancellationToken cancellationToken)
    {
        if (Config.Strategy == FlushStrategy.Manual || Config.Strategy == FlushStrategy.SizeBased)
        {
            // No automatic processing needed
            return;
        }

        while (!cancellationToken.IsCancellationRequested && _isRunning)
        {
            try
            {
                await Task.Delay(Config.FlushInterval, cancellationToken);

                if (!cancellationToken.IsCancellationRequested && _isRunning)
                {
                    if (Config.Strategy == FlushStrategy.TimeBased ||
                        (Config.Strategy == FlushStrategy.Hybrid && _buffer.Count > 0))
                    {
                        await FlushAsync();
                    }
                }
            }
            catch (OperationCanceledException)
            {
                // Expected on shutdown
                break;
            }
            catch (Exception ex)
            {
                // Log error but continue processing
                Console.Error.WriteLine($"Error in event processing loop: {ex.Message}");
            }
        }
    }

    /// <summary>
    /// Updates flush statistics
    /// </summary>
    protected virtual void UpdateFlushStats()
    {
        Interlocked.Increment(ref _totalFlushes);
        _lastFlushTime = DateTime.UtcNow;
    }

    /// <summary>
    /// Disposes of the event collector
    /// </summary>
    public virtual void Dispose()
    {
        Stop();
        _flushLock.Dispose();
        _cancellationTokenSource.Dispose();
        _buffer.Dispose();
        _storageBackend?.Dispose();
        GC.SuppressFinalize(this);
    }
}
