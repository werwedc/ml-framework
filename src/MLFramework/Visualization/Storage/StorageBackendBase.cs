using MachineLearning.Visualization.Events;
using System.Collections.Concurrent;

namespace MachineLearning.Visualization.Storage;

/// <summary>
/// Abstract base class for storage backends with common functionality
/// </summary>
public abstract class StorageBackendBase : IStorageBackend, IDisposable
{
    private readonly object _lock = new();
    private readonly ConcurrentQueue<Event> _eventBuffer;
    private readonly Timer? _flushTimer;
    private long _eventCount;
    private bool _isInitialized;
    private bool _disposed;

    /// <summary>
    /// Gets the storage configuration
    /// </summary>
    protected StorageConfiguration Configuration { get; }

    /// <summary>
    /// Gets whether the backend has been initialized
    /// </summary>
    public bool IsInitialized => _isInitialized;

    /// <summary>
    /// Gets the total number of events stored
    /// </summary>
    public long EventCount => Volatile.Read(ref _eventCount);

    /// <summary>
    /// Initializes a new instance of StorageBackendBase
    /// </summary>
    /// <param name="configuration">Storage configuration</param>
    protected StorageBackendBase(StorageConfiguration configuration)
    {
        Configuration = configuration ?? throw new ArgumentNullException(nameof(configuration));
        Configuration.EnsureValid();

        _eventBuffer = new ConcurrentQueue<Event>();

        // Set up auto-flush timer if enabled
        if (Configuration.FlushInterval > TimeSpan.Zero)
        {
            _flushTimer = new Timer(
                callback: _ => Flush(),
                state: null,
                dueTime: Configuration.FlushInterval,
                period: Configuration.FlushInterval);
        }
    }

    /// <summary>
    /// Initializes the storage backend with a connection string
    /// </summary>
    /// <param name="connectionString">Connection string for the backend</param>
    public void Initialize(string connectionString)
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(StorageBackendBase));
        }

        if (_isInitialized)
        {
            throw new InvalidOperationException("Storage backend is already initialized");
        }

        if (string.IsNullOrWhiteSpace(connectionString))
        {
            throw new ArgumentException("Connection string cannot be null or empty", nameof(connectionString));
        }

        InitializeCore(connectionString);
        _isInitialized = true;
    }

    /// <summary>
    /// Core initialization logic to be implemented by derived classes
    /// </summary>
    /// <param name="connectionString">Connection string for the backend</param>
    protected abstract void InitializeCore(string connectionString);

    /// <summary>
    /// Shuts down the storage backend and releases resources
    /// </summary>
    public void Shutdown()
    {
        if (_disposed)
        {
            return;
        }

        // Flush any remaining events
        Flush();

        if (_isInitialized)
        {
            ShutdownCore();
            _isInitialized = false;
        }
    }

    /// <summary>
    /// Core shutdown logic to be implemented by derived classes
    /// </summary>
    protected abstract void ShutdownCore();

    /// <summary>
    /// Stores a single event synchronously
    /// </summary>
    /// <param name="eventData">Event to store</param>
    public void StoreEvent(Event eventData)
    {
        if (!_isInitialized)
        {
            throw new InvalidOperationException("Storage backend is not initialized");
        }

        if (eventData == null)
        {
            throw new ArgumentNullException(nameof(eventData));
        }

        // Buffer the event
        _eventBuffer.Enqueue(eventData);

        // Increment event count
        Interlocked.Increment(ref _eventCount);

        // Check if we should flush
        if (_eventBuffer.Count >= Configuration.WriteBufferSize)
        {
            Flush();
        }
    }

    /// <summary>
    /// Stores a single event asynchronously
    /// </summary>
    /// <param name="eventData">Event to store</param>
    public async Task StoreEventAsync(Event eventData)
    {
        if (Configuration.EnableAsyncWrites)
        {
            await Task.Run(() => StoreEvent(eventData)).ConfigureAwait(false);
        }
        else
        {
            StoreEvent(eventData);
        }
    }

    /// <summary>
    /// Stores multiple events synchronously
    /// </summary>
    /// <param name="events">Events to store</param>
    public void StoreEvents(IEnumerable<Event> events)
    {
        if (events == null)
        {
            throw new ArgumentNullException(nameof(events));
        }

        foreach (var eventData in events)
        {
            StoreEvent(eventData);
        }
    }

    /// <summary>
    /// Stores multiple events asynchronously
    /// </summary>
    /// <param name="events">Events to store</param>
    public async Task StoreEventsAsync(IEnumerable<Event> events)
    {
        if (Configuration.EnableAsyncWrites)
        {
            await Task.Run(() => StoreEvents(events)).ConfigureAwait(false);
        }
        else
        {
            StoreEvents(events);
        }
    }

    /// <summary>
    /// Retrieves events within a step range synchronously
    /// </summary>
    /// <param name="startStep">Starting step (inclusive)</param>
    /// <param name="endStep">Ending step (inclusive)</param>
    public abstract IEnumerable<Event> GetEvents(long startStep, long endStep);

    /// <summary>
    /// Retrieves events within a step range asynchronously
    /// </summary>
    /// <param name="startStep">Starting step (inclusive)</param>
    /// <param name="endStep">Ending step (inclusive)</param>
    public async Task<IEnumerable<Event>> GetEventsAsync(long startStep, long endStep)
    {
        return await Task.Run(() => GetEvents(startStep, endStep)).ConfigureAwait(false);
    }

    /// <summary>
    /// Flushes any buffered data to storage synchronously
    /// </summary>
    public void Flush()
    {
        if (!_isInitialized || _disposed)
        {
            return;
        }

        lock (_lock)
        {
            if (_eventBuffer.IsEmpty)
            {
                return;
            }

            // Dequeue all buffered events
            var eventsToFlush = new List<Event>();
            while (_eventBuffer.TryDequeue(out var eventData))
            {
                eventsToFlush.Add(eventData);
            }

            // Flush to storage
            if (eventsToFlush.Count > 0)
            {
                FlushCore(eventsToFlush);
            }
        }
    }

    /// <summary>
    /// Core flush logic to be implemented by derived classes
    /// </summary>
    /// <param name="events">Events to flush to storage</param>
    protected abstract void FlushCore(IEnumerable<Event> events);

    /// <summary>
    /// Flushes any buffered data to storage asynchronously
    /// </summary>
    public async Task FlushAsync()
    {
        await Task.Run(Flush).ConfigureAwait(false);
    }

    /// <summary>
    /// Clears all stored events
    /// </summary>
    public void Clear()
    {
        if (!_isInitialized)
        {
            throw new InvalidOperationException("Storage backend is not initialized");
        }

        lock (_lock)
        {
            // Clear buffer
            while (_eventBuffer.TryDequeue(out _)) { }

            // Clear storage
            ClearCore();

            // Reset event count
            Interlocked.Exchange(ref _eventCount, 0);
        }
    }

    /// <summary>
    /// Core clear logic to be implemented by derived classes
    /// </summary>
    protected abstract void ClearCore();

    /// <summary>
    /// Disposes of the storage backend and releases resources
    /// </summary>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Disposes of the storage backend and releases resources
    /// </summary>
    /// <param name="disposing">True if disposing managed resources</param>
    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (disposing)
            {
                // Flush any remaining events
                Flush();

                // Shutdown
                Shutdown();

                // Dispose timer
                _flushTimer?.Dispose();
            }

            _disposed = true;
        }
    }

    /// <summary>
    /// Finalizer
    /// </summary>
    ~StorageBackendBase()
    {
        Dispose(false);
    }
}
