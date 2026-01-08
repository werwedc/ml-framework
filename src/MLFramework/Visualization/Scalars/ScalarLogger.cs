using MachineLearning.Visualization.Events;
using MachineLearning.Visualization.Storage;

namespace MachineLearning.Visualization.Scalars;

/// <summary>
/// Implementation of scalar metrics logger for tracking loss, accuracy, learning rate, etc.
/// </summary>
public class ScalarLogger : IScalarLogger, IDisposable
{
    private readonly IStorageBackend _storage;
    private readonly IEventPublisher _eventPublisher;
    private readonly Dictionary<string, ScalarSeries> _series;
    private readonly Dictionary<string, long> _stepCounters;
    private readonly object _lock = new object();
    private bool _disposed;

    /// <summary>
    /// Whether to automatically smooth logged values
    /// </summary>
    public bool AutoSmooth { get; set; }

    /// <summary>
    /// Default smoothing window size
    /// </summary>
    public int DefaultSmoothingWindow { get; set; } = 10;

    /// <summary>
    /// Maximum number of entries per series (0 for unlimited)
    /// </summary>
    public int MaxEntriesPerSeries { get; set; } = 0;

    /// <summary>
    /// Creates a new scalar logger with a storage backend
    /// </summary>
    /// <param name="storage">Storage backend for persisting events</param>
    public ScalarLogger(IStorageBackend storage)
    {
        _storage = storage ?? throw new ArgumentNullException(nameof(storage));
        _series = new Dictionary<string, ScalarSeries>();
        _stepCounters = new Dictionary<string, long>();
        _eventPublisher = new EventSystem();
    }

    /// <summary>
    /// Creates a new scalar logger with an event publisher
    /// </summary>
    /// <param name="eventPublisher">Event publisher for broadcasting events</param>
    public ScalarLogger(IEventPublisher eventPublisher)
    {
        _eventPublisher = eventPublisher ?? throw new ArgumentNullException(nameof(eventPublisher));
        _storage = new NullStorageBackend();
        _series = new Dictionary<string, ScalarSeries>();
        _stepCounters = new Dictionary<string, long>();
    }

    /// <summary>
    /// Logs a scalar value synchronously
    /// </summary>
    public void LogScalar(string name, float value, long step = -1)
    {
        if (name == null) throw new ArgumentNullException(nameof(name));

        lock (_lock)
        {
            if (step == -1)
            {
                if (!_stepCounters.TryGetValue(name, out step))
                {
                    step = 0;
                }
                _stepCounters[name] = step + 1;
            }

            var entry = new ScalarEntry(step, value);

            if (!_series.TryGetValue(name, out var series))
            {
                series = new ScalarSeries(name);
                _series[name] = series;
            }

            series.Add(entry);

            // Enforce max entries limit
            if (MaxEntriesPerSeries > 0 && series.Count > MaxEntriesPerSeries)
            {
                // Remove oldest entries (simplified approach)
                var entries = series.GetRange(step - MaxEntriesPerSeries + 1, step);
                _series[name] = new ScalarSeries(name, entries);
            }
        }

        // Publish event
        var scalarEvent = new ScalarMetricEvent(name, value, step);
        _eventPublisher.Publish(scalarEvent);
        _storage.StoreEvent(scalarEvent);
    }

    /// <summary>
    /// Logs a scalar value synchronously (double overload)
    /// </summary>
    public void LogScalar(string name, double value, long step = -1)
    {
        LogScalar(name, (float)value, step);
    }

    /// <summary>
    /// Logs a scalar value asynchronously
    /// </summary>
    public Task LogScalarAsync(string name, float value, long step = -1)
    {
        LogScalar(name, value, step);
        return Task.CompletedTask;
    }

    /// <summary>
    /// Gets a scalar series by name
    /// </summary>
    public ScalarSeries? GetSeries(string name)
    {
        if (name == null) throw new ArgumentNullException(nameof(name));

        lock (_lock)
        {
            _series.TryGetValue(name, out var series);
            return series;
        }
    }

    /// <summary>
    /// Gets a scalar series by name asynchronously
    /// </summary>
    public Task<ScalarSeries?> GetSeriesAsync(string name)
    {
        return Task.FromResult(GetSeries(name));
    }

    /// <summary>
    /// Gets all scalar series
    /// </summary>
    public IEnumerable<ScalarSeries> GetAllSeries()
    {
        lock (_lock)
        {
            return _series.Values.ToList();
        }
    }

    /// <summary>
    /// Gets a smoothed version of a scalar series
    /// </summary>
    public ScalarSeries? GetSmoothedSeries(string name, int windowSize)
    {
        if (name == null) throw new ArgumentNullException(nameof(name));

        var series = GetSeries(name);
        if (series == null) return null;

        return series.Smoothed(windowSize);
    }

    /// <summary>
    /// Gets the latest value for all metrics
    /// </summary>
    public Dictionary<string, float> GetLatestValues()
    {
        lock (_lock)
        {
            var latest = new Dictionary<string, float>();
            foreach (var kvp in _series)
            {
                var entries = kvp.Value.Entries;
                if (entries.Count > 0)
                {
                    latest[kvp.Key] = entries.Last().Value;
                }
            }
            return latest;
        }
    }

    /// <summary>
    /// Tags the current run with metadata for comparison
    /// </summary>
    public void TagRun(string runName, Dictionary<string, string> tags)
    {
        // This would be used for multi-run comparison
        // For now, we can store this as metadata in the storage backend
        // This is a placeholder for future implementation
    }

    /// <summary>
    /// Disposes of resources
    /// </summary>
    public void Dispose()
    {
        if (!_disposed)
        {
            _storage.Flush();
            _disposed = true;
        }
    }

    /// <summary>
    /// Null storage backend for event-only mode
    /// </summary>
    private class NullStorageBackend : IStorageBackend
    {
        public void Initialize(string connectionString) { }
        public void Shutdown() { }
        public bool IsInitialized => true;
        public void StoreEvent(Event eventData) { }
        public Task StoreEventAsync(Event eventData) => Task.CompletedTask;
        public void StoreEvents(IEnumerable<Event> events) { }
        public Task StoreEventsAsync(IEnumerable<Event> events) => Task.CompletedTask;
        public IEnumerable<Event> GetEvents(long startStep, long endStep) => Enumerable.Empty<Event>();
        public Task<IEnumerable<Event>> GetEventsAsync(long startStep, long endStep) =>
            Task.FromResult(Enumerable.Empty<Event>());
        public void Flush() { }
        public Task FlushAsync() => Task.CompletedTask;
        public long EventCount => 0;
        public void Clear() { }
        public void Dispose() { }
    }
}
