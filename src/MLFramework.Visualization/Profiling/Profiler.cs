using MachineLearning.Visualization.Events;
using MLFramework.Visualization.Profiling.Statistics;
using System.Collections.Concurrent;

namespace MLFramework.Visualization.Profiling;

/// <summary>
/// Profiler for tracking operation performance metrics
/// </summary>
public class Profiler : IProfiler
{
    private readonly ConcurrentDictionary<string, DurationTracker> _durationTrackers;
    private readonly ConcurrentDictionary<string, string> _parentScopes;
    private readonly IEventPublisher? _eventPublisher;
    private readonly IStorageBackend? _storageBackend;

    /// <summary>
    /// Gets or sets whether profiling is enabled
    /// </summary>
    public bool IsEnabled { get; private set; } = true;

    /// <summary>
    /// Gets or sets whether automatic profiling is enabled
    /// </summary>
    public bool EnableAutomatic { get; set; } = true;

    /// <summary>
    /// Gets or sets the maximum number of operations to store
    /// </summary>
    public int MaxStoredOperations { get; set; } = 10000;

    /// <summary>
    /// Gets the current step number
    /// </summary>
    public long CurrentStep { get; set; } = 0;

    /// <summary>
    /// Creates a new profiler with an event publisher
    /// </summary>
    /// <param name="eventPublisher">Event publisher for profiling events</param>
    public Profiler(IEventPublisher eventPublisher)
    {
        _eventPublisher = eventPublisher ?? throw new ArgumentNullException(nameof(eventPublisher));
        _durationTrackers = new ConcurrentDictionary<string, DurationTracker>();
        _parentScopes = new ConcurrentDictionary<string, string>();
    }

    /// <summary>
    /// Creates a new profiler with a storage backend
    /// </summary>
    /// <param name="storageBackend">Storage backend for persisting profiling data</param>
    public Profiler(IStorageBackend storageBackend)
    {
        _storageBackend = storageBackend ?? throw new ArgumentNullException(nameof(storageBackend));
        _durationTrackers = new ConcurrentDictionary<string, DurationTracker>();
        _parentScopes = new ConcurrentDictionary<string, string>();
    }

    /// <summary>
    /// Creates a new profiler with both event publisher and storage backend
    /// </summary>
    /// <param name="eventPublisher">Event publisher for profiling events</param>
    /// <param name="storageBackend">Storage backend for persisting profiling data</param>
    public Profiler(IEventPublisher eventPublisher, IStorageBackend storageBackend)
    {
        _eventPublisher = eventPublisher;
        _storageBackend = storageBackend;
        _durationTrackers = new ConcurrentDictionary<string, DurationTracker>();
        _parentScopes = new ConcurrentDictionary<string, string>();
    }

    /// <summary>
    /// Starts profiling an operation
    /// </summary>
    /// <param name="name">Name of the operation to profile</param>
    /// <returns>A profiling scope that will record the duration when disposed</returns>
    public IProfilingScope StartProfile(string name)
    {
        return StartProfile(name, new Dictionary<string, string>());
    }

    /// <summary>
    /// Starts profiling an operation with metadata
    /// </summary>
    /// <param name="name">Name of the operation to profile</param>
    /// <param name="metadata">Additional metadata for this profiling operation</param>
    /// <returns>A profiling scope that will record the duration when disposed</returns>
    public IProfilingScope StartProfile(string name, Dictionary<string, string> metadata)
    {
        if (!IsEnabled)
        {
            return new ProfilingScope(this, name, metadata);
        }

        // Ensure duration tracker exists for this operation
        var tracker = _durationTrackers.GetOrAdd(name, _ => new DurationTracker());

        // Publish start event
        if (_eventPublisher != null && EnableAutomatic)
        {
            var startEvent = new ProfilingStartEvent(name, CurrentStep, metadata);
            _eventPublisher.Publish(startEvent);
        }

        return new ProfilingScope(this, name, metadata);
    }

    /// <summary>
    /// Records a duration for an operation (called by ProfilingScope)
    /// </summary>
    /// <param name="name">Name of the operation</param>
    /// <param name="durationNanoseconds">Duration in nanoseconds</param>
    /// <param name="metadata">Metadata for the operation</param>
    internal void RecordDuration(string name, long durationNanoseconds, Dictionary<string, string> metadata)
    {
        if (!IsEnabled)
        {
            return;
        }

        // Record duration in tracker
        var tracker = _durationTrackers.GetOrAdd(name, _ => new DurationTracker());
        tracker.RecordDuration(durationNanoseconds);

        // Publish end event
        if (_eventPublisher != null && EnableAutomatic)
        {
            var endEvent = new ProfilingEndEvent(name, CurrentStep, durationNanoseconds, metadata);
            _eventPublisher.Publish(endEvent);
        }

        // Check if we exceed max operations
        if (tracker.Count > MaxStoredOperations)
        {
            // Remove oldest half of the data to keep memory usage bounded
            var durations = tracker.GetDurations();
            var newTracker = new DurationTracker();
            for (int i = durations.Length / 2; i < durations.Length; i++)
            {
                newTracker.RecordDuration(durations[i]);
            }
            _durationTrackers[name] = newTracker;
        }
    }

    /// <summary>
    /// Records an instant event (e.g., a checkpoint or milestone)
    /// </summary>
    /// <param name="name">Name of the instant event</param>
    public void RecordInstant(string name)
    {
        RecordInstant(name, new Dictionary<string, string>());
    }

    /// <summary>
    /// Records an instant event with metadata
    /// </summary>
    /// <param name="name">Name of the instant event</param>
    /// <param name="metadata">Additional metadata for this event</param>
    public void RecordInstant(string name, Dictionary<string, string> metadata)
    {
        if (!IsEnabled)
        {
            return;
        }

        // For instant events, we could create an InstantEvent type if needed
        // For now, we'll just record a zero-duration event
        if (_eventPublisher != null && EnableAutomatic)
        {
            var startEvent = new ProfilingStartEvent(name, CurrentStep, metadata);
            var endEvent = new ProfilingEndEvent(name, CurrentStep, 0, metadata);
            _eventPublisher.Publish(startEvent);
            _eventPublisher.Publish(endEvent);
        }
    }

    /// <summary>
    /// Gets the profiling result for a specific operation
    /// </summary>
    /// <param name="name">Name of the operation</param>
    /// <returns>Profiling result or null if not found</returns>
    public ProfilingResult? GetResult(string name)
    {
        if (!_durationTrackers.TryGetValue(name, out var tracker))
        {
            return null;
        }

        if (tracker.Count == 0)
        {
            return null;
        }

        var durations = tracker.GetDurations();
        var (p50, p90, p95, p99) = PercentileCalculator.CalculateCommonPercentiles(durations);

        return new ProfilingResult(
            name: name,
            totalDurationNanoseconds: tracker.TotalDurationNanoseconds,
            count: tracker.Count,
            minDurationNanoseconds: tracker.MinDurationNanoseconds,
            maxDurationNanoseconds: tracker.MaxDurationNanoseconds,
            averageDurationNanoseconds: tracker.AverageDurationNanoseconds,
            stdDevNanoseconds: tracker.StdDevNanoseconds,
            p50Nanoseconds: p50,
            p90Nanoseconds: p90,
            p95Nanoseconds: p95,
            p99Nanoseconds: p99
        );
    }

    /// <summary>
    /// Gets all profiling results
    /// </summary>
    /// <returns>Dictionary of operation names to profiling results</returns>
    public Dictionary<string, ProfilingResult> GetAllResults()
    {
        var results = new Dictionary<string, ProfilingResult>();

        foreach (var kvp in _durationTrackers)
        {
            var result = GetResult(kvp.Key);
            if (result != null)
            {
                results[kvp.Key] = result;
            }
        }

        return results;
    }

    /// <summary>
    /// Sets a parent-child relationship between profiling scopes
    /// </summary>
    /// <param name="childName">Name of the child operation</param>
    /// <param name="parentName">Name of the parent operation</param>
    public void SetParentScope(string childName, string parentName)
    {
        if (string.IsNullOrEmpty(childName))
        {
            throw new ArgumentNullException(nameof(childName));
        }

        if (string.IsNullOrEmpty(parentName))
        {
            throw new ArgumentNullException(nameof(parentName));
        }

        _parentScopes[childName] = parentName;
    }

    /// <summary>
    /// Gets the parent scope name for a given child scope
    /// </summary>
    /// <param name="childName">Name of the child operation</param>
    /// <returns>Parent name or null if no parent is set</returns>
    public string? GetParentScope(string childName)
    {
        if (_parentScopes.TryGetValue(childName, out var parentName))
        {
            return parentName;
        }

        return null;
    }

    /// <summary>
    /// Enables profiling
    /// </summary>
    public void Enable()
    {
        IsEnabled = true;
    }

    /// <summary>
    /// Disables profiling
    /// </summary>
    public void Disable()
    {
        IsEnabled = false;
    }

    /// <summary>
    /// Clears all recorded profiling data
    /// </summary>
    public void Clear()
    {
        _durationTrackers.Clear();
        _parentScopes.Clear();
    }
}
