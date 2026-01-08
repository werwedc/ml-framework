using System.Collections.Concurrent;
using System.Diagnostics;
using MachineLearning.Visualization.Collection;
using MachineLearning.Visualization.Storage;

namespace MachineLearning.Visualization.Memory;

/// <summary>
/// Memory profiler for tracking allocations, deallocations, and usage patterns
/// </summary>
public class MemoryProfiler : IMemoryProfiler, IDisposable
{
    private readonly IStorageBackend? _storage;
    private readonly IEventCollector? _eventCollector;
    private readonly ConcurrentDictionary<long, AllocationInfo> _activeAllocations;
    private readonly ConcurrentBag<MemoryEvent> _eventHistory;
    private readonly MemoryStatistics.Builder _statisticsBuilder;
    private readonly object _lock = new object();
    private readonly GCMonitor _gcMonitor;

    private bool _isEnabled = true;
    private bool _captureStackTraces = false;
    private Timer? _snapshotTimer;
    private int _snapshotIntervalMs = 1000;
    private bool _autoSnapshot = true;
    private int _maxStackTraceDepth = 10;
    private bool _disposed = false;

    /// <summary>
    /// Gets or sets the snapshot interval in milliseconds
    /// </summary>
    public int SnapshotIntervalMs
    {
        get => _snapshotIntervalMs;
        set
        {
            _snapshotIntervalMs = value;
            if (_autoSnapshot && _isEnabled)
            {
                RestartSnapshotTimer();
            }
        }
    }

    /// <summary>
    /// Gets or sets whether to automatically capture snapshots
    /// </summary>
    public bool AutoSnapshot
    {
        get => _autoSnapshot;
        set
        {
            _autoSnapshot = value;
            if (value && _isEnabled)
            {
                StartSnapshotTimer();
            }
            else
            {
                StopSnapshotTimer();
            }
        }
    }

    /// <summary>
    /// Gets or sets the maximum stack trace depth to capture
    /// </summary>
    public int MaxStackTraceDepth
    {
        get => _maxStackTraceDepth;
        set => _maxStackTraceDepth = value;
    }

    /// <summary>
    /// Gets whether memory profiling is enabled
    /// </summary>
    public bool IsEnabled => _isEnabled;

    /// <summary>
    /// Gets or sets whether to capture stack traces for allocations
    /// </summary>
    public bool CaptureStackTraces
    {
        get => _captureStackTraces;
        set => _captureStackTraces = value;
    }

    /// <summary>
    /// Creates a new MemoryProfiler with a storage backend
    /// </summary>
    public MemoryProfiler(IStorageBackend storage)
    {
        _storage = storage ?? throw new ArgumentNullException(nameof(storage));
        _eventCollector = null;
        _activeAllocations = new ConcurrentDictionary<long, AllocationInfo>();
        _eventHistory = new ConcurrentBag<MemoryEvent>();
        _statisticsBuilder = MemoryStatistics.CreateBuilder();
        _gcMonitor = new GCMonitor(this);

        if (_autoSnapshot)
        {
            StartSnapshotTimer();
        }
    }

    /// <summary>
    /// Creates a new MemoryProfiler with an event collector
    /// </summary>
    public MemoryProfiler(IEventCollector eventCollector)
    {
        _eventCollector = eventCollector ?? throw new ArgumentNullException(nameof(eventCollector));
        _storage = null;
        _activeAllocations = new ConcurrentDictionary<long, AllocationInfo>();
        _eventHistory = new ConcurrentBag<MemoryEvent>();
        _statisticsBuilder = MemoryStatistics.CreateBuilder();
        _gcMonitor = new GCMonitor(this);

        if (_autoSnapshot)
        {
            StartSnapshotTimer();
        }
    }

    /// <summary>
    /// Tracks a memory allocation
    /// </summary>
    public void TrackAllocation(long address, long sizeBytes, string allocationType)
    {
        if (!_isEnabled)
        {
            return;
        }

        if (sizeBytes < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sizeBytes), "Size bytes cannot be negative");
        }

        if (string.IsNullOrEmpty(allocationType))
        {
            throw new ArgumentException("Allocation type cannot be null or empty", nameof(allocationType));
        }

        var stackTrace = _captureStackTraces ? CaptureStackTrace() : null;
        var allocationInfo = new AllocationInfo
        {
            Address = address,
            SizeBytes = sizeBytes,
            AllocationType = allocationType,
            AllocationTime = DateTime.UtcNow,
            StackTrace = stackTrace
        };

        _activeAllocations.TryAdd(address, allocationInfo);
        _statisticsBuilder.RecordAllocation(sizeBytes, allocationType);

        var statistics = _statisticsBuilder.Build();
        var memoryEvent = new MemoryEvent(
            MemoryEventType.Allocation,
            address,
            sizeBytes,
            statistics.TotalAllocatedBytes,
            statistics.TotalFreedBytes,
            allocationType,
            stackTrace);

        _eventHistory.Add(memoryEvent);
        PublishEvent(memoryEvent);
    }

    /// <summary>
    /// Tracks a memory deallocation
    /// </summary>
    public void TrackDeallocation(long address, long sizeBytes)
    {
        if (!_isEnabled)
        {
            return;
        }

        if (sizeBytes < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sizeBytes), "Size bytes cannot be negative");
        }

        if (_activeAllocations.TryRemove(address, out var allocationInfo))
        {
            _statisticsBuilder.RecordDeallocation(sizeBytes, allocationInfo.AllocationType);

            var statistics = _statisticsBuilder.Build();
            var memoryEvent = new MemoryEvent(
                MemoryEventType.Deallocation,
                address,
                sizeBytes,
                statistics.TotalAllocatedBytes,
                statistics.TotalFreedBytes,
                allocationInfo.AllocationType);

            _eventHistory.Add(memoryEvent);
            PublishEvent(memoryEvent);
        }
        else
        {
            // Deallocation without matching allocation - log a warning
            var statistics = _statisticsBuilder.Build();
            var memoryEvent = new MemoryEvent(
                MemoryEventType.Deallocation,
                address,
                sizeBytes,
                statistics.TotalAllocatedBytes,
                statistics.TotalFreedBytes,
                "Unknown");

            _eventHistory.Add(memoryEvent);
            PublishEvent(memoryEvent);
        }
    }

    /// <summary>
    /// Tracks a snapshot of current memory usage
    /// </summary>
    public void TrackSnapshot()
    {
        if (!_isEnabled)
        {
            return;
        }

        var statistics = _statisticsBuilder.Build();
        var memoryEvent = new MemoryEvent(
            MemoryEventType.Snapshot,
            0,
            0,
            statistics.TotalAllocatedBytes,
            statistics.TotalFreedBytes,
            "Snapshot");

        _eventHistory.Add(memoryEvent);
        PublishEvent(memoryEvent);
    }

    /// <summary>
    /// Gets current memory statistics
    /// </summary>
    public MemoryStatistics GetStatistics()
    {
        return _statisticsBuilder.Build();
    }

    /// <summary>
    /// Gets memory statistics for a specific allocation type
    /// </summary>
    public MemoryStatistics GetStatisticsForType(string allocationType)
    {
        var allStatistics = _statisticsBuilder.Build();
        var typeUsage = allStatistics.UsageByType.TryGetValue(allocationType, out var usage) ? usage : 0;

        return new MemoryStatistics(
            totalAllocatedBytes: typeUsage,
            totalFreedBytes: 0,
            currentUsageBytes: typeUsage,
            peakUsageBytes: typeUsage,
            allocationCount: 0,
            deallocationCount: 0,
            averageAllocationSizeBytes: 0,
            usageByType: new ConcurrentDictionary<string, long>(),
            gcCount: 0,
            totalGcTime: TimeSpan.Zero,
            gcCountByGeneration: new ConcurrentDictionary<int, int>());
    }

    /// <summary>
    /// Gets memory events within a step range
    /// </summary>
    public IEnumerable<MemoryEvent> GetEvents(long startStep, long endStep)
    {
        return _eventHistory
            .Where(e => e.Timestamp.Ticks >= startStep && e.Timestamp.Ticks <= endStep)
            .OrderBy(e => e.Timestamp);
    }

    /// <summary>
    /// Gets allocations since a specific time
    /// </summary>
    public IEnumerable<MemoryEvent> GetAllocationsSince(DateTime startTime)
    {
        return _eventHistory
            .Where(e => e.MemoryEventType == MemoryEventType.Allocation && e.Timestamp >= startTime)
            .OrderBy(e => e.Timestamp);
    }

    /// <summary>
    /// Detects potential memory leaks by finding allocations without matching deallocations
    /// </summary>
    public List<(long address, long size, StackTrace? trace)> DetectPotentialLeaks()
    {
        lock (_lock)
        {
            var potentialLeaks = new List<(long address, long size, StackTrace? trace)>();

            foreach (var allocation in _activeAllocations.Values)
            {
                // Filter allocations that are older than 1 minute to avoid false positives
                if (DateTime.UtcNow - allocation.AllocationTime > TimeSpan.FromMinutes(1))
                {
                    potentialLeaks.Add((allocation.Address, allocation.SizeBytes, allocation.StackTrace));
                }
            }

            return potentialLeaks.OrderByDescending(l => l.size).ToList();
        }
    }

    /// <summary>
    /// Enables memory profiling
    /// </summary>
    public void Enable()
    {
        _isEnabled = true;
        if (_autoSnapshot)
        {
            StartSnapshotTimer();
        }
    }

    /// <summary>
    /// Disables memory profiling
    /// </summary>
    public void Disable()
    {
        _isEnabled = false;
        StopSnapshotTimer();
    }

    /// <summary>
    /// Records a GC event (called by GCMonitor)
    /// </summary>
    internal void RecordGCEvent(int generation, TimeSpan duration)
    {
        _statisticsBuilder.RecordGCEvent(generation, duration);
    }

    /// <summary>
    /// Captures a stack trace with limited depth
    /// </summary>
    private StackTrace CaptureStackTrace()
    {
        return new StackTrace(_maxStackTraceDepth, true);
    }

    /// <summary>
    /// Publishes a memory event to the event system
    /// </summary>
    private async void PublishEvent(MemoryEvent memoryEvent)
    {
        if (_eventCollector != null)
        {
            _eventCollector.Collect(memoryEvent);
        }

        if (_storage != null)
        {
            try
            {
                await _storage.StoreAsync(memoryEvent);
            }
            catch
            {
                // Silently ignore storage errors to avoid disrupting profiling
            }
        }
    }

    /// <summary>
    /// Starts the snapshot timer
    /// </summary>
    private void StartSnapshotTimer()
    {
        StopSnapshotTimer();
        _snapshotTimer = new Timer(_ => TrackSnapshot(), null, _snapshotIntervalMs, _snapshotIntervalMs);
    }

    /// <summary>
    /// Stops the snapshot timer
    /// </summary>
    private void StopSnapshotTimer()
    {
        _snapshotTimer?.Dispose();
        _snapshotTimer = null;
    }

    /// <summary>
    /// Restarts the snapshot timer with the current interval
    /// </summary>
    private void RestartSnapshotTimer()
    {
        if (_autoSnapshot && _isEnabled)
        {
            StartSnapshotTimer();
        }
    }

    /// <summary>
    /// Disposes resources
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        StopSnapshotTimer();
        _gcMonitor.Dispose();
    }

    /// <summary>
    /// Information about an allocation
    /// </summary>
    private class AllocationInfo
    {
        public long Address { get; set; }
        public long SizeBytes { get; set; }
        public string AllocationType { get; set; } = string.Empty;
        public DateTime AllocationTime { get; set; }
        public StackTrace? StackTrace { get; set; }
    }
}
