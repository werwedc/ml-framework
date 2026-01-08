using System.Diagnostics;
using MachineLearning.Visualization.Collection;
using MachineLearning.Visualization.Storage;

namespace MachineLearning.Visualization.Memory;

/// <summary>
/// Monitors .NET garbage collection events and records statistics
/// </summary>
public class GCMonitor : IDisposable
{
    private readonly MemoryProfiler _memoryProfiler;
    private readonly IStorageBackend? _storage;
    private readonly IEventCollector? _eventCollector;
    private readonly Timer _monitorTimer;
    private readonly object _lock = new object();

    private readonly int[] _lastGcCounts = new int[3];
    private readonly DateTime?[] _lastGcStartTimes = new DateTime?[3];

    private bool _disposed = false;
    private const int MonitorIntervalMs = 100;

    /// <summary>
    /// Creates a new GCMonitor for a MemoryProfiler
    /// </summary>
    public GCMonitor(MemoryProfiler memoryProfiler)
    {
        _memoryProfiler = memoryProfiler ?? throw new ArgumentNullException(nameof(memoryProfiler));
        _storage = null;
        _eventCollector = null;

        // Initialize GC counts
        for (int i = 0; i < 3; i++)
        {
            _lastGcCounts[i] = GC.CollectionCount(i);
        }

        // Start monitoring timer
        _monitorTimer = new Timer(MonitorGC, null, MonitorIntervalMs, MonitorIntervalMs);
    }

    /// <summary>
    /// Creates a new GCMonitor for a MemoryProfiler with a storage backend
    /// </summary>
    public GCMonitor(MemoryProfiler memoryProfiler, IStorageBackend storage) : this(memoryProfiler)
    {
        _storage = storage ?? throw new ArgumentNullException(nameof(storage));
    }

    /// <summary>
    /// Creates a new GCMonitor for a MemoryProfiler with an event collector
    /// </summary>
    public GCMonitor(MemoryProfiler memoryProfiler, IEventCollector eventCollector) : this(memoryProfiler)
    {
        _eventCollector = eventCollector ?? throw new ArgumentNullException(nameof(eventCollector));
    }

    /// <summary>
    /// Monitors GC activity and records events
    /// </summary>
    private void MonitorGC(object? state)
    {
        if (_disposed)
        {
            return;
        }

        lock (_lock)
        {
            for (int generation = 0; generation < 3; generation++)
            {
                var currentCount = GC.CollectionCount(generation);

                if (currentCount > _lastGcCounts[generation])
                {
                    // GC occurred for this generation
                    var gcStart = _lastGcStartTimes[generation] ?? DateTime.UtcNow;
                    var gcEnd = DateTime.UtcNow;
                    var duration = gcEnd - gcStart;

                    // Record the GC event
                    _memoryProfiler.RecordGCEvent(generation, duration);

                    // Create and publish GC start event
                    var gcStartEvent = CreateGCEvent(MemoryEventType.GCStart, generation);
                    PublishEvent(gcStartEvent);

                    // Create and publish GC end event
                    var gcEndEvent = CreateGCEvent(MemoryEventType.GCEnd, generation);
                    PublishEvent(gcEndEvent);

                    // Update the last count
                    _lastGcCounts[generation] = currentCount;
                    _lastGcStartTimes[generation] = null;
                }
                else if (_lastGcStartTimes[generation] == null)
                {
                    // Track when a new GC cycle might be starting
                    _lastGcStartTimes[generation] = DateTime.UtcNow;
                }
            }
        }
    }

    /// <summary>
    /// Creates a GC event
    /// </summary>
    private MemoryEvent CreateGCEvent(MemoryEventType eventType, int generation)
    {
        var statistics = _memoryProfiler.GetStatistics();

        return new MemoryEvent(
            eventType,
            0,
            0,
            statistics.TotalAllocatedBytes,
            statistics.TotalFreedBytes,
            "GC",
            null,
            generation);
    }

    /// <summary>
    /// Publishes a GC event to the event system
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
                // Silently ignore storage errors to avoid disrupting GC monitoring
            }
        }
    }

    /// <summary>
    /// Gets the current GC statistics
    /// </summary>
    public GCStatistics GetStatistics()
    {
        var memoryStats = _memoryProfiler.GetStatistics();

        var gen0Count = GC.CollectionCount(0);
        var gen1Count = GC.CollectionCount(1);
        var gen2Count = GC.CollectionCount(2);

        return new GCStatistics
        {
            TotalGCCount = memoryStats.GCCount,
            TotalGCTime = memoryStats.TotalGCTime,
            Gen0GCCount = gen0Count,
            Gen1GCCount = gen1Count,
            Gen2GCCount = gen2Count,
            LastGCTime = DateTime.UtcNow
        };
    }

    /// <summary>
    /// Gets GC count for a specific generation
    /// </summary>
    public int GetGCCount(int generation)
    {
        if (generation < 0 || generation > 2)
        {
            throw new ArgumentOutOfRangeException(nameof(generation), "Generation must be 0, 1, or 2");
        }

        return GC.CollectionCount(generation);
    }

    /// <summary>
    /// Gets the total memory currently allocated (managed heap)
    /// </summary>
    public long GetTotalMemory()
    {
        return GC.GetTotalMemory(false);
    }

    /// <summary>
    /// Gets the total memory including garbage collection
    /// </summary>
    public long GetTotalMemoryWithGC()
    {
        return GC.GetTotalMemory(true);
    }

    /// <summary>
    /// Forces a garbage collection of all generations
    /// </summary>
    public void ForceFullCollection()
    {
        GC.Collect();
        GC.WaitForPendingFinalizers();
    }

    /// <summary>
    /// Forces a garbage collection of a specific generation
    /// </summary>
    public void ForceCollection(int generation)
    {
        if (generation < 0 || generation > 2)
        {
            throw new ArgumentOutOfRangeException(nameof(generation), "Generation must be 0, 1, or 2");
        }

        GC.Collect(generation);
        GC.WaitForPendingFinalizers();
    }

    /// <summary>
    /// Checks if the current GC mode is Server GC
    /// </summary>
    public bool IsServerGC => GCSettings.IsServerGC;

    /// <summary>
    /// Gets or sets whether large object heap compaction is enabled
    /// </summary>
    public bool LargeObjectHeapCompaction
    {
        get => GCSettings.LargeObjectHeapCompactionMode == GCLargeObjectHeapCompactionMode.CompactOnce;
        set => GCSettings.LargeObjectHeapCompactionMode = value ?
            GCLargeObjectHeapCompactionMode.CompactOnce :
            GCLargeObjectHeapCompactionMode.CompactNone;
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
        _monitorTimer?.Dispose();
    }
}

/// <summary>
/// GC statistics
/// </summary>
public class GCStatistics
{
    /// <summary>
    /// Total number of GC collections
    /// </summary>
    public int TotalGCCount { get; set; }

    /// <summary>
    /// Total time spent in GC
    /// </summary>
    public TimeSpan TotalGCTime { get; set; }

    /// <summary>
    /// Number of generation 0 GC collections
    /// </summary>
    public int Gen0GCCount { get; set; }

    /// <summary>
    /// Number of generation 1 GC collections
    /// </summary>
    public int Gen1GCCount { get; set; }

    /// <summary>
    /// Number of generation 2 GC collections
    /// </summary>
    public int Gen2GCCount { get; set; }

    /// <summary>
    /// Timestamp of the last GC
    /// </summary>
    public DateTime LastGCTime { get; set; }
}
