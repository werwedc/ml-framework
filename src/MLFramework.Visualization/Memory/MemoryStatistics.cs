using System.Collections.Concurrent;

namespace MachineLearning.Visualization.Memory;

/// <summary>
/// Represents memory statistics computed from memory events
/// </summary>
public class MemoryStatistics
{
    /// <summary>
    /// Gets the total allocated bytes
    /// </summary>
    public long TotalAllocatedBytes { get; }

    /// <summary>
    /// Gets the total freed bytes
    /// </summary>
    public long TotalFreedBytes { get; }

    /// <summary>
    /// Gets the current usage in bytes
    /// </summary>
    public long CurrentUsageBytes { get; }

    /// <summary>
    /// Gets the peak usage in bytes
    /// </summary>
    public long PeakUsageBytes { get; }

    /// <summary>
    /// Gets the total number of allocations
    /// </summary>
    public int AllocationCount { get; }

    /// <summary>
    /// Gets the total number of deallocations
    /// </summary>
    public int DeallocationCount { get; }

    /// <summary>
    /// Gets the average allocation size in bytes
    /// </summary>
    public long AverageAllocationSizeBytes { get; }

    /// <summary>
    /// Gets the memory usage by allocation type
    /// </summary>
    public ConcurrentDictionary<string, long> UsageByType { get; }

    /// <summary>
    /// Gets the total GC count
    /// </summary>
    public int GCCount { get; }

    /// <summary>
    /// Gets the total time spent in GC
    /// </summary>
    public TimeSpan TotalGCTime { get; }

    /// <summary>
    /// Gets the GC count by generation
    /// </summary>
    public ConcurrentDictionary<int, int> GCCountByGeneration { get; }

    /// <summary>
    /// Private constructor for building statistics
    /// </summary>
    private MemoryStatistics()
    {
        UsageByType = new ConcurrentDictionary<string, long>();
        GCCountByGeneration = new ConcurrentDictionary<int, int>();
    }

    /// <summary>
    /// Creates a new MemoryStatistics instance
    /// </summary>
    public MemoryStatistics(
        long totalAllocatedBytes,
        long totalFreedBytes,
        long currentUsageBytes,
        long peakUsageBytes,
        int allocationCount,
        int deallocationCount,
        long averageAllocationSizeBytes,
        ConcurrentDictionary<string, long> usageByType,
        int gcCount,
        TimeSpan totalGcTime,
        ConcurrentDictionary<int, int> gcCountByGeneration)
    {
        TotalAllocatedBytes = totalAllocatedBytes;
        TotalFreedBytes = totalFreedBytes;
        CurrentUsageBytes = currentUsageBytes;
        PeakUsageBytes = peakUsageBytes;
        AllocationCount = allocationCount;
        DeallocationCount = deallocationCount;
        AverageAllocationSizeBytes = averageAllocationSizeBytes;
        UsageByType = usageByType ?? new ConcurrentDictionary<string, long>();
        GCCount = gcCount;
        TotalGCTime = totalGcTime;
        GCCountByGeneration = gcCountByGeneration ?? new ConcurrentDictionary<int, int>();
    }

    /// <summary>
    /// Creates a builder for incrementally building memory statistics
    /// </summary>
    public static Builder CreateBuilder()
    {
        return new Builder();
    }

    /// <summary>
    /// Builder class for incrementally building memory statistics
    /// </summary>
    public class Builder
    {
        private readonly object _lock = new object();
        private long _totalAllocatedBytes;
        private long _totalFreedBytes;
        private long _peakUsageBytes;
        private int _allocationCount;
        private int _deallocationCount;
        private int _gcCount;
        private TimeSpan _totalGcTime;
        private readonly ConcurrentDictionary<string, long> _usageByType = new ConcurrentDictionary<string, long>();
        private readonly ConcurrentDictionary<int, int> _gcCountByGeneration = new ConcurrentDictionary<int, int>();

        /// <summary>
        /// Gets the current usage in bytes
        /// </summary>
        public long CurrentUsageBytes => Interlocked.Read(ref _totalAllocatedBytes) - Interlocked.Read(ref _totalFreedBytes);

        /// <summary>
        /// Records an allocation event
        /// </summary>
        public void RecordAllocation(long sizeBytes, string allocationType)
        {
            Interlocked.Add(ref _totalAllocatedBytes, sizeBytes);
            Interlocked.Increment(ref _allocationCount);

            lock (_lock)
            {
                var currentUsage = CurrentUsageBytes;
                if (currentUsage > _peakUsageBytes)
                {
                    _peakUsageBytes = currentUsage;
                }
            }

            _usageByType.AddOrUpdate(allocationType, sizeBytes, (_, existing) => existing + sizeBytes);
        }

        /// <summary>
        /// Records a deallocation event
        /// </summary>
        public void RecordDeallocation(long sizeBytes, string allocationType)
        {
            Interlocked.Add(ref _totalFreedBytes, sizeBytes);
            Interlocked.Increment(ref _deallocationCount);

            _usageByType.AddOrUpdate(allocationType, -sizeBytes, (_, existing) => existing - sizeBytes);
        }

        /// <summary>
        /// Records a GC event
        /// </summary>
        public void RecordGCEvent(int generation, TimeSpan duration)
        {
            Interlocked.Increment(ref _gcCount);
            Interlocked.Exchange(ref _totalGcTime, _totalGcTime + duration);

            _gcCountByGeneration.AddOrUpdate(generation, 1, (_, existing) => existing + 1);
        }

        /// <summary>
        /// Builds the final statistics
        /// </summary>
        public MemoryStatistics Build()
        {
            var averageAllocationSize = _allocationCount > 0
                ? _totalAllocatedBytes / _allocationCount
                : 0;

            return new MemoryStatistics(
                _totalAllocatedBytes,
                _totalFreedBytes,
                CurrentUsageBytes,
                _peakUsageBytes,
                _allocationCount,
                _deallocationCount,
                averageAllocationSize,
                new ConcurrentDictionary<string, long>(_usageByType),
                _gcCount,
                _totalGcTime,
                new ConcurrentDictionary<int, int>(_gcCountByGeneration));
        }
    }
}
