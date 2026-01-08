# Spec: Memory Profiler

## Overview
Implement memory profiling to track memory allocations, deallocations, and usage patterns throughout training, helping identify memory leaks and optimize memory usage.

## Objectives
- Track memory allocations and deallocations
- Monitor memory usage over time
- Identify memory leaks and fragmentation
- Provide insights into allocation patterns

## API Design

```csharp
// Memory event types
public enum MemoryEventType
{
    Allocation,      // Memory allocated
    Deallocation,    // Memory freed
    Reallocation,    // Memory resized
    Snapshot,        // Memory usage snapshot
    GCStart,         // Garbage collection started
    GCEnd            // Garbage collection ended
}

// Memory event
public class MemoryEvent : Event
{
    public MemoryEventType MemoryEventType { get; }
    public long Address { get; }
    public long SizeBytes { get; }
    public long TotalAllocatedBytes { get; }
    public long TotalFreeBytes { get; }
    public string AllocationType { get; } // e.g., "GPU", "CPU", "Pinned"

    // Stack trace (optional, for debugging)
    public StackTrace AllocationStackTrace { get; }
}

// Memory statistics
public class MemoryStatistics
{
    public long TotalAllocatedBytes { get; }
    public long TotalFreedBytes { get; }
    public long CurrentUsageBytes { get; }
    public long PeakUsageBytes { get; }

    public int AllocationCount { get; }
    public int DeallocationCount { get; }
    public long AverageAllocationSizeBytes { get; }

    // By allocation type
    public Dictionary<string, long> UsageByType { get; }

    // GC statistics
    public int GCCount { get; }
    public TimeSpan TotalGCTime { get; }
}

// Memory profiler interface
public interface IMemoryProfiler
{
    // Event tracking
    void TrackAllocation(long address, long sizeBytes, string allocationType);
    void TrackDeallocation(long address, long sizeBytes);
    void TrackSnapshot();

    // Statistics
    MemoryStatistics GetStatistics();
    MemoryStatistics GetStatisticsForType(string allocationType);

    // Timeline
    IEnumerable<MemoryEvent> GetEvents(long startStep, long endStep);
    IEnumerable<MemoryEvent> GetAllocationsSince(DateTime startTime);

    // Leak detection
    List<(long address, long size, StackTrace trace)> DetectPotentialLeaks();

    // Configuration
    void Enable();
    void Disable();
    bool IsEnabled { get; }

    // Stack trace capture
    bool CaptureStackTraces { get; set; }
}

public class MemoryProfiler : IMemoryProfiler
{
    public MemoryProfiler(IStorageBackend storage);
    public MemoryProfiler(IEventPublisher eventPublisher);

    // Configuration
    public int SnapshotIntervalMs { get; set; } = 1000;
    public bool AutoSnapshot { get; set; } = true;
    public int MaxStackTraceDepth { get; set; } = 10;
}
```

## Implementation Requirements

### 1. MemoryEvent (20-30 min)
- Implement `MemoryEvent` class inheriting from `Event`
- Include all memory event types
- Store address, size, and allocation type
- Track total allocated/free bytes at event time
- Optionally capture stack trace for debugging
- Add validation for memory sizes

### 2. Memory Statistics Computation (30-45 min)
- Implement `MemoryStatistics` class:
  - Track running totals (allocated, freed, current usage)
  - Track peak usage
  - Compute averages
  - Group by allocation type (GPU, CPU, Pinned)
- Compute GC statistics:
  - Count GC events
  - Sum GC duration
- Update statistics incrementally as events arrive
- Provide thread-safe access to statistics

### 3. MemoryProfiler Core (45-60 min)
- Implement `IMemoryProfiler` interface
- Maintain dictionary of active allocations:
  - Key: memory address
  - Value: size, type, allocation time, stack trace
- Track allocations and deallocations:
  - Update active allocations map
  - Update statistics
  - Detect mismatched allocations/deallocations
- Implement snapshot functionality:
  - Capture current memory state
  - Log as snapshot event
  - Support auto-snapshot on timer
- Implement leak detection:
  - Find allocations without matching deallocations
  - Report address, size, and stack trace
  - Filter by age (ignore recent allocations)
- Integrate with event system (publish `MemoryEvent`)
- Integrate with storage backend

### 4. GC Monitoring (30-45 min)
- Hook into .NET GC events:
  - Subscribe to `GCNotification` if available
  - Monitor `GC.CollectionCount` for changes
- Track GC start/end events:
  - Record timestamp
  - Record generation (0, 1, 2)
  - Calculate duration
- Provide GC statistics:
  - Total GC count
  - Total time spent in GC
  - GC count by generation
- Identify problematic GC patterns:
  - Frequent GC in generation 2
  - Long GC pauses
  - High GC time percentage

### 5. Memory Timeline (20-30 min)
- Store memory events with timestamps
- Provide query interface:
  - Get events by time range
  - Get events by step range
  - Get allocations since specific time
- Support efficient querying:
  - Use appropriate data structures (e.g., sorted list)
  - Implement time-based indexing
- Export timeline for visualization

## File Structure
```
src/
  MLFramework.Visualization/
    Memory/
      MemoryEventType.cs
      MemoryEvent.cs
      MemoryStatistics.cs
      IMemoryProfiler.cs
      MemoryProfiler.cs
      GCMonitor.cs

tests/
  MLFramework.Visualization.Tests/
    Memory/
      MemoryProfilerTests.cs
      MemoryStatisticsTests.cs
      GCMonitorTests.cs
```

## Dependencies
- `MLFramework.Visualization.Events` (Event system)
- `MLFramework.Visualization.Storage` (Storage backend)
- System.Diagnostics (for stack traces)

## Integration Points
- Used by tensor operations to track allocations
- Integrated with memory allocator
- Data consumed by memory usage visualization

## Success Criteria
- Tracking 1M allocations/deallocations completes in <500ms
- Memory overhead of profiling <5% of total memory
- Leak detection accurately identifies unfreed allocations
- Stack trace capture doesn't significantly slow down allocations
- GC monitoring correctly tracks all GC events
- Unit tests verify correctness of all functionality
