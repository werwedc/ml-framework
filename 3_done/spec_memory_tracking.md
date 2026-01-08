# Spec: Memory Tracking System

## Overview
Implement a comprehensive memory tracking system that monitors checkpoint memory usage, tracks peak memory, and provides detailed statistics and profiling information for activation checkpointing.

## Classes

### Location
`src/MLFramework/Checkpointing/MemoryTracking.cs`

### Class: MemoryTracker

```csharp
namespace MLFramework.Checkpointing;

/// <summary>
/// Tracks memory usage for checkpoints and provides statistics
/// </summary>
public class MemoryTracker : IDisposable
{
    /// <summary>
    /// Initializes a new instance of MemoryTracker
    /// </summary>
    public MemoryTracker();

    /// <summary>
    /// Records an allocation for a checkpoint
    /// </summary>
    /// <param name="layerId">Unique identifier for the layer</param>
    /// <param name="sizeBytes">Size of the allocation in bytes</param>
    public void RecordAllocation(string layerId, long sizeBytes);

    /// <summary>
    /// Records a deallocation for a checkpoint
    /// </summary>
    /// <param name="layerId">Unique identifier for the layer</param>
    public void RecordDeallocation(string layerId);

    /// <summary>
    /// Gets current memory usage in bytes
    /// </summary>
    public long CurrentMemoryUsage { get; }

    /// <summary>
    /// Gets peak memory usage in bytes
    /// </summary>
    public long PeakMemoryUsage { get; }

    /// <summary>
    /// Gets total memory allocated since tracker started
    /// </summary>
    public long TotalMemoryAllocated { get; }

    /// <summary>
    /// Gets total memory deallocated since tracker started
    /// </summary>
    public long TotalMemoryDeallocated { get; }

    /// <summary>
    /// Gets the number of active allocations
    /// </summary>
    public int ActiveAllocationCount { get; }

    /// <summary>
    /// Resets the tracker statistics (keeps current allocations)
    /// </summary>
    public void ResetStats();

    /// <summary>
    /// Gets detailed memory statistics
    /// </summary>
    /// <returns>MemoryStats with detailed information</returns>
    public MemoryStats GetStats();

    /// <summary>
    /// Gets memory statistics for a specific layer
    /// </summary>
    /// <param name="layerId">Unique identifier for the layer</param>
    /// <returns>LayerMemoryStats or null if not found</returns>
    public LayerMemoryStats? GetLayerStats(string layerId);

    /// <summary>
    /// Disposes the tracker and releases resources
    /// </summary>
    public void Dispose();
}
```

### Internal Data Structures

```csharp
private class AllocationEntry
{
    public string LayerId { get; set; }
    public long SizeBytes { get; set; }
    public DateTime AllocatedAt { get; set; }
    public long AllocationNumber { get; set; }
}

private readonly Dictionary<string, AllocationEntry> _activeAllocations;
private readonly Dictionary<string, LayerMemoryStats> _layerStats;
private readonly object _lock = new object();
private long _currentMemoryUsage;
private long _peakMemoryUsage;
private long _totalMemoryAllocated;
private long _totalMemoryDeallocated;
private long _allocationCounter;
```

## Implementation Details

### RecordAllocation

```csharp
public void RecordAllocation(string layerId, long sizeBytes)
{
    if (string.IsNullOrWhiteSpace(layerId))
        throw new ArgumentException("Layer ID cannot be null or whitespace");
    if (sizeBytes <= 0)
        throw new ArgumentException("Size must be greater than 0");

    lock (_lock)
    {
        // Create allocation entry
        var entry = new AllocationEntry
        {
            LayerId = layerId,
            SizeBytes = sizeBytes,
            AllocatedAt = DateTime.UtcNow,
            AllocationNumber = ++_allocationCounter
        };

        // Add to active allocations
        _activeAllocations[layerId] = entry;

        // Update memory counters
        _currentMemoryUsage += sizeBytes;
        _totalMemoryAllocated += sizeBytes;

        // Update peak if necessary
        if (_currentMemoryUsage > _peakMemoryUsage)
        {
            _peakMemoryUsage = _currentMemoryUsage;
        }

        // Update layer stats
        if (!_layerStats.ContainsKey(layerId))
        {
            _layerStats[layerId] = new LayerMemoryStats { LayerId = layerId };
        }

        var stats = _layerStats[layerId];
        stats.AllocationCount++;
        stats.TotalBytesAllocated += sizeBytes;
        stats.LastAllocatedAt = DateTime.UtcNow;

        if (sizeBytes > stats.MaxAllocationSize)
        {
            stats.MaxAllocationSize = sizeBytes;
        }
    }
}
```

### RecordDeallocation

```csharp
public void RecordDeallocation(string layerId)
{
    if (string.IsNullOrWhiteSpace(layerId))
        throw new ArgumentException("Layer ID cannot be null or whitespace");

    lock (_lock)
    {
        if (!_activeAllocations.TryGetValue(layerId, out var entry))
        {
            // Silent fail or throw? Choose based on requirements
            return;
        }

        // Remove from active allocations
        _activeAllocations.Remove(layerId);

        // Update memory counters
        _currentMemoryUsage -= entry.SizeBytes;
        _totalMemoryDeallocated += entry.SizeBytes;

        // Update layer stats
        if (_layerStats.TryGetValue(layerId, out var stats))
        {
            stats.DeallocationCount++;
            stats.TotalBytesDeallocated += entry.SizeBytes;
            stats.LastDeallocatedAt = DateTime.UtcNow;
        }
    }
}
```

### GetStats

```csharp
public MemoryStats GetStats()
{
    lock (_lock)
    {
        return new MemoryStats
        {
            CurrentMemoryUsed = _currentMemoryUsage,
            PeakMemoryUsed = _peakMemoryUsage,
            CheckpointCount = _activeAllocations.Count,
            TotalMemoryAllocated = _totalMemoryAllocated,
            TotalMemoryDeallocated = _totalMemoryDeallocated,
            AverageMemoryPerCheckpoint = _activeAllocations.Count > 0
                ? _currentMemoryUsage / _activeAllocations.Count
                : 0,
            AllocationCount = (int)_allocationCounter,
            DeallocationCount = (int)(_allocationCounter - _activeAllocations.Count),
            Timestamp = DateTime.UtcNow
        };
    }
}
```

### GetLayerStats

```csharp
public LayerMemoryStats? GetLayerStats(string layerId)
{
    lock (_lock)
    {
        return _layerStats.TryGetValue(layerId, out var stats) ? stats : null;
    }
}
```

## Enhanced MemoryStats Class

```csharp
namespace MLFramework.Checkpointing;

/// <summary>
/// Detailed memory statistics for checkpoints
/// </summary>
public class MemoryStats
{
    /// <summary>
    /// Total memory currently used by checkpoints (in bytes)
    /// </summary>
    public long CurrentMemoryUsed { get; set; }

    /// <summary>
    /// Peak memory used since tracking started (in bytes)
    /// </summary>
    public long PeakMemoryUsed { get; set; }

    /// <summary>
    /// Number of checkpoints currently stored
    /// </summary>
    public int CheckpointCount { get; set; }

    /// <summary>
    /// Average memory per checkpoint (in bytes)
    /// </summary>
    public long AverageMemoryPerCheckpoint { get; set; }

    /// <summary>
    /// Total memory allocated since tracking started (in bytes)
    /// </summary>
    public long TotalMemoryAllocated { get; set; }

    /// <summary>
    /// Total memory deallocated since tracking started (in bytes)
    /// </summary>
    public long TotalMemoryDeallocated { get; set; }

    /// <summary>
    /// Total number of allocations
    /// </summary>
    public int AllocationCount { get; set; }

    /// <summary>
    /// Total number of deallocations
    /// </summary>
    public int DeallocationCount { get; set; }

    /// <summary>
    /// Timestamp when stats were collected
    /// </summary>
    public DateTime Timestamp { get; set; }

    /// <summary>
    /// Gets memory savings compared to storing all activations (in bytes)
    /// </summary>
    /// <param name="totalActivationSize">Total size of all activations if stored</param>
    /// <returns>Memory savings in bytes</returns>
    public long CalculateMemorySavings(long totalActivationSize)
    {
        return totalActivationSize - CurrentMemoryUsed;
    }

    /// <summary>
    /// Gets memory reduction percentage
    /// </summary>
    /// <param name="totalActivationSize">Total size of all activations if stored</param>
    /// <returns>Memory reduction percentage (0.0 to 1.0)</returns>
    public float CalculateMemoryReductionPercentage(long totalActivationSize)
    {
        if (totalActivationSize == 0)
            return 0f;
        return (float)CalculateMemorySavings(totalActivationSize) / totalActivationSize;
    }
}
```

## LayerMemoryStats Class

```csharp
namespace MLFramework.Checkpointing;

/// <summary>
/// Memory statistics for a specific layer
/// </summary>
public class LayerMemoryStats
{
    /// <summary>
    /// Unique identifier for the layer
    /// </summary>
    public string LayerId { get; set; } = string.Empty;

    /// <summary>
    /// Number of allocations for this layer
    /// </summary>
    public int AllocationCount { get; set; }

    /// <summary>
    /// Number of deallocations for this layer
    /// </summary>
    public int DeallocationCount { get; set; }

    /// <summary>
    /// Total bytes allocated for this layer
    /// </summary>
    public long TotalBytesAllocated { get; set; }

    /// <summary>
    /// Total bytes deallocated for this layer
    /// </summary>
    public long TotalBytesDeallocated { get; set; }

    /// <summary>
    /// Maximum allocation size for this layer
    /// </summary>
    public long MaxAllocationSize { get; set; }

    /// <summary>
    /// Average allocation size for this layer
    /// </summary>
    public long AverageAllocationSize =>
        AllocationCount > 0 ? TotalBytesAllocated / AllocationCount : 0;

    /// <summary>
    /// Timestamp of last allocation
    /// </summary>
    public DateTime LastAllocatedAt { get; set; }

    /// <summary>
    /// Timestamp of last deallocation
    /// </summary>
    public DateTime LastDeallocatedAt { get; set; }

    /// <summary>
    /// Whether this layer is currently allocated
    /// </summary>
    public bool IsCurrentlyAllocated =>
        AllocationCount > DeallocationCount;
}
```

## Memory Event System

### Class: MemoryEventArgs

```csharp
namespace MLFramework.Checkpointing;

public class MemoryEventArgs : EventArgs
{
    public string LayerId { get; set; } = string.Empty;
    public long SizeBytes { get; set; }
    public long CurrentMemoryUsage { get; set; }
    public long PeakMemoryUsage { get; set; }
    public DateTime Timestamp { get; set; }
}

public class MemoryLimitExceededEventArgs : EventArgs
{
    public long CurrentMemoryUsage { get; set; }
    public long MemoryLimit { get; set; }
    public DateTime Timestamp { get; set; }
}
```

### Extended MemoryTracker with Events

```csharp
public class MemoryTracker : IDisposable
{
    // ... existing code ...

    /// <summary>
    /// Event raised when memory allocation occurs
    /// </summary>
    public event EventHandler<MemoryEventArgs>? MemoryAllocated;

    /// <summary>
    /// Event raised when memory deallocation occurs
    /// </summary>
    public event EventHandler<MemoryEventArgs>? MemoryDeallocated;

    /// <summary>
    /// Event raised when peak memory is exceeded
    /// </summary>
    public event EventHandler<MemoryEventArgs>? PeakMemoryExceeded;

    /// <summary>
    /// Sets a memory limit threshold
    /// </summary>
    /// <param name="limitBytes">Memory limit in bytes</param>
    public void SetMemoryLimit(long limitBytes);

    /// <summary>
    /// Gets the current memory limit
    /// </summary>
    public long? MemoryLimit { get; }

    /// <summary>
    /// Event raised when memory limit is exceeded
    /// </summary>
    public event EventHandler<MemoryLimitExceededEventArgs>? MemoryLimitExceeded;

    private long? _memoryLimit;
}
```

## Testing Requirements

### Unit Tests

1. **MemoryTracker Basic Operations Tests**
   - [ ] Successfully record allocation
   - [ ] Successfully record deallocation
   - [ ] Correctly track current memory usage
   - [ ] Correctly track peak memory usage
   - [ ] Correctly update total allocated/deallocated counters
   - [ ] Handle concurrent allocations/deallocations safely

2. **MemoryStats Tests**
   - [ ] Calculate correct average memory per checkpoint
   - [ ] Calculate correct memory savings
   - [ ] Calculate correct memory reduction percentage
   - [ ] Handle zero total activation size gracefully

3. **LayerMemoryStats Tests**
   - [ ] Correctly track allocation/deallocation counts per layer
   - [ ] Correctly track max allocation size
   - [ ] Calculate correct average allocation size
   - [ ] Determine IsCurrentlyAllocated correctly

4. **Event Tests**
   - [ ] MemoryAllocated event fires on allocation
   - [ ] MemoryDeallocated event fires on deallocation
   - [ ] PeakMemoryExceeded event fires on new peak
   - [ ] MemoryLimitExceeded event fires when limit exceeded
   - [ ] Events contain correct data

5. **Memory Limit Tests**
   - [ ] Set and get memory limit
   - [ ] Trigger MemoryLimitExceeded event
   - [ ] Handle null memory limit (no limit)

6. **Edge Cases**
   - [ ] Handle zero-size allocations (should throw)
   - [ ] Handle negative size allocations (should throw)
   - [ ] Handle deallocation of non-existent layer
   - [ ] Handle very large memory values
   - [ ] Handle rapid allocations/deallocations

7. **Thread Safety Tests**
   - [ ] Multiple threads can allocate concurrently
   - [ ] Multiple threads can deallocate concurrently
   - [ ] Stats remain consistent under concurrent access

## Implementation Notes

1. **Thread Safety**:
   - All public methods should be thread-safe
   - Use lock for consistency
   - Consider lock-free approaches for high-performance scenarios

2. **Memory Overhead**:
   - Keep tracking overhead minimal
   - Use efficient data structures (Dictionary)
   - Consider pooling AllocationEntry objects

3. **Event Performance**:
   - Events should not significantly impact performance
   - Consider using value types for event args
   - Allow enabling/disabling events

4. **Disposal**:
   - Clear all dictionaries on disposal
   - Unsubscribe from events
   - Release resources properly

## Dependencies on Other Specs

This spec extends the MemoryStats class mentioned in:
- **Checkpoint Manager Core** (spec_1)

This spec is independent and can be implemented in parallel with other specs.

## Estimated Implementation Time
45-60 minutes
