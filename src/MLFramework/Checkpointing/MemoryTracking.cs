using System;

namespace MLFramework.Checkpointing;

/// <summary>
/// Event arguments for memory events
/// </summary>
public class MemoryEventArgs : EventArgs
{
    public string LayerId { get; set; } = string.Empty;
    public long SizeBytes { get; set; }
    public long CurrentMemoryUsage { get; set; }
    public long PeakMemoryUsage { get; set; }
    public DateTime Timestamp { get; set; }
}

/// <summary>
/// Event arguments for memory limit exceeded events
/// </summary>
public class MemoryLimitExceededEventArgs : EventArgs
{
    public long CurrentMemoryUsage { get; set; }
    public long MemoryLimit { get; set; }
    public DateTime Timestamp { get; set; }
}

/// <summary>
/// Tracks memory usage for checkpoints and provides statistics
/// </summary>
public class MemoryTracker : IDisposable
{
    private class AllocationEntry
    {
        public string LayerId { get; set; } = string.Empty;
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
    private long? _memoryLimit;
    private bool _disposed;

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
    /// Event raised when memory limit is exceeded
    /// </summary>
    public event EventHandler<MemoryLimitExceededEventArgs>? MemoryLimitExceeded;

    /// <summary>
    /// Initializes a new instance of MemoryTracker
    /// </summary>
    public MemoryTracker()
    {
        _activeAllocations = new Dictionary<string, AllocationEntry>();
        _layerStats = new Dictionary<string, LayerMemoryStats>();
        _currentMemoryUsage = 0;
        _peakMemoryUsage = 0;
        _totalMemoryAllocated = 0;
        _totalMemoryDeallocated = 0;
        _allocationCounter = 0;
        _memoryLimit = null;
        _disposed = false;
    }

    /// <summary>
    /// Records an allocation for a checkpoint
    /// </summary>
    /// <param name="layerId">Unique identifier for the layer</param>
    /// <param name="sizeBytes">Size of the allocation in bytes</param>
    public void RecordAllocation(string layerId, long sizeBytes)
    {
        ThrowIfDisposed();

        if (string.IsNullOrWhiteSpace(layerId))
            throw new ArgumentException("Layer ID cannot be null or whitespace", nameof(layerId));
        if (sizeBytes <= 0)
            throw new ArgumentException("Size must be greater than 0", nameof(sizeBytes));

        MemoryEventArgs? eventArgs = null;

        lock (_lock)
        {
            var oldPeak = _peakMemoryUsage;

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

            // Check for memory limit
            if (_memoryLimit.HasValue && _currentMemoryUsage > _memoryLimit.Value)
            {
                OnMemoryLimitExceeded(new MemoryLimitExceededEventArgs
                {
                    CurrentMemoryUsage = _currentMemoryUsage,
                    MemoryLimit = _memoryLimit.Value,
                    Timestamp = DateTime.UtcNow
                });
            }

            // Prepare event args
            eventArgs = new MemoryEventArgs
            {
                LayerId = layerId,
                SizeBytes = sizeBytes,
                CurrentMemoryUsage = _currentMemoryUsage,
                PeakMemoryUsage = _peakMemoryUsage,
                Timestamp = DateTime.UtcNow
            };
        }

        // Fire event outside lock
        OnMemoryAllocated(eventArgs);

        // Fire peak memory exceeded event if peak changed
        lock (_lock)
        {
            if (_peakMemoryUsage > _currentMemoryUsage - sizeBytes) // Peak was updated
            {
                OnPeakMemoryExceeded(eventArgs);
            }
        }
    }

    /// <summary>
    /// Records a deallocation for a checkpoint
    /// </summary>
    /// <param name="layerId">Unique identifier for the layer</param>
    public void RecordDeallocation(string layerId)
    {
        ThrowIfDisposed();

        if (string.IsNullOrWhiteSpace(layerId))
            throw new ArgumentException("Layer ID cannot be null or whitespace", nameof(layerId));

        MemoryEventArgs? eventArgs = null;

        lock (_lock)
        {
            if (!_activeAllocations.TryGetValue(layerId, out var entry))
            {
                // Silent fail - layer not allocated
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

            // Prepare event args
            eventArgs = new MemoryEventArgs
            {
                LayerId = layerId,
                SizeBytes = entry.SizeBytes,
                CurrentMemoryUsage = _currentMemoryUsage,
                PeakMemoryUsage = _peakMemoryUsage,
                Timestamp = DateTime.UtcNow
            };
        }

        // Fire event outside lock
        OnMemoryDeallocated(eventArgs);
    }

    /// <summary>
    /// Gets current memory usage in bytes
    /// </summary>
    public long CurrentMemoryUsage
    {
        get
        {
            ThrowIfDisposed();
            lock (_lock)
            {
                return _currentMemoryUsage;
            }
        }
    }

    /// <summary>
    /// Gets peak memory usage in bytes
    /// </summary>
    public long PeakMemoryUsage
    {
        get
        {
            ThrowIfDisposed();
            lock (_lock)
            {
                return _peakMemoryUsage;
            }
        }
    }

    /// <summary>
    /// Gets total memory allocated since tracker started
    /// </summary>
    public long TotalMemoryAllocated
    {
        get
        {
            ThrowIfDisposed();
            lock (_lock)
            {
                return _totalMemoryAllocated;
            }
        }
    }

    /// <summary>
    /// Gets total memory deallocated since tracker started
    /// </summary>
    public long TotalMemoryDeallocated
    {
        get
        {
            ThrowIfDisposed();
            lock (_lock)
            {
                return _totalMemoryDeallocated;
            }
        }
    }

    /// <summary>
    /// Gets the number of active allocations
    /// </summary>
    public int ActiveAllocationCount
    {
        get
        {
            ThrowIfDisposed();
            lock (_lock)
            {
                return _activeAllocations.Count;
            }
        }
    }

    /// <summary>
    /// Gets the current memory limit
    /// </summary>
    public long? MemoryLimit
    {
        get
        {
            ThrowIfDisposed();
            return _memoryLimit;
        }
    }

    /// <summary>
    /// Sets a memory limit threshold
    /// </summary>
    /// <param name="limitBytes">Memory limit in bytes</param>
    public void SetMemoryLimit(long limitBytes)
    {
        ThrowIfDisposed();

        if (limitBytes <= 0)
            throw new ArgumentException("Memory limit must be greater than 0", nameof(limitBytes));

        lock (_lock)
        {
            _memoryLimit = limitBytes;
        }
    }

    /// <summary>
    /// Resets the tracker statistics (keeps current allocations)
    /// </summary>
    public void ResetStats()
    {
        ThrowIfDisposed();

        lock (_lock)
        {
            _peakMemoryUsage = _currentMemoryUsage; // Reset peak to current
            _totalMemoryAllocated = 0;
            _totalMemoryDeallocated = 0;
            _allocationCounter = 0;

            foreach (var stats in _layerStats.Values)
            {
                stats.AllocationCount = 0;
                stats.DeallocationCount = 0;
                stats.TotalBytesAllocated = 0;
                stats.TotalBytesDeallocated = 0;
                stats.MaxAllocationSize = 0;
            }
        }
    }

    /// <summary>
    /// Gets detailed memory statistics
    /// </summary>
    /// <returns>MemoryStats with detailed information</returns>
    public MemoryStats GetStats()
    {
        ThrowIfDisposed();

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

    /// <summary>
    /// Gets memory statistics for a specific layer
    /// </summary>
    /// <param name="layerId">Unique identifier for the layer</param>
    /// <returns>LayerMemoryStats or null if not found</returns>
    public LayerMemoryStats? GetLayerStats(string layerId)
    {
        ThrowIfDisposed();

        if (string.IsNullOrWhiteSpace(layerId))
            return null;

        lock (_lock)
        {
            return _layerStats.TryGetValue(layerId, out var stats) ? stats : null;
        }
    }

    /// <summary>
    /// Disposes the tracker and releases resources
    /// </summary>
    public void Dispose()
    {
        if (!_disposed)
        {
            lock (_lock)
            {
                _activeAllocations.Clear();
                _layerStats.Clear();
            }
            _disposed = true;
        }
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(MemoryTracker));
    }

    protected virtual void OnMemoryAllocated(MemoryEventArgs e)
    {
        MemoryAllocated?.Invoke(this, e);
    }

    protected virtual void OnMemoryDeallocated(MemoryEventArgs e)
    {
        MemoryDeallocated?.Invoke(this, e);
    }

    protected virtual void OnPeakMemoryExceeded(MemoryEventArgs e)
    {
        PeakMemoryExceeded?.Invoke(this, e);
    }

    protected virtual void OnMemoryLimitExceeded(MemoryLimitExceededEventArgs e)
    {
        MemoryLimitExceeded?.Invoke(this, e);
    }
}

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
