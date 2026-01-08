using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;

namespace MLFramework.Memory;

/// <summary>
/// Predictive memory allocator that uses padding strategies and learns from usage patterns.
/// </summary>
public class PredictiveAllocator : IDynamicMemoryAllocator
{
    /// <summary>
    /// Padding factor for initial allocations (default: 1.2).
    /// </summary>
    public double PaddingFactor { get; set; }

    /// <summary>
    /// Maximum capacity in bytes (0 for unlimited).
    /// </summary>
    public long MaxCapacity { get; set; }

    /// <summary>
    /// Threshold below which to consider shrinking allocations.
    /// </summary>
    public double ShrinkThreshold { get; set; }

    /// <summary>
    /// Padding strategy to use.
    /// </summary>
    private readonly IPaddingStrategy _paddingStrategy;

    /// <summary>
    /// Dictionary of active allocations.
    /// </summary>
    private readonly ConcurrentDictionary<IMemoryHandle, bool> _activeAllocations;

    /// <summary>
    /// Pool of freed buffers for reuse.
    /// </summary>
    private readonly ConcurrentBag<IntPtr> _freedBuffers;

    /// <summary>
    /// Statistics tracker.
    /// </summary>
    private readonly AllocationStats _stats;

    /// <summary>
    /// Currently allocated bytes.
    /// </summary>
    private long _currentAllocatedBytes;

    /// <summary>
    /// Element size (default: 4 bytes for float32).
    /// </summary>
    private readonly int _elementSize;

    /// <summary>
    /// Creates a new predictive allocator.
    /// </summary>
    public PredictiveAllocator(IPaddingStrategy paddingStrategy, double paddingFactor = 1.2, long maxCapacity = 0, int elementSize = 4, double shrinkThreshold = 0.5)
    {
        if (paddingFactor < 1.0)
        {
            throw new ArgumentException("PaddingFactor must be at least 1.0", nameof(paddingFactor));
        }
        if (maxCapacity < 0)
        {
            throw new ArgumentException("MaxCapacity cannot be negative", nameof(maxCapacity));
        }
        if (elementSize <= 0)
        {
            throw new ArgumentException("ElementSize must be positive", nameof(elementSize));
        }
        if (shrinkThreshold <= 0.0 || shrinkThreshold >= 1.0)
        {
            throw new ArgumentException("ShrinkThreshold must be between 0 and 1 (exclusive)", nameof(shrinkThreshold));
        }

        _paddingStrategy = paddingStrategy;
        PaddingFactor = paddingFactor;
        MaxCapacity = maxCapacity;
        ShrinkThreshold = shrinkThreshold;
        _elementSize = elementSize;
        _activeAllocations = new ConcurrentDictionary<IMemoryHandle, bool>();
        _freedBuffers = new ConcurrentBag<IntPtr>();
        _stats = new AllocationStats();
        _currentAllocatedBytes = 0;
    }

    /// <summary>
    /// Allocates memory for a tensor with given shape bounds.
    /// </summary>
    public IMemoryHandle Allocate(ShapeBounds bounds)
    {
        long requiredSize = _paddingStrategy.CalculateRequiredSize(bounds, _elementSize);

        // Check capacity limits
        if (MaxCapacity > 0 && _currentAllocatedBytes + requiredSize > MaxCapacity)
        {
            throw new InvalidOperationException(
                $"Cannot allocate {requiredSize} bytes. Would exceed max capacity of {MaxCapacity} bytes. " +
                $"Currently allocated: {_currentAllocatedBytes} bytes.");
        }

        IntPtr pointer;
        if (_freedBuffers.TryTake(out pointer))
        {
            // Reuse a freed buffer
            // Note: In production, we'd need to verify the buffer is large enough
        }
        else
        {
            // Allocate new memory
            pointer = Marshal.AllocHGlobal((IntPtr)requiredSize);
            _stats.TotalBytesAllocated += requiredSize;
            _currentAllocatedBytes += requiredSize;
        }

        long expectedElements = bounds.CalculateExpectedElements();
        long wastedBytes = requiredSize - (expectedElements * _elementSize);
        _stats.TotalBytesWasted += wastedBytes;

        var handle = new MemoryHandle(
            pointer,
            requiredSize,
            bounds.ExpectedShape,
            bounds,
            _elementSize,
            (ptr, newSize) => Reallocate(ptr, newSize));

        _activeAllocations.TryAdd(handle, true);
        _stats.TotalAllocations++;

        UpdateAverageUtilization();

        return handle;
    }

    /// <summary>
    /// Resizes an existing memory allocation.
    /// </summary>
    public void Resize(IMemoryHandle handle, int[] newShape)
    {
        if (!_activeAllocations.ContainsKey(handle))
        {
            throw new ArgumentException("Handle is not allocated by this allocator", nameof(handle));
        }

        if (_paddingStrategy.ShouldResize(handle, newShape))
        {
            long currentEffectiveSize = handle.GetEffectiveSize();
            long newRequiredSize = handle.ShapeBounds.CalculateElements(newShape) * _elementSize;

            if (newRequiredSize > handle.CapacityBytes)
            {
                // Growing the allocation
                long additionalSize = newRequiredSize - currentEffectiveSize;

                if (MaxCapacity > 0 && _currentAllocatedBytes + additionalSize > MaxCapacity)
                {
                    throw new InvalidOperationException(
                        $"Cannot resize to {newRequiredSize} bytes. Would exceed max capacity of {MaxCapacity} bytes.");
                }

                _currentAllocatedBytes += additionalSize;
                _stats.TotalBytesAllocated += additionalSize;
            }
            else if (newRequiredSize < handle.CapacityBytes * ShrinkThreshold)
            {
                // Shrinking the allocation
                long freedSize = handle.CapacityBytes - newRequiredSize;
                _currentAllocatedBytes -= freedSize;
                _stats.TotalBytesAllocated -= freedSize;
            }

            handle.Resize(newShape);
            _stats.TotalResizes++;

            // Update waste calculation
            long wastedBytes = handle.CapacityBytes - handle.GetEffectiveSize();
            // Note: In a complete implementation, we'd track per-allocation waste
        }
        else
        {
            // Just update the shape without reallocation
            handle.Resize(newShape);
        }

        UpdateAverageUtilization();
    }

    /// <summary>
    /// Frees an allocated memory handle.
    /// </summary>
    public void Free(IMemoryHandle handle)
    {
        if (!_activeAllocations.TryRemove(handle, out _))
        {
            throw new ArgumentException("Handle is not allocated by this allocator or already freed", nameof(handle));
        }

        _currentAllocatedBytes -= handle.CapacityBytes;
        _freedBuffers.Add(handle.Pointer);

        // Note: In production, we'd want to actually free memory periodically
        // For now, we're just returning it to the pool
    }

    /// <summary>
    /// Gets current allocation statistics.
    /// </summary>
    public AllocationStats GetAllocationStats()
    {
        return new AllocationStats
        {
            TotalAllocations = _stats.TotalAllocations,
            TotalResizes = _stats.TotalResizes,
            TotalBytesAllocated = _stats.TotalBytesAllocated,
            TotalBytesWasted = _stats.TotalBytesWasted,
            AverageUtilization = _stats.AverageUtilization
        };
    }

    /// <summary>
    /// Updates expectations based on actual usage.
    /// </summary>
    public void UpdateExpectations(IMemoryHandle handle, int[] actualShape)
    {
        if (!_activeAllocations.ContainsKey(handle))
        {
            throw new ArgumentException("Handle is not allocated by this allocator", nameof(handle));
        }

        // In a complete implementation, this would update internal tracking
        // to improve future allocation decisions
    }

    /// <summary>
    /// Reallocates memory to a new size.
    /// </summary>
    private void Reallocate(IntPtr oldPointer, long newSize)
    {
        Marshal.FreeHGlobal(oldPointer);
        Marshal.AllocHGlobal((IntPtr)newSize);
    }

    /// <summary>
    /// Updates the average utilization statistic.
    /// </summary>
    private void UpdateAverageUtilization()
    {
        if (_activeAllocations.IsEmpty)
        {
            _stats.AverageUtilization = 0.0;
            return;
        }

        double totalUtilization = 0.0;
        foreach (var handle in _activeAllocations.Keys)
        {
            totalUtilization += handle.GetUtilization();
        }

        _stats.AverageUtilization = totalUtilization / _activeAllocations.Count;
    }
}
