using System;
using System.Collections.Generic;

namespace MLFramework.HAL.CUDA;

/// <summary>
/// Memory allocator that supports both regular CUDA allocations and graph-compatible allocations.
/// Uses a graph memory pool when in graph mode to ensure fixed memory addresses.
/// </summary>
public class GraphCompatibleMemoryAllocator : ICUDAMemoryAllocator
{
    private readonly CudaAllocator _baseAllocator;
    private CUDAGraphMemoryPool? _graphPool;
    private bool _graphMode;
    private readonly object _lock = new object();
    private readonly Dictionary<IntPtr, GraphMemoryBlock> _graphAllocations;

    /// <summary>
    /// Initializes a new instance of the GraphCompatibleMemoryAllocator class.
    /// </summary>
    public GraphCompatibleMemoryAllocator()
    {
        _baseAllocator = new CudaAllocator(new CudaDevice(0));
        _graphPool = null;
        _graphMode = false;
        _graphAllocations = new Dictionary<IntPtr, GraphMemoryBlock>();
    }

    /// <summary>
    /// Gets or sets the graph memory pool for graph-compatible allocations.
    /// </summary>
    public CUDAGraphMemoryPool? GraphPool
    {
        get
        {
            lock (_lock)
            {
                return _graphPool;
            }
        }
        set
        {
            lock (_lock)
            {
                _graphPool = value;
            }
        }
    }

    /// <summary>
    /// Gets whether graph mode is enabled.
    /// </summary>
    public bool IsGraphMode => _graphMode;

    /// <summary>
    /// Allocates memory, using graph pool if in graph mode.
    /// </summary>
    /// <param name="size">Size in bytes to allocate</param>
    /// <param name="alignment">Alignment requirement in bytes (default: 256)</param>
    /// <returns>Pointer to the allocated memory</returns>
    public IntPtr Allocate(ulong size, ulong alignment = 256)
    {
        lock (_lock)
        {
            if (_graphMode && _graphPool != null)
            {
                // Allocate from graph pool
                var block = _graphPool.Allocate(size, alignment);
                _graphAllocations[block.Ptr] = block;
                return block.Ptr;
            }
            else
            {
                // Allocate from base allocator
                var buffer = _baseAllocator.Allocate((long)size);
                var ptr = buffer.Pointer;
                // Store the buffer for later freeing
                _graphAllocations[ptr] = new GraphMemoryBlock(ptr, size);
                return ptr;
            }
        }
    }

    /// <summary>
    /// Frees memory, respecting graph mode.
    /// </summary>
    /// <param name="ptr">Pointer to the memory to free</param>
    public void Free(IntPtr ptr)
    {
        lock (_lock)
        {
            if (_graphMode && _graphPool != null && _graphAllocations.TryGetValue(ptr, out var block))
            {
                // In graph mode, don't actually free - return to pool
                _graphPool.ReturnBlock(block.BlockId);
            }
            else
            {
                // Free from base allocator
                // Note: We need to convert IntPtr back to IMemoryBuffer
                // For now, this is a simplified implementation
                _graphAllocations.Remove(ptr);
            }
        }
    }

    /// <summary>
    /// Enables or disables graph mode.
    /// </summary>
    /// <param name="enabled">True to enable graph mode, false to disable</param>
    public void SetGraphMode(bool enabled)
    {
        lock (_lock)
        {
            if (enabled && _graphPool == null)
            {
                throw new InvalidOperationException(
                    "Cannot enable graph mode without a graph memory pool");
            }

            _graphMode = enabled;
        }
    }

    /// <summary>
    /// Resets the graph pool for reuse.
    /// </summary>
    public void ResetGraphPool()
    {
        lock (_lock)
        {
            _graphPool?.Reset();
        }
    }

    /// <summary>
    /// Gets memory usage statistics.
    /// </summary>
    /// <returns>Memory usage statistics for the current mode</returns>
    public MemoryUsageStats GetStats()
    {
        lock (_lock)
        {
            if (_graphMode && _graphPool != null)
            {
                return new MemoryUsageStats
                {
                    TotalAllocated = _graphPool.AllocatedBytes,
                    PoolSize = _graphPool.AllocatedBytes,
                    BlockCount = _graphPool.BlockCount,
                    IsGraphMode = true
                };
            }
            else
            {
                return new MemoryUsageStats
                {
                    TotalAllocated = _baseAllocator.AllocatedSize,
                    PoolSize = _baseAllocator.CacheSize,
                    BlockCount = _graphAllocations.Count,
                    IsGraphMode = false
                };
            }
        }
    }

    /// <summary>
    /// Disposes the allocator and frees all allocated memory.
    /// </summary>
    public void Dispose()
    {
        lock (_lock)
        {
            _graphPool?.Dispose();
            _baseAllocator?.Dispose();
            _graphAllocations.Clear();
        }
    }
}
