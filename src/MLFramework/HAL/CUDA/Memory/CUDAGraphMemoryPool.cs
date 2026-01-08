using System;
using System.Collections.Generic;

namespace MLFramework.HAL.CUDA;

/// <summary>
/// Specialized memory pool for CUDA graphs that ensures fixed memory addresses during graph capture and execution.
/// This is critical because CUDA graphs cannot handle dynamic memory allocations - all memory must be pre-allocated at fixed addresses.
/// </summary>
public class CUDAGraphMemoryPool : IDisposable
{
    private readonly Dictionary<ulong, GraphMemoryBlock> _blocks;
    private readonly object _lock = new object();
    private readonly long _initialCapacity;
    private readonly long _maxCapacity;
    private long _allocatedBytes;
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of the CUDAGraphMemoryPool class.
    /// </summary>
    /// <param name="initialCapacity">Initial capacity in bytes (default: 512MB)</param>
    /// <param name="maxCapacity">Maximum capacity in bytes (default: unlimited)</param>
    public CUDAGraphMemoryPool(long initialCapacity = 512 * 1024 * 1024, long maxCapacity = long.MaxValue)
    {
        _blocks = new Dictionary<ulong, GraphMemoryBlock>();
        _initialCapacity = initialCapacity;
        _maxCapacity = maxCapacity;
        _allocatedBytes = 0;
        _disposed = false;

        // Pre-allocate initial buffer pool
        InitializePool();
    }

    /// <summary>
    /// Allocates a fixed memory block for graph execution.
    /// </summary>
    /// <param name="size">Size of the block in bytes</param>
    /// <param name="alignment">Alignment requirement in bytes (default: 256)</param>
    /// <returns>A graph memory block with fixed address</returns>
    public GraphMemoryBlock Allocate(ulong size, ulong alignment = 256)
    {
        ThrowIfDisposed();

        lock (_lock)
        {
            // Check capacity
            if ((ulong)_allocatedBytes + size > (ulong)_maxCapacity)
                throw new OutOfMemoryException($"Graph memory pool capacity exceeded: {(ulong)_allocatedBytes + size} > {(ulong)_maxCapacity}");

            // Allocate memory using CUDA
            IntPtr ptr = AllocateCudaMemory(size, alignment);

            var block = new GraphMemoryBlock(ptr, size);
            _blocks[block.BlockId] = block;
            _allocatedBytes += (long)size;

            return block;
        }
    }

    /// <summary>
    /// Gets a previously allocated block by ID.
    /// </summary>
    /// <param name="blockId">The block ID to retrieve</param>
    /// <returns>The graph memory block, or null if not found</returns>
    public GraphMemoryBlock? GetBlock(ulong blockId)
    {
        ThrowIfDisposed();

        lock (_lock)
        {
            return _blocks.TryGetValue(blockId, out var block) ? block : null;
        }
    }

    /// <summary>
    /// Returns a block to the pool (doesn't free - keeps for reuse).
    /// </summary>
    /// <param name="blockId">The block ID to return</param>
    public void ReturnBlock(ulong blockId)
    {
        ThrowIfDisposed();

        lock (_lock)
        {
            if (_blocks.TryGetValue(blockId, out var block))
            {
                block.InUse = false;
            }
        }
    }

    /// <summary>
    /// Resets all blocks to available state (between graph iterations).
    /// </summary>
    public void Reset()
    {
        ThrowIfDisposed();

        lock (_lock)
        {
            foreach (var block in _blocks.Values)
            {
                block.InUse = false;
            }
        }
    }

    /// <summary>
    /// Gets the total allocated memory in bytes.
    /// </summary>
    public long AllocatedBytes => _allocatedBytes;

    /// <summary>
    /// Gets the number of allocated blocks.
    /// </summary>
    public int BlockCount => _blocks.Count;

    /// <summary>
    /// Disposes the memory pool and frees all allocated memory.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
            return;

        lock (_lock)
        {
            foreach (var block in _blocks.Values)
            {
                if (!block.Ptr.Equals(IntPtr.Zero))
                {
                    FreeCudaMemory(block.Ptr);
                }
            }

            _blocks.Clear();
            _allocatedBytes = 0;
            _disposed = true;
        }
    }

    private void InitializePool()
    {
        // Pre-allocate some blocks to reduce overhead
        var blockSize = 16 * 1024 * 1024; // 16MB
        var blocksToPreallocate = _initialCapacity / blockSize;

        for (int i = 0; i < blocksToPreallocate; i++)
        {
            try
            {
                Allocate((ulong)blockSize, 256);
            }
            catch (OutOfMemoryException)
            {
                // Stop if we run out of memory
                break;
            }
        }

        // Mark all pre-allocated blocks as available
        Reset();
    }

    private IntPtr AllocateCudaMemory(ulong size, ulong alignment)
    {
        if (alignment <= 1)
        {
            var result = CudaApi.CudaMalloc(out IntPtr ptr, size);
            if (result != CudaError.Success)
                throw new CudaException(result, "Failed to allocate CUDA memory");
            return ptr;
        }
        else
        {
            // Allocate with alignment
            var allocSize = size + alignment - 1;
            var result = CudaApi.CudaMalloc(out IntPtr ptr, allocSize);
            if (result != CudaError.Success)
                throw new CudaException(result, "Failed to allocate aligned CUDA memory");

            // Align pointer
            var alignedPtr = new IntPtr(((ptr.ToInt64() + (long)alignment - 1) / (long)alignment) * (long)alignment);
            return alignedPtr;
        }
    }

    private void FreeCudaMemory(IntPtr ptr)
    {
        var result = CudaApi.CudaFree(ptr);
        if (result != CudaError.Success)
            throw new CudaException(result, "Failed to free CUDA memory");
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(CUDAGraphMemoryPool));
    }
}
