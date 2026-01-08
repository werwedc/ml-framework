# Spec: CUDA Graph Memory Pool

## Overview
Implement a specialized memory pool for CUDA graphs that ensures fixed memory addresses during graph capture and execution. This is critical because CUDA graphs cannot handle dynamic memory allocations - all memory must be pre-allocated at fixed addresses.

## Requirements

### 1. CUDAGraphMemoryPool Class
Implement the graph memory pool.

```csharp
public class CUDAGraphMemoryPool : IDisposable
{
    private readonly Dictionary<ulong, GraphMemoryBlock> _blocks;
    private readonly object _lock = new object();
    private readonly long _initialCapacity;
    private readonly long _maxCapacity;
    private long _allocatedBytes;
    private bool _disposed;

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
    /// Allocates a fixed memory block for graph execution
    /// </summary>
    public GraphMemoryBlock Allocate(ulong size, ulong alignment = 256)
    {
        ThrowIfDisposed();

        lock (_lock)
        {
            // Check capacity
            if (_allocatedBytes + size > _maxCapacity)
                throw new OutOfMemoryException($"Graph memory pool capacity exceeded: {_allocatedBytes + size} > {_maxCapacity}");

            // Allocate memory using CUDA
            IntPtr ptr = CUDAMemory.Allocate(size, alignment);

            var block = new GraphMemoryBlock(ptr, size);
            _blocks[block.BlockId] = block;
            _allocatedBytes += (long)size;

            return block;
        }
    }

    /// <summary>
    /// Gets a previously allocated block by ID
    /// </summary>
    public GraphMemoryBlock GetBlock(ulong blockId)
    {
        ThrowIfDisposed();

        lock (_lock)
        {
            return _blocks.TryGetValue(blockId, out var block) ? block : null;
        }
    }

    /// <summary>
    /// Returns a block to the pool (doesn't free - keeps for reuse)
    /// </summary>
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
    /// Resets all blocks to available state (between graph iterations)
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
    /// Gets the total allocated memory in bytes
    /// </summary>
    public long AllocatedBytes => _allocatedBytes;

    /// <summary>
    /// Gets the number of allocated blocks
    /// </summary>
    public int BlockCount => _blocks.Count;

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
                    CUDAMemory.Free(block.Ptr);
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

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(CUDAGraphMemoryPool));
    }
}
```

### 2. GraphMemoryBlock Class
Represents a fixed memory block for graph execution.

```csharp
public class GraphMemoryBlock : IDisposable
{
    private bool _disposed;

    public ulong BlockId { get; }
    public IntPtr Ptr { get; }
    public ulong Size { get; }

    internal bool InUse { get; set; }

    internal GraphMemoryBlock(IntPtr ptr, ulong size)
    {
        BlockId = GenerateBlockId();
        Ptr = ptr;
        Size = size;
        InUse = true;
        _disposed = false;
    }

    private static ulong _nextBlockId = 1;
    private static readonly object _blockIdLock = new object();

    private static ulong GenerateBlockId()
    {
        lock (_blockIdLock)
        {
            return _nextBlockId++;
        }
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        _disposed = true;
        // Note: Memory is freed by the pool, not the block
    }
}
```

### 3. CUDAMemory Helper
Helper for CUDA memory operations (may already exist, add if needed).

```csharp
internal static class CUDAMemory
{
    [DllImport("nvcuda", EntryPoint = "cuMemAlloc")]
    public static extern CUResult cuMemAlloc(out IntPtr ptr, ulong size);

    [DllImport("nvcuda", EntryPoint = "cuMemFree")]
    public static extern CUResult cuMemFree(IntPtr ptr);

    public static IntPtr Allocate(ulong size, ulong alignment = 256)
    {
        if (alignment <= 1)
        {
            var result = cuMemAlloc(out IntPtr ptr, size);
            if (result != CUResult.Success)
                throw new CUDADriverException($"Failed to allocate CUDA memory: {result}");
            return ptr;
        }
        else
        {
            // Allocate with alignment
            var allocSize = size + alignment - 1;
            var result = cuMemAlloc(out IntPtr ptr, allocSize);
            if (result != CUResult.Success)
                throw new CUDADriverException($"Failed to allocate aligned CUDA memory: {result}");

            // Align pointer
            var alignedPtr = new IntPtr(((ptr.ToInt64() + (long)alignment - 1) / (long)alignment) * (long)alignment);
            return alignedPtr;
        }
    }

    public static void Free(IntPtr ptr)
    {
        var result = cuMemFree(ptr);
        if (result != CUResult.Success)
            throw new CUDADriverException($"Failed to free CUDA memory: {result}");
    }
}
```

## Implementation Details

### File Structure
- **File**: `src/CUDA/Memory/CUDAGraphMemoryPool.cs`
- **File**: `src/CUDA/Memory/GraphMemoryBlock.cs`
- **File**: `src/CUDA/Memory/CUDAMemory.cs` (extend if needed)

### Dependencies
- System.Collections.Generic for Dictionary
- System for IntPtr, ulong, long
- Existing CUDA memory types

### Memory Management Strategy
- Pre-allocate memory in large chunks
- Keep all allocated memory for the lifetime of the graph
- Use blocks that can be reused across graph executions
- Never free memory during graph execution
- Fixed memory addresses are critical for CUDA graphs

### Threading Considerations
- Lock around all allocations and deallocations
- Thread-safe for concurrent graph executions

## Success Criteria
- Memory pool can allocate blocks with fixed addresses
- Allocated blocks maintain same address across resets
- Pool can be reset and reused for multiple iterations
- Memory is properly freed on disposal
- Thread-safe for concurrent allocations

## Testing Requirements

### Unit Tests
- Test memory allocation with various sizes
- Test fixed address retention across resets
- Test allocation capacity limits
- Test concurrent allocations from multiple threads
- Test proper disposal and cleanup
- Test block reuse after reset

### Integration Tests
- Test memory pool with graph capture
- Test memory pool with graph execution
- Test memory constraints with real workloads (requires GPU)
