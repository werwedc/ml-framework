# Spec: Memory Allocator Integration

## Overview
Integrate the CUDA graph memory pool with the existing caching allocator system. This ensures that graph-compatible memory allocations work seamlessly with the framework's memory management.

## Requirements

### 1. Extend Existing Allocator Interface
Modify the existing allocator to support graph-compatible allocations.

```csharp
public interface ICUDAMemoryAllocator
{
    /// <summary>
    /// Allocates GPU memory
    /// </summary>
    IntPtr Allocate(ulong size, ulong alignment = 256);

    /// <summary>
    /// Frees GPU memory
    /// </summary>
    void Free(IntPtr ptr);

    /// <summary>
    /// Gets or sets the graph memory pool for graph-compatible allocations
    /// </summary>
    CUDAGraphMemoryPool GraphPool { get; set; }

    /// <summary>
    /// Gets whether graph mode is enabled
    /// </summary>
    bool IsGraphMode { get; }

    /// <summary>
    /// Enables or disables graph mode
    /// </summary>
    void SetGraphMode(bool enabled);
}
```

### 2. GraphCompatibleMemoryAllocator Class
Implement the integrated allocator.

```csharp
public class GraphCompatibleMemoryAllocator : ICUDAMemoryAllocator
{
    private readonly CUDAMemoryAllocator _baseAllocator;
    private CUDAGraphMemoryPool _graphPool;
    private bool _graphMode;
    private readonly object _lock = new object();

    public GraphCompatibleMemoryAllocator()
    {
        _baseAllocator = new CUDAMemoryAllocator();
        _graphPool = null;
        _graphMode = false;
    }

    public CUDAGraphMemoryPool GraphPool
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

    public bool IsGraphMode => _graphMode;

    /// <summary>
    /// Allocates memory, using graph pool if in graph mode
    /// </summary>
    public IntPtr Allocate(ulong size, ulong alignment = 256)
    {
        lock (_lock)
        {
            if (_graphMode && _graphPool != null)
            {
                // Allocate from graph pool
                var block = _graphPool.Allocate(size, alignment);
                return block.Ptr;
            }
            else
            {
                // Allocate from base allocator
                return _baseAllocator.Allocate(size, alignment);
            }
        }
    }

    /// <summary>
    /// Frees memory, respecting graph mode
    /// </summary>
    public void Free(IntPtr ptr)
    {
        lock (_lock)
        {
            if (_graphMode && _graphPool != null)
            {
                // In graph mode, don't actually free - return to pool
                // Note: This is a simplified approach
                // In practice, we need to track which blocks belong to the pool
            }
            else
            {
                // Free from base allocator
                _baseAllocator.Free(ptr);
            }
        }
    }

    /// <summary>
    /// Enables or disables graph mode
    /// </summary>
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
    /// Resets the graph pool for reuse
    /// </summary>
    public void ResetGraphPool()
    {
        lock (_lock)
        {
            _graphPool?.Reset();
        }
    }

    /// <summary>
    /// Gets memory usage statistics
    /// </summary>
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
                return _baseAllocator.GetStats();
            }
        }
    }

    public void Dispose()
    {
        lock (_lock)
        {
            _graphPool?.Dispose();
            _baseAllocator?.Dispose();
        }
    }
}
```

### 3. MemoryUsageStats Class
Memory usage statistics.

```csharp
public class MemoryUsageStats
{
    /// <summary>
    /// Gets the total allocated memory in bytes
    /// </summary>
    public long TotalAllocated { get; init; }

    /// <summary>
    /// Gets the pool size in bytes
    /// </summary>
    public long PoolSize { get; init; }

    /// <summary>
    /// Gets the number of allocated blocks
    /// </summary>
    public int BlockCount { get; init; }

    /// <summary>
    /// Gets whether graph mode is enabled
    /// </summary>
    public bool IsGraphMode { get; init; }

    public override string ToString()
    {
        return $"Memory: {TotalAllocated / (1024 * 1024):F2} MB, " +
               $"Blocks: {BlockCount}, " +
               $"GraphMode: {IsGraphMode}";
    }
}
```

### 4. MemoryManager Extensions
Add helper methods to the memory manager.

```csharp
public static class MemoryManagerExtensions
{
    /// <summary>
    /// Configures the allocator for graph capture
    /// </summary>
    public static void ConfigureForGraph(
        this ICUDAMemoryAllocator allocator,
        CUDAGraphMemoryPool pool)
    {
        allocator.GraphPool = pool;
        allocator.SetGraphMode(true);
    }

    /// <summary>
    /// Enables graph capture mode
    /// </summary>
    public static void EnableGraphMode(this ICUDAMemoryAllocator allocator)
    {
        allocator.SetGraphMode(true);
    }

    /// <summary>
    /// Disables graph capture mode
    /// </summary>
    public static void DisableGraphMode(this ICUDAMemoryAllocator allocator)
    {
        allocator.SetGraphMode(false);
    }

    /// <summary>
    /// Executes an action with graph mode enabled
    /// </summary>
    public static T WithGraphMode<T>(
        this ICUDAMemoryAllocator allocator,
        Func<T> action)
    {
        var wasGraphMode = allocator.IsGraphMode;
        try
        {
            allocator.EnableGraphMode();
            return action();
        }
        finally
        {
            if (!wasGraphMode)
            {
                allocator.DisableGraphMode();
            }
        }
    }

    /// <summary>
    /// Executes an action with graph mode enabled
    /// </summary>
    public static void WithGraphMode(
        this ICUDAMemoryAllocator allocator,
        Action action)
    {
        var wasGraphMode = allocator.IsGraphMode;
        try
        {
            allocator.EnableGraphMode();
            action();
        }
        finally
        {
            if (!wasGraphMode)
            {
                allocator.DisableGraphMode();
            }
        }
    }
}
```

### 5. Integration with Tensor Class
Ensure tensors work with the graph-compatible allocator.

```csharp
public static class TensorMemoryExtensions
{
    /// <summary>
    /// Creates a tensor that uses graph-compatible memory
    /// </summary>
    public static Tensor WithGraphMemory(
        this Tensor tensor,
        CUDAGraphMemoryPool pool)
    {
        // Allocate memory from the graph pool
        var size = tensor.ElementCount * tensor.ElementSize;
        var block = pool.Allocate((ulong)size);

        // Create a new tensor with the allocated memory
        return new Tensor(block.Ptr, tensor.Shape, tensor.DataType);
    }

    /// <summary>
    /// Ensures tensor memory is allocated for graph execution
    /// </summary>
    public static Tensor EnsureGraphCompatible(
        this Tensor tensor,
        ICUDAMemoryAllocator allocator)
    {
        if (allocator.IsGraphMode && !tensor.IsGraphCompatible)
        {
            // Reallocate with graph-compatible memory
            return tensor.WithGraphMemory(allocator.GraphPool);
        }

        return tensor;
    }
}
```

## Implementation Details

### File Structure
- **File**: `src/CUDA/Memory/ICUDAMemoryAllocator.cs` (extend interface)
- **File**: `src/CUDA/Memory/GraphCompatibleMemoryAllocator.cs`
- **File**: `src/CUDA/Memory/MemoryUsageStats.cs`
- **File**: `src/CUDA/Memory/MemoryManagerExtensions.cs`
- **File**: `src/Tensors/TensorMemoryExtensions.cs`

### Dependencies
- CUDAGraphMemoryPool (from spec_cuda_graph_memory_pool)
- CUDAMemoryAllocator (existing)
- Tensor class (existing)
- System for Action, Func

### Integration Strategy
1. **Non-invasive**: Extend existing interfaces rather than replace them
2. **Transparent**: Switch between graph and regular memory allocation seamlessly
3. **Backward Compatible**: Existing code continues to work without changes
4. **Explicit**: Graph mode must be explicitly enabled

### Memory Allocation Flow
1. Check if graph mode is enabled
2. If yes, allocate from graph memory pool
3. If no, use regular allocator
4. Track allocations for proper cleanup

## Success Criteria
- Allocator can switch between graph and regular mode
- Graph allocations work correctly
- Regular allocations continue to work
- Extension methods work as expected
- Memory statistics are accurate
- Tensor integration works
- Thread-safe operation

## Testing Requirements

### Unit Tests
- Test allocator mode switching
- Test graph allocation
- Test regular allocation
- Test extension methods
- Test memory statistics
- Test tensor integration
- Test concurrent allocations

### Integration Tests
- Test allocator with graph capture (requires GPU)
- Test allocator with training loop
- Test memory pool sharing
