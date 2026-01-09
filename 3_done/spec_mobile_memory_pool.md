# Spec: Tensor Memory Pool

## Overview
Implement a pre-allocated memory pool to reduce malloc/free overhead and fragmentation in the mobile runtime.

## Requirements
- Pre-allocated tensor pool for common sizes
- Thread-safe for concurrent model execution
- Memory-aware model loading
- Buffer reuse strategies
- Low-memory mode for constrained devices
- Track memory usage and enforce limits

## Classes to Implement

### 1. `IMemoryPool` Interface
```csharp
public interface IMemoryPool
{
    // Allocation
    IntPtr Allocate(long size, DataType dataType);
    void Free(IntPtr ptr, long size);

    // Pool management
    void SetMemoryLimit(long maxBytes);
    long GetAvailableMemory();
    long GetUsedMemory();
    MemoryPoolStats GetStats();

    // Configuration
    void EnableLowMemoryMode(bool enable);
    void PreAllocateForTensor(long size);

    // Pool reset
    void Reset();
}
```

### 2. `MemoryPoolStats` Class
```csharp
public class MemoryPoolStats
{
    public long TotalMemory { get; set; }
    public long UsedMemory { get; set; }
    public long AvailableMemory { get; set; }
    public int AllocationCount { get; set; }
    public int FreeCount { get; set; }
    public int CacheHits { get; set; }
    public int CacheMisses { get; set; }
    public long PeakUsage { get; set; }
}
```

### 3. `DefaultMemoryPool` Class
```csharp
public sealed class DefaultMemoryPool : IMemoryPool, IDisposable
{
    private readonly object _lock = new object();
    private readonly Dictionary<long, Stack<IntPtr>> _freeBlocks;
    private readonly Dictionary<IntPtr, long> _allocatedBlocks;
    private long _totalAllocated;
    private long _memoryLimit;
    private bool _lowMemoryMode;
    private long _peakUsage;
    private int _allocationCount;
    private int _freeCount;
    private int _cacheHits;
    private int _cacheMisses;

    public DefaultMemoryPool(long initialCapacity = 16 * 1024 * 1024); // 16MB default

    public IntPtr Allocate(long size, DataType dataType);
    public void Free(IntPtr ptr, long size);

    public void SetMemoryLimit(long maxBytes);
    public long GetAvailableMemory();
    public long GetUsedMemory();
    public MemoryPoolStats GetStats();

    public void EnableLowMemoryMode(bool enable);
    public void PreAllocateForTensor(long size);
    public void Reset();

    public void Dispose();

    private IntPtr AllocateNewBlock(long size);
    private void ReturnToPool(IntPtr ptr, long size);
}
```

### 4. `PreallocatedMemoryPool` Class
```csharp
public sealed class PreallocatedMemoryPool : IMemoryPool, IDisposable
{
    private readonly IntPtr _baseAddress;
    private readonly long _totalSize;
    private readonly MemoryBlock[] _blocks;
    private readonly object _lock = new object();
    private long _offset;
    private bool _disposed;

    public PreallocatedMemoryPool(long totalSize);

    public IntPtr Allocate(long size, DataType dataType);
    public void Free(IntPtr ptr, long size);

    public void SetMemoryLimit(long maxBytes);
    public long GetAvailableMemory();
    public long GetUsedMemory();
    public MemoryPoolStats GetStats();

    public void EnableLowMemoryMode(bool enable);
    public void PreAllocateForTensor(long size);
    public void Reset();

    public void Dispose();

    private struct MemoryBlock
    {
        public IntPtr Address;
        public long Size;
        public bool InUse;
    }
}
```

### 5. `MemoryPoolFactory` Class
```csharp
public static class MemoryPoolFactory
{
    public static IMemoryPool CreateDefault(long capacity = 16 * 1024 * 1024);
    public static IMemoryPool CreatePreallocated(long totalSize);
    public static IMemoryPool CreateLowMemoryMode();

    // Create pool based on platform capabilities
    public static IMemoryPool CreateOptimalForPlatform();
}
```

## Implementation Details

### Block Allocation Strategy
- Cache free blocks by size (bucketed by powers of 2)
- Round up allocations to nearest bucket size
- In low-memory mode: immediate free (no caching)
- Bucket sizes: 32, 64, 128, 256, 512, 1K, 2K, 4K, 8K, 16K, 32K, 64K, 128K, 256K, 512K, 1M, 2M, 4M, 8M, 16M

### Memory Management
- Use `Marshal.AllocHGlobal` for unmanaged allocations
- Track all allocations for debugging and stats
- Double-free detection (throw exception)
- Use-after-free detection (optional, debug mode only)

### Low Memory Mode
- Disable block caching
- Release memory immediately on Free
- Use smaller default capacity (4MB instead of 16MB)
- Aggressive garbage collection hints

### Thread Safety
- All public methods synchronized with lock
- Lock-free fast path for Allocate when free block available
- Use `ReaderWriterLockSlim` for better read performance

## Usage Example

```csharp
// Create pool with 32MB limit
var pool = new DefaultMemoryPool(32 * 1024 * 1024);
pool.SetMemoryLimit(32 * 1024 * 1024);

// Allocate tensor memory
var ptr = pool.Allocate(1024 * 1024, DataType.Float32);

// Use tensor
// ...

// Free when done
pool.Free(ptr, 1024 * 1024);

// Check stats
var stats = pool.GetStats();
Console.WriteLine($"Cache hit rate: {(double)stats.CacheHits / (stats.CacheHits + stats.CacheMisses)}");
```

## File Structure
```
src/MobileRuntime/Memory/
├── Interfaces/
│   └── IMemoryPool.cs
├── DefaultMemoryPool.cs
├── PreallocatedMemoryPool.cs
├── MemoryPoolFactory.cs
└── Models/
    └── MemoryPoolStats.cs
```

## Success Criteria
- Pool reduces allocations by > 90% for typical workloads
- Thread-safe under concurrent load
- Enforces memory limits correctly
- Low-memory mode reduces peak usage by > 50%
- No memory leaks detected under stress tests
- Cache hit rate > 80% for common tensor sizes

## Dependencies
- spec_mobile_runtime_core (DataType enum)

## Testing Requirements
- Unit tests for allocation/free patterns
- Thread-safety tests with concurrent allocations
- Memory limit enforcement tests
- Low-memory mode behavior tests
- Cache efficiency benchmarks
- Leak detection with stress tests (1000+ allocations)

## Performance Targets
- Allocate/Free: < 1µs (cached blocks)
- Allocate: < 10µs (new blocks)
- Cache hit rate: > 80%
- Memory overhead: < 5% of pool size
