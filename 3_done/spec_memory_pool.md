# Spec: Memory Pool

## Overview
Implement an object pooling system for tensors and buffers to minimize GC pressure and improve performance in the data loading pipeline.

## Requirements

### 1. IPool<T> Interface
Generic interface for object pools.

```csharp
public interface IPool<T> : IDisposable
{
    T Rent();
    void Return(T item);
    int AvailableCount { get; }
    int TotalCount { get; }
    void Clear();
}
```

### 2. ObjectPool<T> Class
Generic thread-safe object pool implementation.

**Constructor:**
```csharp
public ObjectPool(
    Func<T> factory,
    Action<T>? reset = null,
    int initialSize = 0,
    int maxSize = 100)
```

**Parameters:**
- `factory`: Function to create new instances
- `reset`: Optional action to reset item when returned to pool
- `initialSize`: Number of items to pre-allocate
- `maxSize`: Maximum number of items to keep in pool

**Properties:**
```csharp
public int AvailableCount { get; }  // Items available for rent
public int TotalCount { get; }      // Total items created by pool
public int MaxSize { get; }         // Maximum pool size
```

**Methods:**

**Rent Item:**
```csharp
public T Rent()
```

**Behavior:**
- Returns available item from pool if available
- Creates new item via factory if pool is empty
- Thread-safe for concurrent renters
- Tracks total count of created items

**Return Item:**
```csharp
public void Return(T item)
```

**Behavior:**
- Returns item to pool for reuse
- Calls `reset` action if provided
- Discards item if pool is at `maxSize`
- Thread-safe for concurrent returners

**Clear Pool:**
```csharp
public void Clear()
```

**Behavior:**
- Removes all items from pool
- Does not dispose of items (caller responsibility)
- Useful for cleanup or memory pressure response

### 3. ArrayPool<T> Specialization
Optimized pool specifically for arrays.

**Constructor:**
```csharp
public ArrayPool<T>(int arrayLength, int initialSize = 0, int maxSize = 50)
```

**Parameters:**
- `arrayLength`: Length of arrays in the pool (all arrays same size)
- `initialSize`: Number of arrays to pre-allocate
- `maxSize`: Maximum number of arrays to keep

**Methods:**

**Rent Array:**
```csharp
public T[] Rent()
```

**Behavior:**
- Returns array of configured length
- Creates new array if none available

**Return Array:**
```csharp
public void Return(T[] array)
```

**Behavior:**
- Validates array length matches pool configuration
- Clears array contents (optional, configurable)
- Returns array to pool

**Resize Pool:**
```csharp
public void Resize(int newArrayLength)
```

**Behavior:**
- Clears existing pool
- Allocates arrays of new length
- Useful for changing batch sizes

### 4. TensorPool (Pseudo-spec, depends on tensor implementation)
Pool for tensor objects (placeholder until tensor spec exists).

```csharp
public class TensorPool : IPool<Tensor>
{
    public TensorPool(TensorShape shape, int initialSize = 0, int maxSize = 20)
    // Implementation depends on actual Tensor class
}
```

### 5. PoolManager
Centralized manager for multiple pools.

**Methods:**

**Get or Create Pool:**
```csharp
public IPool<T> GetPool<T>(string key, Func<T> factory)
public IPool<T> GetOrCreatePool<T>(string key, Func<T> factory)
```

**Behavior:**
- Returns existing pool if key exists
- Creates new pool if key doesn't exist
- Thread-safe for concurrent access

**Clear All Pools:**
```csharp
public void ClearAll()
```

**Behavior:**
- Clears all managed pools
- Useful for cleanup or memory pressure

**Get Statistics:**
```csharp
public PoolManagerStatistics GetStatistics()
```

**Statistics Class:**
```csharp
public class PoolManagerStatistics
{
    public int PoolCount { get; }
    public int TotalAvailableItems { get; }
    public int TotalCreatedItems { get; }
    public Dictionary<string, int> PoolSizes { get; }
}
```

### 6. Pool<T> Statistics

**Metrics to Track:**
- `RentCount`: Total number of rent operations
- `ReturnCount`: Total number of return operations
- `MissCount`: Number of times pool was empty (new item created)
- `DiscardCount`: Number of items discarded due to maxSize
- `HitRate`: Percentage of requests served from pool

**Statistics Class:**
```csharp
public class PoolStatistics
{
    public int RentCount { get; }
    public int ReturnCount { get; }
    public int MissCount { get; }
    public int DiscardCount { get; }
    public double HitRate => (double)(RentCount - MissCount) / RentCount;
}

// In ObjectPool<T>:
public PoolStatistics GetStatistics()
public void ResetStatistics()
```

## File Structure
```
src/
  Data/
    IPool.cs              (Generic pool interface)
    ObjectPool.cs         (Generic pool implementation)
    ArrayPool.cs          (Array-specific pool)
    TensorPool.cs         (Tensor-specific pool - placeholder)
    PoolManager.cs        (Centralized pool manager)
    PoolStatistics.cs     (Pool metrics)
```

## Success Criteria
- [ ] Objects can be rented and returned correctly
- [ ] Pool respects maxSize limit
- [ ] Factory creates new items when pool is empty
- [ ] Reset action is called when returning items
- [ ] Thread-safe for concurrent rent/return operations
- [ ] ArrayPool validates array length
- [ ] PoolManager creates and caches pools
- [ ] Statistics accurately track pool operations
- [ ] Unit tests cover concurrent access scenarios
- [ ] Unit tests verify pool reduces allocations

## Notes
- Use `ConcurrentBag<T>` for thread-safe storage
- Consider using `Interlocked` for statistics counters
- Array clearing can be expensive; make configurable
- PoolManager can be singleton or static for simplicity
- This spec is independent of other dataloader specs
- Consider integrating with `MemoryPressure` API for cleanup
