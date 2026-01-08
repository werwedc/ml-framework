# Spec: Prefetching Strategy

## Overview
Implement an asynchronous prefetching system that prepares future batches while the GPU processes the current batch, ensuring zero GPU idle time.

## Requirements

### 1. IPrefetchStrategy<T> Interface
Defines the contract for prefetching implementations.

```csharp
public interface IPrefetchStrategy<T>
{
    Task<T> GetNextAsync(CancellationToken cancellationToken);
    Task PrefetchAsync(int count, CancellationToken cancellationToken);
    void Reset();
    bool IsAvailable { get; }
}
```

### 2. SimplePrefetchStrategy<T> Implementation
Basic prefetching using a fixed-size buffer.

**Constructor:**
```csharp
public SimplePrefetchStrategy<T>(
    SharedQueue<T> sourceQueue,
    int prefetchCount,
    CancellationToken? cancellationToken = null)
```

**Parameters:**
- `sourceQueue`: Queue to prefetch from
- `prefetchCount`: Number of items to keep preloaded
- `cancellationToken`: Optional cancellation token

**Properties:**
```csharp
public bool IsAvailable { get; }  // True if prefetched items available
```

**Methods:**

**Get Next Item:**
```csharp
public Task<T> GetNextAsync(CancellationToken cancellationToken)
```

**Behavior:**
- Returns next prefetched item immediately if available
- If not available, waits for next item from source queue
- Triggers background task to refill prefetch buffer
- Throws `OperationCanceledException` if cancelled

**Prefetch Multiple Items:**
```csharp
public Task PrefetchAsync(int count, CancellationToken cancellationToken)
```

**Behavior:**
- Background task that pulls `count` items from source queue
- Stores items in internal buffer for fast retrieval
- Can be called multiple times to increase buffer size
- Respects cancellation token

**Reset:**
```csharp
public void Reset()
```

**Behavior:**
- Clears internal buffer
- Can be used when restarting iteration

### 3. PrefetchBuffer<T> Class
Internal buffer that holds prefetched items.

**Properties:**
```csharp
public int Count { get; }
public int Capacity { get; }
public bool IsEmpty { get; }
public bool IsFull { get; }
```

**Methods:**

**Add Item:**
```csharp
public void Add(T item)
```

**Behavior:**
- Adds item to buffer
- Throws `InvalidOperationException` if buffer is full

**Get Next:**
```csharp
public T GetNext()
```

**Behavior:**
- Returns next item in FIFO order
- Throws `InvalidOperationException` if buffer is empty

**Peek:**
```csharp
public T Peek()
```

**Behavior:**
- Returns next item without removing it
- Throws `InvalidOperationException` if buffer is empty

**TryGet:**
```csharp
public bool TryGet(out T item)
```

**Behavior:**
- Returns `false` if buffer is empty
- Returns `true` with item if available

### 4. AdaptivePrefetchStrategy<T> (Optional Enhancement)
Advanced prefetching that adapts to data loading variance.

**Constructor:**
```csharp
public AdaptivePrefetchStrategy<T>(
    SharedQueue<T> sourceQueue,
    int initialPrefetchCount,
    int maxPrefetchCount,
    CancellationToken? cancellationToken = null)
```

**Additional Properties:**
```csharp
public double AverageLoadingTimeMs { get; }
public int CurrentPrefetchCount { get; }
```

**Behavior:**
- Monitors how long items take to load
- Increases prefetch count if loading is slow
- Decreases prefetch count if loading is fast
- Bounded by `maxPrefetchCount`

### 5. PrefetchCoordinator<T> (Optional)
Manages multiple prefetch strategies for different data sources.

**Methods:**

**Register Prefetcher:**
```csharp
public void RegisterPrefetcher(string name, IPrefetchStrategy<T> prefetcher)
```

**Get Next from Specific Source:**
```csharp
public Task<T> GetNextAsync(string sourceName, CancellationToken cancellationToken)
```

**Get Next from Any Source:**
```csharp
public Task<T> GetNextAsync(CancellationToken cancellationToken)
```

**Behavior:**
- Prioritizes sources that are ready
- Useful for multi-dataset training

### 6. Prefetch Statistics

**Metrics to Track:**
- `CacheHitRate`: Percentage of requests served from prefetch buffer
- `AverageLatency`: Average time to get prefetched item
- `RefillCount`: Number of times buffer was refilled
- `StarvationCount`: Number of times buffer was empty

**Statistics Class:**
```csharp
public class PrefetchStatistics
{
    public int CacheHits { get; }
    public int CacheMisses { get; }
    public double CacheHitRate => (double)CacheHits / (CacheHits + CacheMisses);
    public double AverageLatencyMs { get; }
    public int RefillCount { get; }
    public int StarvationCount { get; }
}

// In SimplePrefetchStrategy<T>:
public PrefetchStatistics GetStatistics()
```

## File Structure
```
src/
  Data/
    IPrefetchStrategy.cs          (Interface)
    PrefetchBuffer.cs              (Buffer implementation)
    SimplePrefetchStrategy.cs      (Basic prefetching)
    AdaptivePrefetchStrategy.cs    (Optional adaptive)
    PrefetchCoordinator.cs         (Optional coordinator)
    PrefetchStatistics.cs          (Metrics)
```

## Success Criteria
- [ ] Items can be retrieved immediately when prefetched
- [ ] Prefetch task runs in background without blocking
- [ ] Buffer correctly manages FIFO ordering
- [ ] Cancellation token properly stops prefetching
- [ ] Statistics accurately track cache hits/misses
- [ ] Reset clears buffer correctly
- [ ] Unit tests cover concurrent access patterns
- [ ] Unit tests verify prefetching improves latency

## Notes
- Use `ConcurrentQueue<T>` for thread-safe buffer
- Prefetch task should use `Task.Run` for background execution
- Consider using `ValueTask<T>` for performance-critical scenarios
- Adaptive prefetching is optional; implement only if time permits
- This spec depends on `SharedQueue<T>` from spec_shared_queue.md
