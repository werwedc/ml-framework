# Spec: DataLoader Core Implementation

## Overview
Implement the main DataLoader<T> class that integrates all components (worker pool, shared queue, prefetching, memory management) to provide a high-performance data loading API.

## Requirements

### 1. IDataLoader<T> Interface
High-level interface for data loading.

```csharp
public interface IDataLoader<T> : IDisposable, IEnumerable<T>
{
    void Start();
    void Stop();
    void Reset();
    bool IsRunning { get; }
    DataLoaderConfig Config { get; }
}
```

### 2. DataLoader<T> Class
Main data loader implementation.

**Constructor:**
```csharp
public DataLoader(IDataset<T> dataset, DataLoaderConfig config)
```

**Parameters:**
- `dataset`: The dataset to load data from
- `config`: Configuration for loading behavior

**Properties:**
```csharp
public bool IsRunning { get; }
public DataLoaderConfig Config { get; }
public int BatchCount { get; }  // Number of batches in dataset
```

### 3. Lifecycle Methods

**Start:**
```csharp
public void Start()
```

**Behavior:**
- Validates not already running
- Creates shared queue with configured size
- Initializes worker pool based on NumWorkers
- Starts prefetching strategy
- Begins batch preparation in background
- Throws `InvalidOperationException` if already started

**Stop:**
```csharp
public void Stop()
```

**Behavior:**
- Signals workers to stop
- Stops prefetching
- Marks queue as complete
- Waits for all workers to complete (with timeout)
- Cleans up resources
- Can be called multiple times safely

**Reset:**
```csharp
public void Reset()
```

**Behavior:**
- Stops if currently running
- Clears internal state
- Resets batch iterator
- Can be followed by Start() to restart iteration

**Dispose:**
```csharp
public void Dispose()
```

**Behavior:**
- Calls Stop() if running
- Disposes of all internal resources
- Releases pinned memory if allocated

### 4. Iteration API

**Synchronous Iterator:**
```csharp
public IEnumerator<T> GetEnumerator()
```

**Behavior:**
- Returns batches in order
- Blocks until next batch is available
- Throws `ObjectDisposedException` if disposed
- Throws `InvalidOperationException` if not started

**Async Iterator:**
```csharp
public IAsyncEnumerator<T> GetAsyncEnumerator(CancellationToken cancellationToken = default)
```

**Behavior:**
- Returns batches asynchronously
- Returns awaitable task for each batch
- Respects cancellation token
- Better for async training loops

### 5. Batch Assembly

**Batch Creation:**
```csharp
private T CreateBatch(int[] indices)
```

**Behavior:**
- Fetches items from dataset at specified indices
- Assembles items into batch structure
- Applies shuffling if configured
- Uses pinned memory if PinMemory is enabled

**Index Generation:**
```csharp
private IEnumerable<int[]> GenerateBatchIndices()
```

**Behavior:**
- Generates batches of indices based on BatchSize
- Shuffles indices if Shuffle is enabled
- Uses configured Seed for reproducibility
- Handles uneven final batch

### 6. Worker Function

**Data Loading Worker:**
```csharp
private T DataLoadingWorker(int workerId, CancellationToken cancellationToken)
```

**Behavior:**
- Pulls next batch indices from index generator
- Fetches items from dataset for those indices
- Assembles batch
- Applies preprocessing if needed
- Returns batch for enqueueing

### 7. Integration Components

**Internal Components:**
```csharp
private IDataset<T> _dataset;
private DataLoaderConfig _config;
private SharedQueue<T> _queue;
private WorkerPool<T> _workerPool;
private IPrefetchStrategy<T> _prefetchStrategy;
private PinnedMemoryPool<T>? _pinnedMemoryPool;
private CancellationTokenSource _cancellationTokenSource;
private Random _random;
private int[] _indices;
```

### 8. Shuffling

**Shuffle Indices:**
```csharp
private void ShuffleIndices()
```

**Behavior:**
- Uses Fisher-Yates shuffle algorithm
- Uses configured Seed for reproducibility
- Only called if Shuffle is enabled in config
- Called on Reset() or first Start()

### 9. Statistics and Monitoring

**DataLoader Statistics:**
```csharp
public class DataLoaderStatistics
{
    public int BatchesLoaded { get; }
    public int TotalSamples { get; }
    public TimeSpan AverageBatchTime { get; }
    public double ThroughputSamplesPerSecond { get; }
    public WorkerPoolStatistics WorkerStats { get; }
    public PrefetchStatistics PrefetchStats { get; }
}

public DataLoaderStatistics GetStatistics()
```

### 10. Extension Methods (Optional)

**ToTensorDataLoader:**
```csharp
public static DataLoader<Tensor> ToTensorDataLoader<T>(this IDataLoader<T> dataloader)
```

**Batch Size Adjustment:**
```csharp
public static IDataLoader<T> WithBatchSize<T>(this IDataLoader<T> dataloader, int batchSize)
```

## File Structure
```
src/
  Data/
    IDataLoader.cs              (Interface)
    DataLoader.cs               (Main implementation)
    DataLoaderStatistics.cs     (Statistics class)
    DataLoaderExtensions.cs     (Optional extensions)
```

## Success Criteria
- [ ] DataLoader starts workers and prefetching correctly
- [ ] Iterator returns batches in correct order
- [ ] Shuffling produces different orders (with seed reproducibility)
- [ ] Workers stop gracefully when Stop() is called
- [ ] Resources are properly disposed
- [ ] Pinned memory is used when configured
- [ ] Statistics accurately track performance
- [ ] Multiple iterations work correctly (after Reset)
- [ ] Async iterator works with cancellation
- [ ] Unit tests verify end-to-end behavior
- [ ] Integration tests with real dataset

## Notes
- This spec integrates all previous specs
- Order of implementation matters: start with synchronous, then add async
- Consider using `ValueTask<T>` for async iteration
- Batch structure depends on tensor implementation (placeholder for now)
- Handle edge cases: empty dataset, batch size > dataset size, etc.
- This is the main public API exposed to users
- Ensure comprehensive error messages for common issues
