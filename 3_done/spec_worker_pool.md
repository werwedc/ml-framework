# Spec: Worker Pool

## Overview
Implement a managed pool of worker tasks that load and preprocess data in parallel. Workers communicate with the main process via a shared queue.

## Requirements

### 1. DataWorker<T> Delegate
Delegate that defines the work each worker performs.

```csharp
public delegate T DataWorker<T>(int workerId, CancellationToken cancellationToken);
```

**Parameters:**
- `workerId`: Unique identifier for this worker (0 to NumWorkers-1)
- `cancellationToken`: Token for checking cancellation during work

**Returns:**
- The processed data item to enqueue

### 2. WorkerPool<T> Class
Manages a pool of worker tasks that produce data.

**Constructor:**
```csharp
public WorkerPool<T>(
    DataWorker<T> workerFunc,
    SharedQueue<T> outputQueue,
    int numWorkers,
    CancellationToken? cancellationToken = null)
```

**Parameters:**
- `workerFunc`: Function that defines what each worker does
- `outputQueue`: Queue where workers deposit completed items
- `numWorkers`: Number of parallel workers
- `cancellationToken`: Optional cancellation token for graceful shutdown

**Properties:**
```csharp
public bool IsRunning { get; }
public int NumWorkers { get; }
public int ActiveWorkers { get; }
```

### 3. Lifecycle Methods

**Start Workers:**
```csharp
public void Start()
```

**Behavior:**
- Launches `numWorkers` tasks using `Task.Run`
- Each task continuously calls `workerFunc` and enqueues results
- Validates that pool is not already running
- Throws `InvalidOperationException` if already started

**Stop Workers:**
```csharp
public async Task StopAsync(TimeSpan timeout)
```

**Behavior:**
- Signals all workers to stop via cancellation token
- Waits for all tasks to complete or timeout
- Marks output queue as complete after workers stop
- Throws `TimeoutException` if workers don't stop within timeout
- Can be called multiple times safely

**Wait for Completion:**
```csharp
public async Task WaitAsync()
```

**Behavior:**
- Awaits all worker tasks
- Throws aggregate exception if any worker failed

### 4. Worker Task Logic
Each worker task follows this pattern:

```csharp
while (!cancellationToken.IsCancellationRequested)
{
    try
    {
        // 1. Perform work
        T result = workerFunc(workerId, cancellationToken);

        // 2. Enqueue result
        outputQueue.Enqueue(result);
    }
    catch (OperationCanceledException)
    {
        // Expected during shutdown
        break;
    }
    catch (Exception ex)
    {
        // Log error and restart or fail
        // Handle based on error handling strategy
        break;
    }
}
```

### 5. Work Distribution Strategies

**Static Partitioning:**
- Each worker processes a fixed subset of data
- Worker `i` processes indices: `i, i+numWorkers, i+2*numWorkers, ...`
- Good for balanced workloads

**Dynamic Work Stealing:**
- Workers pull from a shared work queue
- More complex but handles variable workloads better
- (Implement in separate spec if needed)

**Current Spec Focus:**
- Implement static partitioning only
- Work indices are passed to worker via context

### 6. Worker Context
Encapsulates worker-specific state.

```csharp
public class WorkerContext
{
    public int WorkerId { get; }
    public int NumWorkers { get; }
    public int StartIndex { get; }
    public int EndIndex { get; }
    public CancellationToken CancellationToken { get; }
}
```

**Constructor:**
```csharp
public WorkerContext(int workerId, int numWorkers, int totalItems, CancellationToken cancellationToken)
```

**Calculated Properties:**
- `StartIndex`: `workerId * (totalItems / numWorkers)`
- `EndIndex`: `(workerId + 1) * (totalItems / numWorkers)`
- For uneven divisions, last worker gets remainder

### 7. Worker Pool Factory (Optional)
Convenient creation of worker pools for common scenarios.

```csharp
public static class WorkerPoolFactory
{
    public static WorkerPool<T> CreateForDataset<T>(
        IDataset<T> dataset,
        SharedQueue<T> outputQueue,
        DataLoaderConfig config,
        CancellationToken? cancellationToken = null)
}
```

**Behavior:**
- Creates worker function that calls `dataset.GetItem`
- Distributes indices across workers using static partitioning
- Returns configured worker pool

### 8. Event Hooks (Optional)

**Worker Started:**
```csharp
public event Action<int> WorkerStarted;
```

**Worker Completed:**
```csharp
public event Action<int, bool> WorkerCompleted;
```

**Parameters:**
- First param: worker ID
- Second param: true if completed successfully, false if failed

## File Structure
```
src/
  Data/
    DataWorker.cs         (Delegate definition)
    WorkerContext.cs      (Worker context class)
    WorkerPool.cs         (Main worker pool)
    WorkerPoolFactory.cs  (Optional factory)
```

## Success Criteria
- [ ] Worker pool spawns correct number of tasks
- [ ] Workers continuously produce data until stopped
- [ ] Workers respect cancellation token
- [ ] StopAsync gracefully shuts down all workers
- [ ] WorkerContext correctly partitions work
- [ ] Multiple workers can enqueue to same queue safely
- [ ] IsRunning and ActiveWorkers properties are accurate
- [ ] Unit tests cover concurrency scenarios
- [ ] Worker events fire correctly (if implemented)

## Notes
- Use `Task.Run` not `Task.Factory.StartNew` for better exception handling
- Consider `TaskCreationOptions.LongRunning` for CPU-bound workers
- Workers should handle exceptions gracefully (not crash pool)
- This spec depends on `SharedQueue<T>` from spec_shared_queue.md
- This spec depends on `IDataset<T>` from spec_dataset_interface.md
