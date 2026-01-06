# Spec: WorkerPool Architecture

## Overview
Implement a worker pool that manages multiple worker processes for parallel data loading.

## Requirements

### Interfaces

#### IWorkerPool
```csharp
public interface IWorkerPool : IDisposable
{
    void Start();
    void Stop();
    bool IsRunning { get; }
    int NumWorkers { get; }
    void SubmitTask<T>(Func<T> task);
    T GetResult<T>();
    bool TryGetResult<T>(out T result);
}
```

### Implementation

#### WorkerPool
- Manages pool of worker processes using C# Task-based parallelism
- Each worker processes data independently
- Communication via blocking collection queues
- Graceful shutdown support

**Key Fields:**
```csharp
public class WorkerPool : IWorkerPool
{
    private readonly int _numWorkers;
    private readonly CancellationTokenSource _cancellationToken;
    private readonly Task[] _workers;
    private readonly BlockingCollection<object> _taskQueue;
    private readonly BlockingCollection<object> _resultQueue;
    private volatile bool _isRunning;
}
```

**Constructor:**
```csharp
public WorkerPool(int numWorkers = 4)
{
    if (numWorkers <= 0)
        throw new ArgumentOutOfRangeException(nameof(numWorkers));

    _numWorkers = numWorkers;
    _cancellationToken = new CancellationTokenSource();
    _taskQueue = new BlockingCollection<object>();
    _resultQueue = new BlockingCollection<object>();
    _workers = new Task[numWorkers];
    _isRunning = false;
}
```

**Start Method:**
```csharp
public void Start()
{
    if (_isRunning)
        return;

    _isRunning = true;

    for (int i = 0; i < _numWorkers; i++)
    {
        _workers[i] = Task.Run(() => WorkerLoop(_cancellationToken.Token));
    }
}
```

**Worker Loop:**
```csharp
private void WorkerLoop(CancellationToken token)
{
    while (!token.IsCancellationRequested)
    {
        try
        {
            // Get task (with timeout to allow cancellation)
            var taskObj = _taskQueue.Take(token);

            if (taskObj is Func<object> task)
            {
                var result = task();
                _resultQueue.Add(result, token);
            }
        }
        catch (OperationCanceledException)
        {
            break;
        }
        catch (Exception ex)
        {
            // Log error and continue
            Console.WriteLine($"Worker error: {ex.Message}");
        }
    }
}
```

**Submit Task:**
```csharp
public void SubmitTask<T>(Func<T> task)
{
    if (!_isRunning)
        throw new InvalidOperationException("Worker pool is not running");

    if (task == null)
        throw new ArgumentNullException(nameof(task));

    _taskQueue.Add(task);
}
```

**Get Result:**
```csharp
public T GetResult<T>()
{
    if (!_isRunning)
        throw new InvalidOperationException("Worker pool is not running");

    var result = _resultQueue.Take();
    return (T)result;
}

public bool TryGetResult<T>(out T result)
{
    if (!_isRunning)
    {
        result = default;
        return false;
    }

    if (_resultQueue.TryTake(out var resultObj))
    {
        result = (T)resultObj;
        return true;
    }

    result = default;
    return false;
}
```

**Stop/Dispose:**
```csharp
public void Stop()
{
    if (!_isRunning)
        return;

    _cancellationToken.Cancel();

    Task.WaitAll(_workers, TimeSpan.FromSeconds(30));

    _isRunning = false;
}

public void Dispose()
{
    Stop();
    _cancellationToken?.Dispose();
    _taskQueue?.Dispose();
    _resultQueue?.Dispose();
}
```

### Error Handling
- `InvalidOperationException` if pool not running
- `ArgumentNullException` for null tasks
- Graceful handling of worker exceptions
- Timeout in Stop method (30 seconds)

## Acceptance Criteria
1. WorkerPool starts specified number of worker tasks
2. SubmitTask adds tasks to queue
3. GetResult retrieves completed results in order
4. TryGetResult returns false if no result available
5. Stop gracefully cancels all workers
6. Dispose cleans up all resources
7. Multiple tasks execute in parallel
8. Worker exceptions don't crash the pool
9. Unit tests verify correct number of workers spawned
10. Integration tests verify parallel execution

## Files to Create
- `src/Data/Worker/IWorkerPool.cs`
- `src/Data/Worker/WorkerPool.cs`

## Tests
- `tests/Data/Worker/WorkerPoolTests.cs`

## Usage Example
```csharp
using (var pool = new WorkerPool(numWorkers: 4))
{
    pool.Start();

    // Submit tasks
    for (int i = 0; i < 10; i++)
    {
        int index = i;
        pool.SubmitTask(() => ComputeSomething(index));
    }

    // Get results
    for (int i = 0; i < 10; i++)
    {
        var result = pool.GetResult<int>();
        Console.WriteLine($"Result: {result}");
    }
}
```

## Notes
- This uses C# Tasks (not true multiprocessing like Python)
- For true multiprocessing in C#, consider System.Diagnostics.Process
- Current implementation suitable for CPU-bound data loading
- BlockingCollection provides thread-safe queues
- Future specs may add shared memory for zero-copy transfers
- Consider adding priority queues or batch submission
- Monitor queue sizes for backpressure detection
