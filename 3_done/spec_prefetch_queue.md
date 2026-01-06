# Spec: Prefetch Queue Implementation

## Overview
Implement a prefetch queue that proactively prepares data batches before they're needed.

## Requirements

### Interfaces

#### IPrefetchQueue<T>
```csharp
public interface IPrefetchQueue<T> : IDisposable
{
    void Start();
    void Stop();
    bool IsRunning { get; }
    int PrefetchCount { get; }
    int AvailableBatches { get; }
    T GetNext();
    bool TryGetNext(out T batch);
}
```

### Implementation

#### PrefetchQueue<T>
- Background thread continuously loads and prepares batches
- Maintains a buffer of pre-processed batches
- Configurable prefetch depth (number of batches to prepare ahead)
- Seamless transition between prefetching and consumption

**Key Fields:**
```csharp
public class PrefetchQueue<T> : IPrefetchQueue<T>
{
    private readonly int _prefetchCount;
    private readonly SharedMemoryQueue<T> _queue;
    private readonly CancellationTokenSource _cancellationToken;
    private readonly Task _prefetchTask;
    private readonly Func<IEnumerable<T>> _batchGenerator;
    private volatile bool _isRunning;
}
```

**Constructor:**
```csharp
public PrefetchQueue(
    Func<IEnumerable<T>> batchGenerator,
    int prefetchCount = 2)
{
    if (batchGenerator == null)
        throw new ArgumentNullException(nameof(batchGenerator));

    if (prefetchCount <= 0)
        throw new ArgumentOutOfRangeException(nameof(prefetchCount));

    _batchGenerator = batchGenerator;
    _prefetchCount = prefetchCount;
    _cancellationToken = new CancellationTokenSource();
    _queue = new SharedMemoryQueue<T>(maxSize: prefetchCount * 2);
    _isRunning = false;
    _prefetchTask = null;
}
```

**Start Method:**
```csharp
public void Start()
{
    if (_isRunning)
        return;

    _isRunning = true;

    _prefetchTask = Task.Run(() => PrefetchLoop(_cancellationToken.Token));
}
```

**Prefetch Loop:**
```csharp
private void PrefetchLoop(CancellationToken token)
{
    while (!token.IsCancellationRequested)
    {
        try
        {
            // Check if we need more batches
            while (_queue.Count < _prefetchCount && !token.IsCancellationRequested)
            {
                try
                {
                    // Generate next batch
                    var batchIterator = _batchGenerator();

                    foreach (var batch in batchIterator)
                    {
                        _queue.Enqueue(batch);

                        // Check cancellation after each batch
                        if (token.IsCancellationRequested)
                            break;
                    }

                    // If generator returned empty, we're done
                    if (!_queue.Any() && token.IsCancellationRequested)
                        break;
                }
                catch (Exception ex)
                {
                    // Log error but continue
                    Console.WriteLine($"Prefetch error: {ex.Message}");
                    break;
                }
            }

            // Small sleep to prevent busy-waiting
            Thread.Sleep(1);
        }
        catch (OperationCanceledException)
        {
            break;
        }
    }
}
```

**GetNext:**
```csharp
public T GetNext()
{
    if (!_isRunning)
        throw new InvalidOperationException("Prefetch queue is not running");

    return _queue.Dequeue();
}

public bool TryGetNext(out T batch)
{
    if (!_isRunning)
    {
        batch = default;
        return false;
    }

    return _queue.TryDequeue(out batch);
}
```

**Properties:**
```csharp
public bool IsRunning => _isRunning;
public int PrefetchCount => _prefetchCount;
public int AvailableBatches => _queue.Count;
```

**Stop:**
```csharp
public void Stop()
{
    if (!_isRunning)
        return;

    _cancellationToken.Cancel();

    try
    {
        _prefetchTask?.Wait(TimeSpan.FromSeconds(5));
    }
    catch (AggregateException)
    {
        // Ignore task cancellation exceptions
    }

    _isRunning = false;
}
```

**Dispose:**
```csharp
public void Dispose()
{
    Stop();
    _cancellationToken?.Dispose();
    _queue?.Dispose();
    _prefetchTask?.Dispose();
}
```

### Error Handling
- `InvalidOperationException` if queue not running
- Graceful handling of batch generator errors
- Timeout in Stop method (5 seconds)
- Cancellation token propagation

## Acceptance Criteria
1. PrefetchQueue maintains buffer of prefetchCount batches
2. Background thread continuously refills buffer
3. GetNext returns next batch in order
4. TryGetNext returns false if no batch available
5. Stop gracefully terminates prefetch thread
6. Dispose cleans up all resources
7. Prefetch continues until generator exhausted or stopped
8. AvailableBatches correctly reports buffer size
9. Thread-safe for concurrent get operations
10. Unit tests verify prefetch buffer management

## Files to Create
- `src/Data/Worker/IPrefetchQueue.cs`
- `src/Data/Worker/PrefetchQueue.cs`

## Tests
- `tests/Data/Worker/PrefetchQueueTests.cs`

## Usage Example
```csharp
using (var prefetchQueue = new PrefetchQueue<Batch>(
    () => GenerateBatches(),
    prefetchCount: 3))
{
    prefetchQueue.Start();

    // Consumer can immediately get pre-fetched batches
    for (int i = 0; i < 100; i++)
    {
        var batch = prefetchQueue.GetNext();
        ProcessBatch(batch);
    }
}
```

## Notes
- PrefetchCount determines buffer size (typically 2-4)
- Queue maxSize is 2x prefetchCount to accommodate bursts
- BatchGenerator should be an iterator (yield return)
- Background thread sleeps 1ms between checks to avoid CPU spinning
- Consider adding metrics (wait time, queue utilization)
- Future: dynamic prefetch count based on consumption rate
- Prefetch task runs independently of consumer thread
- Critical for GPU utilization - overlap data prep with compute
