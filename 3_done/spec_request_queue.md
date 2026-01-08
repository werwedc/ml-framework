# Spec: Request Queue Management

## Overview
Implement a thread-safe, bounded queue for managing incoming inference requests in the batching system.

## Technical Requirements

### Queue Item Definition
```csharp
namespace MLFramework.Serving;

/// <summary>
/// Encapsulates a request with metadata for batching
/// </summary>
public class QueuedRequest<TRequest>
{
    /// <summary>
    /// Unique identifier for this request
    /// </summary>
    public string RequestId { get; }

    /// <summary>
    /// The actual request payload
    /// </summary>
    public TRequest Request { get; }

    /// <summary>
    /// Timestamp when request was enqueued
    /// </summary>
    public DateTime EnqueuedAt { get; }

    /// <summary>
    /// Task completion source for delivering response
    /// </summary>
    public TaskCompletionSource<TResponse> ResponseSource { get; }

    public QueuedRequest(TRequest request)
    {
        RequestId = Guid.NewGuid().ToString();
        Request = request;
        EnqueuedAt = DateTime.UtcNow;
        ResponseSource = new TaskCompletionSource<TResponse>();
    }
}
```

### Thread-Safe Bounded Queue
```csharp
namespace MLFramework.Serving;

/// <summary>
/// Thread-safe bounded queue for batching requests
/// </summary>
public class BoundedRequestQueue<TRequest, TResponse>
{
    private readonly Queue<QueuedRequest<TRequest>> _queue;
    private readonly SemaphoreSlim _semaphore;
    private readonly object _lock = new object();
    private readonly int _maxSize;

    public BoundedRequestQueue(int maxSize)
    {
        if (maxSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxSize));

        _maxSize = maxSize;
        _queue = new Queue<QueuedRequest<TRequest>>();
        _semaphore = new SemaphoreSlim(maxSize, maxSize);
    }

    /// <summary>
    /// Enqueue a request, returns false if queue is full
    /// </summary>
    public async Task<bool> TryEnqueueAsync(QueuedRequest<TRequest> request, CancellationToken cancellationToken = default)
    {
        // Wait for available slot
        if (!await _semaphore.WaitAsync(0, cancellationToken))
            return false;

        lock (_lock)
        {
            _queue.Enqueue(request);
        }
        return true;
    }

    /// <summary>
    /// Dequeue multiple items up to count
    /// </summary>
    public List<QueuedRequest<TRequest>> Dequeue(int count)
    {
        lock (_lock)
        {
            var items = new List<QueuedRequest<TRequest>>();
            int itemsToDequeue = Math.Min(count, _queue.Count);

            for (int i = 0; i < itemsToDequeue; i++)
            {
                items.Add(_queue.Dequeue());
                _semaphore.Release();
            }

            return items;
        }
    }

    /// <summary>
    /// Get current queue count (thread-safe)
    /// </summary>
    public int Count
    {
        get
        {
            lock (_lock)
            {
                return _queue.Count;
            }
        }
    }

    /// <summary>
    /// Check if queue is empty
    /// </summary>
    public bool IsEmpty
    {
        get
        {
            lock (_lock)
            {
                return _queue.Count == 0;
            }
        }
    }

    /// <summary>
    /// Check if queue is at capacity
    /// </summary>
    public bool IsFull
    {
        get
        {
            lock (_lock)
            {
                return _queue.Count >= _maxSize;
            }
        }
    }
}
```

## File Location
- **Path:** `src/Serving/BoundedRequestQueue.cs`

## Dependencies
- `System.Threading` (SemaphoreSlim, TaskCompletionSource)

## Key Design Decisions

1. **Bounded Queue**: Prevents memory exhaustion by enforcing maximum size
2. **Non-blocking Enqueue**: Returns false immediately if queue is full rather than waiting
3. **Bulk Dequeue**: Efficiently retrieve multiple items for batch processing
4. **Thread Safety**: Uses lock for queue operations, SemaphoreSlim for capacity management

## Success Criteria
- Queue safely handles concurrent enqueue/dequeue operations
- Capacity limits are strictly enforced
- Dequeue operations are atomic
- No deadlocks under high contention
- Queue maintains order (FIFO)

## Testing Requirements
- Test concurrent enqueue operations
- Test bulk dequeue operation
- Test queue capacity limits
- Test IsEmpty and IsFull properties under concurrency
- Test TryEnqueueAsync returns false when queue is full
- Test RequestId uniqueness
- Test EnqueuedAt accuracy
