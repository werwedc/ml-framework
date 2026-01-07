# Spec: Shared Memory Queue

## Overview
Implement a shared memory queue for efficient communication between workers and main process.

## Requirements

### Interfaces

#### ISharedMemoryQueue<T>
```csharp
public interface ISharedMemoryQueue<T> : IDisposable
{
    void Enqueue(T item);
    T Dequeue();
    bool TryDequeue(out T item);
    int Count { get; }
    bool IsEmpty { get; }
    void Clear();
}
```

### Implementation

#### SharedMemoryQueue<T>
- High-performance concurrent queue using lock-free algorithms
- Optimized for producer-consumer scenarios
- Zero-copy design where possible
- Support for batching operations

**Key Fields:**
```csharp
public class SharedMemoryQueue<T> : ISharedMemoryQueue<T>
{
    private readonly ConcurrentQueue<T> _queue;
    private readonly SemaphoreSlim _enqueueSemaphore;
    private readonly SemaphoreSlim _dequeueSemaphore;
    private volatile bool _isDisposed;
    private readonly int _maxSize;
}
```

**Constructor:**
```csharp
public SharedMemoryQueue(int maxSize = 0)
{
    if (maxSize < 0)
        throw new ArgumentOutOfRangeException(nameof(maxSize));

    _queue = new ConcurrentQueue<T>();
    _maxSize = maxSize;

    if (maxSize > 0)
    {
        _enqueueSemaphore = new SemaphoreSlim(maxSize, maxSize);
        _dequeueSemaphore = new SemaphoreSlim(0, maxSize);
    }
    else
    {
        // Unbounded queue
        _enqueueSemaphore = null;
        _dequeueSemaphore = null;
    }

    _isDisposed = false;
}
```

**Enqueue:**
```csharp
public void Enqueue(T item)
{
    if (_isDisposed)
        throw new ObjectDisposedException(nameof(SharedMemoryQueue<T>));

    if (_maxSize > 0)
    {
        // Wait if queue is full
        _enqueueSemaphore.Wait();
    }

    _queue.Enqueue(item);

    if (_maxSize > 0)
    {
        _dequeueSemaphore.Release();
    }
}
```

**Dequeue:**
```csharp
public T Dequeue()
{
    if (_isDisposed)
        throw new ObjectDisposedException(nameof(SharedMemoryQueue<T>));

    if (_maxSize > 0)
    {
        // Wait if queue is empty
        _dequeueSemaphore.Wait();
    }

    if (_queue.TryDequeue(out var item))
    {
        if (_maxSize > 0)
        {
            _enqueueSemaphore.Release();
        }

        return item;
    }

    throw new InvalidOperationException("Queue is empty");
}
```

**TryDequeue:**
```csharp
public bool TryDequeue(out T item)
{
    if (_isDisposed)
    {
        item = default;
        return false;
    }

    if (_maxSize > 0)
    {
        if (!_dequeueSemaphore.Wait(0))
        {
            item = default;
            return false;
        }
    }

    if (_queue.TryDequeue(out item))
    {
        if (_maxSize > 0)
        {
            _enqueueSemaphore.Release();
        }

        return true;
    }

    if (_maxSize > 0)
    {
        _dequeueSemaphore.Release();
    }

    return false;
}
```

**Properties:**
```csharp
public int Count => _queue.Count;
public bool IsEmpty => _queue.IsEmpty;
```

**Clear:**
```csharp
public void Clear()
{
    if (_isDisposed)
        throw new ObjectDisposedException(nameof(SharedMemoryQueue<T>));

    while (_queue.TryDequeue(out _))
    {
        if (_maxSize > 0)
        {
            _enqueueSemaphore.Release();
        }
    }
}
```

**Dispose:**
```csharp
public void Dispose()
{
    if (_isDisposed)
        return;

    _isDisposed = true;

    // Unblock any waiting operations
    if (_maxSize > 0)
    {
        _enqueueSemaphore?.Dispose();
        _dequeueSemaphore?.Dispose();
    }

    // Clear queue
    Clear();
}
```

### Error Handling
- `ObjectDisposedException` if used after disposal
- `InvalidOperationException` if Dequeue on empty queue
- `ArgumentOutOfRangeException` for negative maxSize

## Acceptance Criteria
1. Enqueue adds items to queue
2. Dequeue removes items in FIFO order
3. TryDequeue returns false for empty queue
4. Bounded queue enforces maxSize (blocks when full)
5. Unbounded queue (maxSize=0) never blocks on enqueue
6. Clear empties the queue
7. Dispose unblocks any waiting operations
8. Count property returns correct queue size
9. Thread-safe for multiple producers/consumers
10. Unit tests verify concurrent enqueue/dequeue operations

## Files to Create
- `src/Data/Worker/ISharedMemoryQueue.cs`
- `src/Data/Worker/SharedMemoryQueue.cs`

## Tests
- `tests/Data/Worker/SharedMemoryQueueTests.cs`

## Usage Example
```csharp
using (var queue = new SharedMemoryQueue<int>(maxSize: 10))
{
    // Producer
    for (int i = 0; i < 100; i++)
    {
        queue.Enqueue(i);
    }

    // Consumer
    while (!queue.IsEmpty)
    {
        var item = queue.Dequeue();
        Process(item);
    }
}
```

## Notes
- Uses ConcurrentQueue as foundation (lock-free in .NET Core)
- SemaphoreSlim provides bounded queue semantics
- maxSize = 0 means unbounded (no blocking on enqueue)
- Memory efficient - only stores references
- Consider adding batch operations for higher throughput
- Future optimizations: ring buffer, arena allocation
- Monitor contention under high load
- This is a basic implementation - true shared memory requires unsafe code or MemoryMappedFile
