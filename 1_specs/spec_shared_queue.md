# Spec: Shared Memory Queue

## Overview
Implement a thread-safe, blocking queue for communication between data loading workers and the main training process. This queue forms the backbone of the producer-consumer pattern.

## Requirements

### 1. SharedQueue<T> Class
Thread-safe blocking queue built on `BlockingCollection<T>`.

**Constructor:**
```csharp
public SharedQueue(int capacity, CancellationToken? cancellationToken = null)
```

**Parameters:**
- `capacity`: Maximum number of items in the queue (must be > 0)
- `cancellationToken`: Optional cancellation token for graceful shutdown

**Properties:**
```csharp
public int Count { get; }              // Current number of items in queue
public bool IsCompleted { get; }       // True if queue is marked complete and empty
public int Capacity { get; }           // Maximum capacity
```

### 2. Producer Methods (Called by Workers)

**Enqueue with Blocking:**
```csharp
public void Enqueue(T item)
```

**Behavior:**
- Blocks if queue is full until space becomes available
- Throws `OperationCanceledException` if cancellation token is triggered
- Throws `InvalidOperationException` if queue is marked complete

**Enqueue with Timeout:**
```csharp
public bool TryEnqueue(T item, int timeoutMilliseconds)
```

**Behavior:**
- Returns `false` if timeout elapses without space becoming available
- Returns `true` on successful enqueue
- Throws `OperationCanceledException` if cancellation token is triggered
- Throws `InvalidOperationException` if queue is marked complete

### 3. Consumer Methods (Called by Training Loop)

**Dequeue with Blocking:**
```csharp
public T Dequeue()
```

**Behavior:**
- Blocks if queue is empty until an item is available
- Throws `OperationCanceledException` if cancellation token is triggered
- Returns next item in FIFO order

**Dequeue with Timeout:**
```csharp
public bool TryDequeue(out T item, int timeoutMilliseconds)
```

**Behavior:**
- Returns `false` if timeout elapses without item becoming available
- Returns `true` with item on successful dequeue
- Throws `OperationCanceledException` if cancellation token is triggered

**Peek (Non-blocking):**
```csharp
public bool TryPeek(out T item)
```

**Behavior:**
- Returns `false` if queue is empty
- Returns `true` with item if queue has items
- Does not remove item from queue

### 4. Queue Completion Methods

**Mark Complete (Producers call this when done):**
```csharp
public void CompleteAdding()
```

**Behavior:**
- Signals that no more items will be added
- Existing items can still be dequeued
- Subsequent `Enqueue` calls throw `InvalidOperationException`

**Wait for Completion:**
```csharp
public void WaitForCompletion()
```

**Behavior:**
- Blocks until all items are dequeued
- Returns immediately if already complete
- Useful for graceful shutdown

### 5. Shutdown Methods

**Immediate Shutdown:**
```csharp
public void Shutdown()
```

**Behavior:**
- Marks queue as complete
- Cancels all blocking operations
- Causes all blocked threads to throw `OperationCanceledException`

### 6. Batch Operations (Optimization)

**Enqueue Batch:**
```csharp
public void EnqueueBatch(IEnumerable<T> items)
```

**Behavior:**
- Enqueues multiple items atomically
- More efficient than individual enqueues
- Blocks if insufficient space for entire batch

**Dequeue Batch:**
```csharp
public T[] DequeueBatch(int batchSize, int timeoutMilliseconds)
```

**Behavior:**
- Attempts to dequeue specified number of items
- Returns array with available items (may be fewer if timeout)
- More efficient than individual dequeues

### 7. Statistics and Monitoring

**Queue Statistics:**
```csharp
public class QueueStatistics
{
    public int TotalEnqueued { get; }
    public int TotalDequeued { get; }
    public long AverageWaitTimeMs { get; }
    public long MaxQueueSize { get; }
    public int ProducerWaitCount { get; }
    public int ConsumerWaitCount { get; }
}

public QueueStatistics GetStatistics()
```

## File Structure
```
src/
  Data/
    SharedQueue.cs        (Main queue implementation)
    QueueStatistics.cs    (Statistics class)
```

## Success Criteria
- [ ] Thread-safe for concurrent producers and consumers
- [ ] Blocks appropriately when queue is full or empty
- [ ] Supports cancellation token for graceful shutdown
- [ ] `CompleteAdding()` prevents further additions
- [ ] Batch operations are more efficient than individual operations
- [ ] Statistics tracking is accurate
- [ ] All edge cases handled (timeout, cancellation, etc.)
- [ ] Unit tests cover multi-threaded scenarios

## Notes
- Use `BlockingCollection<T>` as the underlying implementation
- Add XML documentation for all public methods
- Consider implementing `IEnumerable<T>` for iteration
- Statistics should be optional (can be disabled for performance)
- Use `ConcurrentDictionary` for statistics if enabling
- This queue will be used in spec_worker_pool.md
