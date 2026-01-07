# Spec: Request Queue for Continuous Batching

## Overview
Implement a thread-safe priority queue for managing pending inference requests in the continuous batching scheduler. The queue must support priority-based ordering, cancellation handling, and efficient retrieval of available requests.

## Class: RequestQueue
```csharp
public class RequestQueue : IDisposable
{
    private readonly PriorityQueue<QueuedRequest, Priority> _pendingQueue;
    private readonly Dictionary<RequestId, QueuedRequest> _requestMap;
    private readonly SemaphoreSlim _semaphore;
    private readonly object _lock;
    private int _nextSequenceNumber;

    public int Count { get; }
    public bool IsEmpty { get; }

    public RequestQueue(int initialCapacity = 100)
    {
        _pendingQueue = new PriorityQueue<QueuedRequest, Priority>(initialCapacity);
        _requestMap = new Dictionary<RequestId, QueuedRequest>();
        _semaphore = new SemaphoreSlim(0, int.MaxValue);
        _lock = new object();
        _nextSequenceNumber = 0;
    }

    // Enqueue a new request
    public void Enqueue(Request request, Priority priority = Priority.Normal);

    // Dequeue next request (blocking or with timeout)
    public Request? Dequeue(CancellationToken cancellationToken = default);
    public Request? TryDequeue(TimeSpan timeout, CancellationToken cancellationToken = default);

    // Remove a request from queue
    public bool Remove(RequestId requestId);

    // Get pending requests for batch construction
    public List<Request> GetRequests(int maxCount, long memoryBudget);

    // Check if request is in queue
    public bool Contains(RequestId requestId);

    // Clear all pending requests
    public void Clear();

    // Mark request as cancelled (called when cancellation token fires)
    public void MarkCancelled(RequestId requestId);

    public void Dispose();
}
```

---

## Class: QueuedRequest
```csharp
internal class QueuedRequest
{
    public Request Request { get; }
    public Priority Priority { get; }
    public int SequenceNumber { get; }
    public DateTime EnqueuedTime { get; }
    public CancellationTokenRegistration? CancellationRegistration { get; set; }

    public QueuedRequest(Request request, Priority priority, int sequenceNumber)
    {
        Request = request;
        Priority = priority;
        SequenceNumber = sequenceNumber;
        EnqueuedTime = DateTime.UtcNow;
    }
}
```

**Purpose**: Wrapper for requests in the queue with additional metadata.

---

## Implementation Details

### Enqueue Operation
```csharp
public void Enqueue(Request request, Priority priority = Priority.Normal)
{
    lock (_lock)
    {
        var queuedRequest = new QueuedRequest(
            request, priority, _nextSequenceNumber++
        );

        // Register cancellation callback
        queuedRequest.CancellationRegistration = request.CancellationToken.Register(
            () => MarkCancelled(request.Id)
        );

        _pendingQueue.Enqueue(queuedRequest, priority);
        _requestMap[request.Id] = queuedRequest;
        _semaphore.Release();
    }
}
```

**Requirements**:
- Thread-safe insertion
- Register cancellation callback
- Track request in map for O(1) lookup
- Signal semaphore for blocking consumers

---

### Dequeue Operation
```csharp
public Request? Dequeue(CancellationToken cancellationToken = default)
{
    _semaphore.Wait(cancellationToken);

    lock (_lock)
    {
        while (_pendingQueue.TryDequeue(out var queuedRequest, out _))
        {
            // Skip cancelled requests
            if (queuedRequest.Request.CancellationToken.IsCancellationRequested)
            {
                _requestMap.Remove(queuedRequest.Request.Id);
                continue;
            }

            _requestMap.Remove(queuedRequest.Request.Id);
            return queuedRequest.Request;
        }

        return null;
    }
}
```

**Requirements**:
- Wait for available requests
- Skip cancelled requests
- Return null if no valid requests available

---

### GetRequests for Batch Construction
```csharp
public List<Request> GetRequests(int maxCount, long memoryBudget)
{
    lock (_lock)
    {
        var selectedRequests = new List<Request>();
        var tempQueue = new List<QueuedRequest>();
        long totalMemory = 0;

        // Peek at requests in priority order
        while (selectedRequests.Count < maxCount && _pendingQueue.Count > 0)
        {
            var queuedRequest = _pendingQueue.Peek();
            var requestMemory = EstimateRequestMemory(queuedRequest.Request);

            if (totalMemory + requestMemory <= memoryBudget)
            {
                selectedRequests.Add(queuedRequest.Request);
                totalMemory += requestMemory;
                _pendingQueue.Dequeue();
                _requestMap.Remove(queuedRequest.Request.Id);
                tempQueue.Add(queuedRequest);
            }
            else
            {
                break; // Memory budget exceeded
            }
        }

        return selectedRequests;
    }
}
```

**Requirements**:
- Select up to maxCount requests
- Respect memory budget
- Maintain priority order
- Remove selected requests from queue

---

### Remove Operation
```csharp
public bool Remove(RequestId requestId)
{
    lock (_lock)
    {
        if (!_requestMap.TryGetValue(requestId, out var queuedRequest))
            return false;

        // Mark as cancelled to skip during dequeue
        queuedRequest.Request.CancellationToken.ThrowIfCancellationRequested();

        // Note: Can't efficiently remove from PriorityQueue
        // Will be skipped during Dequeue/GetRequests
        _requestMap.Remove(requestId);
        return true;
    }
}
```

**Requirements**:
- O(1) lookup via request map
- Mark for lazy removal (due to PriorityQueue limitations)
- Return success status

---

## Helper Methods

### EstimateRequestMemory
```csharp
private long EstimateRequestMemory(Request request)
{
    // Estimate memory for prompt + generation
    const int bytesPerToken = 2; // FP16
    const int kvCacheMultiplier = 2; // Key + Value

    int promptTokens = Tokenize(request.Prompt).Length;
    int estimatedGeneratedTokens = Math.Min(request.MaxTokens, 256); // Conservative estimate

    return (promptTokens + estimatedGeneratedTokens) * bytesPerToken * kvCacheMultiplier;
}
```

**Note**: Actual tokenization should be delegated to tokenizer service in production.

---

## Files to Create

### Implementation
- `src/MLFramework/Inference/ContinuousBatching/Scheduler/RequestQueue.cs`
- `src/MLFramework/Inference/ContinuousBatching/Scheduler/QueuedRequest.cs`

### Tests
- `tests/MLFramework.Tests/Inference/ContinuousBatching/RequestQueueTests.cs`

---

## Dependencies
- `spec_continuous_batching_models.md` (Request, RequestId, Priority)

---

## Testing Requirements

### Unit Tests
1. **Basic Operations**:
   - Enqueue and dequeue single request
   - Enqueue multiple requests, verify priority ordering
   - Verify Count and IsEmpty properties

2. **Cancellation Handling**:
   - Cancel request before dequeue, verify skipped
   - Cancel request after enqueue, verify proper cleanup

3. **Batch Selection**:
   - GetRequests respects maxCount
   - GetRequests respects memoryBudget
   - Priority order maintained in selection

4. **Thread Safety**:
   - Concurrent enqueue operations
   - Concurrent dequeue operations
   - Enqueue/dequeue race conditions

5. **Edge Cases**:
   - Dequeue from empty queue with timeout
   - Dequeue with cancelled CancellationToken
   - Remove non-existent request
   - Clear and verify queue is empty

---

## Success Criteria
- [ ] All public methods implemented and tested
- [ ] Thread-safe under concurrent access
- [ ] Priority ordering works correctly
- [ ] Cancellation handling works correctly
- [ ] Memory estimation reasonable
- [ ] Unit tests cover all scenarios
- [ ] No memory leaks from cancellation registrations
