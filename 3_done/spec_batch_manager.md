# Spec: Batch Manager for Continuous Batching

## Overview
Implement the batch manager responsible for dynamically constructing and managing batches each iteration. The batch manager handles adding new requests, removing completed requests, and ensuring optimal batch composition for each iteration.

## Class: BatchManager
```csharp
public class BatchManager
{
    private readonly Batch _currentBatch;
    private readonly RequestQueue _requestQueue;
    private readonly IKVCacheManager _kvCacheManager;
    private readonly BatchConstraints _constraints;
    private readonly object _lock;
    private int _nextBatchId;

    public Batch CurrentBatch => _currentBatch;
    public int ActiveRequestCount => _currentBatch.Size;

    public BatchManager(RequestQueue requestQueue, 
                        IKVCacheManager kvCacheManager,
                        BatchConstraints constraints)
    {
        _requestQueue = requestQueue;
        _kvCacheManager = kvCacheManager;
        _constraints = constraints;
        _currentBatch = new Batch(0);
        _lock = new object();
        _nextBatchId = 1;
    }

    // Update batch for next iteration
    public Batch PrepareNextIteration();

    // Remove completed requests from batch
    public void RemoveCompletedRequests(List<RequestId> completedRequestIds);

    // Add new requests from queue to batch
    public void AddNewRequests();

    // Check if batch has capacity for more requests
    public bool HasCapacity();

    // Force batch refresh (e.g., for recovery)
    public void RefreshBatch();

    // Get current batch statistics
    public BatchStats GetStats();
}
```

---

## Class: BatchConstraints
```csharp
public record class BatchConstraints(
    int MaxBatchSize,              // Maximum requests per batch
    long MaxMemoryBytes,           // Maximum KV cache memory
    int MinBatchSize,              // Minimum batch size for efficiency
    int MaxSequenceLength          // Maximum sequence length
)
{
    public static readonly BatchConstraints Default = new(
        MaxBatchSize: 32,
        MaxMemoryBytes: 16L * 1024 * 1024 * 1024, // 16GB
        MinBatchSize: 4,
        MaxSequenceLength: 4096
    );
}
```

**Purpose**: Define batch size and memory constraints.

---

## Implementation Details

### PrepareNextIteration
```csharp
public Batch PrepareNextIteration()
{
    lock (_lock)
    {
        // Remove completed requests from previous iteration
        RemoveCompletedRequestsInternal();

        // Add new requests if capacity allows
        AddNewRequests();

        // Return snapshot of current batch
        return new Batch(_currentBatch)
        {
            Requests = new List<Request>(_currentBatch.Requests),
            EstimatedMemoryBytes = _currentBatch.EstimatedMemoryBytes
        };
    }
}
```

**Requirements**:
- Thread-safe operation
- Remove completed requests
- Fill available capacity
- Return immutable snapshot

---

### RemoveCompletedRequestsInternal
```csharp
private void RemoveCompletedRequestsInternal()
{
    var completedRequestIds = new List<RequestId>();

    foreach (var request in _currentBatch.Requests)
    {
        // Check completion conditions
        bool isCompleted = CheckCompletion(request);
        if (isCompleted)
        {
            completedRequestIds.Add(request.Id);
        }
    }

    // Remove from batch
    foreach (var requestId in completedRequestIds)
    {
        var request = _currentBatch.GetRequest(requestId);
        if (request != null)
        {
            // Release KV cache
            _kvCacheManager.ReleaseCache(requestId);

            // Set completion result
            var result = new RequestResult(
                requestId,
                DecodeTokens(request.GeneratedTokenIds),
                request.GeneratedTokens,
                DetermineCompletionReason(request),
                DateTime.UtcNow - request.EnqueuedTime
            );
            request.CompletionSource.TrySetResult(result.GeneratedText);

            // Mark as completed
            request.IsCompleted = true;
        }

        _currentBatch.RemoveRequest(requestId);
    }
}
```

**Requirements**:
- Check all requests for completion
- Release KV cache memory
- Set completion task results
- Remove from current batch

---

### CheckCompletion
```csharp
private bool CheckCompletion(Request request)
{
    // Check cancellation
    if (request.CancellationToken.IsCancellationRequested)
        return true;

    // Check max tokens
    if (request.GeneratedTokens >= request.MaxTokens)
        return true;

    // Check for EOS token
    if (request.GeneratedTokenIds.Count > 0)
    {
        int lastToken = request.GeneratedTokenIds[^1];
        if (lastToken == EOS_TOKEN_ID)
            return true;
    }

    return false;
}
```

**Requirements**:
- Check all completion conditions
- Return true if any condition met

---

### AddNewRequests
```csharp
private void AddNewRequests()
{
    if (!HasCapacity())
        return;

    int availableSlots = _constraints.MaxBatchSize - _currentBatch.Size;
    long availableMemory = _constraints.MaxMemoryBytes - _currentBatch.EstimatedMemoryBytes;

    if (availableSlots <= 0 || availableMemory <= 0)
        return;

    // Get requests from queue
    var newRequests = _requestQueue.GetRequests(
        availableSlots,
        availableMemory
    );

    // Add to batch
    foreach (var request in newRequests)
    {
        // Allocate KV cache
        long cacheSize = _kvCacheManager.AllocateCache(request.Id, request.MaxTokens);
        if (cacheSize > availableMemory)
        {
            // Rollback allocation and skip
            _kvCacheManager.ReleaseCache(request.Id);
            _requestQueue.Enqueue(request, request.Priority); // Put back in queue
            continue;
        }

        _currentBatch.AddRequest(request);
        _currentBatch.EstimatedMemoryBytes += cacheSize;
        availableMemory -= cacheSize;
    }
}
```

**Requirements**:
- Respect batch size limits
- Respect memory limits
- Handle allocation failures gracefully
- Update memory estimates

---

### HasCapacity
```csharp
public bool HasCapacity()
{
    lock (_lock)
    {
        bool hasSizeCapacity = _currentBatch.Size < _constraints.MaxBatchSize;
        bool hasMemoryCapacity = _currentBatch.EstimatedMemoryBytes < _constraints.MaxMemoryBytes;
        return hasSizeCapacity && hasMemoryCapacity;
    }
}
```

**Requirements**:
- Check both size and memory capacity
- Thread-safe

---

### RefreshBatch
```csharp
public void RefreshBatch()
{
    lock (_lock)
)
    {
        // Complete all current requests with cancellation
        foreach (var request in _currentBatch.Requests)
        {
            if (!request.IsCompleted)
            {
                request.CompletionSource.TrySetCanceled();
                _kvCacheManager.ReleaseCache(request.Id);
            }
        }

        // Clear batch
        _currentBatch.Requests.Clear();
        _currentBatch.EstimatedMemoryBytes = 0;

        // Increment batch ID
        _nextBatchId++;
    }
}
```

**Requirements**:
- Handle error recovery scenarios
- Cancel pending requests
- Release all KV cache
- Reset batch state

---

### GetStats
```csharp
public BatchStats GetStats()
{
    lock (_lock)
    {
        return new BatchStats(
            BatchId: _currentBatch.BatchId,
            RequestCount: _currentBatch.Size,
            MemoryBytesUsed: _currentBatch.EstimatedMemoryBytes,
            UtilizationPercentage: CalculateUtilization(),
            ProcessingTime: TimeSpan.Zero // Set by scheduler
        );
    }
}

private double CalculateUtilization()
{
    if (_constraints.MaxBatchSize == 0)
        return 0.0;

    return (double)_currentBatch.Size / _constraints.MaxBatchSize * 100.0;
}
```

**Requirements**:
- Return comprehensive batch statistics
- Calculate utilization percentage

---

## Interface: IKVCacheManager
```csharp
public interface IKVCacheManager
{
    long AllocateCache(RequestId requestId, int maxTokens);
    void ReleaseCache(RequestId requestId);
    long GetCurrentUsageBytes();
}
```

**Purpose**: Abstraction for KV cache operations. Implementation provided by PagedAttention spec.

---

## Files to Create

### Implementation
- `src/MLFramework/Inference/ContinuousBatching/Scheduler/BatchManager.cs`
- `src/MLFramework/Inference/ContinuousBatching/Scheduler/BatchConstraints.cs`

### Tests
- `tests/MLFramework.Tests/Inference/ContinuousBatching/BatchManagerTests.cs`

---

## Dependencies
- `spec_continuous_batching_models.md` (Request, Batch, RequestResult)
- `spec_request_queue.md` (RequestQueue)
- IKVCacheManager (from PagedAttention implementation)

---

## Testing Requirements

### Unit Tests (with Mock IKVCacheManager)
1. **Basic Operations**:
   - PrepareNextIteration creates valid batch
   - HasCapacity returns correct values
   - GetStats returns accurate statistics

2. **Completion Handling**:
   - Completed requests removed from batch
   - KV cache released on completion
   - Completion task set correctly

3. **Adding New Requests**:
   - New requests added when capacity available
   - Batch size limit respected
   - Memory limit respected
   - Requests skipped if allocation fails

4. **Edge Cases**:
   - Empty queue with capacity
   - Full batch with pending queue
   - Memory exhaustion scenario
   - Concurrent PrepareNextIteration calls

5. **Constraints**:
   - MaxBatchSize enforced
   - MinBatchSize respected (avoid too-small batches)
   - MaxMemoryBytes enforced

---

## Success Criteria
- [ ] All public methods implemented
- [ ] Thread-safe batch management
- [ ] Completion detection works correctly
- [ ] Capacity management accurate
- [ ] KV cache integration tested (with mocks)
- [ ] Unit tests cover all scenarios
- [ ] Statistics calculation correct
