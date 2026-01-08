# Spec: Batch Builder

## Overview
Implement batch construction logic with timeout-based accumulation from the request queue.

## Technical Requirements

### Batch Definition
```csharp
namespace MLFramework.Serving;

/// <summary>
/// Represents a batch of requests ready for processing
/// </summary>
public class RequestBatch<TRequest>
{
    /// <summary>
    /// The requests in this batch
    /// </summary>
    public IReadOnlyList<QueuedRequest<TRequest>> Requests { get; }

    /// <summary>
    /// Timestamp when batch was created
    /// </summary>
    public DateTime BatchCreatedAt { get; }

    public RequestBatch(List<QueuedRequest<TRequest>> requests)
    {
        Requests = requests.AsReadOnly();
        BatchCreatedAt = DateTime.UtcNow;
    }

    /// <summary>
    /// Get the request payloads only
    /// </summary>
    public List<TRequest> GetPayloads()
    {
        return Requests.Select(r => r.Request).ToList();
    }
}
```

### Batch Builder
```csharp
namespace MLFramework.Serving;

/// <summary>
/// Constructs batches from request queue with timeout logic
/// </summary>
public class BatchBuilder<TRequest, TResponse>
{
    private readonly BoundedRequestQueue<TRequest, TResponse> _queue;
    private readonly BatchingConfiguration _config;

    public BatchBuilder(
        BoundedRequestQueue<TRequest, TResponse> queue,
        BatchingConfiguration config)
    {
        _queue = queue ?? throw new ArgumentNullException(nameof(queue));
        _config = config ?? throw new ArgumentNullException(nameof(config));
        _config.Validate();
    }

    /// <summary>
    /// Build a batch from the queue with timeout-based accumulation
    /// </summary>
    public async Task<RequestBatch<TRequest>> BuildBatchAsync(CancellationToken cancellationToken = default)
    {
        var batch = new List<QueuedRequest<TRequest>>();
        var startTime = DateTime.UtcNow;

        while (true)
        {
            // Check cancellation
            cancellationToken.ThrowIfCancellationRequested();

            // Dequeue available items up to MaxBatchSize
            var remainingCapacity = _config.MaxBatchSize - batch.Count;
            var items = _queue.Dequeue(remainingCapacity);
            batch.AddRange(items);

            // Check if batch is full
            if (batch.Count >= _config.MaxBatchSize)
            {
                return new RequestBatch<TRequest>(batch);
            }

            // Check if we've reached PreferBatchSize and queue is empty
            if (batch.Count >= _config.PreferBatchSize && _queue.IsEmpty)
            {
                return new RequestBatch<TRequest>(batch);
            }

            // Check timeout
            var elapsed = DateTime.UtcNow - startTime;
            if (elapsed >= _config.MaxWaitTime)
            {
                return HandleTimeout(batch);
            }

            // Wait a bit before checking again
            await Task.Delay(TimeSpan.FromMilliseconds(1), cancellationToken);
        }
    }

    private RequestBatch<TRequest> HandleTimeout(List<QueuedRequest<TRequest>> batch)
    {
        switch (_config.TimeoutStrategy)
        {
            case TimeoutStrategy.DispatchPartial:
                // Return partial batch
                return new RequestBatch<TRequest>(batch);

            case TimeoutStrategy.WaitForFull:
                // If batch is empty, throw exception
                if (batch.Count == 0)
                    throw new TimeoutException("No requests available within timeout");

                return new RequestBatch<TRequest>(batch);

            case TimeoutStrategy.Adaptive:
                // Simple adaptive strategy: if we have at least one item, dispatch
                // Otherwise wait for timeout
                return batch.Count > 0
                    ? new RequestBatch<TRequest>(batch)
                    : throw new TimeoutException("No requests available within timeout");

            default:
                throw new NotSupportedException($"Unknown timeout strategy: {_config.TimeoutStrategy}");
        }
    }
}
```

## File Location
- **Path:** `src/Serving/BatchBuilder.cs`

## Dependencies
- `BatchingConfiguration` (from spec_batching_config.md)
- `BoundedRequestQueue` (from spec_request_queue.md)

## Key Design Decisions

1. **Polling Approach**: Uses short delay (1ms) between checks to balance responsiveness and CPU usage
2. **PreferBatchSize Check**: Allows early dispatch when preferred size is reached if queue is empty
3. **Timeout Strategies**: Supports multiple strategies based on configuration
4. **Cancellation Support**: Respects cancellation tokens for graceful shutdown

## Success Criteria
- Builder correctly respects MaxBatchSize limit
- Timeout is enforced accurately
- Different timeout strategies behave correctly
- Empty batch handling is appropriate
- Builder is cancellable

## Testing Requirements
- Test batch fills to MaxBatchSize
- Test batch dispatches at PreferBatchSize when queue empties
- Test timeout expiration with partial batch
- Test timeout with no requests (throw exception or return based on strategy)
- Test all timeout strategies
- Test cancellation token behavior
- Test concurrent BuildBatchAsync calls
