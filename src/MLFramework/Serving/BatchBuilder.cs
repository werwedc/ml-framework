using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace MLFramework.Serving;

/// <summary>
/// Represents a batch of requests ready for processing
/// </summary>
public class RequestBatch<TRequest, TResponse>
{
    /// <summary>
    /// The requests in this batch
    /// </summary>
    public IReadOnlyList<QueuedRequest<TRequest, TResponse>> Requests { get; }

    /// <summary>
    /// Timestamp when batch was created
    /// </summary>
    public DateTime BatchCreatedAt { get; }

    public RequestBatch(List<QueuedRequest<TRequest, TResponse>> requests)
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
    public async Task<RequestBatch<TRequest, TResponse>> BuildBatchAsync(CancellationToken cancellationToken = default)
    {
        var batch = new List<QueuedRequest<TRequest, TResponse>>();
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
                return new RequestBatch<TRequest, TResponse>(batch);
            }

            // Check if we've reached PreferBatchSize and queue is empty
            if (batch.Count >= _config.PreferBatchSize && _queue.IsEmpty)
            {
                return new RequestBatch<TRequest, TResponse>(batch);
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

    private RequestBatch<TRequest, TResponse> HandleTimeout(List<QueuedRequest<TRequest, TResponse>> batch)
    {
        switch (_config.TimeoutStrategy)
        {
            case TimeoutStrategy.DispatchPartial:
                // Return partial batch
                return new RequestBatch<TRequest, TResponse>(batch);

            case TimeoutStrategy.WaitForFull:
                // If batch is empty, throw exception
                if (batch.Count == 0)
                    throw new TimeoutException("No requests available within timeout");

                return new RequestBatch<TRequest, TResponse>(batch);

            case TimeoutStrategy.Adaptive:
                // Simple adaptive strategy: if we have at least one item, dispatch
                // Otherwise wait for timeout
                return batch.Count > 0
                    ? new RequestBatch<TRequest, TResponse>(batch)
                    : throw new TimeoutException("No requests available within timeout");

            default:
                throw new NotSupportedException($"Unknown timeout strategy: {_config.TimeoutStrategy}");
        }
    }
}
