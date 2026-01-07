using System.Diagnostics;

namespace MLFramework.Inference.ContinuousBatching;

/// <summary>
/// Manages dynamic batch construction and iteration for continuous batching.
/// </summary>
public class BatchManager
{
    private const int EOS_TOKEN_ID = 0; // Default EOS token ID

    private readonly Batch _currentBatch;
    private readonly IRequestQueue _requestQueue;
    private readonly IKVCacheManager _kvCacheManager;
    private readonly BatchConstraints _constraints;
    private readonly object _lock;
    private int _nextBatchId;

    /// <summary>
    /// Gets the current batch being processed.
    /// </summary>
    public Batch CurrentBatch => _currentBatch;

    /// <summary>
    /// Gets the number of active requests in the current batch.
    /// </summary>
    public int ActiveRequestCount => _currentBatch.Size;

    /// <summary>
    /// Creates a new batch manager.
    /// </summary>
    /// <param name="requestQueue">The request queue to draw new requests from.</param>
    /// <param name="kvCacheManager">The KV cache manager for memory management.</param>
    /// <param name="constraints">Batch size and memory constraints.</param>
    public BatchManager(
        IRequestQueue requestQueue,
        IKVCacheManager kvCacheManager,
        BatchConstraints constraints)
    {
        _requestQueue = requestQueue ?? throw new ArgumentNullException(nameof(requestQueue));
        _kvCacheManager = kvCacheManager ?? throw new ArgumentNullException(nameof(kvCacheManager));
        _constraints = constraints ?? throw new ArgumentNullException(nameof(constraints));
        _currentBatch = new Batch(0);
        _lock = new object();
        _nextBatchId = 1;
    }

    /// <summary>
    /// Updates the batch for the next iteration.
    /// </summary>
    /// <returns>A snapshot of the current batch for processing.</returns>
    public Batch PrepareNextIteration()
    {
        lock (_lock)
        {
            // Remove completed requests from previous iteration
            RemoveCompletedRequestsInternal();

            // Add new requests if capacity allows
            AddNewRequests();

            // Return snapshot of current batch
            return new Batch(_currentBatch.BatchId)
            {
                Requests = new List<Request>(_currentBatch.Requests)
            };
        }
    }

    /// <summary>
    /// Removes completed requests from the batch.
    /// </summary>
    /// <param name="completedRequestIds">List of request IDs that have completed.</param>
    public void RemoveCompletedRequests(List<RequestId> completedRequestIds)
    {
        if (completedRequestIds == null)
            throw new ArgumentNullException(nameof(completedRequestIds));

        lock (_lock)
        {
            foreach (var requestId in completedRequestIds)
            {
                var request = _currentBatch.GetRequest(requestId);
                if (request != null && !request.IsCompleted)
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

            // Update memory estimate
            _currentBatch.EstimatedMemoryBytes = _kvCacheManager.GetCurrentUsageBytes();
        }
    }

    /// <summary>
    /// Adds new requests from the queue to the batch.
    /// </summary>
    public void AddNewRequests()
    {
        lock (_lock)
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
    }

    /// <summary>
    /// Checks if the batch has capacity for more requests.
    /// </summary>
    /// <returns>True if capacity is available, false otherwise.</returns>
    public bool HasCapacity()
    {
        lock (_lock)
        {
            bool hasSizeCapacity = _currentBatch.Size < _constraints.MaxBatchSize;
            bool hasMemoryCapacity = _currentBatch.EstimatedMemoryBytes < _constraints.MaxMemoryBytes;
            return hasSizeCapacity && hasMemoryCapacity;
        }
    }

    /// <summary>
    /// Forces a batch refresh for recovery scenarios.
    /// </summary>
    public void RefreshBatch()
    {
        lock (_lock)
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
            _currentBatch.EstimatedMemoryBytes = 0;

            // Increment batch ID
            _nextBatchId++;
        }
    }

    /// <summary>
    /// Gets current batch statistics.
    /// </summary>
    /// <returns>Statistics about the current batch.</returns>
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

    /// <summary>
    /// Removes completed requests from the batch (internal method).
    /// </summary>
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

        // Update memory estimate
        _currentBatch.EstimatedMemoryBytes = _kvCacheManager.GetCurrentUsageBytes();
    }

    /// <summary>
    /// Checks if a request has completed.
    /// </summary>
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

    /// <summary>
    /// Decodes token IDs to text (placeholder implementation).
    /// </summary>
    private string DecodeTokens(List<int> tokenIds)
    {
        // Placeholder: In a real implementation, this would use a tokenizer
        // For now, return a placeholder string
        return $"[{string.Join(", ", tokenIds)}]";
    }

    /// <summary>
    /// Determines the completion reason for a request.
    /// </summary>
    private CompletionReason DetermineCompletionReason(Request request)
    {
        // Check cancellation first
        if (request.CancellationToken.IsCancellationRequested)
            return CompletionReason.Cancelled;

        // Check for EOS token
        if (request.GeneratedTokenIds.Count > 0)
        {
            int lastToken = request.GeneratedTokenIds[^1];
            if (lastToken == EOS_TOKEN_ID)
                return CompletionReason.EosTokenReached;
        }

        // Check max tokens
        if (request.GeneratedTokens >= request.MaxTokens)
            return CompletionReason.MaxTokensReached;

        // Default case (shouldn't normally happen)
        return CompletionReason.LengthReached;
    }

    /// <summary>
    /// Calculates the batch utilization percentage.
    /// </summary>
    private double CalculateUtilization()
    {
        if (_constraints.MaxBatchSize == 0)
            return 0.0;

        return (double)_currentBatch.Size / _constraints.MaxBatchSize * 100.0;
    }
}
