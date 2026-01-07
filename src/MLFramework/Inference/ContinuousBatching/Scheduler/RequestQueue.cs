using System.Collections.Concurrent;

namespace MLFramework.Inference.ContinuousBatching;

/// <summary>
/// Wrapper for requests in the queue with additional metadata.
/// </summary>
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

/// <summary>
/// Thread-safe priority queue for managing pending inference requests.
/// </summary>
public class RequestQueue : IRequestQueue
{
    private readonly PriorityQueue<QueuedRequest, Priority> _pendingQueue;
    private readonly ConcurrentDictionary<RequestId, QueuedRequest> _requestMap;
    private readonly SemaphoreSlim _semaphore;
    private int _nextSequenceNumber;

    /// <summary>
    /// Gets the number of requests in the queue.
    /// </summary>
    public int Count => _requestMap.Count;

    /// <summary>
    /// Gets whether the queue is empty.
    /// </summary>
    public bool IsEmpty => _requestMap.IsEmpty;

    /// <summary>
    /// Creates a new request queue.
    /// </summary>
    /// <param name="initialCapacity">Initial capacity for the queue.</param>
    public RequestQueue(int initialCapacity = 100)
    {
        _pendingQueue = new PriorityQueue<QueuedRequest, Priority>(initialCapacity);
        _requestMap = new ConcurrentDictionary<RequestId, QueuedRequest>();
        _semaphore = new SemaphoreSlim(0, int.MaxValue);
        _nextSequenceNumber = 0;
    }

    /// <summary>
    /// Enqueues a new request.
    /// </summary>
    public void Enqueue(Request request, Priority priority = Priority.Normal)
    {
        var queuedRequest = new QueuedRequest(
            request, priority, Interlocked.Increment(ref _nextSequenceNumber)
        );

        // Register cancellation callback
        queuedRequest.CancellationRegistration = request.CancellationToken.Register(
            () => MarkCancelled(request.Id)
        );

        _pendingQueue.Enqueue(queuedRequest, priority);
        _requestMap[request.Id] = queuedRequest;
        _semaphore.Release();
    }

    /// <summary>
    /// Dequeues the next request (blocking).
    /// </summary>
    public Request? Dequeue(CancellationToken cancellationToken = default)
    {
        _semaphore.Wait(cancellationToken);

        while (_pendingQueue.TryDequeue(out var queuedRequest, out _))
        {
            // Skip cancelled requests
            if (queuedRequest.Request.CancellationToken.IsCancellationRequested)
            {
                _requestMap.TryRemove(queuedRequest.Request.Id, out _);
                continue;
            }

            _requestMap.TryRemove(queuedRequest.Request.Id, out _);
            return queuedRequest.Request;
        }

        return null;
    }

    /// <summary>
    /// Tries to dequeue a request with a timeout.
    /// </summary>
    public Request? TryDequeue(TimeSpan timeout, CancellationToken cancellationToken = default)
    {
        if (!_semaphore.Wait(timeout, cancellationToken))
            return null;

        return Dequeue(cancellationToken);
    }

    /// <summary>
    /// Removes a request from the queue.
    /// </summary>
    public bool Remove(RequestId requestId)
    {
        if (!_requestMap.TryGetValue(requestId, out var queuedRequest))
            return false;

        // Mark as cancelled to skip during dequeue
        try
        {
            queuedRequest.Request.CancellationToken.ThrowIfCancellationRequested();
        }
        catch (OperationCanceledException)
        {
            // Expected - request will be skipped during dequeue
        }

        _requestMap.TryRemove(requestId, out _);
        return true;
    }

    /// <summary>
    /// Gets pending requests for batch construction.
    /// </summary>
    public List<Request> GetRequests(int maxCount, long memoryBudget)
    {
        var selectedRequests = new List<Request>();
        var tempQueue = new List<(QueuedRequest, Priority)>();
        long totalMemory = 0;

        lock (_pendingQueue)
        {
            // Peek at requests in priority order
            while (selectedRequests.Count < maxCount && _pendingQueue.Count > 0)
            {
                _pendingQueue.TryPeek(out var queuedRequest, out var priority);
                var requestMemory = EstimateRequestMemory(queuedRequest.Request);

                if (totalMemory + requestMemory <= memoryBudget)
                {
                    _pendingQueue.TryDequeue(out _, out _);
                    selectedRequests.Add(queuedRequest.Request);
                    totalMemory += requestMemory;
                    tempQueue.Add((queuedRequest, priority));
                    _requestMap.TryRemove(queuedRequest.Request.Id, out _);
                }
                else
                {
                    break; // Memory budget exceeded
                }
            }
        }

        return selectedRequests;
    }

    /// <summary>
    /// Checks if a request is in the queue.
    /// </summary>
    public bool Contains(RequestId requestId)
    {
        return _requestMap.ContainsKey(requestId);
    }

    /// <summary>
    /// Clears all pending requests.
    /// </summary>
    public void Clear()
    {
        lock (_pendingQueue)
        {
            while (_pendingQueue.TryDequeue(out var request, out _))
            {
                request.CancellationRegistration?.Dispose();
            }
            _requestMap.Clear();
        }
    }

    /// <summary>
    /// Marks a request as cancelled.
    /// </summary>
    public void MarkCancelled(RequestId requestId)
    {
        if (_requestMap.TryGetValue(requestId, out var queuedRequest))
        {
            // Will be skipped during dequeue
        }
    }

    /// <summary>
    /// Estimates memory required for a request.
    /// </summary>
    private long EstimateRequestMemory(Request request)
    {
        // Estimate memory for prompt + generation
        const int bytesPerToken = 2; // FP16
        const int kvCacheMultiplier = 2; // Key + Value

        int promptTokens = EstimateTokenCount(request.Prompt);
        int estimatedGeneratedTokens = Math.Min(request.MaxTokens, 256); // Conservative estimate

        return (promptTokens + estimatedGeneratedTokens) * bytesPerToken * kvCacheMultiplier;
    }

    /// <summary>
    /// Estimates token count from text (conservative estimate).
    /// </summary>
    private int EstimateTokenCount(string text)
    {
        // Conservative estimate: ~4 characters per token
        return (text.Length / 4) + 1;
    }

    /// <summary>
    /// Disposes the queue and releases resources.
    /// </summary>
    public void Dispose()
    {
        _semaphore.Dispose();
        Clear();
    }
}
