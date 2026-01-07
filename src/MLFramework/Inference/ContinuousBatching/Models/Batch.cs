using System.Collections.Concurrent;

namespace MLFramework.Inference.ContinuousBatching;

/// <summary>
/// Represents a batch of active inference requests.
/// </summary>
public class Batch
{
    private readonly ConcurrentDictionary<RequestId, Request> _requests;

    /// <summary>
    /// Sequential identifier for the batch.
    /// </summary>
    public int BatchId { get; }

    /// <summary>
    /// List of active requests in batch.
    /// </summary>
    public IReadOnlyList<Request> Requests => _requests.Values.ToList();

    /// <summary>
    /// Time when the batch was created.
    /// </summary>
    public DateTime CreatedTime { get; }

    /// <summary>
    /// Current number of requests in the batch.
    /// </summary>
    public int Size => _requests.Count;

    /// <summary>
    /// Memory estimate for capacity management.
    /// </summary>
    public long EstimatedMemoryBytes { get; set; }

    /// <summary>
    /// Creates a new batch with the specified ID.
    /// </summary>
    public Batch(int batchId)
    {
        BatchId = batchId;
        _requests = new ConcurrentDictionary<RequestId, Request>();
        CreatedTime = DateTime.UtcNow;
        EstimatedMemoryBytes = 0;
    }

    /// <summary>
    /// Adds a request to the batch.
    /// </summary>
    public void AddRequest(Request request)
    {
        if (request == null)
            throw new ArgumentNullException(nameof(request));

        _requests.TryAdd(request.Id, request);
    }

    /// <summary>
    /// Removes a request from the batch by ID.
    /// </summary>
    public void RemoveRequest(RequestId requestId)
    {
        _requests.TryRemove(requestId, out _);
    }

    /// <summary>
    /// Checks if the batch contains a request with the specified ID.
    /// </summary>
    public bool Contains(RequestId requestId)
    {
        return _requests.ContainsKey(requestId);
    }

    /// <summary>
    /// Retrieves a request by ID, or null if not found.
    /// </summary>
    public Request? GetRequest(RequestId requestId)
    {
        return _requests.TryGetValue(requestId, out var request) ? request : null;
    }
}
