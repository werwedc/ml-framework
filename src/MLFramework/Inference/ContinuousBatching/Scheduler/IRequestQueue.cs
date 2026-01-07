namespace MLFramework.Inference.ContinuousBatching;

/// <summary>
/// Interface for managing a queue of inference requests.
/// </summary>
public interface IRequestQueue
{
    /// <summary>
    /// Gets requests from the queue based on available capacity.
    /// </summary>
    /// <param name="maxRequests">Maximum number of requests to retrieve.</param>
    /// <param name="maxMemoryBytes">Maximum memory available for the requests.</param>
    /// <returns>List of requests to add to the batch.</returns>
    List<Request> GetRequests(int maxRequests, long maxMemoryBytes);

    /// <summary>
    /// Enqueues a request back to the queue (e.g., on allocation failure).
    /// </summary>
    /// <param name="request">The request to enqueue.</param>
    /// <param name="priority">The priority level for the request.</param>
    void Enqueue(Request request, Priority priority);

    /// <summary>
    /// Gets the current count of requests in the queue.
    /// </summary>
    int Count { get; }
}
