namespace MLFramework.Inference.ContinuousBatching;

/// <summary>
/// Tracks resource allocation per request.
/// </summary>
/// <param name="RequestId">The ID of the request this allocation belongs to</param>
/// <param name="AllocatedMemoryBytes">Memory allocated in bytes</param>
/// <param name="AllocatedSlots">Number of slots allocated</param>
/// <param name="AllocatedTime">Timestamp when allocation was created</param>
public record class ResourceAllocation(
    RequestId RequestId,
    long AllocatedMemoryBytes,
    int AllocatedSlots,
    DateTime AllocatedTime
)
{
    /// <summary>
    /// Gets the age of this allocation.
    /// </summary>
    public TimeSpan Age => DateTime.UtcNow - AllocatedTime;
}
