namespace MLFramework.Inference.ContinuousBatching;

/// <summary>
/// Tracks cache allocation details for a specific request.
/// </summary>
public record class CacheAllocation(
    RequestId RequestId,
    List<CachePage> Pages,
    int TotalTokensAllocated,
    long TotalBytesAllocated,
    DateTime AllocatedTime
)
{
    /// <summary>
    /// Gets the number of pages allocated for this request.
    /// </summary>
    public int PageCount => Pages.Count;

    /// <summary>
    /// Gets the age of this allocation (time since it was created).
    /// </summary>
    public TimeSpan Age => DateTime.UtcNow - AllocatedTime;
}
