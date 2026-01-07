namespace MLFramework.Inference.ContinuousBatching;

/// <summary>
/// Tracks individual cache pages for a request.
/// Each page represents a contiguous block of KV cache memory.
/// </summary>
public record class CachePage(
    int PageIndex,
    int StartToken,
    int TokenCount,
    int BlockIndex
)
{
    /// <summary>
    /// Gets the token index immediately following this page (exclusive end).
    /// </summary>
    public int EndToken => StartToken + TokenCount;
}
