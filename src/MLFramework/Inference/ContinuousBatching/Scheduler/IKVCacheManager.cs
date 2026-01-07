namespace MLFramework.Inference.ContinuousBatching;

/// <summary>
/// Interface for managing KV cache allocation and deallocation.
/// </summary>
public interface IKVCacheManager
{
    /// <summary>
    /// Allocates KV cache for a request.
    /// </summary>
    /// <param name="requestId">The request ID to allocate cache for.</param>
    /// <param name="maxTokens">Maximum number of tokens for the request.</param>
    /// <returns>The allocated cache size in bytes.</returns>
    long AllocateCache(RequestId requestId, int maxTokens);

    /// <summary>
    /// Releases KV cache for a request.
    /// </summary>
    /// <param name="requestId">The request ID to release cache for.</param>
    void ReleaseCache(RequestId requestId);

    /// <summary>
    /// Gets the current KV cache usage in bytes.
    /// </summary>
    /// <returns>Current usage in bytes.</returns>
    long GetCurrentUsageBytes();
}
