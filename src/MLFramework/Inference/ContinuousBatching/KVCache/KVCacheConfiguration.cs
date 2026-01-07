namespace MLFramework.Inference.ContinuousBatching;

/// <summary>
/// Configuration for KV cache behavior.
/// </summary>
public record class KVCacheConfiguration(
    int PageSizeTokens,               // Tokens per cache page
    int InitialPagesPerRequest,       // Initial pages to allocate
    int MaxPagesPerRequest,          // Maximum pages per request
    int CacheBlockSizeBytes,         // Size of each cache block in bytes
    double TargetUtilization,        // Target cache utilization (0-1)
    bool EnableCompaction,           // Enable automatic compaction
    int MaxBatchSize                 // Maximum batch size for capacity planning
)
{
    /// <summary>
    /// Default configuration optimized for typical LLM workloads.
    /// </summary>
    public static readonly KVCacheConfiguration Default = new(
        PageSizeTokens: 16,
        InitialPagesPerRequest: 16,     // 256 tokens initially
        MaxPagesPerRequest: 256,        // 4096 tokens max
        CacheBlockSizeBytes: 1024,      // 1KB per block (adjust based on model)
        TargetUtilization: 0.85,
        EnableCompaction: true,
        MaxBatchSize: 64
    );

    /// <summary>
    /// Gets the size of a single page in bytes.
    /// </summary>
    public long PageSizeBytes => PageSizeTokens * CacheBlockSizeBytes;
}
