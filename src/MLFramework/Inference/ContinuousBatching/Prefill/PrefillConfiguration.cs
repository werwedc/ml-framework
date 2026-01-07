namespace MLFramework.Inference.ContinuousBatching.Prefill;

/// <summary>
/// Configuration for prefill behavior.
/// </summary>
public record class PrefillConfiguration(
    int MaxPrefillBatchSize,          // Max requests in prefill batch
    long MaxPrefillMemoryBytes,        // Max memory for prefill
    int PrefillChunkSize,              // Chunk size for long prompts
    bool EnablePrefillCaching,         // Cache common prompts
    int PrefillCacheMaxEntries,        // Max entries in prefill cache
    int PrefillCacheTtlSeconds         // TTL for cache entries
)
{
    /// <summary>
    /// Default prefill configuration.
    /// </summary>
    public static readonly PrefillConfiguration Default = new(
        MaxPrefillBatchSize: 8,
        MaxPrefillMemoryBytes: 4L * 1024 * 1024 * 1024, // 4GB
        PrefillChunkSize: 512,
        EnablePrefillCaching: true,
        PrefillCacheMaxEntries: 100,
        PrefillCacheTtlSeconds: 300
    );
}
