namespace MLFramework.Inference.ContinuousBatching;

/// <summary>
/// Defines capacity limits and safety buffers for resource allocation.
/// </summary>
/// <param name="MaxBatchSize">Maximum requests per batch</param>
/// <param name="MaxMemoryBytes">Maximum memory in bytes</param>
/// <param name="MemoryPerSlotBytes">Estimated memory per request slot</param>
/// <param name="MaxConcurrentRequests">Maximum concurrent requests (across batches)</param>
/// <param name="MemoryBufferRatio">Safety buffer for memory (0.0-1.0)</param>
public record class CapacityConstraints(
    int MaxBatchSize,
    long MaxMemoryBytes,
    long MemoryPerSlotBytes,
    int MaxConcurrentRequests,
    double MemoryBufferRatio
)
{
    /// <summary>
    /// Default capacity constraints.
    /// </summary>
    public static readonly CapacityConstraints Default = new(
        MaxBatchSize: 32,
        MaxMemoryBytes: 16L * 1024 * 1024 * 1024, // 16GB
        MemoryPerSlotBytes: 512L * 1024 * 1024,   // 512MB per request
        MaxConcurrentRequests: 64,
        MemoryBufferRatio: 0.1  // 10% buffer
    );

    /// <summary>
    /// Gets the effective memory bytes accounting for the safety buffer.
    /// </summary>
    public long EffectiveMemoryBytes =>
        (long)(MaxMemoryBytes * (1.0 - MemoryBufferRatio));
}
