namespace MLFramework.Inference.ContinuousBatching;

/// <summary>
/// Defines batch size and memory constraints for batch management.
/// </summary>
public record class BatchConstraints(
    int MaxBatchSize,              // Maximum requests per batch
    long MaxMemoryBytes,           // Maximum KV cache memory
    int MinBatchSize,              // Minimum batch size for efficiency
    int MaxSequenceLength          // Maximum sequence length
)
{
    /// <summary>
    /// Default batch constraints for typical workloads.
    /// </summary>
    public static readonly BatchConstraints Default = new(
        MaxBatchSize: 32,
        MaxMemoryBytes: 16L * 1024 * 1024 * 1024, // 16GB
        MinBatchSize: 4,
        MaxSequenceLength: 4096
    );
}
