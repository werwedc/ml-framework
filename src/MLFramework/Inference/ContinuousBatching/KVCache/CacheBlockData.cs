namespace MLFramework.Inference.ContinuousBatching;

/// <summary>
/// Cache block data containing key and value tensors.
/// Used for transferring data between cache manager and model executor.
/// </summary>
public record class CacheBlockData(
    int BlockIndex,
    float[] KeyCache,
    float[] ValueCache,
    int TokenCount
);
