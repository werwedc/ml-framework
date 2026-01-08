namespace MLFramework.Memory;

/// <summary>
/// Result of a memory eviction operation.
/// </summary>
public class EvictionResult
{
    public List<string> EvictedVersions { get; }
    public long BytesFreed { get; }
    public bool Sufficient { get; }

    public EvictionResult(List<string> evictedVersions, long bytesFreed, bool sufficient)
    {
        EvictedVersions = evictedVersions;
        BytesFreed = bytesFreed;
        Sufficient = sufficient;
    }

    public override string ToString()
    {
        return $"EvictionResult: Evicted {EvictedVersions.Count} models, freed {BytesFreed:N0} bytes, sufficient: {Sufficient}";
    }
}
