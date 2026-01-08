namespace MLFramework.Checkpointing;

/// <summary>
/// Statistics for recomputation cache
/// </summary>
public class CacheStats
{
    public int CachedItemsCount { get; set; }
    public long CurrentSizeBytes { get; set; }
    public long MaxSizeBytes { get; set; }
    public int CacheHits { get; set; }
    public int CacheMisses { get; set; }
    public double HitRate => CacheHits + CacheMisses > 0
        ? (double)CacheHits / (CacheHits + CacheMisses) : 0.0;
}
