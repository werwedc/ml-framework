namespace MLFramework.Compilation;

/// <summary>
/// Statistics for kernel cache
/// </summary>
public class CacheStats
{
    /// <summary>
    /// Gets the total number of kernels in the cache
    /// </summary>
    public int TotalKernels { get; set; }

    /// <summary>
    /// Gets the total number of cache hits
    /// </summary>
    public long TotalHits { get; set; }

    /// <summary>
    /// Gets the total number of cache misses
    /// </summary>
    public long TotalMisses { get; set; }

    /// <summary>
    /// Gets the cache hit rate (0.0 to 1.0)
    /// </summary>
    public double HitRate
    {
        get
        {
            long totalLookups = TotalHits + TotalMisses;
            return totalLookups > 0 ? (double)TotalHits / totalLookups : 0.0;
        }
    }

    /// <summary>
    /// Gets the total compilation time in milliseconds
    /// </summary>
    public long TotalCompilationTimeMs { get; set; }

    /// <summary>
    /// Gets the average compilation time in milliseconds
    /// </summary>
    public double AverageCompilationTimeMs
    {
        get
        {
            return TotalMisses > 0 ? (double)TotalCompilationTimeMs / TotalMisses : 0.0;
        }
    }

    /// <summary>
    /// Creates a new cache stats instance
    /// </summary>
    public CacheStats()
    {
    }

    /// <summary>
    /// Resets all statistics
    /// </summary>
    public void Reset()
    {
        TotalKernels = 0;
        TotalHits = 0;
        TotalMisses = 0;
        TotalCompilationTimeMs = 0;
    }

    /// <summary>
    /// Generates a report of the cache statistics
    /// </summary>
    public string ToReport()
    {
        return $"""
            Cache Statistics:
            ----------------
            Total Kernels: {TotalKernels}
            Total Hits: {TotalHits}
            Total Misses: {TotalMisses}
            Hit Rate: {HitRate:P2}
            Total Compilation Time: {TotalCompilationTimeMs} ms
            Average Compilation Time: {AverageCompilationTimeMs:F2} ms
            """;
    }
}
