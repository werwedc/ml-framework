namespace MLFramework.Compilation;

/// <summary>
/// Statistics for lazy compilation
/// </summary>
public class CompilationStats
{
    /// <summary>
    /// Gets the total number of compilations performed
    /// </summary>
    public int TotalCompilations { get; set; }

    /// <summary>
    /// Gets the number of cache hits
    /// </summary>
    public int CacheHits { get; set; }

    /// <summary>
    /// Gets the number of cache misses
    /// </summary>
    public int CacheMisses { get; set; }

    /// <summary>
    /// Gets the total compilation time in milliseconds
    /// </summary>
    public long TotalCompilationTimeMs { get; set; }

    /// <summary>
    /// Gets the number of unique kernels compiled
    /// </summary>
    public int UniqueKernels { get; set; }

    /// <summary>
    /// Gets the cache hit rate (0.0 to 1.0)
    /// </summary>
    public double HitRate
    {
        get
        {
            int totalLookups = CacheHits + CacheMisses;
            return totalLookups > 0 ? (double)CacheHits / totalLookups : 0.0;
        }
    }

    /// <summary>
    /// Gets the average compilation time in milliseconds
    /// </summary>
    public double AverageCompilationTimeMs
    {
        get
        {
            return TotalCompilations > 0 ? (double)TotalCompilationTimeMs / TotalCompilations : 0.0;
        }
    }

    /// <summary>
    /// Creates a new compilation stats instance
    /// </summary>
    public CompilationStats()
    {
    }

    /// <summary>
    /// Generates a report of the compilation statistics
    /// </summary>
    public string ToReport()
    {
        return $"""
            Compilation Statistics:
            -----------------------
            Total Compilations: {TotalCompilations}
            Cache Hits: {CacheHits}
            Cache Misses: {CacheMisses}
            Hit Rate: {HitRate:P2}
            Unique Kernels: {UniqueKernels}
            Total Compilation Time: {TotalCompilationTimeMs} ms
            Average Compilation Time: {AverageCompilationTimeMs:F2} ms
            """;
    }

    /// <summary>
    /// Resets all statistics
    /// </summary>
    public void Reset()
    {
        TotalCompilations = 0;
        CacheHits = 0;
        CacheMisses = 0;
        TotalCompilationTimeMs = 0;
        UniqueKernels = 0;
    }
}
