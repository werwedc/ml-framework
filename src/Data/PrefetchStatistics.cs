namespace MLFramework.Data
{
    /// <summary>
    /// Statistics for prefetching operations.
    /// Tracks cache hit rate, latency, and other metrics.
    /// </summary>
    public class PrefetchStatistics
    {
        /// <summary>
        /// Gets the number of cache hits (items served from prefetch buffer).
        /// </summary>
        public int CacheHits { get; set; }

        /// <summary>
        /// Gets the number of cache misses (items fetched from source queue).
        /// </summary>
        public int CacheMisses { get; set; }

        /// <summary>
        /// Gets the cache hit rate (percentage of requests served from buffer).
        /// </summary>
        public double CacheHitRate => (CacheHits + CacheMisses) > 0
            ? (double)CacheHits / (CacheHits + CacheMisses)
            : 0.0;

        /// <summary>
        /// Gets the average latency in milliseconds for retrieving items.
        /// </summary>
        public double AverageLatencyMs { get; set; }

        /// <summary>
        /// Gets the number of times the prefetch buffer was refilled.
        /// </summary>
        public int RefillCount { get; set; }

        /// <summary>
        /// Gets the number of times the prefetch buffer was empty when an item was requested.
        /// </summary>
        public int StarvationCount { get; set; }

        /// <summary>
        /// Gets the total number of items requested.
        /// </summary>
        public int TotalRequests => CacheHits + CacheMisses;

        /// <summary>
        /// Resets all statistics to zero.
        /// </summary>
        public void Reset()
        {
            CacheHits = 0;
            CacheMisses = 0;
            AverageLatencyMs = 0;
            RefillCount = 0;
            StarvationCount = 0;
        }

        public override string ToString()
        {
            return $"PrefetchStatistics[CacheHits={CacheHits}, CacheMisses={CacheMisses}, " +
                   $"CacheHitRate={CacheHitRate:P2}, AvgLatency={AverageLatencyMs:F2}ms, " +
                   $"RefillCount={RefillCount}, StarvationCount={StarvationCount}]";
        }
    }
}
