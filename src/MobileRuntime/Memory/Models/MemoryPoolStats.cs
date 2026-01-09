namespace MLFramework.MobileRuntime.Memory
{
    /// <summary>
    /// Statistics and metrics about memory pool usage.
    /// </summary>
    public class MemoryPoolStats
    {
        /// <summary>
        /// Total memory allocated to the pool.
        /// </summary>
        public long TotalMemory { get; set; }

        /// <summary>
        /// Currently used memory.
        /// </summary>
        public long UsedMemory { get; set; }

        /// <summary>
        /// Available memory in the pool.
        /// </summary>
        public long AvailableMemory { get; set; }

        /// <summary>
        /// Total number of allocations performed.
        /// </summary>
        public int AllocationCount { get; set; }

        /// <summary>
        /// Total number of frees performed.
        /// </summary>
        public int FreeCount { get; set; }

        /// <summary>
        /// Number of cache hits (reusing freed blocks).
        /// </summary>
        public int CacheHits { get; set; }

        /// <summary>
        /// Number of cache misses (allocating new blocks).
        /// </summary>
        public int CacheMisses { get; set; }

        /// <summary>
        /// Peak memory usage recorded.
        /// </summary>
        public long PeakUsage { get; set; }

        /// <summary>
        /// Calculates the cache hit rate as a percentage.
        /// </summary>
        /// <returns>Cache hit rate (0-100).</returns>
        public double GetCacheHitRate()
        {
            var total = CacheHits + CacheMisses;
            return total == 0 ? 0.0 : (double)CacheHits / total * 100.0;
        }

        /// <summary>
        /// Calculates the fragmentation rate as a percentage.
        /// </summary>
        /// <returns>Fragmentation rate (0-100).</returns>
        public double GetFragmentationRate()
        {
            if (UsedMemory == 0) return 0.0;
            return (double)(TotalMemory - AvailableMemory) / TotalMemory * 100.0;
        }
    }
}
