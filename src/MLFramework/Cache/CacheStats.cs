using System;

namespace MLFramework.Cache
{
    /// <summary>
    /// Provides statistics about kernel cache performance and usage.
    /// </summary>
    public class CacheStats
    {
        /// <summary>
        /// Gets the total number of kernels currently in the cache.
        /// </summary>
        public int TotalKernels { get; internal set; }

        /// <summary>
        /// Gets the total number of cache hits since the cache was created or last reset.
        /// </summary>
        public long TotalHits { get; internal set; }

        /// <summary>
        /// Gets the total number of cache misses since the cache was created or last reset.
        /// </summary>
        public long TotalMisses { get; internal set; }

        /// <summary>
        /// Gets the cache hit rate (ratio of hits to total lookups).
        /// Returns 0 if no lookups have been made.
        /// </summary>
        public double HitRate
        {
            get
            {
                long totalLookups = TotalHits + TotalMisses;
                if (totalLookups == 0)
                {
                    return 0.0;
                }
                return (double)TotalHits / totalLookups;
            }
        }

        /// <summary>
        /// Gets the total time spent compiling kernels that are currently in the cache (in milliseconds).
        /// </summary>
        public long TotalCompilationTimeMs { get; internal set; }

        /// <summary>
        /// Gets the average compilation time for kernels in the cache (in milliseconds).
        /// Returns 0 if no kernels are in the cache.
        /// </summary>
        public double AverageCompilationTimeMs
        {
            get
            {
                if (TotalKernels == 0)
                {
                    return 0.0;
                }
                return (double)TotalCompilationTimeMs / TotalKernels;
            }
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="CacheStats"/> class.
        /// </summary>
        internal CacheStats()
        {
            TotalKernels = 0;
            TotalHits = 0;
            TotalMisses = 0;
            TotalCompilationTimeMs = 0;
        }

        /// <summary>
        /// Resets all statistics to zero.
        /// </summary>
        public void Reset()
        {
            TotalKernels = 0;
            TotalHits = 0;
            TotalMisses = 0;
            TotalCompilationTimeMs = 0;
        }

        /// <summary>
        /// Returns a string representation of the cache statistics.
        /// </summary>
        /// <returns>A formatted string containing all statistics.</returns>
        public override string ToString()
        {
            long totalLookups = TotalHits + TotalMisses;
            return $"CacheStats: " +
                   $"Kernels={TotalKernels}, " +
                   $"Hits={TotalHits}, " +
                   $"Misses={TotalMisses}, " +
                   $"HitRate={HitRate:P2}, " +
                   $"TotalCompTime={TotalCompilationTimeMs}ms, " +
                   $"AvgCompTime={AverageCompilationTimeMs:F2}ms, " +
                   $"TotalLookups={totalLookups}";
        }

        /// <summary>
        /// Creates a copy of these statistics.
        /// </summary>
        /// <returns>A new <see cref="CacheStats"/> object with the same values.</returns>
        public CacheStats Clone()
        {
            return new CacheStats
            {
                TotalKernels = this.TotalKernels,
                TotalHits = this.TotalHits,
                TotalMisses = this.TotalMisses,
                TotalCompilationTimeMs = this.TotalCompilationTimeMs
            };
        }
    }
}
