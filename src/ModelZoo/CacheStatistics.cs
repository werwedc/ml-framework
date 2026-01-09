using System;
using System.Collections.Generic;
using System.Linq;

namespace MLFramework.ModelZoo
{
    /// <summary>
    /// Tracks and reports cache statistics.
    /// </summary>
    public class CacheStatistics
    {
        /// <summary>
        /// Gets or sets the total number of cached models.
        /// </summary>
        public int TotalModels { get; set; }

        /// <summary>
        /// Gets or sets the total cache size in bytes.
        /// </summary>
        public long TotalCacheSize { get; set; }

        /// <summary>
        /// Gets or sets the number of cache hits.
        /// </summary>
        public int CacheHits { get; set; }

        /// <summary>
        /// Gets or sets the total number of load attempts.
        /// </summary>
        public int TotalLoads { get; set; }

        /// <summary>
        /// Gets the cache hit rate (loads from cache / total loads).
        /// </summary>
        public double CacheHitRate => TotalLoads > 0 ? (double)CacheHits / TotalLoads : 0.0;

        /// <summary>
        /// Gets or sets the list of least recently used models.
        /// </summary>
        public List<ModelAccessInfo> LeastRecentlyUsed { get; set; }

        /// <summary>
        /// Gets or sets the list of most recently used models.
        /// </summary>
        public List<ModelAccessInfo> MostRecentlyUsed { get; set; }

        /// <summary>
        /// Initializes a new instance of the CacheStatistics class.
        /// </summary>
        public CacheStatistics()
        {
            TotalModels = 0;
            TotalCacheSize = 0;
            CacheHits = 0;
            TotalLoads = 0;
            LeastRecentlyUsed = new List<ModelAccessInfo>();
            MostRecentlyUsed = new List<ModelAccessInfo>();
        }

        /// <summary>
        /// Records a cache hit.
        /// </summary>
        public void RecordHit()
        {
            CacheHits++;
            TotalLoads++;
        }

        /// <summary>
        /// Records a cache miss.
        /// </summary>
        public void RecordMiss()
        {
            TotalLoads++;
        }

        /// <summary>
        /// Returns a formatted summary of cache statistics.
        /// </summary>
        /// <returns>Formatted string with cache statistics.</returns>
        public string GetSummary()
        {
            return $"Cache Statistics:\n" +
                   $"  Total Models: {TotalModels}\n" +
                   $"  Total Size: {FormatBytes(TotalCacheSize)}\n" +
                   $"  Cache Hits: {CacheHits}\n" +
                   $"  Total Loads: {TotalLoads}\n" +
                   $"  Hit Rate: {(CacheHitRate * 100):F2}%";
        }

        /// <summary>
        /// Formats bytes to human-readable size (e.g., "2.3 GB").
        /// </summary>
        /// <param name="bytes">Size in bytes.</param>
        /// <returns>Human-readable size string.</returns>
        public static string FormatBytes(long bytes)
        {
            string[] sizes = { "B", "KB", "MB", "GB", "TB" };
            int order = 0;
            double size = bytes;

            while (size >= 1024 && order < sizes.Length - 1)
            {
                order++;
                size /= 1024;
            }

            return $"{size:0.##} {sizes[order]}";
        }
    }

    /// <summary>
    /// Information about model access for tracking usage patterns.
    /// </summary>
    public class ModelAccessInfo
    {
        /// <summary>
        /// Gets or sets the model name.
        /// </summary>
        public string ModelName { get; set; }

        /// <summary>
        /// Gets or sets the model version.
        /// </summary>
        public string Version { get; set; }

        /// <summary>
        /// Gets or sets the last accessed time.
        /// </summary>
        public DateTime LastAccessed { get; set; }

        /// <summary>
        /// Gets or sets the access count.
        /// </summary>
        public int AccessCount { get; set; }

        /// <summary>
        /// Gets or sets the file size in bytes.
        /// </summary>
        public long FileSize { get; set; }

        /// <summary>
        /// Initializes a new instance of the ModelAccessInfo class.
        /// </summary>
        public ModelAccessInfo()
        {
            ModelName = string.Empty;
            Version = string.Empty;
            LastAccessed = DateTime.MinValue;
            AccessCount = 0;
            FileSize = 0;
        }
    }
}
