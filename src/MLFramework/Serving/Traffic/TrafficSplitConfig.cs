using System;
using System.Collections.Generic;

namespace MLFramework.Serving.Traffic
{
    /// <summary>
    /// Configuration for traffic splitting between model versions.
    /// </summary>
    public class TrafficSplitConfig
    {
        /// <summary>
        /// Dictionary mapping version names to their percentage allocations (0.0-1.0).
        /// All percentages must sum to 1.0.
        /// </summary>
        public Dictionary<string, float> VersionPercentages { get; set; }

        /// <summary>
        /// The timestamp when this configuration was last updated.
        /// </summary>
        public DateTime LastUpdated { get; set; }

        /// <summary>
        /// The user or system that last updated this configuration.
        /// </summary>
        public string UpdatedBy { get; set; }

        /// <summary>
        /// Cached percentage ranges for efficient version selection.
        /// Each entry represents [start, end) range for a version.
        /// </summary>
        internal List<VersionRange> CachedRanges { get; set; }

        /// <summary>
        /// Internal structure for caching percentage ranges.
        /// </summary>
        internal class VersionRange
        {
            public string Version { get; set; }
            public float Start { get; set; }
            public float End { get; set; }
        }

        public TrafficSplitConfig()
        {
            VersionPercentages = new Dictionary<string, float>();
            CachedRanges = new List<VersionRange>();
            LastUpdated = DateTime.UtcNow;
            UpdatedBy = "system";
        }

        /// <summary>
        /// Rebuilds the cached percentage ranges based on current VersionPercentages.
        /// </summary>
        internal void RebuildRanges()
        {
            CachedRanges.Clear();

            float cumulative = 0.0f;
            var sortedVersions = new List<KeyValuePair<string, float>>(VersionPercentages);
            sortedVersions.Sort((a, b) => a.Key.CompareTo(b.Key));

            foreach (var kvp in sortedVersions)
            {
                CachedRanges.Add(new VersionRange
                {
                    Version = kvp.Key,
                    Start = cumulative,
                    End = cumulative + kvp.Value
                });
                cumulative += kvp.Value;
            }
        }
    }
}
