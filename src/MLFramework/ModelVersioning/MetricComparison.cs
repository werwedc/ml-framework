using System;
using System.Collections.Generic;

namespace MLFramework.ModelVersioning
{
    /// <summary>
    /// Represents a comparison between two model versions' metrics.
    /// </summary>
    public class MetricComparison
    {
        /// <summary>
        /// Metrics for the first version (baseline).
        /// </summary>
        public VersionMetrics Version1 { get; set; }

        /// <summary>
        /// Metrics for the second version (new).
        /// </summary>
        public VersionMetrics Version2 { get; set; }

        /// <summary>
        /// Difference in latency between versions.
        /// </summary>
        public MetricDelta LatencyDelta { get; set; }

        /// <summary>
        /// Difference in error rate between versions.
        /// </summary>
        public MetricDelta ErrorRateDelta { get; set; }

        /// <summary>
        /// Difference in throughput between versions.
        /// </summary>
        public MetricDelta ThroughputDelta { get; set; }

        /// <summary>
        /// Time when the comparison was made.
        /// </summary>
        public DateTime ComparisonTime { get; set; }

        /// <summary>
        /// Creates a metric comparison between two versions.
        /// </summary>
        public static MetricComparison Create(VersionMetrics version1, VersionMetrics version2)
        {
            if (version1 == null)
            {
                throw new ArgumentNullException(nameof(version1));
            }

            if (version2 == null)
            {
                throw new ArgumentNullException(nameof(version2));
            }

            return new MetricComparison
            {
                Version1 = version1,
                Version2 = version2,
                LatencyDelta = MetricDelta.CreateForLatency(version1.AverageLatencyMs, version2.AverageLatencyMs),
                ErrorRateDelta = MetricDelta.CreateForErrorRate(version1.ErrorRate, version2.ErrorRate),
                ThroughputDelta = MetricDelta.CreateForThroughput(version1.Throughput, version2.Throughput),
                ComparisonTime = DateTime.UtcNow
            };
        }

        /// <summary>
        /// Determines if version 2 is overall better than version 1.
        /// </summary>
        public bool IsVersion2Better()
        {
            // Version 2 is better if at least one metric is better and none are worse
            bool latencyBetter = LatencyDelta.Direction == "better";
            bool errorRateBetter = ErrorRateDelta.Direction == "better";
            bool throughputBetter = ThroughputDelta.Direction == "better";

            bool latencyWorse = LatencyDelta.Direction == "worse";
            bool errorRateWorse = ErrorRateDelta.Direction == "worse";
            bool throughputWorse = ThroughputDelta.Direction == "worse";

            bool anyBetter = latencyBetter || errorRateBetter || throughputBetter;
            bool anyWorse = latencyWorse || errorRateWorse || throughputWorse;

            return anyBetter && !anyWorse;
        }

        /// <summary>
        /// Gets a summary of the comparison as a dictionary.
        /// </summary>
        public Dictionary<string, object> GetSummary()
        {
            return new Dictionary<string, object>
            {
                { "version1", Version1?.Version },
                { "version2", Version2?.Version },
                { "latencyDelta", LatencyDelta?.AbsoluteDifference },
                { "latencyPercentChange", LatencyDelta?.PercentageChange },
                { "latencyDirection", LatencyDelta?.Direction },
                { "errorRateDelta", ErrorRateDelta?.AbsoluteDifference },
                { "errorRatePercentChange", ErrorRateDelta?.PercentageChange },
                { "errorRateDirection", ErrorRateDelta?.Direction },
                { "throughputDelta", ThroughputDelta?.AbsoluteDifference },
                { "throughputPercentChange", ThroughputDelta?.PercentageChange },
                { "throughputDirection", ThroughputDelta?.Direction },
                { "isVersion2Better", IsVersion2Better() }
            };
        }
    }
}
