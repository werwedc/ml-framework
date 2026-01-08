using System;
using System.Collections.Generic;

namespace MLFramework.Serving
{
    /// <summary>
    /// Metrics collected for a specific experiment and model version during A/B testing.
    /// </summary>
    public class ExperimentMetrics
    {
        /// <summary>
        /// Unique identifier for the experiment.
        /// </summary>
        public string ExperimentId { get; set; }

        /// <summary>
        /// Name of the model being tested.
        /// </summary>
        public string ModelName { get; set; }

        /// <summary>
        /// Model version being tracked.
        /// </summary>
        public string Version { get; set; }

        /// <summary>
        /// When the experiment started.
        /// </summary>
        public DateTime StartTime { get; set; }

        /// <summary>
        /// When the experiment ended (null if still active).
        /// </summary>
        public DateTime? EndTime { get; set; }

        /// <summary>
        /// Total number of inference requests.
        /// </summary>
        public int RequestCount { get; set; }

        /// <summary>
        /// Number of successful inference requests.
        /// </summary>
        public int SuccessCount { get; set; }

        /// <summary>
        /// Number of failed inference requests.
        /// </summary>
        public int ErrorCount { get; set; }

        /// <summary>
        /// Average latency in milliseconds.
        /// </summary>
        public double AverageLatencyMs { get; set; }

        /// <summary>
        /// 50th percentile latency (median) in milliseconds.
        /// </summary>
        public double P50LatencyMs { get; set; }

        /// <summary>
        /// 95th percentile latency in milliseconds.
        /// </summary>
        public double P95LatencyMs { get; set; }

        /// <summary>
        /// 99th percentile latency in milliseconds.
        /// </summary>
        public double P99LatencyMs { get; set; }

        /// <summary>
        /// Custom metrics recorded during the experiment.
        /// </summary>
        public Dictionary<string, double> CustomMetrics { get; set; }

        /// <summary>
        /// Success rate as a percentage.
        /// </summary>
        public double SuccessRate => RequestCount > 0 ? (SuccessCount * 100.0 / RequestCount) : 0;

        /// <summary>
        /// Error rate as a percentage.
        /// </summary>
        public double ErrorRate => RequestCount > 0 ? (ErrorCount * 100.0 / RequestCount) : 0;

        public ExperimentMetrics()
        {
            CustomMetrics = new Dictionary<string, double>();
        }
    }
}
