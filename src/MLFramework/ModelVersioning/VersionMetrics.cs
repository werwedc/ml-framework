using System;

namespace MLFramework.ModelVersioning
{
    /// <summary>
    /// Aggregated metrics for a specific model version over a time period.
    /// </summary>
    public class VersionMetrics
    {
        /// <summary>
        /// Unique identifier for the model.
        /// </summary>
        public string ModelId { get; set; }

        /// <summary>
        /// Version tag for the model.
        /// </summary>
        public string Version { get; set; }

        /// <summary>
        /// Start time of the metrics collection period.
        /// </summary>
        public DateTime StartTime { get; set; }

        /// <summary>
        /// End time of the metrics collection period.
        /// </summary>
        public DateTime EndTime { get; set; }

        /// <summary>
        /// Total number of requests processed.
        /// </summary>
        public int TotalRequests { get; set; }

        /// <summary>
        /// Average request latency in milliseconds.
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
        /// Error rate as a fraction (0-1).
        /// </summary>
        public double ErrorRate { get; set; }

        /// <summary>
        /// Throughput in requests per second.
        /// </summary>
        public double Throughput { get; set; }

        /// <summary>
        /// Average memory usage in megabytes.
        /// </summary>
        public double MemoryUsageMB { get; set; }

        /// <summary>
        /// Validates that the metrics data is consistent.
        /// </summary>
        public bool IsValid()
        {
            // EndTime must be >= StartTime
            if (EndTime < StartTime)
            {
                return false;
            }

            // Latency values must be non-negative
            if (AverageLatencyMs < 0 || P50LatencyMs < 0 || P95LatencyMs < 0 || P99LatencyMs < 0)
            {
                return false;
            }

            // Throughput must be non-negative
            if (Throughput < 0)
            {
                return false;
            }

            // Error rate must be 0-1
            if (ErrorRate < 0 || ErrorRate > 1)
            {
                return false;
            }

            // Memory usage must be non-negative
            if (MemoryUsageMB < 0)
            {
                return false;
            }

            // Total requests must be non-negative
            if (TotalRequests < 0)
            {
                return false;
            }

            return true;
        }

        /// <summary>
        /// Calculates the duration of the metrics collection period.
        /// </summary>
        public TimeSpan GetDuration()
        {
            return EndTime - StartTime;
        }

        /// <summary>
        /// Creates a copy of this instance.
        /// </summary>
        public VersionMetrics Clone()
        {
            return new VersionMetrics
            {
                ModelId = ModelId,
                Version = Version,
                StartTime = StartTime,
                EndTime = EndTime,
                TotalRequests = TotalRequests,
                AverageLatencyMs = AverageLatencyMs,
                P50LatencyMs = P50LatencyMs,
                P95LatencyMs = P95LatencyMs,
                P99LatencyMs = P99LatencyMs,
                ErrorRate = ErrorRate,
                Throughput = Throughput,
                MemoryUsageMB = MemoryUsageMB
            };
        }
    }
}
