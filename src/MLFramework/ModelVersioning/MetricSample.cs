using System;

namespace MLFramework.ModelVersioning
{
    /// <summary>
    /// Represents a single metric sample collected during request processing.
    /// </summary>
    public class MetricSample
    {
        /// <summary>
        /// Timestamp when the sample was recorded.
        /// </summary>
        public DateTime Timestamp { get; set; }

        /// <summary>
        /// Request latency in milliseconds.
        /// </summary>
        public double LatencyMs { get; set; }

        /// <summary>
        /// Whether the request was successful.
        /// </summary>
        public bool Success { get; set; }

        /// <summary>
        /// Memory usage at sample time in megabytes.
        /// </summary>
        public double MemoryUsageMB { get; set; }

        /// <summary>
        /// Creates a new metric sample with current timestamp.
        /// </summary>
        public static MetricSample Create(double latencyMs, bool success, double memoryUsageMB)
        {
            return new MetricSample
            {
                Timestamp = DateTime.UtcNow,
                LatencyMs = latencyMs,
                Success = success,
                MemoryUsageMB = memoryUsageMB
            };
        }

        /// <summary>
        /// Creates a copy of this sample.
        /// </summary>
        public MetricSample Clone()
        {
            return new MetricSample
            {
                Timestamp = Timestamp,
                LatencyMs = LatencyMs,
                Success = Success,
                MemoryUsageMB = MemoryUsageMB
            };
        }
    }
}
