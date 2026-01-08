using System;
using System.ComponentModel.DataAnnotations;
using System.Text.Json.Serialization;

namespace MLFramework.ModelVersioning
{
    /// <summary>
    /// Performance metrics for a model
    /// </summary>
    public class PerformanceMetrics
    {
        /// <summary>
        /// Accuracy of the model (0.0 to 1.0)
        /// </summary>
        [JsonPropertyName("accuracy")]
        [Range(0.0, 1.0, ErrorMessage = "Accuracy must be between 0 and 1")]
        public float Accuracy { get; set; }

        /// <summary>
        /// Average latency in milliseconds
        /// </summary>
        [JsonPropertyName("latencyMs")]
        [Range(0.0, float.MaxValue, ErrorMessage = "Latency must be non-negative")]
        public float LatencyMs { get; set; }

        /// <summary>
        /// Throughput (requests per second)
        /// </summary>
        [JsonPropertyName("throughput")]
        [Range(0.0, float.MaxValue, ErrorMessage = "Throughput must be non-negative")]
        public float Throughput { get; set; }

        /// <summary>
        /// Memory usage in megabytes
        /// </summary>
        [JsonPropertyName("memoryUsageMB")]
        [Range(0.0, float.MaxValue, ErrorMessage = "Memory usage must be non-negative")]
        public float MemoryUsageMB { get; set; }

        /// <summary>
        /// Creates a string representation of the performance metrics
        /// </summary>
        public override string ToString()
        {
            return $"PerformanceMetrics(Accuracy: {Accuracy:P2}, Latency: {LatencyMs}ms, Throughput: {Throughput:F2} req/s, Memory: {MemoryUsageMB:F2}MB)";
        }
    }
}
