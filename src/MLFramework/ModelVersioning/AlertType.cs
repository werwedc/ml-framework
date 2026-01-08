namespace MLFramework.ModelVersioning
{
    /// <summary>
    /// Type of performance or operational alert for a model version.
    /// </summary>
    public enum AlertType
    {
        /// <summary>
        /// Request latency exceeds defined threshold.
        /// </summary>
        HighLatency,

        /// <summary>
        /// Error rate exceeds defined threshold.
        /// </summary>
        HighErrorRate,

        /// <summary>
        /// Throughput falls below defined threshold.
        /// </summary>
        LowThroughput,

        /// <summary>
        /// Memory usage exceeds defined limit.
        /// </summary>
        MemoryExceeded,

        /// <summary>
        /// Statistical anomaly detected in metrics.
        /// </summary>
        AnomalyDetected
    }
}
