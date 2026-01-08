namespace MLFramework.Data.Metrics;

/// <summary>
/// Represents aggregated metrics for a specific model version over a time window.
/// </summary>
public class VersionMetrics
{
    /// <summary>
    /// Gets the model name.
    /// </summary>
    public string ModelName { get; }

    /// <summary>
    /// Gets the model version.
    /// </summary>
    public string Version { get; }

    /// <summary>
    /// Gets the start time of the metrics window.
    /// </summary>
    public DateTime WindowStart { get; }

    /// <summary>
    /// Gets the end time of the metrics window.
    /// </summary>
    public DateTime WindowEnd { get; }

    /// <summary>
    /// Gets the total number of inference requests in the window.
    /// </summary>
    public long RequestCount { get; }

    /// <summary>
    /// Gets the number of requests per second in the window.
    /// </summary>
    public double RequestsPerSecond { get; }

    /// <summary>
    /// Gets the average latency in milliseconds for requests in the window.
    /// </summary>
    public double AverageLatencyMs { get; }

    /// <summary>
    /// Gets the 50th percentile latency in milliseconds.
    /// </summary>
    public double P50LatencyMs { get; }

    /// <summary>
    /// Gets the 95th percentile latency in milliseconds.
    /// </summary>
    public double P95LatencyMs { get; }

    /// <summary>
    /// Gets the 99th percentile latency in milliseconds.
    /// </summary>
    public double P99LatencyMs { get; }

    /// <summary>
    /// Gets the error rate as a percentage (0-100).
    /// </summary>
    public double ErrorRate { get; }

    /// <summary>
    /// Gets the number of active connections for this version.
    /// </summary>
    public long ActiveConnections { get; }

    /// <summary>
    /// Gets the memory usage in megabytes.
    /// </summary>
    public double MemoryUsageMB { get; }

    /// <summary>
    /// Gets the total number of errors encountered.
    /// </summary>
    public long ErrorCount { get; }

    /// <summary>
    /// Gets the breakdown of errors by type.
    /// </summary>
    public Dictionary<string, long> ErrorCountsByType { get; }

    public VersionMetrics(
        string modelName,
        string version,
        DateTime windowStart,
        DateTime windowEnd,
        long requestCount,
        double requestsPerSecond,
        double averageLatencyMs,
        double p50LatencyMs,
        double p95LatencyMs,
        double p99LatencyMs,
        double errorRate,
        long activeConnections,
        double memoryUsageMB,
        long errorCount,
        Dictionary<string, long> errorCountsByType)
    {
        ModelName = modelName ?? throw new ArgumentNullException(nameof(modelName));
        Version = version ?? throw new ArgumentNullException(nameof(version));
        WindowStart = windowStart;
        WindowEnd = windowEnd;
        RequestCount = requestCount;
        RequestsPerSecond = requestsPerSecond;
        AverageLatencyMs = averageLatencyMs;
        P50LatencyMs = p50LatencyMs;
        P95LatencyMs = p95LatencyMs;
        P99LatencyMs = p99LatencyMs;
        ErrorRate = errorRate;
        ActiveConnections = activeConnections;
        MemoryUsageMB = memoryUsageMB;
        ErrorCount = errorCount;
        ErrorCountsByType = errorCountsByType ?? new Dictionary<string, long>();
    }

    public override string ToString()
    {
        return $"VersionMetrics({ModelName}:{Version}, " +
               $"Window: {WindowStart:HH:mm:ss} - {WindowEnd:HH:mm:ss}, " +
               $"Requests: {RequestCount}, RPS: {RequestsPerSecond:F2}, " +
               $"Latency: P50={P50LatencyMs:F2}ms, P95={P95LatencyMs:F2}ms, P99={P99LatencyMs:F2}ms, " +
               $"ErrorRate: {ErrorRate:F2}%, ActiveConnections: {ActiveConnections}, " +
               $"Memory: {MemoryUsageMB:F2}MB)";
    }
}
