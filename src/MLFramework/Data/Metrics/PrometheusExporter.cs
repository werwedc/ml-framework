using System.Text;

namespace MLFramework.Data.Metrics;

/// <summary>
/// Exports metrics in Prometheus text format.
/// </summary>
public class PrometheusExporter : IMetricsExporter
{
    private readonly TextWriter _output;

    /// <summary>
    /// Initializes a new instance of the PrometheusExporter that writes to console output.
    /// </summary>
    public PrometheusExporter() : this(Console.Out)
    {
    }

    /// <summary>
    /// Initializes a new instance of the PrometheusExporter that writes to the specified text writer.
    /// </summary>
    /// <param name="output">The text writer to write metrics to.</param>
    public PrometheusExporter(TextWriter output)
    {
        _output = output ?? throw new ArgumentNullException(nameof(output));
    }

    /// <summary>
    /// Exports metrics in Prometheus text format.
    /// </summary>
    /// <param name="metrics">A dictionary mapping model keys to metrics.</param>
    public void Export(Dictionary<string, VersionMetrics> metrics)
    {
        if (metrics == null)
            throw new ArgumentNullException(nameof(metrics));

        var builder = new StringBuilder();

        // HELP and TYPE definitions
        builder.AppendLine("# HELP model_inference_requests_total Total number of inference requests");
        builder.AppendLine("# TYPE model_inference_requests_total counter");

        builder.AppendLine("# HELP model_inference_latency_ms Inference latency in milliseconds");
        builder.AppendLine("# TYPE model_inference_latency_ms histogram");

        builder.AppendLine("# HELP model_inference_errors_total Total number of inference errors");
        builder.AppendLine("# TYPE model_inference_errors_total counter");

        builder.AppendLine("# HELP model_memory_usage_bytes Memory usage in bytes");
        builder.AppendLine("# TYPE model_memory_usage_bytes gauge");

        builder.AppendLine("# HELP model_active_connections Number of active connections");
        builder.AppendLine("# TYPE model_active_connections gauge");

        builder.AppendLine("# HELP model_requests_per_second Requests per second");
        builder.AppendLine("# TYPE model_requests_per_second gauge");

        builder.AppendLine("# HELP model_error_rate Error rate percentage");
        builder.AppendLine("# TYPE model_error_rate gauge");

        // Metrics data
        foreach (var kvp in metrics)
        {
            var metricsData = kvp.Value;
            var labels = GetLabels(metricsData.ModelName, metricsData.Version);

            // Counter metrics
            builder.AppendLine($"model_inference_requests_total{labels} {metricsData.RequestCount}");

            // Latency histogram summary (using percentiles)
            builder.AppendLine($"model_inference_latency_ms_sum{labels} {metricsData.AverageLatencyMs * metricsData.RequestCount}");
            builder.AppendLine($"model_inference_latency_ms_count{labels} {metricsData.RequestCount}");
            builder.AppendLine($"model_inference_latency_ms{{model_name=\"{metricsData.ModelName}\",version=\"{metricsData.Version}\",quantile=\"0.5\"}} {metricsData.P50LatencyMs}");
            builder.AppendLine($"model_inference_latency_ms{{model_name=\"{metricsData.ModelName}\",version=\"{metricsData.Version}\",quantile=\"0.95\"}} {metricsData.P95LatencyMs}");
            builder.AppendLine($"model_inference_latency_ms{{model_name=\"{metricsData.ModelName}\",version=\"{metricsData.Version}\",quantile=\"0.99\"}} {metricsData.P99LatencyMs}");

            // Error metrics
            builder.AppendLine($"model_inference_errors_total{labels} {metricsData.ErrorCount}");

            // Error breakdown by type
            foreach (var errorKvp in metricsData.ErrorCountsByType)
            {
                var errorLabels = $"{{model_name=\"{metricsData.ModelName}\",version=\"{metricsData.Version}\",error_type=\"{errorKvp.Key}\"}}";
                builder.AppendLine($"model_inference_errors_total{errorLabels} {errorKvp.Value}");
            }

            // Gauge metrics
            builder.AppendLine($"model_memory_usage_bytes{labels} {(long)(metricsData.MemoryUsageMB * 1024 * 1024)}");
            builder.AppendLine($"model_active_connections{labels} {metricsData.ActiveConnections}");
            builder.AppendLine($"model_requests_per_second{labels} {metricsData.RequestsPerSecond:F2}");
            builder.AppendLine($"model_error_rate{labels} {metricsData.ErrorRate:F2}");

            builder.AppendLine();
        }

        _output.Write(builder.ToString());
    }

    /// <summary>
    /// Exports metrics in Prometheus text format asynchronously.
    /// </summary>
    /// <param name="metrics">A dictionary mapping model keys to metrics.</param>
    /// <returns>A task representing the asynchronous export operation.</returns>
    public Task ExportAsync(Dictionary<string, VersionMetrics> metrics)
    {
        return Task.Run(() => Export(metrics));
    }

    private string GetLabels(string modelName, string version)
    {
        return $"{{model_name=\"{modelName}\",version=\"{version}\"}}";
    }
}
