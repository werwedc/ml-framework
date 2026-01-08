using System.Net;
using System.Net.Sockets;
using System.Text;

namespace MLFramework.Data.Metrics;

/// <summary>
/// Exports metrics to a StatsD server via UDP.
/// </summary>
public class StatsDExporter : IMetricsExporter, IDisposable
{
    private readonly string _host;
    private readonly int _port;
    private readonly UdpClient _udpClient;
    private readonly string _prefix;
    private readonly TextWriter _output;
    private readonly bool _useUdp;
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of the StatsDExporter that writes to console (for testing).
    /// </summary>
    /// <param name="prefix">Optional prefix for all metric names.</param>
    public StatsDExporter(string prefix = "") : this(null, 8125, prefix, true)
    {
    }

    /// <summary>
    /// Initializes a new instance of the StatsDExporter that sends metrics to a StatsD server.
    /// </summary>
    /// <param name="host">The StatsD server hostname or IP address.</param>
    /// <param name="port">The StatsD server port.</param>
    /// <param name="prefix">Optional prefix for all metric names.</param>
    public StatsDExporter(string host, int port, string prefix = "") : this(host, port, prefix, false)
    {
    }

    private StatsDExporter(string host, int port, string prefix, bool useConsole)
    {
        _host = host ?? "localhost";
        _port = port;
        _prefix = prefix;
        _useUdp = !useConsole;

        if (_useUdp)
        {
            _udpClient = new UdpClient();
            _output = null;
        }
        else
        {
            _udpClient = null;
            _output = Console.Out;
        }
    }

    /// <summary>
    /// Exports metrics to StatsD.
    /// </summary>
    /// <param name="metrics">A dictionary mapping model keys to metrics.</param>
    public void Export(Dictionary<string, VersionMetrics> metrics)
    {
        if (metrics == null)
            throw new ArgumentNullException(nameof(metrics));

        foreach (var kvp in metrics)
        {
            var metricsData = kvp.Value;
            var sanitizedModelName = SanitizeMetricName(metricsData.ModelName);
            var sanitizedVersion = SanitizeMetricName(metricsData.Version);

            // Counter metrics
            SendMetric($"model.{sanitizedModelName}.{sanitizedVersion}.inference.requests", metricsData.RequestCount, "c");

            // Timing metrics (latency)
            SendMetric($"model.{sanitizedModelName}.{sanitizedVersion}.inference.latency", (long)metricsData.AverageLatencyMs, "ms");
            SendMetric($"model.{sanitizedModelName}.{sanitizedVersion}.inference.latency.p50", (long)metricsData.P50LatencyMs, "ms");
            SendMetric($"model.{sanitizedModelName}.{sanitizedVersion}.inference.latency.p95", (long)metricsData.P95LatencyMs, "ms");
            SendMetric($"model.{sanitizedModelName}.{sanitizedVersion}.inference.latency.p99", (long)metricsData.P99LatencyMs, "ms");

            // Error metrics
            SendMetric($"model.{sanitizedModelName}.{sanitizedVersion}.inference.errors", metricsData.ErrorCount, "c");

            // Error breakdown by type
            foreach (var errorKvp in metricsData.ErrorCountsByType)
            {
                var sanitizedErrorType = SanitizeMetricName(errorKvp.Key);
                SendMetric($"model.{sanitizedModelName}.{sanitizedVersion}.inference.errors.{sanitizedErrorType}", errorKvp.Value, "c");
            }

            // Gauge metrics
            SendMetric($"model.{sanitizedModelName}.{sanitizedVersion}.memory.bytes", (long)(metricsData.MemoryUsageMB * 1024 * 1024), "g");
            SendMetric($"model.{sanitizedModelName}.{sanitizedVersion}.connections.active", metricsData.ActiveConnections, "g");
            SendMetric($"model.{sanitizedModelName}.{sanitizedVersion}.requests_per_second", metricsData.RequestsPerSecond, "g");
            SendMetric($"model.{sanitizedModelName}.{sanitizedVersion}.error_rate", metricsData.ErrorRate, "g");
        }
    }

    /// <summary>
    /// Exports metrics to StatsD asynchronously.
    /// </summary>
    /// <param name="metrics">A dictionary mapping model keys to metrics.</param>
    /// <returns>A task representing the asynchronous export operation.</returns>
    public Task ExportAsync(Dictionary<string, VersionMetrics> metrics)
    {
        return Task.Run(() => Export(metrics));
    }

    private void SendMetric(string name, double value, string type)
    {
        var metricName = string.IsNullOrEmpty(_prefix) ? name : $"{_prefix}.{name}";
        var metric = $"{metricName}:{value}|{type}";

        if (_useUdp)
        {
            try
            {
                var data = Encoding.UTF8.GetBytes(metric);
                _udpClient.Send(data, data.Length, _host, _port);
            }
            catch (Exception ex)
            {
                // Silently fail to not block metrics collection
                // In production, you might want to log this
                Console.Error.WriteLine($"Failed to send metric to StatsD: {ex.Message}");
            }
        }
        else
        {
            _output?.WriteLine(metric);
        }
    }

    private void SendMetric(string name, long value, string type)
    {
        var metricName = string.IsNullOrEmpty(_prefix) ? name : $"{_prefix}.{name}";
        var metric = $"{metricName}:{value}|{type}";

        if (_useUdp)
        {
            try
            {
                var data = Encoding.UTF8.GetBytes(metric);
                _udpClient.Send(data, data.Length, _host, _port);
            }
            catch (Exception ex)
            {
                // Silently fail to not block metrics collection
                Console.Error.WriteLine($"Failed to send metric to StatsD: {ex.Message}");
            }
        }
        else
        {
            _output?.WriteLine(metric);
        }
    }

    private string SanitizeMetricName(string name)
    {
        if (string.IsNullOrEmpty(name))
            return "unknown";

        // Replace dots and other special characters with underscores
        return name
            .Replace('.', '_')
            .Replace('-', '_')
            .Replace(' ', '_')
            .Replace(':', '_')
            .Replace('/', '_');
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        _udpClient?.Dispose();
        _disposed = true;
    }
}
