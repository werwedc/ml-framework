using System.Text;

namespace MLFramework.Data.Metrics;

/// <summary>
/// Exports metrics to the console for debugging and monitoring.
/// </summary>
public class ConsoleExporter : IMetricsExporter
{
    /// <summary>
    /// Exports metrics to the console.
    /// </summary>
    /// <param name="metrics">A dictionary mapping model keys to metrics.</param>
    public void Export(Dictionary<string, VersionMetrics> metrics)
    {
        if (metrics == null)
            throw new ArgumentNullException(nameof(metrics));

        Console.WriteLine("=== Metrics Export ===");
        Console.WriteLine($"Timestamp: {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss} UTC");
        Console.WriteLine($"Model Versions: {metrics.Count}");
        Console.WriteLine();

        foreach (var kvp in metrics.OrderBy(m => m.Key))
        {
            var metricsData = kvp.Value;
            Console.WriteLine($"Model: {metricsData.ModelName}");
            Console.WriteLine($"  Version: {metricsData.Version}");
            Console.WriteLine($"  Window: {metricsData.WindowStart:HH:mm:ss} - {metricsData.WindowEnd:HH:mm:ss}");
            Console.WriteLine($"  Requests: {metricsData.RequestCount}");
            Console.WriteLine($"  Requests/Second: {metricsData.RequestsPerSecond:F2}");
            Console.WriteLine($"  Average Latency: {metricsData.AverageLatencyMs:F2}ms");
            Console.WriteLine($"  P50 Latency: {metricsData.P50LatencyMs:F2}ms");
            Console.WriteLine($"  P95 Latency: {metricsData.P95LatencyMs:F2}ms");
            Console.WriteLine($"  P99 Latency: {metricsData.P99LatencyMs:F2}ms");
            Console.WriteLine($"  Error Rate: {metricsData.ErrorRate:F2}%");
            Console.WriteLine($"  Error Count: {metricsData.ErrorCount}");
            Console.WriteLine($"  Active Connections: {metricsData.ActiveConnections}");
            Console.WriteLine($"  Memory Usage: {metricsData.MemoryUsageMB:F2}MB");

            if (metricsData.ErrorCountsByType.Count > 0)
            {
                Console.WriteLine("  Errors by Type:");
                foreach (var errorKvp in metricsData.ErrorCountsByType)
                {
                    Console.WriteLine($"    {errorKvp.Key}: {errorKvp.Value}");
                }
            }

            Console.WriteLine();
        }

        Console.WriteLine("=== End of Export ===");
    }

    /// <summary>
    /// Exports metrics to the console asynchronously.
    /// </summary>
    /// <param name="metrics">A dictionary mapping model keys to metrics.</param>
    /// <returns>A task representing the asynchronous export operation.</returns>
    public Task ExportAsync(Dictionary<string, VersionMetrics> metrics)
    {
        return Task.Run(() => Export(metrics));
    }
}
