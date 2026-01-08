namespace MLFramework.Data.Metrics;

/// <summary>
/// Interface for collecting and managing metrics per model version.
/// </summary>
public interface IMetricsCollector
{
    /// <summary>
    /// Records an inference request with its latency and success status.
    /// </summary>
    /// <param name="modelName">The name of the model.</param>
    /// <param name="version">The version of the model.</param>
    /// <param name="latencyMs">The inference latency in milliseconds.</param>
    /// <param name="success">Whether the inference was successful.</param>
    /// <param name="errorType">The type of error, if the inference failed.</param>
    void RecordInference(string modelName, string version, double latencyMs, bool success, string errorType = null);

    /// <summary>
    /// Records the number of active connections for a model version.
    /// </summary>
    /// <param name="modelName">The name of the model.</param>
    /// <param name="version">The version of the model.</param>
    /// <param name="count">The number of active connections.</param>
    void RecordActiveConnections(string modelName, string version, int count);

    /// <summary>
    /// Records memory usage for a model version.
    /// </summary>
    /// <param name="modelName">The name of the model.</param>
    /// <param name="version">The version of the model.</param>
    /// <param name="bytes">The memory usage in bytes.</param>
    void RecordMemoryUsage(string modelName, string version, long bytes);

    /// <summary>
    /// Gets aggregated metrics for a specific model version over a time window.
    /// </summary>
    /// <param name="modelName">The name of the model.</param>
    /// <param name="version">The version of the model.</param>
    /// <param name="window">The time window for aggregation.</param>
    /// <returns>The aggregated metrics for the specified version and window.</returns>
    VersionMetrics GetMetrics(string modelName, string version, TimeSpan window);

    /// <summary>
    /// Gets aggregated metrics for all model versions over a time window.
    /// </summary>
    /// <param name="window">The time window for aggregation.</param>
    /// <returns>A dictionary mapping model keys (modelName:version) to metrics.</returns>
    Dictionary<string, VersionMetrics> GetAllMetrics(TimeSpan window);

    /// <summary>
    /// Exports current metrics to the configured exporter.
    /// </summary>
    void ExportMetrics();

    /// <summary>
    /// Sets the metrics exporter.
    /// </summary>
    /// <param name="exporter">The exporter to use for exporting metrics.</param>
    void SetExporter(IMetricsExporter exporter);

    /// <summary>
    /// Starts automatic periodic export of metrics.
    /// </summary>
    /// <param name="interval">The interval between exports.</param>
    void StartAutoExport(TimeSpan interval);

    /// <summary>
    /// Stops automatic periodic export of metrics.
    /// </summary>
    void StopAutoExport();
}
