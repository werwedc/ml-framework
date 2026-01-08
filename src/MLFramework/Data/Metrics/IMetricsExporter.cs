namespace MLFramework.Data.Metrics;

/// <summary>
/// Interface for exporting metrics to external monitoring systems.
/// </summary>
public interface IMetricsExporter
{
    /// <summary>
    /// Exports metrics synchronously.
    /// </summary>
    /// <param name="metrics">A dictionary mapping model keys (modelName:version) to metrics.</param>
    void Export(Dictionary<string, VersionMetrics> metrics);

    /// <summary>
    /// Exports metrics asynchronously.
    /// </summary>
    /// <param name="metrics">A dictionary mapping model keys (modelName:version) to metrics.</param>
    /// <returns>A task representing the asynchronous export operation.</returns>
    Task ExportAsync(Dictionary<string, VersionMetrics> metrics);
}
