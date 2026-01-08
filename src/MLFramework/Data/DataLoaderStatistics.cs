namespace MLFramework.Data;

/// <summary>
/// Statistics and monitoring information for DataLoader operations.
/// </summary>
public sealed class DataLoaderStatistics
{
    /// <summary>
    /// Gets the number of batches that have been loaded.
    /// </summary>
    public int BatchesLoaded { get; }

    /// <summary>
    /// Gets the total number of samples in the dataset.
    /// </summary>
    public int TotalSamples { get; }

    /// <summary>
    /// Gets the average time in milliseconds to load a batch.
    /// </summary>
    public double AverageBatchTimeMs { get; }

    /// <summary>
    /// Gets the throughput in samples per second.
    /// </summary>
    public double ThroughputSamplesPerSecond { get; }

    /// <summary>
    /// Gets statistics about the internal shared queue.
    /// </summary>
    public QueueStatistics? QueueStatistics { get; }

    /// <summary>
    /// Gets statistics about the prefetch strategy.
    /// </summary>
    public PrefetchStatistics? PrefetchStatistics { get; }

    /// <summary>
    /// Initializes a new instance of the DataLoaderStatistics class.
    /// </summary>
    /// <param name="batchesLoaded">Number of batches loaded.</param>
    /// <param name="totalSamples">Total number of samples in the dataset.</param>
    /// <param name="averageBatchTimeMs">Average batch loading time in milliseconds.</param>
    /// <param name="throughputSamplesPerSecond">Throughput in samples per second.</param>
    /// <param name="queueStatistics">Queue statistics (optional).</param>
    /// <param name="prefetchStatistics">Prefetch statistics (optional).</param>
    internal DataLoaderStatistics(
        int batchesLoaded,
        int totalSamples,
        double averageBatchTimeMs,
        double throughputSamplesPerSecond,
        QueueStatistics? queueStatistics = null,
        PrefetchStatistics? prefetchStatistics = null)
    {
        BatchesLoaded = batchesLoaded;
        TotalSamples = totalSamples;
        AverageBatchTimeMs = averageBatchTimeMs;
        ThroughputSamplesPerSecond = throughputSamplesPerSecond;
        QueueStatistics = queueStatistics;
        PrefetchStatistics = prefetchStatistics;
    }

    /// <summary>
    /// Returns a human-readable string representation of the statistics.
    /// </summary>
    public override string ToString()
    {
        return $"DataLoaderStatistics {{ BatchesLoaded: {BatchesLoaded}, TotalSamples: {TotalSamples}, " +
               $"AverageBatchTimeMs: {AverageBatchTimeMs:F2}, ThroughputSamplesPerSecond: {ThroughputSamplesPerSecond:F2}, " +
               $"QueueStats: {QueueStatistics?.ToString() ?? "N/A"}, PrefetchStats: {PrefetchStatistics?.ToString() ?? "N/A"} }}";
    }
}
