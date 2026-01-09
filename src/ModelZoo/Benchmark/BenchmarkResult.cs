namespace ModelZoo.Benchmark;

/// <summary>
/// Results from a model benchmark run.
/// </summary>
public class BenchmarkResult
{
    /// <summary>
    /// Gets or sets the model identifier.
    /// </summary>
    public string ModelName { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the dataset used for benchmarking.
    /// </summary>
    public string Dataset { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the number of samples processed during benchmarking.
    /// </summary>
    public int TotalSamples { get; set; }

    /// <summary>
    /// Gets or sets the throughput in samples per second.
    /// </summary>
    public float Throughput { get; set; }

    /// <summary>
    /// Gets or sets the average latency per sample in milliseconds.
    /// </summary>
    public float AvgLatency { get; set; }

    /// <summary>
    /// Gets or sets the minimum latency in milliseconds.
    /// </summary>
    public float MinLatency { get; set; }

    /// <summary>
    /// Gets or sets the maximum latency in milliseconds.
    /// </summary>
    public float MaxLatency { get; set; }

    /// <summary>
    /// Gets or sets the 50th percentile latency (median) in milliseconds.
    /// </summary>
    public float P50Latency { get; set; }

    /// <summary>
    /// Gets or sets the 95th percentile latency in milliseconds.
    /// </summary>
    public float P95Latency { get; set; }

    /// <summary>
    /// Gets or sets the 99th percentile latency in milliseconds.
    /// </summary>
    public float P99Latency { get; set; }

    /// <summary>
    /// Gets or sets the peak memory usage during benchmarking in bytes.
    /// </summary>
    public long MemoryPeak { get; set; }

    /// <summary>
    /// Gets or sets the average memory usage during benchmarking in bytes.
    /// </summary>
    public long MemoryAvg { get; set; }

    /// <summary>
    /// Gets or sets the accuracy on the dataset if labels were available.
    /// </summary>
    public float Accuracy { get; set; }

    /// <summary>
    /// Gets or sets the total duration of the benchmark.
    /// </summary>
    public TimeSpan BenchmarkDuration { get; set; }

    /// <summary>
    /// Gets or sets the timestamp when the benchmark was run.
    /// </summary>
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets raw latency measurements for statistical analysis.
    /// </summary>
    public List<float> LatencyMeasurements { get; set; } = new List<float>();

    /// <summary>
    /// Calculates percentile from latency measurements.
    /// </summary>
    /// <param name="percentile">The percentile to calculate (e.g., 50, 95, 99).</param>
    /// <returns>The calculated percentile value.</returns>
    public float CalculatePercentile(int percentile)
    {
        if (LatencyMeasurements.Count == 0)
        {
            return 0f;
        }

        var sorted = LatencyMeasurements.OrderBy(x => x).ToList();
        var index = (int)Math.Ceiling((percentile / 100.0) * sorted.Count) - 1;
        index = Math.Max(0, Math.Min(index, sorted.Count - 1));

        return sorted[index];
    }

    /// <summary>
    /// Returns a summary string of the benchmark results.
    /// </summary>
    /// <returns>A human-readable summary.</returns>
    public string GetSummary()
    {
        return $"Model: {ModelName}, Dataset: {Dataset}, " +
               $"Throughput: {Throughput:F2} samples/s, " +
               $"Avg Latency: {AvgLatency:F2}ms, " +
               $"P95 Latency: {P95Latency:F2}ms, " +
               $"Accuracy: {Accuracy:F4}";
    }
}
