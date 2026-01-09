namespace ModelZoo.Benchmark;

/// <summary>
/// Configuration options for comparing multiple models.
/// </summary>
public class ComparisonOptions
{
    /// <summary>
    /// Gets or sets the list of model names to compare.
    /// </summary>
    public string[] Models { get; set; } = Array.Empty<string>();

    /// <summary>
    /// Gets or sets the dataset to use for comparison.
    /// </summary>
    public string Dataset { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the subset of the dataset to use.
    /// </summary>
    public string Subset { get; set; } = "test";

    /// <summary>
    /// Gets or sets the batch size for inference.
    /// </summary>
    public int BatchSize { get; set; } = 1;

    /// <summary>
    /// Gets or sets the device to run on.
    /// </summary>
    public MLFramework.Core.DeviceId Device { get; set; } = MLFramework.Core.DeviceId.CPU;

    /// <summary>
    /// Gets or sets the metrics to compare.
    /// Valid values: "latency", "throughput", "accuracy", "memory".
    /// </summary>
    public string[] Metrics { get; set; } = new[] { "latency", "throughput", "accuracy" };

    /// <summary>
    /// Gets or sets the primary metric for ranking models.
    /// </summary>
    public string PrimaryMetric { get; set; } = "throughput";

    /// <summary>
    /// Gets or sets whether to run comparisons in parallel.
    /// </summary>
    public bool Parallel { get; set; } = true;

    /// <summary>
    /// Gets or sets the number of batches to benchmark per model.
    /// </summary>
    public int? NumBatches { get; set; }

    /// <summary>
    /// Gets or sets the number of iterations per batch for averaging.
    /// </summary>
    public int NumIterations { get; set; } = 1;

    /// <summary>
    /// Gets or sets the number of warmup iterations.
    /// </summary>
    public int WarmupIterations { get; set; } = 10;

    /// <summary>
    /// Validates the comparison options.
    /// </summary>
    /// <exception cref="ArgumentException">Thrown when options are invalid.</exception>
    public void Validate()
    {
        if (Models == null || Models.Length == 0)
        {
            throw new ArgumentException("At least one model must be specified.", nameof(Models));
        }

        if (string.IsNullOrWhiteSpace(Dataset))
        {
            throw new ArgumentException("Dataset must be specified.", nameof(Dataset));
        }

        if (BatchSize <= 0)
        {
            throw new ArgumentException("BatchSize must be positive.", nameof(BatchSize));
        }

        if (Metrics == null || Metrics.Length == 0)
        {
            throw new ArgumentException("At least one metric must be specified.", nameof(Metrics));
        }

        var validMetrics = new[] { "latency", "throughput", "accuracy", "memory" };
        var invalidMetrics = Metrics.Where(m => !validMetrics.Contains(m.ToLower())).ToList();

        if (invalidMetrics.Any())
        {
            throw new ArgumentException(
                $"Invalid metrics: {string.Join(", ", invalidMetrics)}. " +
                $"Valid metrics are: {string.Join(", ", validMetrics)}.",
                nameof(Metrics));
        }

        if (!validMetrics.Contains(PrimaryMetric.ToLower()))
        {
            throw new ArgumentException(
                $"Invalid primary metric: {PrimaryMetric}. " +
                $"Valid metrics are: {string.Join(", ", validMetrics)}.",
                nameof(PrimaryMetric));
        }
    }
}
