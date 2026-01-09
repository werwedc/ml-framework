namespace ModelZoo.Benchmark;

/// <summary>
/// Configuration options for running model benchmarks.
/// </summary>
public class BenchmarkOptions
{
    /// <summary>
    /// Gets or sets the dataset name or path to use for benchmarking.
    /// </summary>
    public string Dataset { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the subset of the dataset to use ("train", "val", "test", or custom subset).
    /// </summary>
    public string Subset { get; set; } = "test";

    /// <summary>
    /// Gets or sets the batch size for inference.
    /// </summary>
    public int BatchSize { get; set; } = 1;

    /// <summary>
    /// Gets or sets the number of batches to benchmark (null = all batches).
    /// </summary>
    public int? NumBatches { get; set; }

    /// <summary>
    /// Gets or sets the number of iterations per batch for averaging latency measurements.
    /// </summary>
    public int NumIterations { get; set; } = 1;

    /// <summary>
    /// Gets or sets the device to run the benchmark on.
    /// Defaults to GPU if available, otherwise CPU.
    /// </summary>
    public MLFramework.Core.DeviceId Device { get; set; } = MLFramework.Core.DeviceId.CPU;

    /// <summary>
    /// Gets or sets the number of warmup iterations to run before measurement.
    /// </summary>
    public int WarmupIterations { get; set; } = 10;

    /// <summary>
    /// Gets or sets whether to include preprocessing time in latency measurements.
    /// </summary>
    public bool Preprocess { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to include postprocessing time in latency measurements.
    /// </summary>
    public bool Postprocess { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to profile memory usage during benchmarking.
    /// </summary>
    public bool IncludeMemoryProfile { get; set; } = false;

    /// <summary>
    /// Validates the benchmark options.
    /// </summary>
    /// <exception cref="ArgumentException">Thrown when options are invalid.</exception>
    public void Validate()
    {
        if (string.IsNullOrWhiteSpace(Dataset))
        {
            throw new ArgumentException("Dataset must be specified.", nameof(Dataset));
        }

        if (BatchSize <= 0)
        {
            throw new ArgumentException("BatchSize must be positive.", nameof(BatchSize));
        }

        if (NumBatches.HasValue && NumBatches.Value <= 0)
        {
            throw new ArgumentException("NumBatches must be positive if specified.", nameof(NumBatches));
        }

        if (NumIterations <= 0)
        {
            throw new ArgumentException("NumIterations must be positive.", nameof(NumIterations));
        }

        if (WarmupIterations < 0)
        {
            throw new ArgumentException("WarmupIterations must be non-negative.", nameof(WarmupIterations));
        }
    }
}
