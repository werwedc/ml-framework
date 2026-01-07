using MLFramework.Core;
using MLFramework.Fusion;
using Backends = MLFramework.Fusion.Backends;
using RitterFramework.Core;
using Tensor = RitterFramework.Core.Tensor.Tensor;

namespace MLFramework.Autotuning;

/// <summary>
/// Interface for kernel autotuning
/// </summary>
public interface IAutotuner
{
    /// <summary>
    /// Tunes a fused operation and selects best configuration
    /// </summary>
    AutotuningResult Tune(
        MLFramework.Fusion.FusedOperation fusedOp,
        AutotuningOptions options,
        Tensor? benchmarkInput = null);

    /// <summary>
    /// Gets cached tuning result if available
    /// </summary>
    TuningCacheEntry? GetCachedResult(MLFramework.Fusion.FusedOperation fusedOp, DeviceInfo device);

    /// <summary>
    /// Stores tuning result in cache
    /// </summary>
    void CacheResult(MLFramework.Fusion.FusedOperation fusedOp, AutotuningResult result, DeviceInfo device);
}

/// <summary>
/// Result of autotuning operation
/// </summary>
public record AutotuningResult
{
    public required Backends.KernelLaunchConfiguration BestConfiguration { get; init; }
    public required double BestExecutionTimeMs { get; init; }
    public required IReadOnlyList<TuningBenchmarkResult> Benchmarks { get; init; }
    public required bool CacheHit { get; init; }
    public required DateTime Timestamp { get; init; }
}

/// <summary>
/// Result of a single tuning benchmark
/// </summary>
public record TuningBenchmarkResult
{
    public required Backends.KernelLaunchConfiguration Configuration { get; init; }
    public required double ExecutionTimeMs { get; init; }
    public required double MemoryBandwidthGBps { get; init; }
    public required int SMUtilizationPercent { get; init; }
}

/// <summary>
/// Options for autotuning
/// </summary>
public record AutotuningOptions
{
    public int MaxIterations { get; init; } = 10;
    public int WarmupRuns { get; init; } = 3;
    public int MeasurementRuns { get; init; } = 5;
    public bool EnableMemoryMeasurement { get; init; } = true;
    public bool EnableSMUtilization { get; init; } = true;
    public SearchStrategy SearchStrategy { get; init; } = SearchStrategy.GridSearch;
}

/// <summary>
/// Search strategy for autotuning
/// </summary>
public enum SearchStrategy
{
    GridSearch,
    RandomSearch,
    BayesianOptimization,
    GeneticAlgorithm
}

/// <summary>
/// Information about a device for autotuning
/// </summary>
public record DeviceInfo
{
    public required string DeviceName { get; init; }
    public required int ComputeCapability { get; init; }
    public required int SMCount { get; init; }
    public required long TotalMemoryBytes { get; init; }
    public required int MaxThreadsPerBlock { get; init; }
    public required int MaxSharedMemoryPerBlock { get; init; }
    public required string Architecture { get; init; }
}
