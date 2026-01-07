using MLFramework.Fusion;
using Backends = MLFramework.Fusion.Backends;
using RitterFramework.Core;
using Tensor = RitterFramework.Core.Tensor.Tensor;

namespace MLFramework.Autotuning;

/// <summary>
/// Interface for running benchmarks
/// </summary>
public interface IBenchmarkRunner
{
    TuningBenchmarkResult Benchmark(
        MLFramework.Fusion.FusedOperation fusedOp,
        Backends.KernelLaunchConfiguration config,
        Tensor input,
        AutotuningOptions options);

    IReadOnlyList<TuningBenchmarkResult> BenchmarkAll(
        MLFramework.Fusion.FusedOperation fusedOp,
        IReadOnlyList<Backends.KernelLaunchConfiguration> configs,
        Tensor input,
        AutotuningOptions options);
}

/// <summary>
/// Runner for benchmarking kernel configurations
/// </summary>
public class BenchmarkRunner : IBenchmarkRunner
{
    private readonly IKernelExecutor _executor;
    private readonly IPerformanceProfiler _profiler;

    public BenchmarkRunner(IKernelExecutor executor, IPerformanceProfiler profiler)
    {
        _executor = executor;
        _profiler = profiler;
    }

    public TuningBenchmarkResult Benchmark(
        MLFramework.Fusion.FusedOperation fusedOp,
        Backends.KernelLaunchConfiguration config,
        Tensor input,
        AutotuningOptions options)
    {
        // Warmup runs
        for (int i = 0; i < options.WarmupRuns; i++)
        {
            _executor.ExecuteFusedKernel(fusedOp, config, input);
        }

        // Synchronize to ensure accurate timing
        _executor.Synchronize();

        var measurements = new List<double>();

        // Measurement runs
        for (int i = 0; i < options.MeasurementRuns; i++)
        {
            var measurement = _profiler.MeasureExecution(() =>
            {
                _executor.ExecuteFusedKernel(fusedOp, config, input);
            });

            measurements.Add(measurement);
        }

        // Compute statistics
        var executionTimeMs = measurements.Average();
        var memoryBandwidthGBps = options.EnableMemoryMeasurement
            ? _profiler.MeasureMemoryBandwidth(fusedOp, config)
            : 0.0;
        var smUtilization = options.EnableSMUtilization
            ? _profiler.MeasureSMUtilization(fusedOp, config)
            : 0;

        return new TuningBenchmarkResult
        {
            Configuration = config,
            ExecutionTimeMs = executionTimeMs,
            MemoryBandwidthGBps = memoryBandwidthGBps,
            SMUtilizationPercent = smUtilization
        };
    }

    public IReadOnlyList<TuningBenchmarkResult> BenchmarkAll(
        MLFramework.Fusion.FusedOperation fusedOp,
        IReadOnlyList<Backends.KernelLaunchConfiguration> configs,
        Tensor input,
        AutotuningOptions options)
    {
        var results = new List<TuningBenchmarkResult>();

        foreach (var config in configs)
        {
            var result = Benchmark(fusedOp, config, input, options);
            results.Add(result);
        }

        return results;
    }
}
