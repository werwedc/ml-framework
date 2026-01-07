using MLFramework.Core;
using MLFramework.Fusion;
using Backends = MLFramework.Fusion.Backends;

namespace MLFramework.Autotuning;

/// <summary>
/// Interface for performance profiling
/// </summary>
public interface IPerformanceProfiler
{
    double MeasureExecution(Action action);
    double MeasureMemoryBandwidth(MLFramework.Fusion.FusedOperation fusedOp, Backends.KernelLaunchConfiguration config);
    int MeasureSMUtilization(MLFramework.Fusion.FusedOperation fusedOp, Backends.KernelLaunchConfiguration config);
}

/// <summary>
/// CUDA performance profiler
/// </summary>
public class CudaPerformanceProfiler : IPerformanceProfiler
{
    private readonly IKernelExecutor _executor;

    public CudaPerformanceProfiler(IKernelExecutor executor)
    {
        _executor = executor;
    }

    public double MeasureExecution(Action action)
    {
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();

        action();

        _executor.Synchronize();
        stopwatch.Stop();

        return stopwatch.Elapsed.TotalMilliseconds;
    }

    public double MeasureMemoryBandwidth(MLFramework.Fusion.FusedOperation fusedOp, Backends.KernelLaunchConfiguration config)
    {
        // Estimate based on tensor sizes and execution time
        var totalBytes = fusedOp.InputShape.Size * GetDataTypeSize(fusedOp.DataType) * 2; // read + write

        // This would need actual profiling via CUDA events
        return totalBytes / 1e9; // Simplified
    }

    public int MeasureSMUtilization(MLFramework.Fusion.FusedOperation fusedOp, Backends.KernelLaunchConfiguration config)
    {
        // Estimate SM utilization based on thread count and device SM count
        var totalThreads = config.BlockDim.Total() * (config.GridDim.X * config.GridDim.Y * config.GridDim.Z);
        var maxThreadsPerSM = 2048; // Typical value

        var smEstimate = (int)((totalThreads / (double)maxThreadsPerSM) * 100);
        return Math.Min(smEstimate, 100);
    }

    private int GetDataTypeSize(DataType dtype)
    {
        return dtype switch
        {
            DataType.Float32 or DataType.Int32 => 4,
            DataType.Float16 or DataType.Int16 => 2,
            DataType.Int8 => 1,
            DataType.Int64 => 8,
            _ => 4
        };
    }
}
