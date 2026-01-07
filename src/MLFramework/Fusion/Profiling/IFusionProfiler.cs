using MLFramework.Fusion.Backends;

namespace MLFramework.Fusion.Profiling;

/// <summary>
/// Interface for profiling fusion operations
/// </summary>
public interface IFusionProfiler
{
    /// <summary>
    /// Starts profiling a fusion operation
    /// </summary>
    FusionProfilingSession StartProfiling(FusedOperation fusedOp);

    /// <summary>
    /// Records a fusion decision
    /// </summary>
    void RecordDecision(FusionDecision decision);

    /// <summary>
    /// Records kernel execution time
    /// </summary>
    void RecordKernelExecution(string kernelName, double durationMs);

    /// <summary>
    /// Gets profiling report
    /// </summary>
    FusionProfilingReport GetReport();
}

/// <summary>
/// Represents a profiling session for a single fusion operation
/// </summary>
public class FusionProfilingSession : IDisposable
{
    private readonly IFusionProfiler _profiler;
    private readonly string _kernelName;
    private readonly System.Diagnostics.Stopwatch _stopwatch;

    public FusionProfilingSession(IFusionProfiler profiler, string kernelName)
    {
        _profiler = profiler;
        _kernelName = kernelName;
        _stopwatch = System.Diagnostics.Stopwatch.StartNew();
    }

    public void Dispose()
    {
        _stopwatch.Stop();
        _profiler.RecordKernelExecution(_kernelName, _stopwatch.Elapsed.TotalMilliseconds);
    }
}

/// <summary>
/// Represents a fusion decision with context
/// </summary>
public record FusionDecision
{
    public required string OperationChain { get; init; }
    public required bool Fused { get; init; }
    public required FusionPatternType? PatternType { get; init; }
    public required string? RejectionReason { get; init; }
    public required DateTime Timestamp { get; init; }
    public required IReadOnlyDictionary<string, object> Metadata { get; init; }
}
