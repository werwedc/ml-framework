namespace MLFramework.Fusion.Profiling;

/// <summary>
/// Comprehensive profiling report for fusion operations
/// </summary>
public record FusionProfilingReport
{
    public required IReadOnlyList<FusionDecision> Decisions { get; init; }
    public required IReadOnlyList<KernelExecutionRecord> KernelExecutions { get; init; }
    public required FusionSummary Summary { get; init; }
    public required IReadOnlyDictionary<string, FusionPatternMetrics> PatternMetrics { get; init; }
}

/// <summary>
/// Record of a single kernel execution
/// </summary>
public record KernelExecutionRecord
{
    public required string KernelName { get; init; }
    public required double DurationMs { get; init; }
    public required int ThreadCount { get; init; }
    public required int BlockCount { get; init; }
    public required int SharedMemoryBytes { get; init; }
    public required DateTime Timestamp { get; init; }
}

/// <summary>
/// Summary statistics for fusion operations
/// </summary>
public record FusionSummary
{
    public required int TotalOperations { get; init; }
    public required int FusedOperations { get; init; }
    public required int FusedGroups { get; init; }
    public required double FusionRate { get; init; }
    public required double TotalKernelTimeMs { get; init; }
    public required double AverageKernelTimeMs { get; init; }
    public required int SuccessfulFusions { get; init; }
    public required int FailedFusions { get; init; }
}

/// <summary>
/// Metrics for a specific fusion pattern
/// </summary>
public record FusionPatternMetrics
{
    public required string PatternName { get; init; }
    public required int Count { get; init; }
    public required double TotalTimeMs { get; init; }
    public required double AverageTimeMs { get; init; }
    public required double MinTimeMs { get; init; }
    public required double MaxTimeMs { get; init; }
    public required double EstimatedSpeedup { get; init; }
}
