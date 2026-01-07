namespace MLFramework.Fusion.Profiling;

/// <summary>
/// Main profiler implementation for fusion operations
/// </summary>
public class FusionProfiler : IFusionProfiler
{
    private readonly List<FusionDecision> _decisions = new();
    private readonly List<KernelExecutionRecord> _kernelExecutions = new();
    private readonly Dictionary<string, List<KernelExecutionRecord>> _kernelRecordsMap = new();
    private readonly Dictionary<string, FusionPatternMetrics> _patternMetrics = new();
    private readonly object _lock = new();

    public FusionProfilingSession StartProfiling(FusedOperation fusedOp)
    {
        return new FusionProfilingSession(this, fusedOp.KernelSpec.KernelName);
    }

    public void RecordDecision(FusionDecision decision)
    {
        lock (_lock)
        {
            _decisions.Add(decision);

            if (decision.Fused && decision.PatternType.HasValue)
            {
                var patternName = decision.PatternType.Value.ToString();
                UpdatePatternMetrics(patternName);
            }
        }
    }

    public void RecordKernelExecution(string kernelName, double durationMs)
    {
        lock (_lock)
        {
            var record = new KernelExecutionRecord
            {
                KernelName = kernelName,
                DurationMs = durationMs,
                ThreadCount = 0, // Would need to be passed in
                BlockCount = 0,
                SharedMemoryBytes = 0,
                Timestamp = DateTime.UtcNow
            };

            _kernelExecutions.Add(record);

            if (!_kernelRecordsMap.ContainsKey(kernelName))
            {
                _kernelRecordsMap[kernelName] = new List<KernelExecutionRecord>();
            }
            _kernelRecordsMap[kernelName].Add(record);

            // Update pattern metrics if kernel name contains pattern
            UpdatePatternMetricsForKernel(kernelName, durationMs);
        }
    }

    public FusionProfilingReport GetReport()
    {
        lock (_lock)
        {
            var summary = ComputeSummary();
            var patternMetrics = _patternMetrics.ToDictionary();

            return new FusionProfilingReport
            {
                Decisions = _decisions.ToList(),
                KernelExecutions = _kernelExecutions.ToList(),
                Summary = summary,
                PatternMetrics = patternMetrics
            };
        }
    }

    private FusionSummary ComputeSummary()
    {
        var totalOps = _decisions.Sum(d =>
            ParseOperationCount(d.OperationChain));

        var fusedOps = _decisions
            .Where(d => d.Fused)
            .Sum(d => ParseOperationCount(d.OperationChain));

        var fusedGroups = _decisions.Count(d => d.Fused);

        return new FusionSummary
        {
            TotalOperations = totalOps,
            FusedOperations = fusedOps,
            FusedGroups = fusedGroups,
            FusionRate = totalOps > 0 ? (fusedOps * 100.0 / totalOps) : 0.0,
            TotalKernelTimeMs = _kernelExecutions.Sum(e => e.DurationMs),
            AverageKernelTimeMs = _kernelExecutions.Any()
                ? _kernelExecutions.Average(e => e.DurationMs)
                : 0.0,
            SuccessfulFusions = _decisions.Count(d => d.Fused),
            FailedFusions = _decisions.Count(d => !d.Fused)
        };
    }

    private void UpdatePatternMetrics(string patternName)
    {
        if (!_patternMetrics.ContainsKey(patternName))
        {
            _patternMetrics[patternName] = new FusionPatternMetrics
            {
                PatternName = patternName,
                Count = 0,
                TotalTimeMs = 0,
                AverageTimeMs = 0,
                MinTimeMs = double.MaxValue,
                MaxTimeMs = 0,
                EstimatedSpeedup = 0
            };
        }

        _patternMetrics[patternName] = _patternMetrics[patternName] with
        {
            Count = _patternMetrics[patternName].Count + 1
        };
    }

    private void UpdatePatternMetricsForKernel(string kernelName, double durationMs)
    {
        foreach (var patternName in _patternMetrics.Keys)
        {
            if (kernelName.Contains(patternName, StringComparison.OrdinalIgnoreCase))
            {
                var current = _patternMetrics[patternName];
                var newCount = current.Count;
                var newTotalTime = current.TotalTimeMs + durationMs;
                var newMinTime = Math.Min(current.MinTimeMs, durationMs);
                var newMaxTime = Math.Max(current.MaxTimeMs, durationMs);

                _patternMetrics[patternName] = current with
                {
                    TotalTimeMs = newTotalTime,
                    AverageTimeMs = newTotalTime / newCount,
                    MinTimeMs = newMinTime,
                    MaxTimeMs = newMaxTime
                };
            }
        }
    }

    private int ParseOperationCount(string operationChain)
    {
        // Parse "Add -> Mul -> ReLU" as 3
        var parts = operationChain.Split(new[] { " -> " }, StringSplitOptions.RemoveEmptyEntries);
        return parts.Length;
    }
}
