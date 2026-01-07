# Spec: Profiling Integration

## Overview
Implement profiling integration for the fusion system, providing visibility into fusion decisions, kernel timings, and performance metrics.

## Requirements

### 1. Fusion Profiling Interface
Interface for profiling fusion operations.

```csharp
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

public record FusionDecision
{
    public required string OperationChain { get; init; }
    public required bool Fused { get; init; }
    public required FusionPatternType? PatternType { get; init; }
    public required string? RejectionReason { get; init; }
    public required DateTime Timestamp { get; init; }
    public required IReadOnlyDictionary<string, object> Metadata { get; init; }
}
```

### 2. Fusion Profiling Report
Comprehensive profiling report.

```csharp
public record FusionProfilingReport
{
    public required IReadOnlyList<FusionDecision> Decisions { get; init; }
    public required IReadOnlyList<KernelExecutionRecord> KernelExecutions { get; init; }
    public required FusionSummary Summary { get; init; }
    public required IReadOnlyDictionary<string, FusionPatternMetrics> PatternMetrics { get; init; }
}

public record KernelExecutionRecord
{
    public required string KernelName { get; init; }
    public required double DurationMs { get; init; }
    public required int ThreadCount { get; init; }
    public required int BlockCount { get; init; }
    public required int SharedMemoryBytes { get; init; }
    public required DateTime Timestamp { get; init; }
}

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
```

### 3. Profiling Implementation
Main profiler implementation.

```csharp
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
```

### 4. Kernel Timing Instrumentation
Instrument kernels for timing.

```csharp
public interface IKernelTimingInstrument
{
    /// <summary>
    /// Records kernel launch and completion
    /// </summary>
    void RecordKernelLaunch(string kernelName, KernelLaunchConfiguration config);

    /// <summary>
    /// Records kernel completion
    /// </summary>
    void RecordKernelComplete(string kernelName, double durationMs);

    /// <summary>
    /// Gets timing statistics for a kernel
    /// </summary>
    KernelTimingStatistics GetTimingStatistics(string kernelName);
}

public record KernelTimingStatistics
{
    public required string KernelName { get; init; }
    public required int ExecutionCount { get; init; }
    public required double TotalTimeMs { get; init; }
    public required double AverageTimeMs { get; init; }
    public required double MinTimeMs { get; init; }
    public required double MaxTimeMs { get; init; }
    public required double StdDevMs { get; init; }
}

public class KernelTimingInstrument : IKernelTimingInstrument
{
    private readonly Dictionary<string, List<double>> _timingRecords = new();
    private readonly Dictionary<string, KernelLaunchConfiguration> _launchConfigs = new();
    private readonly object _lock = new();

    public void RecordKernelLaunch(string kernelName, KernelLaunchConfiguration config)
    {
        lock (_lock)
        {
            _launchConfigs[kernelName] = config;
        }
    }

    public void RecordKernelComplete(string kernelName, double durationMs)
    {
        lock (_lock)
        {
            if (!_timingRecords.ContainsKey(kernelName))
            {
                _timingRecords[kernelName] = new List<double>();
            }
            _timingRecords[kernelName].Add(durationMs);
        }
    }

    public KernelTimingStatistics GetTimingStatistics(string kernelName)
    {
        lock (_lock)
        {
            if (!_timingRecords.TryGetValue(kernelName, out var timings))
            {
                return new KernelTimingStatistics
                {
                    KernelName = kernelName,
                    ExecutionCount = 0,
                    TotalTimeMs = 0,
                    AverageTimeMs = 0,
                    MinTimeMs = 0,
                    MaxTimeMs = 0,
                    StdDevMs = 0
                };
            }

            var count = timings.Count;
            var total = timings.Sum();
            var average = total / count;
            var min = timings.Min();
            var max = timings.Max();
            var variance = timings.Sum(t => Math.Pow(t - average, 2)) / count;
            var stdDev = Math.Sqrt(variance);

            return new KernelTimingStatistics
            {
                KernelName = kernelName,
                ExecutionCount = count,
                TotalTimeMs = total,
                AverageTimeMs = average,
                MinTimeMs = min,
                MaxTimeMs = max,
                StdDevMs = stdDev
            };
        }
    }
}
```

### 5. Fusion Decision Logger
Logs fusion decisions for debugging and analysis.

```csharp
public interface IFusionDecisionLogger
{
    void LogDecision(FusionDecision decision);
    void LogFusionResult(FusedOperation fusedOp, FusionResult result);
    void LogTiming(string message, double durationMs);
}

public class ConsoleFusionDecisionLogger : IFusionDecisionLogger
{
    private readonly bool _verbose;
    private readonly ILogger _logger;

    public ConsoleFusionDecisionLogger(bool verbose = false, ILogger? logger = null)
    {
        _verbose = verbose;
        _logger = logger ?? new ConsoleLogger();
    }

    public void LogDecision(FusionDecision decision)
    {
        var status = decision.Fused ? "FUSED" : "REJECTED";
        var pattern = decision.PatternType?.ToString() ?? "N/A";
        var reason = decision.RejectionReason ?? "N/A";

        var message = $"[{status}] {decision.OperationChain} (Pattern: {pattern})";

        if (!decision.Fused)
        {
            message += $" Reason: {reason}";
        }

        _logger.LogInformation(message);

        if (_verbose && decision.Metadata.Count > 0)
        {
            _logger.LogInformation("  Metadata:");
            foreach (var (key, value) in decision.Metadata)
            {
                _logger.LogInformation($"    {key}: {value}");
            }
        }
    }

    public void LogFusionResult(FusedOperation fusedOp, FusionResult result)
    {
        _logger.LogInformation($"Fusion Result: {fusedOp.KernelSpec.KernelName}");
        _logger.LogInformation($"  Original ops: {result.OriginalOpCount}");
        _logger.LogInformation($"  Fused ops: {result.FusedOpCount}");
        _logger.LogInformation($"  Fused groups: {result.FusedOperations.Count}");

        if (result.RejectedFusions.Count > 0)
        {
            _logger.LogInformation($"  Rejected: {result.RejectedFusions.Count}");
            foreach (var rejected in result.RejectedFusions)
            {
                _logger.LogInformation($"    - {rejected.RejectionReason}");
            }
        }
    }

    public void LogTiming(string message, double durationMs)
    {
        _logger.LogInformation($"[TIME] {message}: {durationMs:F3}ms");
    }
}

public class FileFusionDecisionLogger : IFusionDecisionLogger
{
    private readonly string _logFilePath;
    private readonly object _lock = new();

    public FileFusionDecisionLogger(string logFilePath)
    {
        _logFilePath = logFilePath;
    }

    public void LogDecision(FusionDecision decision)
    {
        var entry = $"{DateTime.UtcNow:O},{decision.Fused},{decision.OperationChain}," +
                   $"{decision.PatternType},{decision.RejectionReason}\n";

        lock (_lock)
        {
            File.AppendAllText(_logFilePath, entry);
        }
    }

    public void LogFusionResult(FusedOperation fusedOp, FusionResult result)
    {
        var entry = $"{DateTime.UtcNow:O},FUSION_RESULT,{fusedOp.KernelSpec.KernelName}," +
                   $"{result.OriginalOpCount},{result.FusedOpCount},{result.FusedOperations.Count}\n";

        lock (_lock)
        {
            File.AppendAllText(_logFilePath, entry);
        }
    }

    public void LogTiming(string message, double durationMs)
    {
        var entry = $"{DateTime.UtcNow:O},TIMING,{message},{durationMs:F3}\n";

        lock (_lock)
        {
            File.AppendAllText(_logFilePath, entry);
        }
    }
}
```

### 6. Performance Report Generator
Generate detailed performance reports.

```csharp
public interface IPerformanceReportGenerator
{
    string GenerateTextReport(FusionProfilingReport report);
    string GenerateJsonReport(FusionProfilingReport report);
    string GenerateMarkdownReport(FusionProfilingReport report);
}

public class PerformanceReportGenerator : IPerformanceReportGenerator
{
    public string GenerateTextReport(FusionProfilingReport report)
    {
        var sb = new StringBuilder();

        sb.AppendLine("=== Fusion Profiling Report ===");
        sb.AppendLine();

        // Summary
        sb.AppendLine("Summary:");
        sb.AppendLine($"  Total Operations: {report.Summary.TotalOperations}");
        sb.AppendLine($"  Fused Operations: {report.Summary.FusedOperations} ({report.Summary.FusionRate:F2}%)");
        sb.AppendLine($"  Fused Groups: {report.Summary.FusedGroups}");
        sb.AppendLine($"  Successful Fusions: {report.Summary.SuccessfulFusions}");
        sb.AppendLine($"  Failed Fusions: {report.Summary.FailedFusions}");
        sb.AppendLine($"  Total Kernel Time: {report.Summary.TotalKernelTimeMs:F3}ms");
        sb.AppendLine($"  Average Kernel Time: {report.Summary.AverageKernelTimeMs:F3}ms");
        sb.AppendLine();

        // Pattern Metrics
        sb.AppendLine("Pattern Metrics:");
        foreach (var (pattern, metrics) in report.PatternMetrics.OrderByDescending(kv => kv.Value.Count))
        {
            sb.AppendLine($"  {pattern}:");
            sb.AppendLine($"    Count: {metrics.Count}");
            sb.AppendLine($"    Total Time: {metrics.TotalTimeMs:F3}ms");
            sb.AppendLine($"    Average Time: {metrics.AverageTimeMs:F3}ms");
            sb.AppendLine($"    Min Time: {metrics.MinTimeMs:F3}ms");
            sb.AppendLine($"    Max Time: {metrics.MaxTimeMs:F3}ms");
            sb.AppendLine($"    Estimated Speedup: {metrics.EstimatedSpeedup:F2}x");
        }

        sb.AppendLine();

        // Top Kernels
        sb.AppendLine("Top 10 Slowest Kernels:");
        var slowestKernels = report.KernelExecutions
            .OrderByDescending(k => k.DurationMs)
            .Take(10);

        foreach (var kernel in slowestKernels)
        {
            sb.AppendLine($"  {kernel.KernelName}: {kernel.DurationMs:F3}ms");
        }

        return sb.ToString();
    }

    public string GenerateJsonReport(FusionProfilingReport report)
    {
        var options = new System.Text.Json.JsonSerializerOptions
        {
            WriteIndented = true
        };

        return System.Text.Json.JsonSerializer.Serialize(report, options);
    }

    public string GenerateMarkdownReport(FusionProfilingReport report)
    {
        var sb = new StringBuilder();

        sb.AppendLine("# Fusion Profiling Report");
        sb.AppendLine();

        sb.AppendLine("## Summary");
        sb.AppendLine();
        sb.AppendLine("| Metric | Value |");
        sb.AppendLine("|--------|-------|");
        sb.AppendLine($"| Total Operations | {report.Summary.TotalOperations} |");
        sb.AppendLine($"| Fused Operations | {report.Summary.FusedOperations} ({report.Summary.FusionRate:F2}%) |");
        sb.AppendLine($"| Fused Groups | {report.Summary.FusedGroups} |");
        sb.AppendLine($"| Successful Fusions | {report.Summary.SuccessfulFusions} |");
        sb.AppendLine($"| Failed Fusions | {report.Summary.FailedFusions} |");
        sb.AppendLine($"| Total Kernel Time | {report.Summary.TotalKernelTimeMs:F3}ms |");
        sb.AppendLine($"| Average Kernel Time | {report.Summary.AverageKernelTimeMs:F3}ms |");
        sb.AppendLine();

        sb.AppendLine("## Pattern Metrics");
        sb.AppendLine();
        sb.AppendLine("| Pattern | Count | Total Time (ms) | Avg Time (ms) | Min Time (ms) | Max Time (ms) | Speedup |");
        sb.AppendLine("|---------|-------|-----------------|---------------|---------------|---------------|---------|");

        foreach (var (pattern, metrics) in report.PatternMetrics.OrderByDescending(kv => kv.Value.Count))
        {
            sb.AppendLine($"| {pattern} | {metrics.Count} | {metrics.TotalTimeMs:F3} | " +
                         $"{metrics.AverageTimeMs:F3} | {metrics.MinTimeMs:F3} | {metrics.MaxTimeMs:F3} | " +
                         $"{metrics.EstimatedSpeedup:F2}x |");
        }

        return sb.ToString();
    }
}
```

## Implementation Tasks

1. **Create profiling interfaces** (20 min)
   - IFusionProfiler interface
   - FusionProfilingSession class
   - FusionDecision record

2. **Create profiling report structures** (15 min)
   - FusionProfilingReport record
   - KernelExecutionRecord record
   - FusionSummary record
   - FusionPatternMetrics record

3. **Implement FusionProfiler** (35 min)
   - Main profiler implementation
   - Thread-safe recording
   - Report generation
   - Summary computation

4. **Implement IKernelTimingInstrument** (30 min)
   - KernelTimingInstrument class
   - Timing record management
   - Statistics computation

5. **Implement IFusionDecisionLogger** (25 min)
   - Console logger
   - File logger
   - Decision logging

6. **Implement IPerformanceReportGenerator** (25 min)
   - Text report generation
   - JSON report generation
   - Markdown report generation

## Test Cases

```csharp
[Test]
public void Profiler_RecordDecision_Retrievable()
{
    var profiler = new FusionProfiler();
    var decision = new FusionDecision
    {
        OperationChain = "Add -> Mul",
        Fused = true,
        PatternType = FusionPatternType.ElementWise,
        RejectionReason = null,
        Timestamp = DateTime.UtcNow,
        Metadata = new Dictionary<string, object>()
    };

    profiler.RecordDecision(decision);
    var report = profiler.GetReport();

    Assert.AreEqual(1, report.Decisions.Count);
    Assert.IsTrue(report.Decisions[0].Fused);
}

[Test]
public void KernelTimingInstrument_ComputesStatistics()
{
    var instrument = new KernelTimingInstrument();
    instrument.RecordKernelLaunch("test_kernel", CreateLaunchConfig());

    instrument.RecordKernelComplete("test_kernel", 10.0);
    instrument.RecordKernelComplete("test_kernel", 20.0);
    instrument.RecordKernelComplete("test_kernel", 30.0);

    var stats = instrument.GetTimingStatistics("test_kernel");

    Assert.AreEqual(3, stats.ExecutionCount);
    Assert.AreEqual(20.0, stats.AverageTimeMs);
    Assert.AreEqual(10.0, stats.MinTimeMs);
    Assert.AreEqual(30.0, stats.MaxTimeMs);
}

[Test]
public void PerformanceReportGenerator_GeneratesTextReport()
{
    var generator = new PerformanceReportGenerator();
    var report = CreateSampleReport();

    var text = generator.GenerateTextReport(report);

    Assert.IsNotEmpty(text);
    Assert.IsTrue(text.Contains("Fusion Profiling Report"));
    Assert.IsTrue(text.Contains("Total Operations"));
}

[Test]
public void PerformanceReportGenerator_GeneratesJsonReport()
{
    var generator = new PerformanceReportGenerator();
    var report = CreateSampleReport();

    var json = generator.GenerateJsonReport(report);

    Assert.IsNotEmpty(json);
    // Verify it's valid JSON by deserializing
    var deserialized = System.Text.Json.JsonSerializer.Deserialize<FusionProfilingReport>(json);
    Assert.IsNotNull(deserialized);
}
```

## Success Criteria
- Profiler accurately records fusion decisions and kernel timings
- Timing statistics are computed correctly
- Decision logs capture all relevant information
- Report generation produces readable output in all formats
- All operations are thread-safe
- JSON serialization works correctly

## Dependencies
- FusedOperation from fusion engine
- FusionDecision types
- KernelLaunchConfiguration
- ILogger interface
