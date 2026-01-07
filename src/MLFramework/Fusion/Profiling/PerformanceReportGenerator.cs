using System.Text;

namespace MLFramework.Fusion.Profiling;

/// <summary>
/// Interface for generating performance reports
/// </summary>
public interface IPerformanceReportGenerator
{
    string GenerateTextReport(FusionProfilingReport report);
    string GenerateJsonReport(FusionProfilingReport report);
    string GenerateMarkdownReport(FusionProfilingReport report);
}

/// <summary>
/// Implementation of performance report generator
/// </summary>
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
