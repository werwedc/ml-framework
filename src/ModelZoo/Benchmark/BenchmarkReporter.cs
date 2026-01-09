using System.Text.Json;
using System.Text.Json.Serialization;

namespace ModelZoo.Benchmark;

/// <summary>
/// Report format options.
/// </summary>
public enum ReportFormat
{
    /// <summary>
    /// Plain text format.
    /// </summary>
    Text,

    /// <summary>
    /// Markdown format.
    /// </summary>
    Markdown,

    /// <summary>
    /// JSON format.
    /// </summary>
    Json,

    /// <summary>
    /// CSV format.
    /// </summary>
    Csv
}

/// <summary>
/// Generates benchmark reports in various formats.
/// </summary>
public class BenchmarkReporter
{
    /// <summary>
    /// Prints a benchmark result to the console.
    /// </summary>
    /// <param name="result">The benchmark result to print.</param>
    public void PrintResult(BenchmarkResult result)
    {
        if (result == null)
        {
            throw new ArgumentNullException(nameof(result));
        }

        Console.WriteLine(GenerateTextReport(result));
    }

    /// <summary>
    /// Prints a comparison result to the console.
    /// </summary>
    /// <param name="result">The comparison result to print.</param>
    public void PrintComparison(ComparisonResult result)
    {
        if (result == null)
        {
            throw new ArgumentNullException(nameof(result));
        }

        Console.WriteLine(GenerateComparisonTextReport(result));
    }

    /// <summary>
    /// Generates a benchmark report and saves it to a file.
    /// </summary>
    /// <param name="result">The benchmark result.</param>
    /// <param name="outputPath">The file path to save the report to.</param>
    /// <param name="format">The report format.</param>
    public void GenerateReport(BenchmarkResult result, string outputPath, ReportFormat format = ReportFormat.Markdown)
    {
        if (result == null)
        {
            throw new ArgumentNullException(nameof(result));
        }

        if (string.IsNullOrWhiteSpace(outputPath))
        {
            throw new ArgumentException("Output path must not be empty.", nameof(outputPath));
        }

        var content = format switch
        {
            ReportFormat.Text => GenerateTextReport(result),
            ReportFormat.Markdown => GenerateMarkdownReport(result),
            ReportFormat.Json => GenerateJsonReport(result),
            ReportFormat.Csv => GenerateCsvReport(result),
            _ => throw new ArgumentException($"Unsupported format: {format}", nameof(format))
        };

        File.WriteAllText(outputPath, content);
    }

    /// <summary>
    /// Generates a comparison report and saves it to a file.
    /// </summary>
    /// <param name="result">The comparison result.</param>
    /// <param name="outputPath">The file path to save the report to.</param>
    /// <param name="format">The report format.</param>
    public void GenerateComparisonReport(ComparisonResult result, string outputPath, ReportFormat format = ReportFormat.Markdown)
    {
        if (result == null)
        {
            throw new ArgumentNullException(nameof(result));
        }

        if (string.IsNullOrWhiteSpace(outputPath))
        {
            throw new ArgumentException("Output path must not be empty.", nameof(outputPath));
        }

        var content = format switch
        {
            ReportFormat.Text => GenerateComparisonTextReport(result),
            ReportFormat.Markdown => GenerateComparisonMarkdownReport(result),
            ReportFormat.Json => GenerateComparisonJsonReport(result),
            ReportFormat.Csv => GenerateComparisonCsvReport(result),
            _ => throw new ArgumentException($"Unsupported format: {format}", nameof(format))
        };

        File.WriteAllText(outputPath, content);
    }

    /// <summary>
    /// Generates a text format benchmark report.
    /// </summary>
    private static string GenerateTextReport(BenchmarkResult result)
    {
        var report = new System.Text.StringBuilder();

        report.AppendLine("=".PadRight(80, '='));
        report.AppendLine("MODEL BENCHMARK REPORT");
        report.AppendLine("=".PadRight(80, '='));
        report.AppendLine();
        report.AppendLine($"Model Name:      {result.ModelName}");
        report.AppendLine($"Dataset:         {result.Dataset}");
        report.AppendLine($"Timestamp:       {result.Timestamp:yyyy-MM-dd HH:mm:ss UTC}");
        report.AppendLine($"Duration:        {result.BenchmarkDuration.TotalSeconds:F2}s");
        report.AppendLine();
        report.AppendLine("-".PadRight(80, '-'));
        report.AppendLine("RESULTS");
        report.AppendLine("-".PadRight(80, '-'));
        report.AppendLine($"Total Samples:       {result.TotalSamples:N0}");
        report.AppendLine($"Throughput:          {result.Throughput:F2} samples/s");
        report.AppendLine($"Accuracy:            {result.Accuracy:F4}");
        report.AppendLine();
        report.AppendLine("-".PadRight(80, '-'));
        report.AppendLine("LATENCY STATISTICS");
        report.AppendLine("-".PadRight(80, '-'));
        report.AppendLine($"Average:             {result.AvgLatency:F2} ms");
        report.AppendLine($"Minimum:             {result.MinLatency:F2} ms");
        report.AppendLine($"Maximum:             {result.MaxLatency:F2} ms");
        report.AppendLine($"Median (P50):        {result.P50Latency:F2} ms");
        report.AppendLine($"P95:                 {result.P95Latency:F2} ms");
        report.AppendLine($"P99:                 {result.P99Latency:F2} ms");
        report.AppendLine();
        report.AppendLine("-".PadRight(80, '-'));
        report.AppendLine("MEMORY USAGE");
        report.AppendLine("-".PadRight(80, '-'));
        report.AppendLine($"Peak:                {result.MemoryPeak:N0} bytes ({FormatBytes(result.MemoryPeak)})");
        report.AppendLine($"Average:             {result.MemoryAvg:N0} bytes ({FormatBytes(result.MemoryAvg)})");
        report.AppendLine();
        report.AppendLine("=".PadRight(80, '='));

        return report.ToString();
    }

    /// <summary>
    /// Generates a markdown format benchmark report.
    /// </summary>
    private static string GenerateMarkdownReport(BenchmarkResult result)
    {
        var report = new System.Text.StringBuilder();

        report.AppendLine("# Model Benchmark Report");
        report.AppendLine();
        report.AppendLine("## Model Information");
        report.AppendLine();
        report.AppendLine($"- **Model Name:** {result.ModelName}");
        report.AppendLine($"- **Dataset:** {result.Dataset}");
        report.AppendLine($"- **Timestamp:** {result.Timestamp:yyyy-MM-dd HH:mm:ss UTC}");
        report.AppendLine($"- **Duration:** {result.BenchmarkDuration.TotalSeconds:F2}s");
        report.AppendLine();

        report.AppendLine("## Results");
        report.AppendLine();
        report.AppendLine("| Metric | Value |");
        report.AppendLine("|--------|-------|");
        report.AppendLine($"| Total Samples | {result.TotalSamples:N0} |");
        report.AppendLine($"| Throughput | {result.Throughput:F2} samples/s |");
        report.AppendLine($"| Accuracy | {result.Accuracy:F4} |");
        report.AppendLine();

        report.AppendLine("## Latency Statistics");
        report.AppendLine();
        report.AppendLine("| Percentile | Latency (ms) |");
        report.AppendLine("|------------|--------------|");
        report.AppendLine($"| Min | {result.MinLatency:F2} |");
        report.AppendLine($"| P50 | {result.P50Latency:F2} |");
        report.AppendLine($"| P95 | {result.P95Latency:F2} |");
        report.AppendLine($"| P99 | {result.P99Latency:F2} |");
        report.AppendLine($"| Max | {result.MaxLatency:F2} |");
        report.AppendLine();

        report.AppendLine("## Memory Usage");
        report.AppendLine();
        report.AppendLine("| Metric | Value |");
        report.AppendLine("|--------|-------|");
        report.AppendLine($"| Peak | {FormatBytes(result.MemoryPeak)} ({result.MemoryPeak:N0} bytes) |");
        report.AppendLine($"| Average | {FormatBytes(result.MemoryAvg)} ({result.MemoryAvg:N0} bytes) |");

        return report.ToString();
    }

    /// <summary>
    /// Generates a JSON format benchmark report.
    /// </summary>
    private static string GenerateJsonReport(BenchmarkResult result)
    {
        var options = new JsonSerializerOptions
        {
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        };

        return JsonSerializer.Serialize(result, options);
    }

    /// <summary>
    /// Generates a CSV format benchmark report.
    /// </summary>
    private static string GenerateCsvReport(BenchmarkResult result)
    {
        var csv = new System.Text.StringBuilder();

        csv.AppendLine("metric,value");
        csv.AppendLine($"model_name,{result.ModelName}");
        csv.AppendLine($"dataset,{result.Dataset}");
        csv.AppendLine($"timestamp,{result.Timestamp:yyyy-MM-dd HH:mm:ss UTC}");
        csv.AppendLine($"duration_seconds,{result.BenchmarkDuration.TotalSeconds:F2}");
        csv.AppendLine($"total_samples,{result.TotalSamples}");
        csv.AppendLine($"throughput_samples_per_second,{result.Throughput:F2}");
        csv.AppendLine($"accuracy,{result.Accuracy:F4}");
        csv.AppendLine($"avg_latency_ms,{result.AvgLatency:F2}");
        csv.AppendLine($"min_latency_ms,{result.MinLatency:F2}");
        csv.AppendLine($"max_latency_ms,{result.MaxLatency:F2}");
        csv.AppendLine($"p50_latency_ms,{result.P50Latency:F2}");
        csv.AppendLine($"p95_latency_ms,{result.P95Latency:F2}");
        csv.AppendLine($"p99_latency_ms,{result.P99Latency:F2}");
        csv.AppendLine($"memory_peak_bytes,{result.MemoryPeak}");
        csv.AppendLine($"memory_avg_bytes,{result.MemoryAvg}");

        return csv.ToString();
    }

    /// <summary>
    /// Generates a text format comparison report.
    /// </summary>
    private static string GenerateComparisonTextReport(ComparisonResult result)
    {
        var report = new System.Text.StringBuilder();

        report.AppendLine("=".PadRight(80, '='));
        report.AppendLine("MODEL COMPARISON REPORT");
        report.AppendLine("=".PadRight(80, '='));
        report.AppendLine();
        report.AppendLine($"Timestamp:       {result.Timestamp:yyyy-MM-dd HH:mm:ss UTC}");
        report.AppendLine($"Total Duration:  {result.TotalDuration.TotalSeconds:F2}s");
        report.AppendLine($"Models Compared: {result.ModelCount}");
        report.AppendLine($"Winner:          {result.Winner ?? "N/A"}");
        report.AppendLine();
        report.AppendLine("-".PadRight(80, '-'));
        report.AppendLine("RANKINGS BY METRIC");
        report.AppendLine("-".PadRight(80, '-'));

        foreach (var ranking in result.RankByMetric)
        {
            report.AppendLine($"{ranking.Key.ToUpper()}: {string.Join(" > ", ranking.Value)}");
        }

        report.AppendLine();
        report.AppendLine("-".PadRight(80, '-'));
        report.AppendLine("DETAILED RESULTS");
        report.AppendLine("-".PadRight(80, '-'));

        foreach (var kvp in result.ModelResults)
        {
            report.AppendLine();
            report.AppendLine($"Model: {kvp.Key}");
            report.AppendLine($"  Throughput:  {kvp.Value.Throughput:F2} samples/s");
            report.AppendLine($"  Avg Latency: {kvp.Value.AvgLatency:F2} ms");
            report.AppendLine($"  Accuracy:    {kvp.Value.Accuracy:F4}");
            report.AppendLine($"  Memory Peak: {FormatBytes(kvp.Value.MemoryPeak)}");
        }

        report.AppendLine();
        report.AppendLine("-".PadRight(80, '-'));
        report.AppendLine("STATISTICAL SIGNIFICANCE");
        report.AppendLine("-".PadRight(80, '-'));

        foreach (var sig in result.StatisticalSignificance)
        {
            var status = sig.Value ? "YES" : "NO";
            report.AppendLine($"{sig.Key}: {status}");
        }

        report.AppendLine();
        report.AppendLine("=".PadRight(80, '='));

        return report.ToString();
    }

    /// <summary>
    /// Generates a markdown format comparison report.
    /// </summary>
    private static string GenerateComparisonMarkdownReport(ComparisonResult result)
    {
        var report = new System.Text.StringBuilder();

        report.AppendLine("# Model Comparison Report");
        report.AppendLine();
        report.AppendLine("## Summary");
        report.AppendLine();
        report.AppendLine($"- **Timestamp:** {result.Timestamp:yyyy-MM-dd HH:mm:ss UTC}");
        report.AppendLine($"- **Total Duration:** {result.TotalDuration.TotalSeconds:F2}s");
        report.AppendLine($"- **Models Compared:** {result.ModelCount}");
        report.AppendLine($"- **Winner:** {result.Winner ?? "N/A"}");
        report.AppendLine();

        report.AppendLine("## Rankings by Metric");
        report.AppendLine();

        foreach (var ranking in result.RankByMetric)
        {
            report.AppendLine($"### {ranking.Key.ToUpper()}");
            report.AppendLine();
            report.AppendLine(string.Join(" > ", ranking.Value));
            report.AppendLine();
        }

        report.AppendLine("## Detailed Results");
        report.AppendLine();

        report.AppendLine("| Model | Throughput (samples/s) | Avg Latency (ms) | Accuracy | Memory Peak |");
        report.AppendLine("|-------|----------------------|------------------|----------|-------------|");

        foreach (var kvp in result.ModelResults.OrderBy(r => r.Key))
        {
            report.AppendLine($"| {kvp.Key} | {kvp.Value.Throughput:F2} | {kvp.Value.AvgLatency:F2} | {kvp.Value.Accuracy:F4} | {FormatBytes(kvp.Value.MemoryPeak)} |");
        }

        report.AppendLine();
        report.AppendLine("## Statistical Significance");
        report.AppendLine();
        report.AppendLine("| Comparison | Significant |");
        report.AppendLine("|------------|-------------|");

        foreach (var sig in result.StatisticalSignificance.OrderBy(s => s.Key))
        {
            var status = sig.Value ? "✓" : "✗";
            report.AppendLine($"| {sig.Key} | {status} |");
        }

        return report.ToString();
    }

    /// <summary>
    /// Generates a JSON format comparison report.
    /// </summary>
    private static string GenerateComparisonJsonReport(ComparisonResult result)
    {
        var options = new JsonSerializerOptions
        {
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        };

        return JsonSerializer.Serialize(result, options);
    }

    /// <summary>
    /// Generates a CSV format comparison report.
    /// </summary>
    private static string GenerateComparisonCsvReport(ComparisonResult result)
    {
        var csv = new System.Text.StringBuilder();

        csv.AppendLine("model,throughput_samples_per_second,avg_latency_ms,accuracy,memory_peak_bytes");

        foreach (var kvp in result.ModelResults.OrderBy(r => r.Key))
        {
            csv.AppendLine($"{kvp.Key},{kvp.Value.Throughput:F2},{kvp.Value.AvgLatency:F2},{kvp.Value.Accuracy:F4},{kvp.Value.MemoryPeak}");
        }

        return csv.ToString();
    }

    /// <summary>
    /// Formats bytes to a human-readable string.
    /// </summary>
    private static string FormatBytes(long bytes)
    {
        string[] sizes = { "B", "KB", "MB", "GB", "TB" };
        double len = bytes;
        int order = 0;

        while (len >= 1024 && order < sizes.Length - 1)
        {
            order++;
            len /= 1024;
        }

        return $"{len:0.##} {sizes[order]}";
    }
}
