using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using MobileRuntime.Benchmarking.Models;

namespace MobileRuntime.Benchmarking.Export;

public static class BenchmarkExporter
{
    public static void Export(BenchmarkResults results, string filePath, string format = ReportFormat.Json)
    {
        switch (format.ToLower())
        {
            case ReportFormat.Json:
                ExportToJson(results, filePath);
                break;
            case ReportFormat.Csv:
                ExportToCsv(results, filePath);
                break;
            case ReportFormat.Markdown:
                ExportToMarkdown(results, filePath);
                break;
            case ReportFormat.Html:
                ExportToHtml(results, filePath);
                break;
            default:
                throw new ArgumentException($"Unsupported format: {format}");
        }
    }

    public static void ExportProfile(ProfileReport report, string filePath, string format = ReportFormat.Json)
    {
        switch (format.ToLower())
        {
            case ReportFormat.Json:
                ExportProfileToJson(report, filePath);
                break;
            case ReportFormat.Markdown:
                ExportProfileToMarkdown(report, filePath);
                break;
            case ReportFormat.Html:
                ExportProfileToHtml(report, filePath);
                break;
            default:
                throw new ArgumentException($"Unsupported format for profiles: {format}");
        }
    }

    public static void ExportToJson(BenchmarkResults results, string filePath)
    {
        var json = SerializeBenchmarkResultsToJson(results);
        File.WriteAllText(filePath, json);
    }

    public static void ExportToCsv(BenchmarkResults results, string filePath)
    {
        var sb = new StringBuilder();

        // Header
        sb.AppendLine("Name,Iterations,MinTimeMs,MaxTimeMs,AverageTimeMs,MedianTimeMs,StdDevMs," +
                      "MinMemoryMB,MaxMemoryMB,AverageMemoryMB," +
                      "MinEnergyJ,MaxEnergyJ,AverageEnergyJ,Timestamp");

        // Data rows
        foreach (var result in results.Results)
        {
            sb.AppendLine($"{EscapeCsv(result.Name)},{result.Iterations}," +
                         $"{result.MinTime.TotalMilliseconds:F4},{result.MaxTime.TotalMilliseconds:F4}," +
                         $"{result.AverageTime.TotalMilliseconds:F4},{result.MedianTime.TotalMilliseconds:F4}," +
                         $"{result.StdDev:F4}," +
                         $"{result.MinMemoryBytes / (1024.0 * 1024.0):F4},{result.MaxMemoryBytes / (1024.0 * 1024.0):F4}," +
                         $"{result.AverageMemoryBytes / (1024.0 * 1024.0):F4}," +
                         $"{result.MinEnergyJoules:F4},{result.MaxEnergyJoules:F4},{result.AverageEnergyJoules:F4}," +
                         $"{result.Timestamp:O}");
        }

        File.WriteAllText(filePath, sb.ToString());
    }

    public static void ExportToMarkdown(BenchmarkResults results, string filePath)
    {
        var sb = new StringBuilder();

        sb.AppendLine($"# Benchmark Results - {results.SuiteName}");
        sb.AppendLine();
        sb.AppendLine($"Generated: {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss UTC}");
        sb.AppendLine();

        // Summary
        sb.AppendLine("## Summary");
        sb.AppendLine();
        sb.AppendLine($"- Total Time: {results.Summary.TotalTime.TotalMilliseconds:F2} ms");
        sb.AppendLine($"- Total Memory: {results.Summary.TotalMemoryBytes / (1024.0 * 1024.0):F2} MB");
        sb.AppendLine($"- Total Energy: {results.Summary.TotalEnergyJoules:F4} J");
        sb.AppendLine($"- Passed: {results.Summary.PassedCount}");
        sb.AppendLine($"- Failed: {results.Summary.FailedCount}");
        sb.AppendLine();

        // Results table
        sb.AppendLine("## Results");
        sb.AppendLine();
        sb.AppendLine("| Name | Iterations | Min (ms) | Max (ms) | Avg (ms) | Median (ms) | StdDev (ms) | Memory (MB) | Energy (J) |");
        sb.AppendLine("|------|------------|----------|----------|----------|-------------|-------------|-------------|------------|");

        foreach (var result in results.Results)
        {
            sb.AppendLine($"| {EscapeMarkdown(result.Name)} | {result.Iterations} | " +
                         $"{result.MinTime.TotalMilliseconds:F2} | {result.MaxTime.TotalMilliseconds:F2} | " +
                         $"{result.AverageTime.TotalMilliseconds:F2} | {result.MedianTime.TotalMilliseconds:F2} | " +
                         $"{result.StdDev:F2} | {result.AverageMemoryBytes / (1024.0 * 1024.0):F2} | " +
                         $"{result.AverageEnergyJoules:F4} |");
        }

        File.WriteAllText(filePath, sb.ToString());
    }

    public static void ExportToHtml(BenchmarkResults results, string filePath)
    {
        var html = CreateHtmlReport(results);
        File.WriteAllText(filePath, html);
    }

    private static void ExportProfileToJson(ProfileReport report, string filePath)
    {
        var json = SerializeProfileReportToJson(report);
        File.WriteAllText(filePath, json);
    }

    private static string SerializeBenchmarkResultsToJson(BenchmarkResults results)
    {
        var sb = new StringBuilder();
        sb.AppendLine("{");
        sb.AppendLine($"  \"suiteName\": {JsonEscape(results.SuiteName)},");

        sb.AppendLine("  \"results\": [");
        for (int i = 0; i < results.Results.Count; i++)
        {
            var result = results.Results[i];
            sb.AppendLine("    {");
            sb.AppendLine($"      \"name\": {JsonEscape(result.Name)},");
            sb.AppendLine($"      \"iterations\": {result.Iterations},");
            sb.AppendLine($"      \"minTimeMs\": {result.MinTime.TotalMilliseconds:F4},");
            sb.AppendLine($"      \"maxTimeMs\": {result.MaxTime.TotalMilliseconds:F4},");
            sb.AppendLine($"      \"averageTimeMs\": {result.AverageTime.TotalMilliseconds:F4},");
            sb.AppendLine($"      \"medianTimeMs\": {result.MedianTime.TotalMilliseconds:F4},");
            sb.AppendLine($"      \"stdDev\": {result.StdDev:F4},");
            sb.AppendLine($"      \"minMemoryBytes\": {result.MinMemoryBytes},");
            sb.AppendLine($"      \"maxMemoryBytes\": {result.MaxMemoryBytes},");
            sb.AppendLine($"      \"averageMemoryBytes\": {result.AverageMemoryBytes},");
            sb.AppendLine($"      \"minEnergyJoules\": {result.MinEnergyJoules:F4},");
            sb.AppendLine($"      \"maxEnergyJoules\": {result.MaxEnergyJoules:F4},");
            sb.AppendLine($"      \"averageEnergyJoules\": {result.AverageEnergyJoules:F4},");
            sb.AppendLine($"      \"timestamp\": {JsonEscape(result.Timestamp.ToString("O"))}");
            sb.Append($"    }}");

            if (i < results.Results.Count - 1)
                sb.AppendLine(",");
            else
                sb.AppendLine();
        }
        sb.AppendLine("  ],");

        sb.AppendLine("  \"summary\": {");
        sb.AppendLine($"    \"totalTimeMs\": {results.Summary.TotalTime.TotalMilliseconds:F2},");
        sb.AppendLine($"    \"totalMemoryBytes\": {results.Summary.TotalMemoryBytes},");
        sb.AppendLine($"    \"totalEnergyJoules\": {results.Summary.TotalEnergyJoules:F4},");
        sb.AppendLine($"    \"passedCount\": {results.Summary.PassedCount},");
        sb.AppendLine($"    \"failedCount\": {results.Summary.FailedCount}");
        sb.AppendLine("  }");
        sb.Append("}");

        return sb.ToString();
    }

    private static string SerializeProfileReportToJson(ProfileReport report)
    {
        var sb = new StringBuilder();
        sb.AppendLine("{");
        sb.AppendLine($"  \"timestamp\": {JsonEscape(report.Timestamp.ToString("O"))},");
        sb.AppendLine($"  \"totalTimeMs\": {report.TotalTime.TotalMilliseconds:F2},");
        sb.AppendLine("  \"entries\": [");

        for (int i = 0; i < report.Entries.Count; i++)
        {
            var entry = report.Entries[i];
            sb.AppendLine("    {");
            sb.AppendLine($"      \"name\": {JsonEscape(entry.Name)},");
            sb.AppendLine($"      \"callCount\": {entry.CallCount},");
            sb.AppendLine($"      \"totalTimeMs\": {entry.TotalTime.TotalMilliseconds:F4},");
            sb.AppendLine($"      \"averageTimeMs\": {entry.AverageTime.TotalMilliseconds:F4},");
            sb.AppendLine($"      \"minTimeMs\": {entry.MinTime.TotalMilliseconds:F4},");
            sb.AppendLine($"      \"maxTimeMs\": {entry.MaxTime.TotalMilliseconds:F4},");
            sb.AppendLine($"      \"totalMemoryBytes\": {entry.TotalMemoryBytes}");
            sb.Append($"    }}");

            if (i < report.Entries.Count - 1)
                sb.AppendLine(",");
            else
                sb.AppendLine();
        }

        sb.AppendLine("  ]");
        sb.Append("}");

        return sb.ToString();
    }

    private static string JsonEscape(string value)
    {
        if (value == null)
            return "null";
        return $"\"{value.Replace("\\", "\\\\").Replace("\"", "\\\"").Replace("\n", "\\n").Replace("\r", "\\r").Replace("\t", "\\t")}\"";
    }

    private static void ExportProfileToMarkdown(ProfileReport report, string filePath)
    {
        var sb = new StringBuilder();

        sb.AppendLine("# Profile Report");
        sb.AppendLine();
        sb.AppendLine($"Generated: {report.Timestamp:yyyy-MM-dd HH:mm:ss UTC}");
        sb.AppendLine();
        sb.AppendLine($"Total Time: {report.TotalTime.TotalMilliseconds:F2} ms");
        sb.AppendLine();
        sb.AppendLine("## Profile Entries");
        sb.AppendLine();
        sb.AppendLine("| Name | Calls | Total (ms) | Avg (ms) | Min (ms) | Max (ms) | Memory (MB) |");
        sb.AppendLine("|------|-------|------------|----------|----------|----------|-------------|");

        foreach (var entry in report.Entries)
        {
            sb.AppendLine($"| {EscapeMarkdown(entry.Name)} | {entry.CallCount} | " +
                         $"{entry.TotalTime.TotalMilliseconds:F2} | {entry.AverageTime.TotalMilliseconds:F2} | " +
                         $"{entry.MinTime.TotalMilliseconds:F2} | {entry.MaxTime.TotalMilliseconds:F2} | " +
                         $"{entry.TotalMemoryBytes / (1024.0 * 1024.0):F2} |");
        }

        File.WriteAllText(filePath, sb.ToString());
    }

    private static void ExportProfileToHtml(ProfileReport report, string filePath)
    {
        var html = CreateProfileHtmlReport(report);
        File.WriteAllText(filePath, html);
    }

    private static string CreateHtmlReport(BenchmarkResults results)
    {
        var sb = new StringBuilder();

        sb.AppendLine("<!DOCTYPE html>");
        sb.AppendLine("<html lang=\"en\">");
        sb.AppendLine("<head>");
        sb.AppendLine("    <meta charset=\"UTF-8\">");
        sb.AppendLine("    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">");
        sb.AppendLine("    <title>Benchmark Results</title>");
        sb.AppendLine("    <style>");
        sb.AppendLine("        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }");
        sb.AppendLine("        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }");
        sb.AppendLine("        h1 { color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }");
        sb.AppendLine("        h2 { color: #666; margin-top: 30px; }");
        sb.AppendLine("        .summary { display: flex; gap: 20px; margin: 20px 0; }");
        sb.AppendLine("        .summary-card { flex: 1; padding: 15px; background: #f8f9fa; border-radius: 6px; border-left: 4px solid #4CAF50; }");
        sb.AppendLine("        .summary-card h3 { margin: 0 0 10px 0; color: #333; }");
        sb.AppendLine("        .summary-card p { margin: 5px 0; color: #666; font-size: 1.1em; }");
        sb.AppendLine("        table { width: 100%; border-collapse: collapse; margin: 20px 0; }");
        sb.AppendLine("        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }");
        sb.AppendLine("        th { background-color: #4CAF50; color: white; font-weight: bold; }");
        sb.AppendLine("        tr:nth-child(even) { background-color: #f8f9fa; }");
        sb.AppendLine("        tr:hover { background-color: #e8f5e9; }");
        sb.AppendLine("        .timestamp { color: #666; font-size: 0.9em; }");
        sb.AppendLine("        .passed { color: #4CAF50; font-weight: bold; }");
        sb.AppendLine("        .failed { color: #f44336; font-weight: bold; }");
        sb.AppendLine("    </style>");
        sb.AppendLine("</head>");
        sb.AppendLine("<body>");
        sb.AppendLine("    <div class=\"container\">");
        sb.AppendLine($"        <h1>Benchmark Results - {results.SuiteName}</h1>");
        sb.AppendLine($"        <p class=\"timestamp\">Generated: {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss UTC}</p>");

        // Summary section
        sb.AppendLine("        <h2>Summary</h2>");
        sb.AppendLine("        <div class=\"summary\">");
        sb.AppendLine($"            <div class=\"summary-card\"><h3>Total Time</h3><p>{results.Summary.TotalTime.TotalMilliseconds:F2} ms</p></div>");
        sb.AppendLine($"            <div class=\"summary-card\"><h3>Total Memory</h3><p>{results.Summary.TotalMemoryBytes / (1024.0 * 1024.0):F2} MB</p></div>");
        sb.AppendLine($"            <div class=\"summary-card\"><h3>Total Energy</h3><p>{results.Summary.TotalEnergyJoules:F4} J</p></div>");
        sb.AppendLine($"            <div class=\"summary-card\"><h3>Status</h3><p><span class=\"passed\">Passed: {results.Summary.PassedCount}</span> | <span class=\"failed\">Failed: {results.Summary.FailedCount}</span></p></div>");
        sb.AppendLine("        </div>");

        // Results table
        sb.AppendLine("        <h2>Results</h2>");
        sb.AppendLine("        <table>");
        sb.AppendLine("            <thead>");
        sb.AppendLine("                <tr>");
        sb.AppendLine("                    <th>Name</th>");
        sb.AppendLine("                    <th>Iterations</th>");
        sb.AppendLine("                    <th>Min (ms)</th>");
        sb.AppendLine("                    <th>Max (ms)</th>");
        sb.AppendLine("                    <th>Avg (ms)</th>");
        sb.AppendLine("                    <th>Median (ms)</th>");
        sb.AppendLine("                    <th>StdDev (ms)</th>");
        sb.AppendLine("                    <th>Memory (MB)</th>");
        sb.AppendLine("                    <th>Energy (J)</th>");
        sb.AppendLine("                </tr>");
        sb.AppendLine("            </thead>");
        sb.AppendLine("            <tbody>");

        foreach (var result in results.Results)
        {
            sb.AppendLine("                <tr>");
            sb.AppendLine($"                    <td>{result.Name}</td>");
            sb.AppendLine($"                    <td>{result.Iterations}</td>");
            sb.AppendLine($"                    <td>{result.MinTime.TotalMilliseconds:F2}</td>");
            sb.AppendLine($"                    <td>{result.MaxTime.TotalMilliseconds:F2}</td>");
            sb.AppendLine($"                    <td>{result.AverageTime.TotalMilliseconds:F2}</td>");
            sb.AppendLine($"                    <td>{result.MedianTime.TotalMilliseconds:F2}</td>");
            sb.AppendLine($"                    <td>{result.StdDev:F2}</td>");
            sb.AppendLine($"                    <td>{result.AverageMemoryBytes / (1024.0 * 1024.0):F2}</td>");
            sb.AppendLine($"                    <td>{result.AverageEnergyJoules:F4}</td>");
            sb.AppendLine("                </tr>");
        }

        sb.AppendLine("            </tbody>");
        sb.AppendLine("        </table>");
        sb.AppendLine("    </div>");
        sb.AppendLine("</body>");
        sb.AppendLine("</html>");

        return sb.ToString();
    }

    private static string CreateProfileHtmlReport(ProfileReport report)
    {
        var sb = new StringBuilder();

        sb.AppendLine("<!DOCTYPE html>");
        sb.AppendLine("<html lang=\"en\">");
        sb.AppendLine("<head>");
        sb.AppendLine("    <meta charset=\"UTF-8\">");
        sb.AppendLine("    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">");
        sb.AppendLine("    <title>Profile Report</title>");
        sb.AppendLine("    <style>");
        sb.AppendLine("        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }");
        sb.AppendLine("        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }");
        sb.AppendLine("        h1 { color: #333; border-bottom: 2px solid #2196F3; padding-bottom: 10px; }");
        sb.AppendLine("        h2 { color: #666; margin-top: 30px; }");
        sb.AppendLine("        .summary { margin: 20px 0; padding: 15px; background: #e3f2fd; border-radius: 6px; border-left: 4px solid #2196F3; }");
        sb.AppendLine("        table { width: 100%; border-collapse: collapse; margin: 20px 0; }");
        sb.AppendLine("        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }");
        sb.AppendLine("        th { background-color: #2196F3; color: white; font-weight: bold; }");
        sb.AppendLine("        tr:nth-child(even) { background-color: #f8f9fa; }");
        sb.AppendLine("        tr:hover { background-color: #e3f2fd; }");
        sb.AppendLine("        .timestamp { color: #666; font-size: 0.9em; }");
        sb.AppendLine("    </style>");
        sb.AppendLine("</head>");
        sb.AppendLine("<body>");
        sb.AppendLine("    <div class=\"container\">");
        sb.AppendLine("        <h1>Profile Report</h1>");
        sb.AppendLine($"        <p class=\"timestamp\">Generated: {report.Timestamp:yyyy-MM-dd HH:mm:ss UTC}</p>");
        sb.AppendLine($"        <div class=\"summary\"><strong>Total Time:</strong> {report.TotalTime.TotalMilliseconds:F2} ms</div>");
        sb.AppendLine("        <h2>Profile Entries</h2>");
        sb.AppendLine("        <table>");
        sb.AppendLine("            <thead>");
        sb.AppendLine("                <tr>");
        sb.AppendLine("                    <th>Name</th>");
        sb.AppendLine("                    <th>Calls</th>");
        sb.AppendLine("                    <th>Total (ms)</th>");
        sb.AppendLine("                    <th>Avg (ms)</th>");
        sb.AppendLine("                    <th>Min (ms)</th>");
        sb.AppendLine("                    <th>Max (ms)</th>");
        sb.AppendLine("                    <th>Memory (MB)</th>");
        sb.AppendLine("                </tr>");
        sb.AppendLine("            </thead>");
        sb.AppendLine("            <tbody>");

        foreach (var entry in report.Entries)
        {
            sb.AppendLine("                <tr>");
            sb.AppendLine($"                    <td>{entry.Name}</td>");
            sb.AppendLine($"                    <td>{entry.CallCount}</td>");
            sb.AppendLine($"                    <td>{entry.TotalTime.TotalMilliseconds:F2}</td>");
            sb.AppendLine($"                    <td>{entry.AverageTime.TotalMilliseconds:F2}</td>");
            sb.AppendLine($"                    <td>{entry.MinTime.TotalMilliseconds:F2}</td>");
            sb.AppendLine($"                    <td>{entry.MaxTime.TotalMilliseconds:F2}</td>");
            sb.AppendLine($"                    <td>{entry.TotalMemoryBytes / (1024.0 * 1024.0):F2}</td>");
            sb.AppendLine("                </tr>");
        }

        sb.AppendLine("            </tbody>");
        sb.AppendLine("        </table>");
        sb.AppendLine("    </div>");
        sb.AppendLine("</body>");
        sb.AppendLine("</html>");

        return sb.ToString();
    }

    private static string EscapeCsv(string value)
    {
        if (value.Contains(',') || value.Contains('"') || value.Contains('\n'))
        {
            return $"\"{value.Replace("\"", "\"\"")}\"";
        }
        return value;
    }

    private static string EscapeMarkdown(string value)
    {
        return value.Replace("|", "\\|");
    }
}
