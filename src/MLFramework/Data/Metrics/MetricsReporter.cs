using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;

namespace MLFramework.Data.Metrics
{
    /// <summary>
    /// Provides reporting functionality for data loading metrics.
    /// </summary>
    public static class MetricsReporter
    {
        /// <summary>
        /// Generates a human-readable text report.
        /// </summary>
        /// <param name="metrics">The metrics to report on.</param>
        /// <returns>A formatted text report.</returns>
        public static string GenerateTextReport(DataLoadingMetrics metrics)
        {
            var summary = metrics.GetMetricsSummary();
            var sb = new StringBuilder();

            sb.AppendLine("=== Data Loading Metrics ===");
            sb.AppendLine();

            if (summary.Count == 0)
            {
                sb.AppendLine("No metrics recorded.");
                return sb.ToString();
            }

            foreach (var kvp in summary.OrderByDescending(x => x.Value.Total))
            {
                sb.AppendLine($"{kvp.Key}:");
                sb.AppendLine($"  Count: {kvp.Value.Count}");
                sb.AppendLine($"  Avg:   {kvp.Value.Average:F2} ms");
                sb.AppendLine($"  Min:   {kvp.Value.Min:F2} ms");
                sb.AppendLine($"  Max:   {kvp.Value.Max:F2} ms");
                sb.AppendLine($"  Total: {kvp.Value.Total:F2} ms");
                sb.AppendLine();
            }

            return sb.ToString();
        }

        /// <summary>
        /// Exports metrics to a JSON file.
        /// </summary>
        /// <param name="metrics">The metrics to export.</param>
        /// <param name="filePath">The file path to write to.</param>
        public static void ExportToJson(DataLoadingMetrics metrics, string filePath)
        {
            var summary = metrics.GetMetricsSummary();
            var json = JsonSerializer.Serialize(summary, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(filePath, json);
        }

        /// <summary>
        /// Exports metrics to a CSV file.
        /// </summary>
        /// <param name="metrics">The metrics to export.</param>
        /// <param name="filePath">The file path to write to.</param>
        public static void ExportToCsv(DataLoadingMetrics metrics, string filePath)
        {
            var summary = metrics.GetMetricsSummary();
            var sb = new StringBuilder();

            sb.AppendLine("Metric,Count,Average,Min,Max,Total");

            foreach (var kvp in summary.OrderBy(x => x.Key))
            {
                sb.AppendLine($"{kvp.Key},{kvp.Value.Count},{kvp.Value.Average:F2},{kvp.Value.Min:F2},{kvp.Value.Max:F2},{kvp.Value.Total:F2}");
            }

            File.WriteAllText(filePath, sb.ToString());
        }

        /// <summary>
        /// Generates a summary of key performance indicators.
        /// </summary>
        /// <param name="metrics">The metrics to analyze.</param>
        /// <returns>A formatted summary of key metrics.</returns>
        public static string GenerateSummaryReport(DataLoadingMetrics metrics)
        {
            var summary = metrics.GetMetricsSummary();
            var sb = new StringBuilder();

            sb.AppendLine("=== Key Performance Indicators ===");
            sb.AppendLine();

            if (summary.Count == 0)
            {
                sb.AppendLine("No metrics recorded.");
                return sb.ToString();
            }

            // Find top time-consuming operations
            var topOps = summary.OrderByDescending(x => x.Value.Total).Take(3);
            sb.AppendLine("Top Time-Consuming Operations:");
            foreach (var kvp in topOps)
            {
                sb.AppendLine($"  {kvp.Key}: {kvp.Value.Total:F2} ms");
            }
            sb.AppendLine();

            // Find slowest operations (by average time)
            var slowestOps = summary.Where(x => x.Value.Count > 0).OrderByDescending(x => x.Value.Average).Take(3);
            sb.AppendLine("Slowest Operations (Avg Time):");
            foreach (var kvp in slowestOps)
            {
                sb.AppendLine($"  {kvp.Key}: {kvp.Value.Average:F2} ms");
            }
            sb.AppendLine();

            // Find most variable operations (by max-min spread)
            var variableOps = summary.Where(x => x.Value.Count > 0).OrderByDescending(x => x.Value.Max - x.Value.Min).Take(3);
            sb.AppendLine("Most Variable Operations:");
            foreach (var kvp in variableOps)
            {
                sb.AppendLine($"  {kvp.Key}: {kvp.Value.Max - kvp.Value.Min:F2} ms spread");
            }

            return sb.ToString();
        }
    }
}
