using System.Text.Json;
using System.Text.Json.Serialization;

namespace MLFramework.Quantization.Evaluation;

/// <summary>
/// Generates reports from accuracy evaluation results.
/// </summary>
public class AccuracyReporter
{
    private readonly JsonSerializerOptions _jsonOptions;

    /// <summary>
    /// Initializes a new instance of the AccuracyReporter.
    /// </summary>
    public AccuracyReporter()
    {
        _jsonOptions = new JsonSerializerOptions
        {
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
        };
    }

    /// <summary>
    /// Generates a human-readable text report.
    /// </summary>
    /// <param name="report">Accuracy report</param>
    /// <returns>Formatted text report</returns>
    public string GenerateTextReport(AccuracyReport report)
    {
        if (report == null)
            throw new ArgumentNullException(nameof(report));

        var sb = new System.Text.StringBuilder();

        sb.AppendLine("Model Accuracy Evaluation Report");
        sb.AppendLine(new string('=', 70));
        sb.AppendLine();

        // Timestamp and metadata
        sb.AppendLine($"Generated: {report.Timestamp:yyyy-MM-dd HH:mm:ss} UTC");
        if (report.Metadata.Count > 0)
        {
            sb.AppendLine();
            sb.AppendLine("Metadata:");
            foreach (var kvp in report.Metadata)
            {
                sb.AppendLine($"  {kvp.Key}: {kvp.Value}");
            }
        }
        sb.AppendLine();

        // Accuracy summary
        sb.AppendLine("Accuracy Summary:");
        sb.AppendLine(new string('-', 70));
        sb.AppendLine($"  FP32 Model Accuracy:       {report.FP32Accuracy:F4}");
        sb.AppendLine($"  Quantized Model Accuracy:   {report.QuantizedAccuracy:F4}");
        sb.AppendLine($"  Accuracy Drop:              {report.AccuracyDrop:F4}");
        sb.AppendLine($"  Acceptable Threshold:       {report.AcceptableThreshold:F4}");
        sb.AppendLine();

        // Acceptance status
        string status = report.IsAcceptable ? "✓ PASS" : "✗ FAIL";
        sb.AppendLine($"Status: {status}");

        if (report.IsAcceptable)
        {
            sb.AppendLine("  Accuracy drop is within acceptable limits.");
        }
        else
        {
            float excessDrop = report.AccuracyDrop - report.AcceptableThreshold;
            sb.AppendLine($"  WARNING: Accuracy drop exceeds threshold by {excessDrop:F4}");
        }

        sb.AppendLine();

        // Per-layer results
        if (report.PerLayerResults.Length > 0)
        {
            sb.AppendLine("Per-Layer Sensitivity Analysis:");
            sb.AppendLine(new string('-', 70));

            // Sort by accuracy impact (descending)
            var sortedResults = report.PerLayerResults
                .OrderByDescending(r => r.AccuracyImpact)
                .ToList();

            foreach (var result in sortedResults)
            {
                string icon = result.IsSensitive ? "⚠" : "✓";
                sb.AppendLine($"  {icon} {result.LayerName}");
                sb.AppendLine($"     Accuracy Impact: {result.AccuracyImpact:F4}");
                sb.AppendLine($"     Recommended:     {result.RecommendedAction}");
                sb.AppendLine();
            }

            // Summary statistics
            int sensitiveCount = sortedResults.Count(r => r.IsSensitive);
            int safeCount = sortedResults.Count - sensitiveCount;

            sb.AppendLine("Summary:");
            sb.AppendLine($"  Total layers analyzed: {sortedResults.Count}");
            sb.AppendLine($"  Sensitive layers:       {sensitiveCount}");
            sb.AppendLine($"  Safe to quantize:       {safeCount}");
            sb.AppendLine();
        }

        // Recommendations
        sb.AppendLine("Recommendations:");
        sb.AppendLine(new string('-', 70));

        var recommendedFP32 = report.GetRecommendedFP32Layers();
        var quantizable = report.GetQuantizableLayers();

        if (recommendedFP32.Length > 0)
        {
            sb.AppendLine($"  Layers recommended for FP32 fallback:");
            foreach (var layer in recommendedFP32)
            {
                sb.AppendLine($"    - {layer}");
            }
        }

        if (quantizable.Length > 0)
        {
            sb.AppendLine();
            sb.AppendLine($"  Layers safe to quantize:");
            foreach (var layer in quantizable)
            {
                sb.AppendLine($"    - {layer}");
            }
        }

        if (recommendedFP32.Length == 0 && quantizable.Length == 0)
        {
            sb.AppendLine("  No per-layer analysis available.");
        }

        sb.AppendLine();
        sb.AppendLine(new string('=', 70));
        sb.AppendLine("End of Report");

        return sb.ToString();
    }

    /// <summary>
    /// Generates a JSON report.
    /// </summary>
    /// <param name="report">Accuracy report</param>
    /// <returns>JSON string</returns>
    public string GenerateJSONReport(AccuracyReport report)
    {
        if (report == null)
            throw new ArgumentNullException(nameof(report));

        var jsonReport = new SerializableAccuracyReport
        {
            Timestamp = report.Timestamp,
            FP32Accuracy = report.FP32Accuracy,
            QuantizedAccuracy = report.QuantizedAccuracy,
            AccuracyDrop = report.AccuracyDrop,
            AcceptableThreshold = report.AcceptableThreshold,
            IsAcceptable = report.IsAcceptable,
            PerLayerResults = report.PerLayerResults.Select(r => new SerializableLayerResult
            {
                LayerName = r.LayerName,
                AccuracyImpact = r.AccuracyImpact,
                IsSensitive = r.IsSensitive,
                RecommendedAction = r.RecommendedAction
            }).ToArray(),
            Metadata = report.Metadata,
            RecommendedFP32Layers = report.GetRecommendedFP32Layers(),
            QuantizableLayers = report.GetQuantizableLayers()
        };

        return JsonSerializer.Serialize(jsonReport, _jsonOptions);
    }

    /// <summary>
    /// Saves a report to file.
    /// </summary>
    /// <param name="report">Accuracy report</param>
    /// <param name="path">File path</param>
    /// <param name="format">Report format (text or json)</param>
    public async Task SaveReportAsync(
        AccuracyReport report,
        string path,
        ReportFormat format = ReportFormat.Text)
    {
        if (report == null)
            throw new ArgumentNullException(nameof(report));

        if (string.IsNullOrWhiteSpace(path))
            throw new ArgumentException("Path cannot be empty", nameof(path));

        string content;

        switch (format)
        {
            case ReportFormat.JSON:
                content = GenerateJSONReport(report);
                break;
            case ReportFormat.Text:
            default:
                content = GenerateTextReport(report);
                break;
        }

        // Ensure directory exists
        var directory = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }

        await File.WriteAllTextAsync(path, content);
    }

    /// <summary>
    /// Saves a report to file (synchronous version).
    /// </summary>
    public void SaveReport(
        AccuracyReport report,
        string path,
        ReportFormat format = ReportFormat.Text)
    {
        SaveReportAsync(report, path, format).GetAwaiter().GetResult();
    }
}

/// <summary>
/// Report format options.
/// </summary>
public enum ReportFormat
{
    /// <summary>
    /// Human-readable text format.
    /// </summary>
    Text,

    /// <summary>
    /// JSON format for programmatic access.
    /// </summary>
    JSON
}

/// <summary>
/// Serializable version of accuracy report for JSON serialization.
/// </summary>
internal class SerializableAccuracyReport
{
    public DateTime Timestamp { get; set; }
    public float FP32Accuracy { get; set; }
    public float QuantizedAccuracy { get; set; }
    public float AccuracyDrop { get; set; }
    public float AcceptableThreshold { get; set; }
    public bool IsAcceptable { get; set; }
    public SerializableLayerResult[] PerLayerResults { get; set; } = Array.Empty<SerializableLayerResult>();
    public Dictionary<string, object> Metadata { get; set; } = new();
    public string[] RecommendedFP32Layers { get; set; } = Array.Empty<string>();
    public string[] QuantizableLayers { get; set; } = Array.Empty<string>();
}

/// <summary>
/// Serializable layer result for JSON serialization.
/// </summary>
internal class SerializableLayerResult
{
    public string LayerName { get; set; } = string.Empty;
    public float AccuracyImpact { get; set; }
    public bool IsSensitive { get; set; }
    public string RecommendedAction { get; set; } = string.Empty;
}
