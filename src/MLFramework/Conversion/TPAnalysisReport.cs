using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MLFramework.Conversion;

/// <summary>
/// Contains the analysis report for tensor parallelism conversion of a model.
/// </summary>
public class TPAnalysisReport
{
    /// <summary>
    /// Gets or sets the list of analyzed layers.
    /// </summary>
    public List<LayerAnalysisResult> Layers { get; set; } = new();

    /// <summary>
    /// Gets or sets the total memory in bytes for all layers.
    /// </summary>
    public long TotalMemoryBytes { get; set; }

    /// <summary>
    /// Gets or sets the memory in bytes for parallelizable layers.
    /// </summary>
    public long ParallelizableMemoryBytes { get; set; }

    /// <summary>
    /// Gets the percentage of memory that can be parallelized.
    /// </summary>
    public double ParallelizablePercentage => TotalMemoryBytes > 0
        ? (double)ParallelizableMemoryBytes / TotalMemoryBytes * 100
        : 0;

    /// <summary>
    /// Gets or sets recommendations for the model.
    /// </summary>
    public Dictionary<string, string> Recommendations { get; set; } = new();

    /// <summary>
    /// Gets or sets the suggested world size for tensor parallelism.
    /// </summary>
    public int SuggestedWorldSize { get; set; }

    /// <summary>
    /// Adds a layer analysis result to the report.
    /// </summary>
    public void AddLayer(LayerAnalysisResult layer)
    {
        Layers.Add(layer);
        TotalMemoryBytes += layer.MemoryBytes;
        if (layer.IsParallelizable)
        {
            ParallelizableMemoryBytes += layer.MemoryBytes;
        }
    }

    /// <summary>
    /// Generates a human-readable summary of the analysis report.
    /// </summary>
    public string GenerateSummary()
    {
        var sb = new StringBuilder();
        sb.AppendLine("=== TP Model Analysis ===");
        sb.AppendLine($"Total layers: {Layers.Count}");
        sb.AppendLine($"Parallelizable layers: {Layers.Count(l => l.IsParallelizable)}");
        sb.AppendLine($"Total memory: {TotalMemoryBytes / 1024 / 1024} MB");
        sb.AppendLine($"Parallelizable: {ParallelizableMemoryBytes / 1024 / 1024} MB ({ParallelizablePercentage:F1}%)");
        sb.AppendLine($"Suggested world size: {SuggestedWorldSize}");
        sb.AppendLine();

        sb.AppendLine("Recommendations:");
        foreach (var rec in Recommendations)
        {
            sb.AppendLine($"  - {rec.Key}: {rec.Value}");
        }

        return sb.ToString();
    }
}
