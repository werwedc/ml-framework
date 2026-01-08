using System;
using System.Text;

namespace MLFramework.Checkpointing.Extensions;

/// <summary>
/// Statistics for checkpointing
/// </summary>
public class CheckpointStatistics
{
    /// <summary>
    /// Layer ID (if applicable)
    /// </summary>
    public string LayerId { get; set; } = string.Empty;

    /// <summary>
    /// Current memory used by checkpoints (in bytes)
    /// </summary>
    public long MemoryUsed { get; set; }

    /// <summary>
    /// Peak memory used by checkpoints (in bytes)
    /// </summary>
    public long PeakMemoryUsed { get; set; }

    /// <summary>
    /// Total number of recomputations
    /// </summary>
    public int RecomputationCount { get; set; }

    /// <summary>
    /// Total time spent on recomputation (in milliseconds)
    /// </summary>
    public long RecomputationTimeMs { get; set; }

    /// <summary>
    /// Whether checkpointing is enabled
    /// </summary>
    public bool IsCheckpointingEnabled { get; set; }

    /// <summary>
    /// Number of checkpoints stored
    /// </summary>
    public int CheckpointCount { get; set; }

    /// <summary>
    /// Memory savings compared to full storage (in bytes)
    /// </summary>
    public long MemorySavings { get; set; }

    /// <summary>
    /// Memory reduction percentage (0.0 to 1.0)
    /// </summary>
    public float MemoryReductionPercentage { get; set; }

    /// <summary>
    /// Timestamp when statistics were collected
    /// </summary>
    public DateTime Timestamp { get; set; }

    /// <summary>
    /// Creates a string summary of the statistics
    /// </summary>
    /// <returns>Summary string</returns>
    public override string ToString()
    {
        var sb = new StringBuilder();
        sb.AppendLine("Checkpoint Statistics:");
        sb.AppendLine($"  Memory Used: {FormatBytes(MemoryUsed)}");
        sb.AppendLine($"  Peak Memory: {FormatBytes(PeakMemoryUsed)}");
        sb.AppendLine($"  Recomputations: {RecomputationCount}");
        sb.AppendLine($"  Recomputation Time: {RecomputationTimeMs}ms");
        sb.AppendLine($"  Checkpoint Count: {CheckpointCount}");
        sb.AppendLine($"  Memory Savings: {FormatBytes(MemorySavings)}");
        sb.AppendLine($"  Memory Reduction: {MemoryReductionPercentage:P0}");
        sb.AppendLine($"  Enabled: {IsCheckpointingEnabled}");
        return sb.ToString();
    }

    private string FormatBytes(long bytes)
    {
        if (bytes < 1024) return $"{bytes}B";
        if (bytes < 1024 * 1024) return $"{bytes / 1024.0:F2}KB";
        if (bytes < 1024 * 1024 * 1024) return $"{bytes / (1024.0 * 1024):F2}MB";
        return $"{bytes / (1024.0 * 1024 * 1024):F2}GB";
    }
}
