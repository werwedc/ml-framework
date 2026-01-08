using System.Text;

namespace MLFramework.Checkpointing.Distributed;

/// <summary>
/// Memory statistics for distributed checkpointing
/// </summary>
public class DistributedMemoryStats
{
    /// <summary>
    /// Total current memory used across all ranks (in bytes)
    /// </summary>
    public long TotalCurrentMemoryUsed { get; set; }

    /// <summary>
    /// Total peak memory used across all ranks (in bytes)
    /// </summary>
    public long TotalPeakMemoryUsed { get; set; }

    /// <summary>
    /// Memory used per rank (in bytes)
    /// </summary>
    public List<long> PerRankMemoryUsed { get; set; } = new List<long>();

    /// <summary>
    /// Average memory per rank (in bytes)
    /// </summary>
    public long AverageMemoryPerRank { get; set; }

    /// <summary>
    /// Maximum memory used by any rank (in bytes)
    /// </summary>
    public long MaxMemoryUsed { get; set; }

    /// <summary>
    /// Minimum memory used by any rank (in bytes)
    /// </summary>
    public long MinMemoryUsed { get; set; }

    /// <summary>
    /// Total checkpoint count across all ranks
    /// </summary>
    public int TotalCheckpointCount { get; set; }

    /// <summary>
    /// Checkpoint count per rank
    /// </summary>
    public List<int> PerRankCheckpointCount { get; set; } = new List<int>();

    /// <summary>
    /// Timestamp when stats were collected
    /// </summary>
    public DateTime Timestamp { get; set; }

    /// <summary>
    /// Creates a string summary of the statistics
    /// </summary>
    /// <returns>Summary string</returns>
    public override string ToString()
    {
        var sb = new StringBuilder();
        sb.AppendLine("Distributed Memory Statistics:");
        sb.AppendLine($"  Total Memory Used: {FormatBytes(TotalCurrentMemoryUsed)}");
        sb.AppendLine($"  Total Peak Memory: {FormatBytes(TotalPeakMemoryUsed)}");
        sb.AppendLine($"  Average Memory Per Rank: {FormatBytes(AverageMemoryPerRank)}");
        sb.AppendLine($"  Max Memory Used: {FormatBytes(MaxMemoryUsed)}");
        sb.AppendLine($"  Min Memory Used: {FormatBytes(MinMemoryUsed)}");
        sb.AppendLine($"  Total Checkpoints: {TotalCheckpointCount}");
        sb.AppendLine($"  Per-Rank Memory: [{string.Join(", ", PerRankMemoryUsed.Select(FormatBytes))}]");
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
