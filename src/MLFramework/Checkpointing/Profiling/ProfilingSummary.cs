using System.Text;
using System.Text.Json;

namespace MLFramework.Checkpointing.Profiling;

/// <summary>
/// Summary of profiling data
/// </summary>
public class ProfilingSummary
{
    /// <summary>
    /// Start time of profiling
    /// </summary>
    public DateTime StartTime { get; set; }

    /// <summary>
    /// End time of profiling
    /// </summary>
    public DateTime EndTime { get; set; }

    /// <summary>
    /// Total duration in milliseconds
    /// </summary>
    public double Duration { get; set; }

    /// <summary>
    /// Total number of events
    /// </summary>
    public int TotalEvents { get; set; }

    /// <summary>
    /// Total checkpoint time in milliseconds
    /// </summary>
    public long TotalCheckpointTime { get; set; }

    /// <summary>
    /// Total recomputation time in milliseconds
    /// </summary>
    public long TotalRecomputeTime { get; set; }

    /// <summary>
    /// Total memory saved in bytes
    /// </summary>
    public long TotalMemorySaved { get; set; }

    /// <summary>
    /// Layer profiles
    /// </summary>
    public List<LayerProfile> LayerProfiles { get; set; } = new List<LayerProfile>();

    /// <summary>
    /// Creates a string summary
    /// </summary>
    /// <returns>Summary string</returns>
    public override string ToString()
    {
        var sb = new StringBuilder();
        sb.AppendLine("Checkpoint Profiling Summary");
        sb.AppendLine("============================");
        sb.AppendLine($"Duration: {Duration:F2}ms");
        sb.AppendLine($"Total Events: {TotalEvents}");
        sb.AppendLine($"Total Checkpoint Time: {TotalCheckpointTime}ms");
        sb.AppendLine($"Total Recompute Time: {TotalRecomputeTime}ms");
        sb.AppendLine($"Total Memory Saved: {FormatBytes(TotalMemorySaved)}");
        sb.AppendLine();
        sb.AppendLine("Layer Profiles:");
        sb.AppendLine("===============");
        foreach (var profile in LayerProfiles.OrderByDescending(p => p.TotalMemorySaved))
        {
            sb.AppendLine($"{profile.LayerId}:");
            sb.AppendLine($"  Checkpoints: {profile.CheckpointCount}");
            sb.AppendLine($"  Avg Checkpoint Time: {profile.AverageCheckpointTimeMs:F2}ms");
            sb.AppendLine($"  Recomputations: {profile.RecomputeCount}");
            sb.AppendLine($"  Avg Recompute Time: {profile.AverageRecomputeTimeMs:F2}ms");
            sb.AppendLine($"  Cache Hit Rate: {profile.CacheHitRate:P2}");
            sb.AppendLine($"  Memory Saved: {FormatBytes(profile.TotalMemorySaved)}");
        }

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
