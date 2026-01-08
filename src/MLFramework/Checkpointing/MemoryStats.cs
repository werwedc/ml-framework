namespace MLFramework.Checkpointing;

/// <summary>
/// Memory statistics for checkpoints
/// </summary>
public class MemoryStats
{
    /// <summary>
    /// Total memory used in bytes
    /// </summary>
    public long TotalMemoryUsed { get; set; }

    /// <summary>
    /// Peak memory used in bytes
    /// </summary>
    public long PeakMemoryUsed { get; set; }

    /// <summary>
    /// Number of checkpoints stored
    /// </summary>
    public int CheckpointCount { get; set; }
}
