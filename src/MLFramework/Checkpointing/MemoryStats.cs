namespace MLFramework.Checkpointing;

/// <summary>
/// Memory statistics for checkpoints
/// </summary>
public class MemoryStats
{
    /// <summary>
    /// Total memory currently used by checkpoints (in bytes)
    /// </summary>
    public long CurrentMemoryUsed { get; set; }

    /// <summary>
    /// Peak memory used since last clear (in bytes)
    /// </summary>
    public long PeakMemoryUsed { get; set; }

    /// <summary>
    /// Number of checkpoints currently stored
    /// </summary>
    public int CheckpointCount { get; set; }

    /// <summary>
    /// Average memory per checkpoint (in bytes)
    /// </summary>
    public long AverageMemoryPerCheckpoint { get; set; }

    /// <summary>
    /// Memory savings compared to storing all activations (in bytes)
    /// </summary>
    public long MemorySavings { get; set; }
}
