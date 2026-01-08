namespace MachineLearning.Checkpointing;

/// <summary>
/// Information about a checkpoint
/// </summary>
public class CheckpointInfo
{
    /// <summary>
    /// Checkpoint prefix/path
    /// </summary>
    public string Prefix { get; set; } = string.Empty;

    /// <summary>
    /// Checkpoint version
    /// </summary>
    public string Version { get; set; } = string.Empty;

    /// <summary>
    /// Timestamp of the checkpoint
    /// </summary>
    public DateTime Timestamp { get; set; }

    /// <summary>
    /// World size when checkpoint was saved
    /// </summary>
    public int WorldSize { get; set; }

    /// <summary>
    /// Size of the checkpoint in bytes
    /// </summary>
    public long SizeBytes { get; set; }

    /// <summary>
    /// Format of the checkpoint
    /// </summary>
    public string Format { get; set; } = string.Empty;
}
