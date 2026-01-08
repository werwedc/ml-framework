namespace MachineLearning.Checkpointing;

/// <summary>
/// Checkpoint format types
/// </summary>
public enum CheckpointFormat
{
    /// <summary>
    /// Single file format
    /// </summary>
    SingleFile,

    /// <summary>
    /// Multi-shard format (default)
    /// </summary>
    MultiShard,

    /// <summary>
    /// Consolidated format
    /// </summary>
    Consolidated
}

/// <summary>
/// Compression levels for checkpoints
/// </summary>
public enum CheckpointCompressionLevel
{
    /// <summary>
    /// No compression
    /// </summary>
    None,

    /// <summary>
    /// Fast compression
    /// </summary>
    Fast,

    /// <summary>
    /// Balanced compression
    /// </summary>
    Balanced,

    /// <summary>
    /// Maximum compression
    /// </summary>
    Maximum
}
