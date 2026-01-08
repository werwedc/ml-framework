namespace MachineLearning.Checkpointing;

/// <summary>
/// Checkpoint metadata
/// </summary>
public class CheckpointMetadata
{
    /// <summary>
    /// Checkpoint version
    /// </summary>
    public string Version { get; set; } = "1.0.0";

    /// <summary>
    /// Timestamp when the checkpoint was created
    /// </summary>
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Number of GPUs that saved the checkpoint
    /// </summary>
    public int WorldSize { get; set; }

    /// <summary>
    /// Rank that created this metadata
    /// </summary>
    public int DdpRank { get; set; }

    /// <summary>
    /// Sharding metadata
    /// </summary>
    public ShardingMetadata? Sharding { get; set; }

    /// <summary>
    /// List of shard metadata
    /// </summary>
    public List<ShardMetadata>? Shards { get; set; }

    /// <summary>
    /// Training metadata
    /// </summary>
    public TrainingMetadata? Training { get; set; }

    /// <summary>
    /// Custom metadata fields
    /// </summary>
    public Dictionary<string, string>? CustomFields { get; set; }
}

/// <summary>
/// Training state metadata
/// </summary>
public class TrainingMetadata
{
    /// <summary>
    /// Current epoch
    /// </summary>
    public int Epoch { get; set; }

    /// <summary>
    /// Current training step
    /// </summary>
    public long Step { get; set; }

    /// <summary>
    /// Learning rate
    /// </summary>
    public float LearningRate { get; set; }

    /// <summary>
    /// Optimizer type
    /// </summary>
    public string? OptimizerType { get; set; }

    /// <summary>
    /// Optimizer state
    /// </summary>
    public Dictionary<string, object>? OptimizerState { get; set; }
}

/// <summary>
/// Shard metadata
/// </summary>
public class ShardMetadata
{
    /// <summary>
    /// Rank of the shard
    /// </summary>
    public int Rank { get; set; }

    /// <summary>
    /// Path to the shard file
    /// </summary>
    public string FilePath { get; set; } = string.Empty;

    /// <summary>
    /// Size of the shard file in bytes
    /// </summary>
    public long FileSize { get; set; }

    /// <summary>
    /// List of tensors in this shard
    /// </summary>
    public List<TensorMetadata>? Tensors { get; set; }

    /// <summary>
    /// Checksum of the shard file
    /// </summary>
    public string? Checksum { get; set; }
}
