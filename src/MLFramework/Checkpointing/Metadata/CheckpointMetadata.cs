namespace MachineLearning.Checkpointing;

/// <summary>
/// Checkpoint metadata
/// </summary>
public class CheckpointMetadata
{
    /// <summary>
    /// Checkpoint version
    /// </summary>
    public string? Version { get; set; }

    /// <summary>
    /// Timestamp when the checkpoint was created
    /// </summary>
    public DateTime Timestamp { get; set; }

    /// <summary>
    /// Total number of processes in the distributed group
    /// </summary>
    public int WorldSize { get; set; }

    /// <summary>
    /// Rank of the process that created this checkpoint
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
