namespace MachineLearning.Checkpointing;

/// <summary>
/// Represents a shard of model data for distributed checkpointing
/// </summary>
public class ShardData
{
    /// <summary>
    /// Rank of the shard
    /// </summary>
    public int Rank { get; set; }

    /// <summary>
    /// Serialized tensor data
    /// </summary>
    public byte[] Data { get; set; } = Array.Empty<byte>();

    /// <summary>
    /// Metadata for each tensor in the shard
    /// </summary>
    public List<TensorMetadata> TensorInfo { get; set; } = new();
}

/// <summary>
/// Metadata for a single tensor
/// </summary>
public class TensorMetadata
{
    /// <summary>
    /// Name of the tensor
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Shape of the tensor
    /// </summary>
    public long[] Shape { get; set; } = Array.Empty<long>();

    /// <summary>
    /// Data type of the tensor
    /// </summary>
    public TensorDataType DataType { get; set; }

    /// <summary>
    /// Offset in the serialized data
    /// </summary>
    public long Offset { get; set; }

    /// <summary>
    /// Size of the tensor in bytes
    /// </summary>
    public long Size { get; set; }
}

/// <summary>
/// Metadata for sharding strategy
/// </summary>
public class ShardingMetadata
{
    /// <summary>
    /// Sharding strategy type (e.g., "fsdp", "ddp", "tensor_parallel")
    /// </summary>
    public string Strategy { get; set; } = string.Empty;

    /// <summary>
    /// Number of shards
    /// </summary>
    public int ShardCount { get; set; }

    /// <summary>
    /// Precision/DataType used
    /// </summary>
    public string Precision { get; set; } = string.Empty;

    /// <summary>
    /// Strategy-specific information
    /// </summary>
    public Dictionary<string, object> StrategySpecificInfo { get; set; } = new();
}
