namespace MachineLearning.Checkpointing;

/// <summary>
/// Interface for checkpoint format implementations
/// </summary>
public interface ICheckpointFormat
{
    /// <summary>
    /// File extension for this checkpoint format
    /// </summary>
    string Extension { get; }

    /// <summary>
    /// Serialize shards and metadata into a byte array
    /// </summary>
    /// <param name="shards">List of shard data to serialize</param>
    /// <param name="metadata">Checkpoint metadata</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Serialized byte array</returns>
    Task<byte[]> SerializeAsync(
        List<ShardData> shards,
        CheckpointMetadata metadata,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Deserialize byte array into shards and metadata
    /// </summary>
    /// <param name="data">Byte array to deserialize</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Tuple of shard data list and checkpoint metadata</returns>
    Task<(List<ShardData>, CheckpointMetadata)> DeserializeAsync(
        byte[] data,
        CancellationToken cancellationToken = default);
}
