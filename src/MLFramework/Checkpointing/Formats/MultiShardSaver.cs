namespace MachineLearning.Checkpointing;

using System.Text;

/// <summary>
/// Saves shards in multi-shard checkpoint format
/// Each shard is written independently to its own file
/// </summary>
public class MultiShardSaver
{
    private readonly MultiShardCheckpointFormat _format;
    private readonly ICheckpointStorage _storage;

    /// <summary>
    /// Create a new multi-shard saver
    /// </summary>
    /// <param name="format">Multi-shard checkpoint format</param>
    /// <param name="storage">Checkpoint storage implementation</param>
    public MultiShardSaver(
        MultiShardCheckpointFormat format,
        ICheckpointStorage storage)
    {
        _format = format ?? throw new ArgumentNullException(nameof(format));
        _storage = storage ?? throw new ArgumentNullException(nameof(storage));
    }

    /// <summary>
    /// Save a single shard (called by each rank for its local shard)
    /// </summary>
    /// <param name="checkpointPrefix">Checkpoint prefix (e.g., "checkpoint_step_100")</param>
    /// <param name="rank">Rank of the shard</param>
    /// <param name="shard">Shard data to save</param>
    /// <param name="shardMeta">Shard metadata</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Path to the saved shard file</returns>
    public async Task<string> SaveShardAsync(
        string checkpointPrefix,
        int rank,
        ShardData shard,
        ShardMetadata shardMeta,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(checkpointPrefix))
            throw new ArgumentException("Checkpoint prefix cannot be empty", nameof(checkpointPrefix));

        if (shard == null)
            throw new ArgumentNullException(nameof(shard));

        if (shardMeta == null)
            throw new ArgumentNullException(nameof(shardMeta));

        cancellationToken.ThrowIfCancellationRequested();

        // Serialize shard
        var data = await _format.SerializeShardAsync(shard, shardMeta, cancellationToken);

        // Write to storage
        var shardPath = $"{checkpointPrefix}_shard_{rank}.shard";
        await _storage.WriteAsync(shardPath, data, cancellationToken);

        return shardPath;
    }

    /// <summary>
    /// Save metadata file (called by rank 0 only)
    /// </summary>
    /// <param name="checkpointPrefix">Checkpoint prefix</param>
    /// <param name="metadata">Checkpoint metadata</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Path to the saved metadata file</returns>
    public async Task<string> SaveMetadataAsync(
        string checkpointPrefix,
        CheckpointMetadata metadata,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(checkpointPrefix))
            throw new ArgumentException("Checkpoint prefix cannot be empty", nameof(checkpointPrefix));

        if (metadata == null)
            throw new ArgumentNullException(nameof(metadata));

        cancellationToken.ThrowIfCancellationRequested();

        var metadataJson = MetadataSerializer.Serialize(metadata);
        var metadataBytes = Encoding.UTF8.GetBytes(metadataJson);

        var metadataPath = $"{checkpointPrefix}.metadata.json";
        await _storage.WriteAsync(metadataPath, metadataBytes, cancellationToken);

        return metadataPath;
    }
}
