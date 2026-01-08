namespace MachineLearning.Checkpointing;

using System.Text;

/// <summary>
/// Loads shards in multi-shard checkpoint format
/// Each shard is loaded independently from its own file
/// </summary>
public class MultiShardLoader
{
    private readonly MultiShardCheckpointFormat _format;
    private readonly ICheckpointStorage _storage;

    /// <summary>
    /// Create a new multi-shard loader
    /// </summary>
    /// <param name="format">Multi-shard checkpoint format</param>
    /// <param name="storage">Checkpoint storage implementation</param>
    public MultiShardLoader(
        MultiShardCheckpointFormat format,
        ICheckpointStorage storage)
    {
        _format = format ?? throw new ArgumentNullException(nameof(format));
        _storage = storage ?? throw new ArgumentNullException(nameof(storage));
    }

    /// <summary>
    /// Load a specific shard
    /// </summary>
    /// <param name="checkpointPrefix">Checkpoint prefix</param>
    /// <param name="rank">Rank of the shard to load</param>
    /// <param name="shardMeta">Shard metadata</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Loaded shard data</returns>
    public async Task<ShardData> LoadShardAsync(
        string checkpointPrefix,
        int rank,
        ShardMetadata shardMeta,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(checkpointPrefix))
            throw new ArgumentException("Checkpoint prefix cannot be empty", nameof(checkpointPrefix));

        if (shardMeta == null)
            throw new ArgumentNullException(nameof(shardMeta));

        cancellationToken.ThrowIfCancellationRequested();

        // Read from storage
        var shardPath = $"{checkpointPrefix}_shard_{rank}.shard";
        var data = await _storage.ReadAsync(shardPath, cancellationToken);

        // Deserialize
        return await _format.DeserializeShardAsync(data, shardMeta, cancellationToken);
    }

    /// <summary>
    /// Load metadata file
    /// </summary>
    /// <param name="checkpointPrefix">Checkpoint prefix</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Loaded checkpoint metadata</returns>
    public async Task<CheckpointMetadata> LoadMetadataAsync(
        string checkpointPrefix,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(checkpointPrefix))
            throw new ArgumentException("Checkpoint prefix cannot be empty", nameof(checkpointPrefix));

        cancellationToken.ThrowIfCancellationRequested();

        // Read from storage
        var metadataPath = $"{checkpointPrefix}.metadata.json";
        var data = await _storage.ReadAsync(metadataPath, cancellationToken);

        // Deserialize
        var metadataJson = Encoding.UTF8.GetString(data);
        return MetadataSerializer.Deserialize(metadataJson);
    }

    /// <summary>
    /// Load all shards (for consolidation or validation)
    /// </summary>
    /// <param name="checkpointPrefix">Checkpoint prefix</param>
    /// <param name="metadata">Checkpoint metadata containing shard information</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>List of all loaded shards</returns>
    public async Task<List<ShardData>> LoadAllShardsAsync(
        string checkpointPrefix,
        CheckpointMetadata metadata,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(checkpointPrefix))
            throw new ArgumentException("Checkpoint prefix cannot be empty", nameof(checkpointPrefix));

        if (metadata == null)
            throw new ArgumentNullException(nameof(metadata));

        cancellationToken.ThrowIfCancellationRequested();

        var shards = new List<ShardData>();

        if (metadata.Shards != null)
        {
            foreach (var shardMeta in metadata.Shards)
            {
                var shard = await LoadShardAsync(
                    checkpointPrefix,
                    shardMeta.Rank,
                    shardMeta,
                    cancellationToken);
                shards.Add(shard);
            }
        }

        return shards;
    }
}
