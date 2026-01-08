namespace MachineLearning.Checkpointing;

/// <summary>
/// Handles saving checkpoints in single-file format
/// </summary>
public class SingleFileSaver
{
    private readonly ICheckpointFormat _format;
    private readonly ICheckpointStorage _storage;

    /// <summary>
    /// Create a new single-file checkpoint saver
    /// </summary>
    /// <param name="format">The checkpoint format to use</param>
    /// <param name="storage">The storage backend to use</param>
    public SingleFileSaver(
        ICheckpointFormat format,
        ICheckpointStorage storage)
    {
        _format = format ?? throw new ArgumentNullException(nameof(format));
        _storage = storage ?? throw new ArgumentNullException(nameof(storage));
    }

    /// <summary>
    /// Save all shards as a single checkpoint file
    /// </summary>
    /// <param name="checkpointPrefix">Prefix for the checkpoint file path</param>
    /// <param name="shards">List of shard data to save</param>
    /// <param name="metadata">Checkpoint metadata</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Path to the saved checkpoint file</returns>
    public async Task<string> SaveAsync(
        string checkpointPrefix,
        List<ShardData> shards,
        CheckpointMetadata metadata,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(checkpointPrefix))
            throw new ArgumentException("Checkpoint prefix cannot be empty", nameof(checkpointPrefix));

        if (shards == null)
            throw new ArgumentNullException(nameof(shards));

        if (metadata == null)
            throw new ArgumentNullException(nameof(metadata));

        cancellationToken.ThrowIfCancellationRequested();

        // Serialize all shards into a single file
        var data = await _format.SerializeAsync(shards, metadata, cancellationToken);

        // Write to storage
        var filePath = $"{checkpointPrefix}{_format.Extension}";
        await _storage.WriteAsync(filePath, data, cancellationToken);

        return filePath;
    }

    /// <summary>
    /// Save shards with custom file path
    /// </summary>
    /// <param name="filePath">Full file path for the checkpoint</param>
    /// <param name="shards">List of shard data to save</param>
    /// <param name="metadata">Checkpoint metadata</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Path to the saved checkpoint file</returns>
    public async Task<string> SaveToPathAsync(
        string filePath,
        List<ShardData> shards,
        CheckpointMetadata metadata,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentException("File path cannot be empty", nameof(filePath));

        if (shards == null)
            throw new ArgumentNullException(nameof(shards));

        if (metadata == null)
            throw new ArgumentNullException(nameof(metadata));

        cancellationToken.ThrowIfCancellationRequested();

        // Serialize all shards into a single file
        var data = await _format.SerializeAsync(shards, metadata, cancellationToken);

        // Write to storage
        await _storage.WriteAsync(filePath, data, cancellationToken);

        return filePath;
    }
}
