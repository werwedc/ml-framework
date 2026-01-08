namespace MachineLearning.Checkpointing;

/// <summary>
/// Handles loading checkpoints in single-file format
/// </summary>
public class SingleFileLoader
{
    private readonly ICheckpointFormat _format;
    private readonly ICheckpointStorage _storage;

    /// <summary>
    /// Create a new single-file checkpoint loader
    /// </summary>
    /// <param name="format">The checkpoint format to use</param>
    /// <param name="storage">The storage backend to use</param>
    public SingleFileLoader(
        ICheckpointFormat format,
        ICheckpointStorage storage)
    {
        _format = format ?? throw new ArgumentNullException(nameof(format));
        _storage = storage ?? throw new ArgumentNullException(nameof(storage));
    }

    /// <summary>
    /// Load a checkpoint from a file path
    /// </summary>
    /// <param name="checkpointPath">Path to the checkpoint file</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Tuple of shard data list and checkpoint metadata</returns>
    public async Task<(List<ShardData>, CheckpointMetadata)> LoadAsync(
        string checkpointPath,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(checkpointPath))
            throw new ArgumentException("Checkpoint path cannot be empty", nameof(checkpointPath));

        cancellationToken.ThrowIfCancellationRequested();

        // Check if file exists
        var exists = await _storage.ExistsAsync(checkpointPath, cancellationToken);
        if (!exists)
        {
            throw new FileNotFoundException($"Checkpoint file not found: {checkpointPath}");
        }

        // Read from storage
        var data = await _storage.ReadAsync(checkpointPath, cancellationToken);

        // Deserialize
        return await _format.DeserializeAsync(data, cancellationToken);
    }

    /// <summary>
    /// Load a checkpoint with prefix
    /// </summary>
    /// <param name="checkpointPrefix">Prefix for the checkpoint file path</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Tuple of shard data list and checkpoint metadata</returns>
    public async Task<(List<ShardData>, CheckpointMetadata)> LoadByPrefixAsync(
        string checkpointPrefix,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(checkpointPrefix))
            throw new ArgumentException("Checkpoint prefix cannot be empty", nameof(checkpointPrefix));

        var checkpointPath = $"{checkpointPrefix}{_format.Extension}";
        return await LoadAsync(checkpointPath, cancellationToken);
    }

    /// <summary>
    /// Load only metadata from a checkpoint file
    /// </summary>
    /// <param name="checkpointPath">Path to the checkpoint file</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Checkpoint metadata</returns>
    public async Task<CheckpointMetadata> LoadMetadataAsync(
        string checkpointPath,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(checkpointPath))
            throw new ArgumentException("Checkpoint path cannot be empty", nameof(checkpointPath));

        cancellationToken.ThrowIfCancellationRequested();

        // Check if file exists
        var exists = await _storage.ExistsAsync(checkpointPath, cancellationToken);
        if (!exists)
        {
            throw new FileNotFoundException($"Checkpoint file not found: {checkpointPath}");
        }

        // Read from storage (only read header portion)
        var data = await _storage.ReadAsync(checkpointPath, cancellationToken);

        // Read magic number
        if (data.Length < 4)
        {
            throw new InvalidDataException("Checkpoint file is too small to be valid");
        }

        using var memoryStream = new MemoryStream(data);
        using var reader = new BinaryReader(memoryStream);

        // Verify magic number
        var magic = reader.ReadInt32();
        const int expectedMagic = 0x4D4C4350; // "MLCP" in hex
        if (magic != expectedMagic)
        {
            throw new InvalidDataException($"Invalid checkpoint file: magic number mismatch (expected 0x{expectedMagic:X8}, found 0x{magic:X8})");
        }

        // Read version
        var version = reader.ReadString();

        // Read metadata
        var metadataLength = reader.ReadInt32();
        var metadataBytes = reader.ReadBytes(metadataLength);
        var metadataJson = System.Text.Encoding.UTF8.GetString(metadataBytes);
        var metadata = MetadataSerializer.Deserialize(metadataJson);

        return metadata;
    }

    /// <summary>
    /// Check if a checkpoint exists at the given path
    /// </summary>
    /// <param name="checkpointPath">Path to the checkpoint file</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>True if the checkpoint exists, false otherwise</returns>
    public async Task<bool> ExistsAsync(
        string checkpointPath,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(checkpointPath))
            throw new ArgumentException("Checkpoint path cannot be empty", nameof(checkpointPath));

        return await _storage.ExistsAsync(checkpointPath, cancellationToken);
    }

    /// <summary>
    /// Get metadata about a checkpoint file
    /// </summary>
    /// <param name="checkpointPath">Path to the checkpoint file</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Storage metadata for the checkpoint file</returns>
    public async Task<StorageMetadata> GetCheckpointMetadataAsync(
        string checkpointPath,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(checkpointPath))
            throw new ArgumentException("Checkpoint path cannot be empty", nameof(checkpointPath));

        return await _storage.GetMetadataAsync(checkpointPath, cancellationToken);
    }
}
