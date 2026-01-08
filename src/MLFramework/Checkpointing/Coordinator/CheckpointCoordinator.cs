namespace MachineLearning.Checkpointing;

/// <summary>
/// Coordinator for distributed checkpoint save operations
/// </summary>
public class CheckpointCoordinator
{
    private readonly IDistributedCoordinator _coordinator;
    private readonly ICheckpointStorage _storage;
    private readonly int _currentRank;

    /// <summary>
    /// Create a new CheckpointCoordinator
    /// </summary>
    public CheckpointCoordinator(
        IDistributedCoordinator coordinator,
        ICheckpointStorage storage)
    {
        _coordinator = coordinator ?? throw new ArgumentNullException(nameof(coordinator));
        _storage = storage ?? throw new ArgumentNullException(nameof(storage));
        _currentRank = coordinator.Rank;
    }

    /// <summary>
    /// Coordinate a save operation across all ranks
    /// </summary>
    public async Task<CheckpointMetadata> CoordinateSaveAsync(
        string checkpointPrefix,
        Func<Task<ShardData>> saveShardAsync,
        CancellationToken cancellationToken = default)
    {
        // Phase 1: Each rank saves its shard
        var shardData = await saveShardAsync();

        // Save shard data to storage
        var shardPath = $"{checkpointPrefix}/shard_{_currentRank}.bin";
        await _storage.WriteAsync(shardPath, shardData.Data, cancellationToken);

        // Phase 2: Barrier to ensure all ranks have saved their shards
        await _coordinator.BarrierAsync(cancellationToken);

        // Phase 3: Rank 0 creates the metadata file
        CheckpointMetadata? metadata = null;
        if (_currentRank == 0)
        {
            metadata = await CreateMetadataAsync(
                checkpointPrefix,
                shardData.TensorInfo,
                cancellationToken);
        }

        // Phase 4: Barrier to ensure metadata is created
        await _coordinator.BarrierAsync(cancellationToken);

        // Phase 5: Non-rank 0 processes load the metadata
        if (_currentRank != 0)
        {
            var metadataPath = $"{checkpointPrefix}/metadata.json";
            var metadataBytes = await _storage.ReadAsync(metadataPath, cancellationToken);
            var metadataJson = System.Text.Encoding.UTF8.GetString(metadataBytes);
            metadata = MetadataSerializer.Deserialize(metadataJson);
        }
        else
        {
            // metadata is already created in phase 3
            metadata = metadata ?? throw new InvalidOperationException("Metadata should not be null");
        }

        return metadata ?? throw new InvalidOperationException("Metadata should not be null");
    }

    /// <summary>
    /// Create checkpoint metadata
    /// </summary>
    private async Task<CheckpointMetadata> CreateMetadataAsync(
        string checkpointPrefix,
        List<TensorMetadata> tensorInfo,
        CancellationToken cancellationToken)
    {
        var metadata = new CheckpointMetadata
        {
            Version = "1.0.0",
            Timestamp = DateTime.UtcNow,
            WorldSize = _coordinator.WorldSize,
            DdpRank = 0,
            Format = "multishard",
            ShardingStrategy = new ShardingMetadata
            {
                Strategy = "ddp",
                ShardCount = _coordinator.WorldSize,
                Precision = "fp32"
            },
            TensorInfos = new List<TensorMetadata>()
        };

        // Save metadata file
        var metadataJson = MetadataSerializer.Serialize(metadata);
        var metadataBytes = System.Text.Encoding.UTF8.GetBytes(metadataJson);
        var metadataPath = $"{checkpointPrefix}/metadata.json";
        await _storage.WriteAsync(metadataPath, metadataBytes, cancellationToken);

        return metadata;
    }
}
