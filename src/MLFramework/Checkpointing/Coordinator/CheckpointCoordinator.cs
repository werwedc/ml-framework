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
    /// Coordinate save operation across all ranks
    /// </summary>
    public async Task<CheckpointMetadata?> CoordinateSaveAsync(
        string checkpointPrefix,
        Func<Task<ShardData>> localSaveFunc,
        CancellationToken cancellationToken = default)
    {
        // Phase 1: Prepare - all ranks indicate readiness
        await _coordinator.BarrierAsync(cancellationToken);

        // Phase 2: Write local shard
        var localShard = await localSaveFunc();
        var shardPath = $"{checkpointPrefix}_shard_{_coordinator.Rank}.bin";
        await _storage.WriteAsync(shardPath, localShard.Data, cancellationToken);

        // Phase 3: Collect shard metadata from all ranks
        var shardMetadata = new ShardMetadata
        {
            Rank = _coordinator.Rank,
            FilePath = shardPath,
            FileSize = localShard.Data.Length,
            Tensors = localShard.TensorInfo,
            Checksum = ComputeChecksum(localShard.Data)
        };

        // Gather all shard metadata to rank 0
        var allShards = await _coordinator.GatherAsync(shardMetadata, cancellationToken);

        // Phase 4: Rank 0 writes metadata file
        if (_coordinator.Rank == 0)
        {
            var metadata = CreateCheckpointMetadata(allShards!);
            var metadataPath = $"{checkpointPrefix}.metadata.json";
            await _storage.WriteAsync(
                metadataPath,
                System.Text.Encoding.UTF8.GetBytes(MetadataSerializer.Serialize(metadata)),
                cancellationToken);
        }

        // Phase 5: Final barrier - ensure all ranks complete
        await _coordinator.BarrierAsync(cancellationToken);

        return _coordinator.Rank == 0
            ? await LoadMetadataAsync(checkpointPrefix, cancellationToken)
            : null;
    }

    /// <summary>
    /// Compute checksum for data integrity validation
    /// </summary>
    private string ComputeChecksum(byte[] data)
    {
        using var sha256 = System.Security.Cryptography.SHA256.Create();
        var hashBytes = sha256.ComputeHash(data);
        return Convert.ToHexString(hashBytes).ToLowerInvariant();
    }

    /// <summary>
    /// Create checkpoint metadata from gathered shard information
    /// </summary>
    private CheckpointMetadata CreateCheckpointMetadata(IList<ShardMetadata> allShards)
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
            Shards = allShards.ToList(),
            TensorInfos = allShards.SelectMany(s => s.Tensors ?? new List<TensorMetadata>()).ToList()
        };

        return metadata;
    }

    /// <summary>
    /// Load metadata from storage
    /// </summary>
    private async Task<CheckpointMetadata> LoadMetadataAsync(string checkpointPrefix, CancellationToken cancellationToken)
    {
        var metadataPath = $"{checkpointPrefix}.metadata.json";
        var metadataBytes = await _storage.ReadAsync(metadataPath, cancellationToken);
        var metadataJson = System.Text.Encoding.UTF8.GetString(metadataBytes);
        return MetadataSerializer.Deserialize(metadataJson);
    }
}
