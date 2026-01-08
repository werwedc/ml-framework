namespace MachineLearning.Checkpointing;

/// <summary>
/// Result of a checkpoint load operation
/// </summary>
public class CheckpointLoadResult
{
    /// <summary>
    /// Checkpoint metadata
    /// </summary>
    public CheckpointMetadata? Metadata { get; set; }

    /// <summary>
    /// Model state dictionary
    /// </summary>
    public StateDict? ModelState { get; set; }

    /// <summary>
    /// Optimizer state dictionary
    /// </summary>
    public StateDict? OptimizerState { get; set; }

    /// <summary>
    /// Shards loaded from storage
    /// </summary>
    public List<ShardData> Shards { get; set; } = new();

    /// <summary>
    /// Source world size
    /// </summary>
    public int SourceWorldSize { get; set; }

    /// <summary>
    /// Target world size
    /// </summary>
    public int TargetWorldSize { get; set; }

    /// <summary>
    /// Whether the checkpoint was resharded
    /// </summary>
    public bool WasResharded { get; set; }

    /// <summary>
    /// Whether the load was successful
    /// </summary>
    public bool Success { get; set; }

    /// <summary>
    /// Error message if load failed
    /// </summary>
    public string? Error { get; set; }
}

/// <summary>
/// Loader for distributed checkpoint load operations
/// </summary>
public class CheckpointLoader
{
    private readonly IDistributedCoordinator _coordinator;
    private readonly ICheckpointStorage _storage;
    private readonly int _currentRank;

    /// <summary>
    /// Create a new CheckpointLoader
    /// </summary>
    public CheckpointLoader(
        IDistributedCoordinator coordinator,
        ICheckpointStorage storage)
    {
        _coordinator = coordinator ?? throw new ArgumentNullException(nameof(coordinator));
        _storage = storage ?? throw new ArgumentNullException(nameof(storage));
        _currentRank = coordinator.Rank;
    }

    /// <summary>
    /// Coordinate a load operation across all ranks
    /// </summary>
    public async Task<CheckpointLoadResult> CoordinateLoadAsync(
        string checkpointPrefix,
        int targetWorldSize,
        CancellationToken cancellationToken = default)
    {
        // Phase 1: Load metadata
        var metadataPath = $"{checkpointPrefix}/metadata.json";
        var metadataBytes = await _storage.ReadAsync(metadataPath, cancellationToken);
        var metadataJson = System.Text.Encoding.UTF8.GetString(metadataBytes);
        var metadata = MetadataSerializer.Deserialize(metadataJson);

        // Phase 2: Determine if resharding is needed
        var sourceWorldSize = metadata.WorldSize;
        bool needsResharding = sourceWorldSize != targetWorldSize;

        // Phase 3: Load appropriate shard(s)
        var shards = new List<ShardData>();

        if (needsResharding)
        {
            // Load all shards for resharding
            for (int i = 0; i < sourceWorldSize; i++)
            {
                var shardPath = $"{checkpointPrefix}/shard_{i}.bin";
                var shardBytes = await _storage.ReadAsync(shardPath, cancellationToken);
                shards.Add(new ShardData
                {
                    Rank = i,
                    Data = shardBytes,
                    TensorInfo = metadata.TensorInfos
                });
            }
        }
        else
        {
            // Load only the current rank's shard
            var shardPath = $"{checkpointPrefix}/shard_{_currentRank}.bin";
            var shardBytes = await _storage.ReadAsync(shardPath, cancellationToken);
            shards.Add(new ShardData
            {
                Rank = _currentRank,
                Data = shardBytes,
                TensorInfo = metadata.TensorInfos
            });
        }

        return new CheckpointLoadResult
        {
            Metadata = metadata,
            Shards = shards,
            Success = true
        };
    }
}
