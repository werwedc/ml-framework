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
    /// Coordinate load operation across all ranks
    /// </summary>
    public async Task<CheckpointLoadResult> CoordinateLoadAsync(
        string checkpointPrefix,
        int targetWorldSize,
        CancellationToken cancellationToken = default)
    {
        // Phase 1: Load metadata (rank 0 only, then broadcast)
        CheckpointMetadata? metadata = null;
        if (_coordinator.Rank == 0)
        {
            metadata = await LoadMetadataAsync(checkpointPrefix, cancellationToken);
            MetadataValidator.ValidateOrThrow(metadata);
        }

        metadata = await _coordinator.BroadcastAsync(metadata!, cancellationToken);

        // Phase 2: Validate cross-topology compatibility
        await ValidateCrossTopologyAsync(metadata, targetWorldSize, cancellationToken);

        // Phase 3: Determine which shard to load for each rank
        var shardAssignments = ComputeShardAssignments(metadata, targetWorldSize);
        var myAssignment = shardAssignments[_coordinator.Rank];

        // Phase 4: Load assigned shards
        var loadedShards = new List<ShardData>();
        foreach (var shardRank in myAssignment)
        {
            var shardPath = $"{checkpointPrefix}_shard_{shardRank}.bin";
            var data = await _storage.ReadAsync(shardPath, cancellationToken);
            loadedShards.Add(new ShardData { Data = data });
        }

        return new CheckpointLoadResult
        {
            Metadata = metadata,
            Shards = loadedShards,
            Success = true
        };
    }

    /// <summary>
    /// Compute shard assignments for cross-topology loading
    /// </summary>
    private List<int>[] ComputeShardAssignments(CheckpointMetadata metadata, int targetWorldSize)
    {
        // Simple round-robin assignment
        var sourceShardCount = metadata.Sharding?.ShardCount ?? metadata.WorldSize;
        var assignments = new List<int>[targetWorldSize];

        for (int i = 0; i < sourceShardCount; i++)
        {
            var targetRank = i % targetWorldSize;
            assignments[targetRank] ??= new List<int>();
            assignments[targetRank].Add(i);
        }

        return assignments!;
    }

    /// <summary>
    /// Validate cross-topology compatibility
    /// </summary>
    private async Task ValidateCrossTopologyAsync(CheckpointMetadata metadata, int targetWorldSize, CancellationToken cancellationToken)
    {
        var sourceWorldSize = metadata.WorldSize;

        // Validate that the checkpoint can be loaded with the target world size
        if (targetWorldSize <= 0)
        {
            throw new InvalidOperationException($"Target world size must be positive, got {targetWorldSize}");
        }

        if (sourceWorldSize <= 0)
        {
            throw new InvalidOperationException($"Source world size must be positive, got {sourceWorldSize}");
        }

        // Warn about resharding if world sizes don't match
        if (sourceWorldSize != targetWorldSize)
        {
            // Log warning (in a real implementation, use proper logging)
            // Console.WriteLine($"Warning: Resharding from {sourceWorldSize} to {targetWorldSize} ranks");
        }
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
