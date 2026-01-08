# Spec: Distributed Checkpoint Coordinator

## Overview
Implement the coordination protocol for distributed checkpointing, including barriers, handshake, and atomic commit operations to ensure all ranks complete writes before checkpoint is valid.

## Scope
- 45-60 minutes coding time
- Focus on distributed synchronization
- Target: `src/MLFramework/Checkpointing/Coordination/`

## Classes

### 1. IDistributedCoordinator (Interface)
```csharp
public interface IDistributedCoordinator
{
    int Rank { get; }
    int WorldSize { get; }
    Task BarrierAsync(CancellationToken cancellationToken = default);
    Task BroadcastAsync<T>(T data, CancellationToken cancellationToken = default) where T : class;
    Task AllReduceAsync<T>(T data, Func<T, T, T> reducer, CancellationToken cancellationToken = default) where T : class;
}
```

### 2. CheckpointCoordinator (Main Coordination Logic)
```csharp
public class CheckpointCoordinator
{
    private readonly IDistributedCoordinator _coordinator;
    private readonly ICheckpointStorage _storage;

    public CheckpointCoordinator(
        IDistributedCoordinator coordinator,
        ICheckpointStorage storage)
    {
        _coordinator = coordinator;
        _storage = storage;
    }

    /// <summary>
    /// Coordinate save operation across all ranks
    /// </summary>
    public async Task<CheckpointMetadata> CoordinateSaveAsync(
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
            var metadata = CreateCheckpointMetadata(allShards);
            var metadataPath = $"{checkpointPrefix}.metadata.json";
            await _storage.WriteAsync(
                metadataPath,
                Encoding.UTF8.GetBytes(MetadataSerializer.Serialize(metadata)),
                cancellationToken);
        }

        // Phase 5: Final barrier - ensure all ranks complete
        await _coordinator.BarrierAsync(cancellationToken);

        return _coordinator.Rank == 0
            ? await LoadMetadataAsync(checkpointPrefix, cancellationToken)
            : null;
    }
}
```

### 3. CheckpointLoader (Load Coordination)
```csharp
public class CheckpointLoader
{
    private readonly IDistributedCoordinator _coordinator;
    private readonly ICheckpointStorage _storage;

    public CheckpointLoader(
        IDistributedCoordinator coordinator,
        ICheckpointStorage storage)
    {
        _coordinator = coordinator;
        _storage = storage;
    }

    /// <summary>
    /// Coordinate load operation across all ranks
    /// </summary>
    public async Task<LoadResult> CoordinateLoadAsync(
        string checkpointPrefix,
        int targetWorldSize,
        CancellationToken cancellationToken = default)
    {
        // Phase 1: Load metadata (rank 0 only, then broadcast)
        CheckpointMetadata metadata;
        if (_coordinator.Rank == 0)
        {
            metadata = await LoadMetadataAsync(checkpointPrefix, cancellationToken);
            await MetadataValidator.ValidateOrThrow(metadata);
        }

        metadata = await _coordinator.BroadcastAsync(metadata, cancellationToken);

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

        return new LoadResult
        {
            Metadata = metadata,
            Shards = loadedShards
        };
    }

    private List<int>[] ComputeShardAssignments(CheckpointMetadata metadata, int targetWorldSize)
    {
        // Simple round-robin assignment
        var sourceShardCount = metadata.Sharding.ShardCount;
        var assignments = new List<int>[targetWorldSize];

        for (int i = 0; i < sourceShardCount; i++)
        {
            var targetRank = i % targetWorldSize;
            assignments[targetRank] ??= new List<int>();
            assignments[targetRank].Add(i);
        }

        return assignments;
    }
}
```

### 4. ShardData (Data Transfer Object)
```csharp
public class ShardData
{
    public byte[] Data { get; set; } = Array.Empty<byte>();
    public List<TensorMetadata> TensorInfo { get; set; } = new();
}
```

### 5. LoadResult (Load Result)
```csharp
public class LoadResult
{
    public CheckpointMetadata Metadata { get; set; }
    public List<ShardData> Shards { get; set; }
}
```

### 6. AtomicCommitProtocol (Commit Mechanism)
```csharp
public class AtomicCommitProtocol
{
    private readonly ICheckpointStorage _storage;

    public AtomicCommitProtocol(ICheckpointStorage storage)
    {
        _storage = storage;
    }

    /// <summary>
    /// Implement two-phase commit: write to temp, then rename
    /// </summary>
    public async Task CommitAsync(
        string finalPath,
        byte[] data,
        CancellationToken cancellationToken = default)
    {
        var tempPath = $"{finalPath}.tmp";

        try
        {
            // Write to temporary file
            await _storage.WriteAsync(tempPath, data, cancellationToken);

            // Atomically move to final location
            if (_storage is LocalFileSystemStorage localStorage)
            {
                File.Move(
                    tempPath,
                    finalPath,
                    overwrite: true);
            }
            else
            {
                // For cloud storage, copy and delete
                await _storage.WriteAsync(finalPath, data, cancellationToken);
                await _storage.DeleteAsync(tempPath, cancellationToken);
            }
        }
        catch
        {
            // Cleanup on failure
            try
            {
                await _storage.DeleteAsync(tempPath, cancellationToken);
            }
            catch { }
            throw;
        }
    }
}
```

## Integration Points
- Depends on: `ICheckpointStorage`, `IDistributedCoordinator`
- Used by: `DistributedCheckpoint.SaveAsync()`, `DistributedCheckpoint.LoadAsync()`

## Error Handling
- Timeout handling for unresponsive ranks
- Retry logic with exponential backoff
- Cleanup of temporary files on failure
- Detailed error messages for coordination failures

## Testing Requirements
- Test barrier synchronization
- Test broadcast from rank 0
- Test gather operation
- Test cross-topology shard assignment
- Test atomic commit success and failure scenarios

## Success Criteria
- All ranks synchronize correctly during save/load
- Atomic commit prevents partial checkpoints
- Cross-topology loading assigns shards correctly
- Handles timeouts and unresponsive ranks gracefully
