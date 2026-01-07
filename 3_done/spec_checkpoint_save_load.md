# Spec: Checkpoint Save/Load for Sharded Models

## Overview
Implement checkpoint saving and loading utilities for tensor-parallel models. This includes saving sharded model parameters (each rank saves its own shard) and loading them back correctly, with support for both distributed and centralized checkpoint formats.

## Context
For TP models, each rank holds a different shard of parameters. Checkpointing must:
1. Save each rank's parameters individually (distributed format)
2. Optionally gather all shards on rank 0 for single-file checkpoints
3. Support loading sharded checkpoints back onto correct ranks
4. Handle optimizer state (if applicable)

## Implementation Details

### 1. Checkpoint Metadata

```csharp
namespace MLFramework.Checkpoint;

public class TPCheckpointMetadata
{
    public string ModelName { get; set; } = "";
    public DateTime SavedAt { get; set; } = DateTime.UtcNow;
    public int TPWorldSize { get; set; }
    public int[] MeshShape { get; set; } = Array.Empty<int>();
    public int Version { get; set; } = 1;
    public Dictionary<string, object> AdditionalInfo { get; set; } = new();

    public void Serialize(BinaryWriter writer)
    {
        writer.Write(ModelName);
        writer.Write(SavedAt.ToBinary());
        writer.Write(TPWorldSize);
        writer.Write(MeshShape.Length);
        foreach (var dim in MeshShape)
        {
            writer.Write(dim);
        }
        writer.Write(Version);
        // Simplified: skip AdditionalInfo for now
    }

    public static TPCheckpointMetadata Deserialize(BinaryReader reader)
    {
        var metadata = new TPCheckpointMetadata
        {
            ModelName = reader.ReadString(),
            SavedAt = DateTime.FromBinary(reader.ReadInt64()),
            TPWorldSize = reader.ReadInt32(),
            Version = reader.ReadInt32()
        };

        int meshDims = reader.ReadInt32();
        metadata.MeshShape = new int[meshDims];
        for (int i = 0; i < meshDims; i++)
        {
            metadata.MeshShape[i] = reader.ReadInt32();
        }

        return metadata;
    }
}
```

### 2. Distributed Checkpoint (Each Rank Saves Shard)

```csharp
public class DistributedTPCheckpoint
{
    private readonly string _checkpointDir;
    private readonly TPCheckpointMetadata _metadata;

    public DistributedTPCheckpoint(string checkpointDir)
    {
        _checkpointDir = checkpointDir;
        Directory.CreateDirectory(_checkpointDir);

        var metadataFile = Path.Combine(_checkpointDir, "metadata.bin");
        if (File.Exists(metadataFile))
        {
            using var reader = new BinaryReader(File.OpenRead(metadataFile));
            _metadata = TPCheckpointMetadata.Deserialize(reader);
        }
        else
        {
            _metadata = new TPCheckpointMetadata();
        }
    }

    /// <summary>
    /// Save model checkpoint (each rank saves its own shard)
    /// </summary>
    public async Task SaveAsync(Module model, int rank, string? customName = null)
    {
        var metadata = new TPCheckpointMetadata
        {
            ModelName = customName ?? model.GetType().Name,
            TPWorldSize = TensorParallel.GetWorldSize(),
            SavedAt = DateTime.UtcNow,
            MeshShape = Array.Empty<int>()
        };

        // Save metadata on rank 0
        if (rank == 0)
        {
            var metadataFile = Path.Combine(_checkpointDir, "metadata.bin");
            using var writer = new BinaryWriter(File.OpenWrite(metadataFile));
            metadata.Serialize(writer);
        }

        // Save each rank's shard
        var shardFile = Path.Combine(_checkpointDir, $"shard_rank{rank}.pt");
        await SaveShardAsync(model, shardFile);

        // Save optimizer state if available
        // await SaveOptimizerStateAsync(optimizer, checkpointDir, rank);
    }

    private async Task SaveShardAsync(Module model, string filePath)
    {
        var stateDict = new Dictionary<string, Tensor>();

        // Collect all parameters from model
        CollectParameters(model, stateDict);

        // Save tensors to file
        using var fileStream = File.Create(filePath);
        await SaveTensorsAsync(fileStream, stateDict);
    }

    private void CollectParameters(Module module, Dictionary<string, Tensor> stateDict)
    {
        foreach (var param in module.Parameters)
        {
            if (param.Data != null)
            {
                stateDict[param.Name] = param.Data;
            }
        }

        foreach (var submodule in module.Modules)
        {
            CollectParameters(submodule, stateDict);
        }
    }

    private async Task SaveTensorsAsync(Stream stream, Dictionary<string, Tensor> tensors)
    {
        // Simple binary format
        using var writer = new BinaryWriter(stream);
        writer.Write(tensors.Count);

        foreach (var kvp in tensors)
        {
            writer.Write(kvp.Key);
            await SaveTensorAsync(writer, kvp.Value);
        }
    }

    private async Task SaveTensorAsync(BinaryWriter writer, Tensor tensor)
    {
        // Write shape
        writer.Write(tensor.Shape.Length);
        foreach (var dim in tensor.Shape)
        {
            writer.Write(dim);
        }

        // Write data type
        writer.Write((int)tensor.DataType);

        // Write data
        // Assuming tensor has a method to get raw data
        byte[] data = await tensor.GetRawDataAsync();
        writer.Write(data.Length);
        writer.Write(data);
    }

    /// <summary>
    /// Load model checkpoint (each rank loads its own shard)
    /// </summary>
    public async Task LoadAsync(Module model, int rank)
    {
        var shardFile = Path.Combine(_checkpointDir, $"shard_rank{rank}.pt");

        if (!File.Exists(shardFile))
        {
            throw new FileNotFoundException($"Shard file not found: {shardFile}");
        }

        await LoadShardAsync(model, shardFile);
    }

    private async Task LoadShardAsync(Module model, string filePath)
    {
        using var fileStream = File.OpenRead(filePath);
        var stateDict = await LoadTensorsAsync(fileStream);

        // Load tensors into model
        LoadParameters(model, stateDict);
    }

    private async Task<Dictionary<string, Tensor>> LoadTensorsAsync(Stream stream)
    {
        var tensors = new Dictionary<string, Tensor>();
        using var reader = new BinaryReader(stream);

        int count = reader.ReadInt32();
        for (int i = 0; i < count; i++)
        {
            string key = reader.ReadString();
            var tensor = await LoadTensorAsync(reader);
            tensors[key] = tensor;
        }

        return tensors;
    }

    private async Task<Tensor> LoadTensorAsync(BinaryReader reader)
    {
        // Read shape
        int shapeLength = reader.ReadInt32();
        var shape = new int[shapeLength];
        for (int i = 0; i < shapeLength; i++)
        {
            shape[i] = reader.ReadInt32();
        }

        // Read data type
        var dataType = (TensorDataType)reader.ReadInt32();

        // Read data
        int dataLength = reader.ReadInt32();
        byte[] data = reader.ReadBytes(dataLength);

        // Create tensor from data
        return await Tensor.FromRawDataAsync(data, shape, dataType);
    }

    private void LoadParameters(Module module, Dictionary<string, Tensor> stateDict)
    {
        foreach (var param in module.Parameters)
        {
            if (stateDict.TryGetValue(param.Name, out var tensor))
            {
                param.Data.CopyFrom(tensor);
            }
        }

        foreach (var submodule in module.Modules)
        {
            LoadParameters(submodule, stateDict);
        }
    }

    /// <summary>
    /// Get checkpoint metadata
    /// </summary>
    public TPCheckpointMetadata? GetMetadata()
    {
        var metadataFile = Path.Combine(_checkpointDir, "metadata.bin");
        if (!File.Exists(metadataFile))
            return null;

        using var reader = new BinaryReader(File.OpenRead(metadataFile));
        return TPCheckpointMetadata.Deserialize(reader);
    }
}
```

### 3. Centralized Checkpoint (All Shards in One File on Rank 0)

```csharp
public class CentralizedTPCheckpoint
{
    private readonly string _checkpointFile;

    public CentralizedTPCheckpoint(string checkpointFile)
    {
        _checkpointFile = checkpointFile;
    }

    /// <summary>
    /// Save checkpoint with all shards gathered on rank 0
    /// </summary>
    public async Task SaveAsync(Module model, int rank, int worldSize, string? customName = null)
    {
        var metadata = new TPCheckpointMetadata
        {
            ModelName = customName ?? model.GetType().Name,
            TPWorldSize = worldSize,
            SavedAt = DateTime.UtcNow,
            MeshShape = Array.Empty<int>()
        };

        // Each rank serializes its shard
        using var memoryStream = new MemoryStream();
        var localData = memoryStream.ToArray();

        // Gather all shards on rank 0
        Tensor? gatheredData = null;
        if (rank == 0)
        {
            // Gather from all ranks would need proper implementation
            // For now, assume we have a way to gather
            // gatheredData = comm.AllGather(localData);
        }

        // Rank 0 saves the consolidated checkpoint
        if (rank == 0 && gatheredData != null)
        {
            using var fileStream = File.Create(_checkpointFile);
            using var writer = new BinaryWriter(fileStream);

            // Write metadata
            metadata.Serialize(writer);

            // Write number of shards
            writer.Write(worldSize);

            // Write each shard
            // for (int i = 0; i < worldSize; i++)
            // {
            //     WriteShardData(writer, shards[i]);
            // }
        }
    }

    /// <summary>
    /// Load checkpoint and scatter shards to appropriate ranks
    /// </summary>
    public async Task LoadAsync(Module model, int rank, int worldSize)
    {
        if (rank == 0)
        {
            // Read all shards
            using var fileStream = File.OpenRead(_checkpointFile);
            using var reader = new BinaryReader(fileStream);

            var metadata = TPCheckpointMetadata.Deserialize(reader);

            // Read shards
            int shardCount = reader.ReadInt32();
            var shards = new byte[shardCount][];

            for (int i = 0; i < shardCount; i++)
            {
                int shardSize = reader.ReadInt32();
                shards[i] = reader.ReadBytes(shardSize);
            }

            // Scatter to all ranks
            // comm.Scatter(shards, out myShard);
        }

        // Load local shard
        // LoadShard(model, myShard);
    }
}
```

### 4. Checkpoint Manager

```csharp
public static class TPCheckpointManager
{
    /// <summary>
    /// Save checkpoint (distributed format)
    /// </summary>
    public static async Task SaveDistributedAsync(
        Module model,
        string checkpointDir,
        string? checkpointName = null)
    {
        var rank = TensorParallel.GetRank();
        var checkpoint = new DistributedTPCheckpoint(checkpointDir);
        await checkpoint.SaveAsync(model, rank, checkpointName);

        await TensorParallel.GetCommunicator().BarrierAsync();
    }

    /// <summary>
    /// Load checkpoint (distributed format)
    /// </summary>
    public static async Task LoadDistributedAsync(
        Module model,
        string checkpointDir)
    {
        var rank = TensorParallel.GetRank();
        var checkpoint = new DistributedTPCheckpoint(checkpointDir);

        await checkpoint.LoadAsync(model, rank);

        await TensorParallel.GetCommunicator().BarrierAsync();
    }

    /// <summary>
    /// Save checkpoint (centralized format)
    /// </summary>
    public static async Task SaveCentralizedAsync(
        Module model,
        string checkpointFile,
        string? checkpointName = null)
    {
        var rank = TensorParallel.GetRank();
        var worldSize = TensorParallel.GetWorldSize();

        var checkpoint = new CentralizedTPCheckpoint(checkpointFile);
        await checkpoint.SaveAsync(model, rank, worldSize, checkpointName);

        await TensorParallel.GetCommunicator().BarrierAsync();
    }

    /// <summary>
    /// Load checkpoint (centralized format)
    /// </summary>
    public static async Task LoadCentralizedAsync(
        Module model,
        string checkpointFile)
    {
        var rank = TensorParallel.GetRank();
        var worldSize = TensorParallel.GetWorldSize();

        var checkpoint = new CentralizedTPCheckpoint(checkpointFile);
        await checkpoint.LoadAsync(model, rank, worldSize);

        await TensorParallel.GetCommunicator().BarrierAsync();
    }

    /// <summary>
    /// Check if checkpoint exists
    /// </summary>
    public static bool CheckpointExists(string checkpointPath, bool isDistributed = true)
    {
        if (isDistributed)
        {
            var metadataFile = Path.Combine(checkpointPath, "metadata.bin");
            return Directory.Exists(checkpointPath) && File.Exists(metadataFile);
        }
        else
        {
            return File.Exists(checkpointFile);
        }
    }

    /// <summary>
    /// Get checkpoint metadata
    /// </summary>
    public static TPCheckpointMetadata? GetMetadata(string checkpointDir)
    {
        var checkpoint = new DistributedTPCheckpoint(checkpointDir);
        return checkpoint.GetMetadata();
    }

    /// <summary>
    /// List available checkpoints in a directory
    /// </summary>
    public static List<string> ListCheckpoints(string checkpointDir)
    {
        var checkpoints = new List<string>();

        if (!Directory.Exists(checkpointDir))
            return checkpoints;

        foreach (var subDir in Directory.GetDirectories(checkpointDir))
        {
            var metadataFile = Path.Combine(subDir, "metadata.bin");
            if (File.Exists(metadataFile))
            {
                checkpoints.Add(Path.GetFileName(subDir));
            }
        }

        return checkpoints;
    }
}
```

### 5. Tensor Data Type Enum

```csharp
public enum TensorDataType
{
    Float32,
    Float64,
    Int32,
    Int64,
    Float16
}
```

## Files to Create

### Source Files
- `src/MLFramework/Checkpoint/TPCheckpointMetadata.cs`
- `src/MLFramework/Checkpoint/DistributedTPCheckpoint.cs`
- `src/MLFramework/Checkpoint/CentralizedTPCheckpoint.cs`
- `src/MLFramework/Checkpoint/TPCheckpointManager.cs`
- `src/MLFramework/Checkpoint/TensorDataType.cs`

### Test Files
- `tests/MLFramework.Tests/Checkpoint/DistributedTPCheckpointTests.cs`
- `tests/MLFramework.Tests/Checkpoint/TPCheckpointManagerTests.cs`

## Test Requirements

1. **Distributed Checkpoint Tests**
   - Test saving checkpoint from each rank
   - Test loading checkpoint onto correct rank
   - Test metadata is saved correctly
   - Test checkpoint directory structure

2. **Centralized Checkpoint Tests**
   - Test saving all shards to single file (on rank 0)
   - Test loading and scattering shards
   - Test file format is correct

3. **Checkpoint Manager Tests**
   - Test SaveDistributedAsync and LoadDistributedAsync
   - Test SaveCentralizedAsync and LoadCentralizedAsync
   - Test CheckpointExists works correctly
   - Test ListCheckpoints returns correct list

4. **Model Compatibility Tests**
   - Test checkpointing TP models
   - Test loading checkpoint onto same architecture
   - Test parameter shapes match after load

5. **Edge Cases**
   - Test with non-existent checkpoints
   - Test with corrupted checkpoint files
   - Test with different world sizes

## Dependencies
- `Tensor` class with serialization support
- `Module` base class
- `TensorParallel` context manager
- System.IO for file operations

## Success Criteria
- [ ] Distributed checkpoint saves each rank's shard correctly
- [ ] Distributed checkpoint loads onto correct rank
- [ ] Metadata is saved and loaded correctly
- [ ] Centralized checkpoint gathers all shards on rank 0
- [ ] CheckpointManager provides convenient API
- [ ] CheckpointExists and ListCheckpoints work correctly
- [ ] Checkpointed models can be loaded and produce same outputs
- [ ] Unit tests pass for all scenarios

## Estimated Time
45-60 minutes
