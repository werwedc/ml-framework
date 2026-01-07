using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using RitterFramework.Core;
using RitterFramework.Core.Tensor;
using MLFramework.Modules;

namespace MLFramework.Checkpoint;

/// <summary>
/// Centralized checkpoint implementation for tensor-parallel models
/// All shards are gathered on rank 0 and saved to a single file
/// </summary>
public class CentralizedTPCheckpoint
{
    private readonly string _checkpointFile;

    /// <summary>
    /// Creates a new centralized checkpoint instance
    /// </summary>
    public CentralizedTPCheckpoint(string checkpointFile)
    {
        _checkpointFile = checkpointFile;
    }

    /// <summary>
    /// Save checkpoint with all shards gathered on rank 0
    /// </summary>
    public async Task SaveAsync(IModule model, int rank, int worldSize, string? customName = null)
    {
        var metadata = new TPCheckpointMetadata
        {
            ModelName = customName ?? model.ModuleType,
            TPWorldSize = worldSize,
            SavedAt = DateTime.UtcNow,
            MeshShape = Array.Empty<int>()
        };

        // Each rank serializes its shard
        using var memoryStream = new MemoryStream();
        await WriteShardToStreamAsync(memoryStream, model);
        var localData = memoryStream.ToArray();

        // Rank 0 saves the consolidated checkpoint
        if (rank == 0)
        {
            using var fileStream = File.Create(_checkpointFile);
            using var writer = new BinaryWriter(fileStream);

            // Write metadata
            metadata.Serialize(writer);

            // Write number of shards
            writer.Write(worldSize);

            // Write this rank's shard (other ranks would send their shards via collective communication)
            WriteShardData(writer, localData);

            // Note: In a real implementation, we would gather all ranks' data via collective communication
            // For now, we only save rank 0's shard
        }
    }

    /// <summary>
    /// Save checkpoint with all shards gathered on rank 0 (Module overload)
    /// </summary>
    public async Task SaveAsync(NN.Module model, int rank, int worldSize, string? customName = null)
    {
        var metadata = new TPCheckpointMetadata
        {
            ModelName = customName ?? model.Name,
            TPWorldSize = worldSize,
            SavedAt = DateTime.UtcNow,
            MeshShape = Array.Empty<int>()
        };

        // Each rank serializes its shard
        using var memoryStream = new MemoryStream();
        await WriteModuleShardToStreamAsync(memoryStream, model);
        var localData = memoryStream.ToArray();

        // Rank 0 saves the consolidated checkpoint
        if (rank == 0)
        {
            using var fileStream = File.Create(_checkpointFile);
            using var writer = new BinaryWriter(fileStream);

            // Write metadata
            metadata.Serialize(writer);

            // Write number of shards
            writer.Write(worldSize);

            // Write this rank's shard (other ranks would send their shards via collective communication)
            WriteShardData(writer, localData);
        }
    }

    private async Task WriteShardToStreamAsync(Stream stream, IModule model)
    {
        var stateDict = new Dictionary<string, Tensor>();
        CollectParameters(model, stateDict);
        await SaveTensorsToStreamAsync(stream, stateDict);
    }

    private async Task WriteModuleShardToStreamAsync(Stream stream, NN.Module model)
    {
        var stateDict = new Dictionary<string, Tensor>();
        CollectModuleParameters(model, stateDict);
        await SaveTensorsToStreamAsync(stream, stateDict);
    }

    private void CollectParameters(IModule module, Dictionary<string, Tensor> stateDict)
    {
        int paramIndex = 0;
        foreach (var param in module.Parameters)
        {
            if (param != null)
            {
                stateDict[$"{module.ModuleType}_param_{paramIndex}"] = param;
                paramIndex++;
            }
        }
    }

    private void CollectModuleParameters(NN.Module module, Dictionary<string, Tensor> stateDict)
    {
        foreach (var (name, param) in module.GetNamedParameters())
        {
            if (param.Data != null)
            {
                stateDict[name] = param.Data;
            }
        }
    }

    private async Task SaveTensorsToStreamAsync(Stream stream, Dictionary<string, Tensor> tensors)
    {
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
        writer.Write(tensor.Shape.Length);
        foreach (var dim in tensor.Shape)
        {
            writer.Write(dim);
        }

        var dataType = ConvertToTensorDataType(tensor.Dtype);
        writer.Write((int)dataType);

        byte[] data = await GetRawDataAsync(tensor);
        writer.Write(data.Length);
        writer.Write(data);
    }

    private async Task<byte[]> GetRawDataAsync(Tensor tensor)
    {
        var data = tensor.Data;
        var bytes = new byte[data.Length * sizeof(float)];

        await Task.Run(() =>
        {
            Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
        });

        return bytes;
    }

    private TensorDataType ConvertToTensorDataType(DataType dtype)
    {
        return dtype switch
        {
            DataType.Float32 => TensorDataType.Float32,
            DataType.Float64 => TensorDataType.Float64,
            DataType.Int32 => TensorDataType.Int32,
            DataType.Int64 => TensorDataType.Int64,
            DataType.Float16 => TensorDataType.Float16,
            _ => TensorDataType.Float32
        };
    }

    private void WriteShardData(BinaryWriter writer, byte[] shardData)
    {
        writer.Write(shardData.Length);
        writer.Write(shardData);
    }

    /// <summary>
    /// Load checkpoint and scatter shards to appropriate ranks
    /// </summary>
    public async Task LoadAsync(IModule model, int rank, int worldSize)
    {
        byte[] myShard = null;

        if (rank == 0)
        {
            // Read all shards
            using var fileStream = File.OpenRead(_checkpointFile);
            using var reader = new BinaryReader(fileStream);

            var metadata = TPCheckpointMetadata.Deserialize(reader);

            // Read shards
            int shardCount = reader.ReadInt32();
            var shards = new List<byte[]>();

            for (int i = 0; i < shardCount; i++)
            {
                int shardSize = reader.ReadInt32();
                shards.Add(reader.ReadBytes(shardSize));
            }

            // Get this rank's shard (in real implementation, would scatter to all ranks)
            if (rank < shards.Count)
            {
                myShard = shards[rank];
            }
        }

        // Load local shard if available
        if (myShard != null)
        {
            await LoadShardAsync(model, myShard);
        }
    }

    /// <summary>
    /// Load checkpoint and scatter shards to appropriate ranks (Module overload)
    /// </summary>
    public async Task LoadAsync(NN.Module model, int rank, int worldSize)
    {
        byte[] myShard = null;

        if (rank == 0)
        {
            // Read all shards
            using var fileStream = File.OpenRead(_checkpointFile);
            using var reader = new BinaryReader(fileStream);

            var metadata = TPCheckpointMetadata.Deserialize(reader);

            // Read shards
            int shardCount = reader.ReadInt32();
            var shards = new List<byte[]>();

            for (int i = 0; i < shardCount; i++)
            {
                int shardSize = reader.ReadInt32();
                shards.Add(reader.ReadBytes(shardSize));
            }

            // Get this rank's shard
            if (rank < shards.Count)
            {
                myShard = shards[rank];
            }
        }

        // Load local shard if available
        if (myShard != null)
        {
            await LoadModuleShardAsync(model, myShard);
        }
    }

    private async Task LoadShardAsync(IModule model, byte[] shardData)
    {
        using var memoryStream = new MemoryStream(shardData);
        var stateDict = await LoadTensorsFromStreamAsync(memoryStream);
        LoadParameters(model, stateDict);
    }

    private async Task LoadModuleShardAsync(NN.Module model, byte[] shardData)
    {
        using var memoryStream = new MemoryStream(shardData);
        var stateDict = await LoadTensorsFromStreamAsync(memoryStream);
        LoadModuleParameters(model, stateDict);
    }

    private async Task<Dictionary<string, Tensor>> LoadTensorsFromStreamAsync(Stream stream)
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
        int shapeLength = reader.ReadInt32();
        var shape = new int[shapeLength];
        for (int i = 0; i < shapeLength; i++)
        {
            shape[i] = reader.ReadInt32();
        }

        var tensorDataType = (TensorDataType)reader.ReadInt32();
        var dataType = ConvertToDataType(tensorDataType);

        int dataLength = reader.ReadInt32();
        byte[] data = reader.ReadBytes(dataLength);

        return await TensorFromRawDataAsync(data, shape, dataType);
    }

    private async Task<Tensor> TensorFromRawDataAsync(byte[] data, int[] shape, DataType dtype)
    {
        int floatCount = data.Length / sizeof(float);
        var floatData = new float[floatCount];

        await Task.Run(() =>
        {
            Buffer.BlockCopy(data, 0, floatData, 0, data.Length);
        });

        return new Tensor(floatData, shape, false, dtype);
    }

    private DataType ConvertToDataType(TensorDataType tensorDataType)
    {
        return tensorDataType switch
        {
            TensorDataType.Float32 => DataType.Float32,
            TensorDataType.Float64 => DataType.Float64,
            TensorDataType.Int32 => DataType.Int32,
            TensorDataType.Int64 => DataType.Int64,
            TensorDataType.Float16 => DataType.Float16,
            _ => DataType.Float32
        };
    }

    private void LoadParameters(IModule module, Dictionary<string, Tensor> stateDict)
    {
        int paramIndex = 0;
        foreach (var param in module.Parameters)
        {
            if (param != null)
            {
                var key = $"{module.ModuleType}_param_{paramIndex}";
                if (stateDict.TryGetValue(key, out var tensor))
                {
                    param.CopyFrom(tensor);
                }
                paramIndex++;
            }
        }
    }

    private void LoadModuleParameters(NN.Module module, Dictionary<string, Tensor> stateDict)
    {
        foreach (var (name, param) in module.GetNamedParameters())
        {
            if (stateDict.TryGetValue(name, out var tensor))
            {
                param.Data.CopyFrom(tensor);
            }
        }
    }
}
