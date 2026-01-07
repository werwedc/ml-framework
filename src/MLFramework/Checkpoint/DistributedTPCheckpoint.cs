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
/// Distributed checkpoint implementation for tensor-parallel models
/// Each rank saves and loads its own shard independently
/// </summary>
public class DistributedTPCheckpoint
{
    private readonly string _checkpointDir;
    private readonly TPCheckpointMetadata _metadata;

    /// <summary>
    /// Creates a new distributed checkpoint instance
    /// </summary>
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
    public async Task SaveAsync(IModule model, int rank, string? customName = null)
    {
        var metadata = new TPCheckpointMetadata
        {
            ModelName = customName ?? model.ModuleType,
            TPWorldSize = 1, // Will be updated by TensorParallel context
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
    }

    /// <summary>
    /// Save model checkpoint (each rank saves its own shard) with Module (abstract class)
    /// </summary>
    public async Task SaveAsync(NN.Module model, int rank, string? customName = null)
    {
        var metadata = new TPCheckpointMetadata
        {
            ModelName = customName ?? model.Name,
            TPWorldSize = 1, // Will be updated by TensorParallel context
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
        await SaveShardModuleAsync(model, shardFile);
    }

    private async Task SaveShardAsync(IModule model, string filePath)
    {
        var stateDict = new Dictionary<string, Tensor>();

        // Collect all parameters from model
        CollectParameters(model, stateDict);

        // Save tensors to file
        using var fileStream = File.Create(filePath);
        await SaveTensorsAsync(fileStream, stateDict);
    }

    private async Task SaveShardModuleAsync(NN.Module model, string filePath)
    {
        var stateDict = new Dictionary<string, Tensor>();

        // Collect all parameters from model
        CollectModuleParameters(model, stateDict);

        // Save tensors to file
        using var fileStream = File.Create(filePath);
        await SaveTensorsAsync(fileStream, stateDict);
    }

    private void CollectParameters(IModule module, Dictionary<string, Tensor> stateDict)
    {
        // Use reflection to get Parameters property if it exists
        var parametersProperty = module.GetType().GetProperty("Parameters");

        if (parametersProperty != null && parametersProperty.PropertyType.IsGenericType)
        {
            var parameters = parametersProperty.GetValue(module);
            if (parameters != null)
            {
                foreach (var param in (System.Collections.IEnumerable)parameters)
                {
                    if (param != null && param is Tensor tensor)
                    {
                        stateDict[$"{module.ModuleType}_param_{stateDict.Count}"] = tensor;
                    }
                }
            }
        }
    }

    private void CollectModuleParameters(NN.Module module, Dictionary<string, Tensor> stateDict)
    {
        foreach (var (name, param) in module.GetNamedParameters())
        {
            if (param.Data != null)
            {
                // Create a Tensor from the data array
                stateDict[name] = new Tensor(param.Data, param.Shape, param.RequiresGrad, param.Dtype);
            }
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
        var dataType = ConvertToTensorDataType(tensor.Dtype);
        writer.Write((int)dataType);

        // Write data
        byte[] data = await GetRawDataAsync(tensor);
        writer.Write(data.Length);
        writer.Write(data);
    }

    private async Task<byte[]> GetRawDataAsync(Tensor tensor)
    {
        // For Float32 tensors, convert float[] to byte[]
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
            _ => TensorDataType.Float32 // Default to Float32
        };
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
            _ => DataType.Float32 // Default to Float32
        };
    }

    /// <summary>
    /// Load model checkpoint (each rank loads its own shard)
    /// </summary>
    public async Task LoadAsync(IModule model, int rank)
    {
        var shardFile = Path.Combine(_checkpointDir, $"shard_rank{rank}.pt");

        if (!File.Exists(shardFile))
        {
            throw new FileNotFoundException($"Shard file not found: {shardFile}");
        }

        await LoadShardAsync(model, shardFile);
    }

    /// <summary>
    /// Load model checkpoint (each rank loads its own shard) with Module (abstract class)
    /// </summary>
    public async Task LoadAsync(NN.Module model, int rank)
    {
        var shardFile = Path.Combine(_checkpointDir, $"shard_rank{rank}.pt");

        if (!File.Exists(shardFile))
        {
            throw new FileNotFoundException($"Shard file not found: {shardFile}");
        }

        await LoadShardModuleAsync(model, shardFile);
    }

    private async Task LoadShardAsync(IModule model, string filePath)
    {
        using var fileStream = File.OpenRead(filePath);
        var stateDict = await LoadTensorsAsync(fileStream);

        // Load tensors into model
        LoadParameters(model, stateDict);
    }

    private async Task LoadShardModuleAsync(NN.Module model, string filePath)
    {
        using var fileStream = File.OpenRead(filePath);
        var stateDict = await LoadTensorsAsync(fileStream);

        // Load tensors into model
        LoadModuleParameters(model, stateDict);
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
        var tensorDataType = (TensorDataType)reader.ReadInt32();
        var dataType = ConvertToDataType(tensorDataType);

        // Read data
        int dataLength = reader.ReadInt32();
        byte[] data = reader.ReadBytes(dataLength);

        // Create tensor from data
        return await TensorFromRawDataAsync(data, shape, dataType);
    }

    private async Task<Tensor> TensorFromRawDataAsync(byte[] data, int[] shape, DataType dtype)
    {
        // For Float32 tensors, convert byte[] to float[]
        int floatCount = data.Length / sizeof(float);
        var floatData = new float[floatCount];

        await Task.Run(() =>
        {
            Buffer.BlockCopy(data, 0, floatData, 0, data.Length);
        });

        return new Tensor(floatData, shape, false, dtype);
    }

    private void LoadParameters(IModule module, Dictionary<string, Tensor> stateDict)
    {
        // Use reflection to get Parameters property if it exists
        var parametersProperty = module.GetType().GetProperty("Parameters");

        if (parametersProperty != null && parametersProperty.PropertyType.IsGenericType)
        {
            var parameters = parametersProperty.GetValue(module);
            if (parameters != null)
            {
                int paramIndex = 0;
                foreach (var param in (System.Collections.IEnumerable)parameters)
                {
                    if (param != null && param is Tensor tensor)
                    {
                        var key = $"{module.ModuleType}_param_{paramIndex}";
                        if (stateDict.TryGetValue(key, out var stateTensor))
                        {
                            tensor.CopyFrom(stateTensor);
                        }
                        paramIndex++;
                    }
                }
            }
        }
    }

    private void LoadModuleParameters(NN.Module module, Dictionary<string, Tensor> stateDict)
    {
        foreach (var (name, param) in module.GetNamedParameters())
        {
            if (stateDict.TryGetValue(name, out var tensor))
            {
                param.CopyFrom(tensor);
            }
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
