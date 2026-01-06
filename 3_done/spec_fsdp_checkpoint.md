# Spec: FSDP Checkpointing

## Overview
Implement distributed checkpointing support for saving and loading sharded model states.

## Requirements

### 1. FSDPCheckpointConfig Class
Define checkpointing configuration:

```csharp
public class FSDPCheckpointConfig
{
    /// <summary>Checkpoint directory</summary>
    public string CheckpointDir { get; set; }

    /// <summary>Checkpoint file prefix</summary>
    public string FilePrefix { get; set; } = "fsdp_checkpoint";

    /// <summary>Whether to save optimizer states</summary>
    public bool SaveOptimizerStates { get; set; } = true;

    /// <summary>Whether to save training state (epoch, step)</summary>
    public bool SaveTrainingState { get; set; } = true;

    /// <summary>Whether to use async checkpointing</summary>
    public bool AsyncCheckpoint { get; set; } = false;

    /// <summary>Checkpoint format (binary, json, torch)</summary>
    public string CheckpointFormat { get; set; } = "binary";

    /// <summary>Maximum number of checkpoints to keep</summary>
    public int MaxCheckpoints { get; set; } = 5;

    /// <summary>Validate configuration</summary>
    public void Validate()
    {
        if (string.IsNullOrEmpty(CheckpointDir))
            throw new ArgumentException("CheckpointDir cannot be empty", nameof(CheckpointDir));

        if (MaxCheckpoints < 1)
            throw new ArgumentException("MaxCheckpoints must be at least 1", nameof(MaxCheckpoints));
    }
}
```

### 2. FSDPCheckpointMetadata Class
Define checkpoint metadata:

```csharp
public class FSDPCheckpointMetadata
{
    /// <summary>Checkpoint version</summary>
    public int Version { get; set; } = 1;

    /// <summary>World size when checkpoint was created</summary>
    public int WorldSize { get; set; }

    /// <summary>Sharding strategy used</summary>
    public string ShardingStrategy { get; set; }

    /// <summary>Number of parameters</summary>
    public int NumParameters { get; set; }

    /// <summary>Timestamp of checkpoint</summary>
    public DateTime Timestamp { get; set; }

    /// <summary>Training epoch</summary>
    public int Epoch { get; set; }

    /// <summary>Training step</summary>
    public int Step { get; set; }

    /// <summary>Loss value</summary>
    public float Loss { get; set; }

    /// <summary>Mixed precision enabled</summary>
    public bool MixedPrecision { get; set; }

    /// <summary>CPU offloading enabled</summary>
    public bool CpuOffloading { get; set; }

    /// <summary>Parameter shapes</summary>
    public Dictionary<string, long[]> ParameterShapes { get; set; }

    public FSDPCheckpointMetadata()
    {
        ParameterShapes = new Dictionary<string, long[]>();
    }
}
```

### 3. FSDPCheckpoint Class
Create a class to manage checkpoint operations:

```csharp
public class FSDPCheckpoint : IDisposable
{
    private readonly FSDP _fsdp;
    private readonly IProcessGroup _processGroup;
    private readonly FSDPCheckpointConfig _config;

    /// <summary>
    /// Initialize checkpoint manager for FSDP.
    /// </summary>
    /// <param name="fsdp">FSDP wrapper instance</param>
    /// <param name="config">Checkpoint configuration</param>
    public FSDPCheckpoint(FSDP fsdp, FSDPCheckpointConfig config)
    {
        _fsdp = fsdp ?? throw new ArgumentNullException(nameof(fsdp));
        _config = config ?? new FSDPCheckpointConfig();

        _config.Validate();
        _processGroup = fsdp.ProcessGroup;

        // Create checkpoint directory if it doesn't exist
        if (!Directory.Exists(_config.CheckpointDir))
        {
            Directory.CreateDirectory(_config.CheckpointDir);
        }
    }

    /// <summary>
    /// Save a checkpoint of the current model state.
    /// </summary>
    /// <param name="epoch">Current training epoch</param>
    /// <param name="step">Current training step</param>
    /// <param name="loss">Current loss value</param>
    /// <returns>Checkpoint path</returns>
    public string SaveCheckpoint(int epoch, int step, float loss)
    {
        var checkpointName = $"{_config.FilePrefix}_epoch{epoch}_step{step}";
        var checkpointPath = Path.Combine(_config.CheckpointDir, checkpointName);

        if (_config.AsyncCheckpoint)
        {
            // Async checkpointing
            return SaveCheckpointAsync(epoch, step, loss, checkpointPath).GetAwaiter().GetResult();
        }
        else
        {
            // Synchronous checkpointing
            return SaveCheckpointSync(epoch, step, loss, checkpointPath);
        }
    }

    /// <summary>
    /// Save checkpoint synchronously.
    /// </summary>
    private string SaveCheckpointSync(int epoch, int step, float loss, string checkpointPath)
    {
        // Only rank 0 saves the checkpoint
        if (_processGroup.Rank != 0)
        {
            // Other ranks participate in gathering but don't write files
            GatherStatesToRank0();
            return checkpointPath;
        }

        // Gather all states to rank 0
        var gatheredStates = GatherStatesToRank0();

        // Create metadata
        var metadata = new FSDPCheckpointMetadata
        {
            WorldSize = _processGroup.WorldSize,
            ShardingStrategy = _fsdp.Config.ShardingStrategy.ToString(),
            NumParameters = gatheredStates.Parameters.Count,
            Timestamp = DateTime.UtcNow,
            Epoch = epoch,
            Step = step,
            Loss = loss,
            MixedPrecision = _fsdp.Config.MixedPrecision,
            CpuOffloading = _fsdp.Config.OffloadToCPU
        };

        // Save metadata
        var metadataPath = $"{checkpointPath}_metadata.json";
        File.WriteAllText(metadataPath, JsonSerializer.Serialize(metadata, new JsonSerializerOptions { WriteIndented = true }));

        // Save parameters
        var parametersPath = $"{checkpointPath}_parameters.bin";
        SaveParameters(gatheredStates.Parameters, parametersPath);

        // Save optimizer states if configured
        if (_config.SaveOptimizerStates)
        {
            var optimizerPath = $"{checkpointPath}_optimizer.bin";
            SaveOptimizerStates(gatheredStates.OptimizerStates, optimizerPath);
        }

        // Clean up old checkpoints
        CleanOldCheckpoints();

        return checkpointPath;
    }

    /// <summary>
    /// Save checkpoint asynchronously.
    /// </summary>
    private async Task<string> SaveCheckpointAsync(int epoch, int step, float loss, string checkpointPath)
    {
        return await Task.Run(() => SaveCheckpointSync(epoch, step, loss, checkpointPath));
    }

    /// <summary>
    /// Gather all states to rank 0.
    /// </summary>
    private GatheredStates GatherStatesToRank0()
    {
        var states = new GatheredStates
        {
            Parameters = new Dictionary<string, Tensor>(),
            OptimizerStates = new Dictionary<string, OptimizerState>()
        };

        // Gather all parameters to rank 0
        var shardingUnits = _fsdp.GetShardingUnits();
        foreach (var unit in shardingUnits.Values)
        {
            // Gather parameter from all ranks
            var fullParameter = GatherParameter(unit);
            if (_processGroup.Rank == 0 && fullParameter != null)
            {
                states.Parameters[unit.ParameterName] = fullParameter;
            }
        }

        // Gather optimizer states to rank 0
        if (_config.SaveOptimizerStates)
        {
            var optimizerStateManager = _fsdp.GetOptimizerStateManager();
            foreach (var paramName in optimizerStateManager.GetAllParameterNames())
            {
                var state = optimizerStateManager.GatherOptimizerState(paramName);
                if (_processGroup.Rank == 0 && state != null)
                {
                    states.OptimizerStates[paramName] = state;
                }
            }
        }

        return states;
    }

    /// <summary>
    /// Gather a parameter from all ranks.
    /// </summary>
    private Tensor GatherParameter(FSDPShardingUnit shardingUnit)
    {
        var allGatherOp = new AllGatherOperation(
            _processGroup,
            shardingUnit.Shape,
            shardingUnit.DataType,
            shardingUnit.State.ShardIndex
        );

        return allGatherOp.AllGather(shardingUnit.ShardedParameter!);
    }

    /// <summary>
    /// Save parameters to file.
    /// </summary>
    private void SaveParameters(Dictionary<string, Tensor> parameters, string filePath)
    {
        using var writer = new BinaryWriter(File.Open(filePath, FileMode.Create));

        // Write header
        writer.Write(0x46454450); // Magic number: "FSDP"
        writer.Write(parameters.Count);

        // Write each parameter
        foreach (var kvp in parameters)
        {
            var paramName = kvp.Key;
            var tensor = kvp.Value;

            // Write parameter name length and name
            var nameBytes = Encoding.UTF8.GetBytes(paramName);
            writer.Write(nameBytes.Length);
            writer.Write(nameBytes);

            // Write tensor shape
            writer.Write(tensor.Shape.Length);
            foreach (var dim in tensor.Shape)
            {
                writer.Write(dim);
            }

            // Write tensor data type
            writer.Write((int)tensor.DataType);

            // Write tensor data
            foreach (var val in tensor.Data)
            {
                writer.Write(val);
            }
        }
    }

    /// <summary>
    /// Save optimizer states to file.
    /// </summary>
    private void SaveOptimizerStates(Dictionary<string, OptimizerState> optimizerStates, string filePath)
    {
        using var writer = new BinaryWriter(File.Open(filePath, FileMode.Create));

        // Write header
        writer.Write(0x4F505453); // Magic number: "OPTS"
        writer.Write(optimizerStates.Count);

        // Write each optimizer state
        foreach (var kvp in optimizerStates)
        {
            var paramName = kvp.Key;
            var state = kvp.Value;

            // Write parameter name length and name
            var nameBytes = Encoding.UTF8.GetBytes(paramName);
            writer.Write(nameBytes.Length);
            writer.Write(nameBytes);

            // Write optimizer state type
            writer.Write((int)state.StateType);

            // Write optimizer state data
            if (state is AdamOptimizerState adamState)
            {
                // Write momentum buffer
                WriteTensor(writer, adamState.MomentumBuffer);

                // Write variance buffer
                WriteTensor(writer, adamState.VarianceBuffer);

                // Write step count
                writer.Write(adamState.StepCount);
            }
        }
    }

    /// <summary>
    /// Write a tensor to a binary writer.
    /// </summary>
    private void WriteTensor(BinaryWriter writer, Tensor tensor)
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
        foreach (var val in tensor.Data)
        {
            writer.Write(val);
        }
    }

    /// <summary>
    /// Load a checkpoint.
    /// </summary>
    /// <param name="checkpointPath">Checkpoint directory path</param>
    public void LoadCheckpoint(string checkpointPath)
    {
        // Load metadata
        var metadataPath = $"{checkpointPath}_metadata.json";
        var metadataJson = File.ReadAllText(metadataPath);
        var metadata = JsonSerializer.Deserialize<FSDPCheckpointMetadata>(metadataJson);

        // Load parameters
        var parametersPath = $"{checkpointPath}_parameters.bin";
        var parameters = LoadParameters(parametersPath);

        // Reshard and distribute to all ranks
        ReshardParameters(parameters, metadata.WorldSize);

        // Load optimizer states if configured
        if (_config.SaveOptimizerStates)
        {
            var optimizerPath = $"{checkpointPath}_optimizer.bin";
            var optimizerStates = LoadOptimizerStates(optimizerPath);

            // Reshard and distribute to all ranks
            ReshardOptimizerStates(optimizerStates, metadata.WorldSize);
        }
    }

    /// <summary>
    /// Load parameters from file.
    /// </summary>
    private Dictionary<string, Tensor> LoadParameters(string filePath)
    {
        var parameters = new Dictionary<string, Tensor>();

        using var reader = new BinaryReader(File.Open(filePath, FileMode.Open));

        // Read header
        var magic = reader.ReadInt32();
        if (magic != 0x46454450)
            throw new InvalidDataException("Invalid checkpoint file format");

        var numParameters = reader.ReadInt32();

        // Read each parameter
        for (int i = 0; i < numParameters; i++)
        {
            // Read parameter name
            var nameLength = reader.ReadInt32();
            var nameBytes = reader.ReadBytes(nameLength);
            var paramName = Encoding.UTF8.GetString(nameBytes);

            // Read shape
            var shapeLength = reader.ReadInt32();
            var shape = new long[shapeLength];
            for (int j = 0; j < shapeLength; j++)
            {
                shape[j] = reader.ReadInt64();
            }

            // Read data type
            var dataType = (TensorDataType)reader.ReadInt32();

            // Create tensor and read data
            var tensor = Tensor.Zeros(shape, dataType);
            for (int j = 0; j < tensor.Size; j++)
            {
                tensor.Data[j] = reader.ReadSingle();
            }

            parameters[paramName] = tensor;
        }

        return parameters;
    }

    /// <summary>
    /// Load optimizer states from file.
    /// </summary>
    private Dictionary<string, OptimizerState> LoadOptimizerStates(string filePath)
    {
        var optimizerStates = new Dictionary<string, OptimizerState>();

        using var reader = new BinaryReader(File.Open(filePath, FileMode.Open));

        // Read header
        var magic = reader.ReadInt32();
        if (magic != 0x4F505453)
            throw new InvalidDataException("Invalid optimizer state file format");

        var numStates = reader.ReadInt32();

        // Read each optimizer state
        for (int i = 0; i < numStates; i++)
        {
            // Read parameter name
            var nameLength = reader.ReadInt32();
            var nameBytes = reader.ReadBytes(nameLength);
            var paramName = Encoding.UTF8.GetString(nameBytes);

            // Read optimizer state type
            var stateType = (OptimizerStateType)reader.ReadInt32();

            // Read optimizer state data
            OptimizerState state = stateType switch
            {
                OptimizerStateType.Adam => ReadAdamState(reader),
                _ => throw new NotSupportedException($"Unsupported optimizer state type: {stateType}")
            };

            optimizerStates[paramName] = state;
        }

        return optimizerStates;
    }

    /// <summary>
    /// Read Adam optimizer state from binary reader.
    /// </summary>
    private AdamOptimizerState ReadAdamState(BinaryReader reader)
    {
        // Read momentum buffer
        var momentum = ReadTensor(reader);

        // Read variance buffer
        var variance = ReadTensor(reader);

        // Read step count
        var stepCount = reader.ReadInt32();

        var state = new AdamOptimizerState(momentum, 0, 1);
        state.MomentumBuffer = momentum;
        state.VarianceBuffer = variance;
        state.StepCount = stepCount;

        return state;
    }

    /// <summary>
    /// Read a tensor from a binary reader.
    /// </summary>
    private Tensor ReadTensor(BinaryReader reader)
    {
        // Read shape
        var shapeLength = reader.ReadInt32();
        var shape = new long[shapeLength];
        for (int i = 0; i < shapeLength; i++)
        {
            shape[i] = reader.ReadInt64();
        }

        // Read data type
        var dataType = (TensorDataType)reader.ReadInt32();

        // Create tensor and read data
        var tensor = Tensor.Zeros(shape, dataType);
        for (int i = 0; i < tensor.Size; i++)
        {
            tensor.Data[i] = reader.ReadSingle();
        }

        return tensor;
    }

    /// <summary>
    /// Reshard parameters to match current world size.
    /// </summary>
    private void ReshardParameters(Dictionary<string, Tensor> parameters, int checkpointWorldSize)
    {
        var shardingUnits = _fsdp.GetShardingUnits();
        var currentWorldSize = _processGroup.WorldSize;

        foreach (var kvp in shardingUnits)
        {
            var paramName = kvp.Key;
            var shardingUnit = kvp.Value;

            if (!parameters.TryGetValue(paramName, out var fullParam))
                continue;

            if (checkpointWorldSize == currentWorldSize)
            {
                // Same world size, just broadcast and extract shard
                _processGroup.Broadcast(fullParam, 0);
                ExtractShard(fullParam, shardingUnit);
            }
            else
            {
                // Elastic resharding: different world size
                ElasticReshard(fullParam, shardingUnit, checkpointWorldSize, currentWorldSize);
            }
        }
    }

    /// <summary>
    /// Extract a shard from a full parameter.
    /// </summary>
    private void ExtractShard(Tensor fullParam, FSDPShardingUnit shardingUnit)
    {
        var shardSize = shardingUnit.ShardedParameter!.Size;
        var shardOffset = shardingUnit.State.ShardIndex * shardSize;

        Array.Copy(fullParam.Data, shardOffset, shardingUnit.ShardedParameter.Data, 0, shardSize);
    }

    /// <summary>
    /// Elastic resharding for different world sizes.
    /// </summary>
    private void ElasticReshard(Tensor fullParam, FSDPShardingUnit shardingUnit, int oldWorldSize, int newWorldSize)
    {
        // Broadcast full parameter to all ranks
        _processGroup.Broadcast(fullParam, 0);

        // Calculate new shard distribution
        var totalSize = fullParam.Size;
        var newShardSize = (totalSize + newWorldSize - 1) / newWorldSize;
        var newShardOffset = shardingUnit.State.ShardIndex * newShardSize;

        // Resize sharded parameter if needed
        if (shardingUnit.ShardedParameter!.Size != newShardSize)
        {
            shardingUnit.ShardedParameter.Dispose();
            shardingUnit.ShardedParameter = Tensor.Zeros(new[] { newShardSize }, fullParam.DataType);
        }

        // Extract new shard
        var actualShardSize = Math.Min(newShardSize, totalSize - newShardOffset);
        Array.Copy(fullParam.Data, newShardOffset, shardingUnit.ShardedParameter.Data, 0, actualShardSize);
    }

    /// <summary>
    /// Reshard optimizer states to match current world size.
    /// </summary>
    private void ReshardOptimizerStates(Dictionary<string, OptimizerState> optimizerStates, int checkpointWorldSize)
    {
        var optimizerStateManager = _fsdp.GetOptimizerStateManager();

        foreach (var kvp in optimizerStates)
        {
            var paramName = kvp.Key;
            var fullState = kvp.Value;

            if (fullState is AdamOptimizerState adamState)
            {
                ReshardAdamState(paramName, adamState, optimizerStateManager, checkpointWorldSize);
            }
        }
    }

    /// <summary>
    /// Reshard Adam optimizer state.
    /// </summary>
    private void ReshardAdamState(string paramName, AdamOptimizerState fullState, FSDPOptimizerStateManager manager, int checkpointWorldSize)
    {
        // Broadcast full optimizer state to all ranks
        _processGroup.Broadcast(fullState.MomentumBuffer, 0);
        _processGroup.Broadcast(fullState.VarianceBuffer, 0);

        // Extract local shard
        var localState = manager.GetOptimizerState(paramName) as AdamOptimizerState;
        if (localState != null)
        {
            ExtractShard(fullState.MomentumBuffer, localState.MomentumBuffer);
            ExtractShard(fullState.VarianceBuffer, localState.VarianceBuffer);
            localState.StepCount = fullState.StepCount;
        }
    }

    /// <summary>
    /// Clean up old checkpoints.
    /// </summary>
    private void CleanOldCheckpoints()
    {
        var checkpointFiles = Directory.GetFiles(_config.CheckpointDir, $"{_config.FilePrefix}_*")
            .Where(f => File.GetCreationTime(f) < DateTime.UtcNow.AddDays(-1))
            .OrderByDescending(f => File.GetCreationTime(f))
            .Skip(_config.MaxCheckpoints);

        foreach (var file in checkpointFiles)
        {
            try
            {
                File.Delete(file);
            }
            catch (Exception)
            {
                // Ignore errors when deleting old checkpoints
            }
        }
    }

    /// <summary>
    /// Dispose of resources.
    /// </summary>
    public void Dispose()
    {
        // No resources to dispose
    }
}

public class GatheredStates
{
    public Dictionary<string, Tensor> Parameters { get; set; }
    public Dictionary<string, OptimizerState> OptimizerStates { get; set; }
}
```

## Directory Structure
- **File**: `src/MLFramework/Distributed/FSDP/FSDPCheckpoint.cs`
- **Namespace**: `MLFramework.Distributed.FSDP`

## Dependencies
- `MLFramework.Distributed.ProcessGroup`
- `MLFramework.Distributed.FSDP.FSDP`
- `MLFramework.Distributed.FSDP.FSDPShardingUnit`
- `MLFramework.Distributed.FSDP.FSDPOptimizerStateManager`
- `MLFramework.Distributed.FSDP.AllGatherOperation`
- `RitterFramework.Core.Tensor`
- `System.Text.Json`

## Implementation Notes
1. Only rank 0 writes checkpoint files
2. Support elastic resharding (different world sizes)
3. Save and load parameters, optimizer states, and training state
4. Support both synchronous and asynchronous checkpointing
5. Clean up old checkpoints to save space

## Testing Requirements
- Test saving checkpoints
- Test loading checkpoints
- Test elastic resharding (different world sizes)
- Test optimizer state saving/loading
- Test checkpoint metadata
- Test old checkpoint cleanup
- Test edge cases (invalid checkpoint, missing files)

## Estimated Time
60 minutes
