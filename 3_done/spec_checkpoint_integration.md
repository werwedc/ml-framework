# Spec: Checkpoint Integration with Distributed Training

## Overview
Implement integration points between distributed checkpointing and distributed training strategies (FSDP, DDP, Tensor Parallelism) to enable seamless state management.

## Scope
- 30-45 minutes coding time
- Focus on integration interfaces
- Target: `src/MLFramework/Checkpointing/Integration/`

## Classes

### 1. IDistributedModel (Interface for Distributed Models)
```csharp
public interface IDistributedModel : IStateful
{
    DistributedStrategy Strategy { get; }
    int WorldSize { get; }
    int Rank { get; }

    /// <summary>
    /// Get the local shard of the model (for FSDP)
    /// </summary>
    StateDict GetLocalStateDict();

    /// <summary>
    /// Get the full state dict (for DDP, gathered)
    /// </summary>
    StateDict GetFullStateDict();

    /// <summary>
    /// Load local shard (for FSDP)
    /// </summary>
    void LoadLocalStateDict(StateDict state);

    /// <summary>
    /// Load full state dict (for DDP)
    /// </summary>
    void LoadFullStateDict(StateDict state);
}
```

### 2. DistributedStrategy (Strategy Enum)
```csharp
public enum DistributedStrategy
{
    DataParallel,      // DDP - full model on each rank
    FullyShardedDataParallel,  // FSDP - model sharded across ranks
    TensorParallel,    // TP - model split along tensor dimensions
    PipelineParallel,  // PP - model split across layers
    Hybrid             // Combination of strategies
}
```

### 3. FSDPCheckpointHelper (FSDP-Specific Helper)
```csharp
public class FSDPCheckpointHelper
{
    private readonly IDistributedModel _model;
    private readonly IDistributedCoordinator _coordinator;
    private readonly ILogger<FSDPCheckpointHelper> _logger;

    public FSDPCheckpointHelper(
        IDistributedModel model,
        IDistributedCoordinator coordinator,
        ILogger<FSDPCheckpointHelper>? logger = null)
    {
        if (model.Strategy != DistributedStrategy.FullyShardedDataParallel)
        {
            throw new ArgumentException("Model must use FSDP strategy", nameof(model));
        }

        _model = model;
        _coordinator = coordinator;
        _logger = logger;
    }

    /// <summary>
    /// Collect local shard for checkpointing
    /// </summary>
    public ShardData CollectLocalShard()
    {
        _logger?.LogDebug("Collecting local FSDP shard (rank: {Rank})", _coordinator.Rank);

        var localState = _model.GetLocalStateDict();

        var shardData = new ShardData
        {
            Data = SerializeState(localState),
            TensorInfo = CollectTensorMetadata(localState)
        };

        _logger?.LogDebug("Collected {TensorCount} tensors in local shard", shardData.TensorInfo.Count);

        return shardData;
    }

    /// <summary>
    /// Load shard into local FSDP model
    /// </summary>
    public void LoadLocalShard(ShardData shardData)
    {
        _logger?.LogDebug("Loading local FSDP shard (rank: {Rank})", _coordinator.Rank);

        var localState = DeserializeState(shardData.Data);
        _model.LoadLocalStateDict(localState);

        _logger?.LogDebug("Loaded {TensorCount} tensors into local shard", shardData.TensorInfo.Count);
    }

    /// <summary>
    /// Get metadata for FSDP checkpoint
    /// </summary>
    public ShardingMetadata GetShardingMetadata()
    {
        return new ShardingMetadata
        {
            Strategy = "fsdp",
            ShardCount = _coordinator.WorldSize,
            Precision = GetPrecision(),
            StrategySpecificInfo = new Dictionary<string, object>
            {
                ["zero_stage"] = GetZeroStage(),
                ["sharding_strategy"] = GetShardingStrategy()
            }
        };
    }

    private string GetPrecision()
    {
        // Determine model precision
        // Simplified: return fp16
        return "fp16";
    }

    private int GetZeroStage()
    {
        // Get FSDP ZeRO stage (1, 2, or 3)
        // Simplified: return 3
        return 3;
    }

    private string GetShardingStrategy()
    {
        // Get specific sharding strategy
        // Simplified: return full_shard
        return "full_shard";
    }

    private byte[] SerializeState(StateDict state)
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream);
        // TODO: Implement proper serialization
        return stream.ToArray();
    }

    private StateDict DeserializeState(byte[] data)
    {
        using var stream = new MemoryStream(data);
        using var reader = new BinaryReader(stream);
        // TODO: Implement proper deserialization
        return new StateDict();
    }

    private List<TensorMetadata> CollectTensorMetadata(StateDict state)
    {
        var metadata = new List<TensorMetadata>();
        foreach (var (name, tensor) in state)
        {
            metadata.Add(new TensorMetadata
            {
                Name = name,
                Shape = tensor.Shape,
                DataType = tensor.DataType,
                Offset = 0,  // Will be set during serialization
                Size = tensor.GetSizeInBytes()
            });
        }
        return metadata;
    }
}
```

### 4. DDPCheckpointHelper (DDP-Specific Helper)
```csharp
public class DDPCheckpointHelper
{
    private readonly IDistributedModel _model;
    private readonly IDistributedCoordinator _coordinator;
    private readonly ILogger<DDPCheckpointHelper> _logger;

    public DDPCheckpointHelper(
        IDistributedModel model,
        IDistributedCoordinator coordinator,
        ILogger<DDPCheckpointHelper>? logger = null)
    {
        if (model.Strategy != DistributedStrategy.DataParallel)
        {
            throw new ArgumentException("Model must use DDP strategy", nameof(model));
        }

        _model = model;
        _coordinator = coordinator;
        _logger = logger;
    }

    /// <summary>
    /// Collect full state (same on all ranks)
    /// </summary>
    public ShardData CollectFullState()
    {
        _logger?.LogDebug("Collecting full DDP state (rank: {Rank})", _coordinator.Rank);

        var fullState = _model.GetFullStateDict();

        var shardData = new ShardData
        {
            Data = SerializeState(fullState),
            TensorInfo = CollectTensorMetadata(fullState)
        };

        _logger?.LogDebug("Collected {TensorCount} tensors in full state", shardData.TensorInfo.Count);

        return shardData;
    }

    /// <summary>
    /// Load full state into DDP model
    /// </summary>
    public void LoadFullState(ShardData shardData)
    {
        _logger?.LogDebug("Loading full DDP state (rank: {Rank})", _coordinator.Rank);

        var fullState = DeserializeState(shardData.Data);
        _model.LoadFullStateDict(fullState);

        _logger?.LogDebug("Loaded {TensorCount} tensors into full state", shardData.TensorInfo.Count);
    }

    /// <summary>
    /// Get metadata for DDP checkpoint
    /// </summary>
    public ShardingMetadata GetShardingMetadata()
    {
        return new ShardingMetadata
        {
            Strategy = "ddp",
            ShardCount = _coordinator.WorldSize,
            Precision = GetPrecision(),
            StrategySpecificInfo = new Dictionary<string, object>
            {
                ["bucket_size"] = GetBucketSize()
            }
        };
    }

    private string GetPrecision()
    {
        return "fp16";
    }

    private long GetBucketSize()
    {
        // Get gradient bucket size
        return 25 * 1024 * 1024;  // 25 MB default
    }

    private byte[] SerializeState(StateDict state)
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream);
        return stream.ToArray();
    }

    private StateDict DeserializeState(byte[] data)
    {
        using var stream = new MemoryStream(data);
        using var reader = new BinaryReader(stream);
        return new StateDict();
    }

    private List<TensorMetadata> CollectTensorMetadata(StateDict state)
    {
        var metadata = new List<TensorMetadata>();
        foreach (var (name, tensor) in state)
        {
            metadata.Add(new TensorMetadata
            {
                Name = name,
                Shape = tensor.Shape,
                DataType = tensor.DataType,
                Offset = 0,
                Size = tensor.GetSizeInBytes()
            });
        }
        return metadata;
    }
}
```

### 5. CheckpointIntegrationHelperFactory (Factory Pattern)
```csharp
public static class CheckpointIntegrationHelperFactory
{
    public static object CreateHelper(
        IDistributedModel model,
        IDistributedCoordinator coordinator,
        ILogger? logger = null)
    {
        return model.Strategy switch
        {
            DistributedStrategy.FullyShardedDataParallel =>
                new FSDPCheckpointHelper(model, coordinator, logger),
            DistributedStrategy.DataParallel =>
                new DDPCheckpointHelper(model, coordinator, logger),
            DistributedStrategy.TensorParallel =>
                new TensorParallelCheckpointHelper(model, coordinator, logger),
            _ => throw new ArgumentException($"Unsupported distributed strategy: {model.Strategy}")
        };
    }
}
```

### 6. TensorParallelCheckpointHelper (TP-Specific Helper)
```csharp
public class TensorParallelCheckpointHelper
{
    private readonly IDistributedModel _model;
    private readonly IDistributedCoordinator _coordinator;
    private readonly ILogger<TensorParallelCheckpointHelper> _logger;

    public TensorParallelCheckpointHelper(
        IDistributedModel model,
        IDistributedCoordinator coordinator,
        ILogger<TensorParallelCheckpointHelper>? logger = null)
    {
        if (model.Strategy != DistributedStrategy.TensorParallel)
        {
            throw new ArgumentException("Model must use TP strategy", nameof(model));
        }

        _model = model;
        _coordinator = coordinator;
        _logger = logger;
    }

    /// <summary>
    /// Collect tensor parallel shard
    /// </summary>
    public ShardData CollectTPShard()
    {
        _logger?.LogDebug("Collecting TP shard (rank: {Rank})", _coordinator.Rank);

        var localState = _model.GetLocalStateDict();

        var shardData = new ShardData
        {
            Data = SerializeState(localState),
            TensorInfo = CollectTensorMetadata(localState)
        };

        return shardData;
    }

    /// <summary>
    /// Load tensor parallel shard
    /// </summary>
    public void LoadTPShard(ShardData shardData)
    {
        _logger?.LogDebug("Loading TP shard (rank: {Rank})", _coordinator.Rank);

        var localState = DeserializeState(shardData.Data);
        _model.LoadLocalStateDict(localState);
    }

    /// <summary>
    /// Get metadata for TP checkpoint
    /// </summary>
    public ShardingMetadata GetShardingMetadata()
    {
        return new ShardingMetadata
        {
            Strategy = "tensor_parallel",
            ShardCount = _coordinator.WorldSize,
            Precision = GetPrecision(),
            StrategySpecificInfo = new Dictionary<string, object>
            {
                ["tp_degree"] = GetTPDegree(),
                ["axis"] = GetTPAxis()
            }
        };
    }

    private string GetPrecision()
    {
        return "bf16";
    }

    private int GetTPDegree()
    {
        return _coordinator.WorldSize;
    }

    private int GetTPAxis()
    {
        // Axis to shard along (usually 0 for row parallel, 1 for column parallel)
        return 0;
    }

    private byte[] SerializeState(StateDict state)
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream);
        return stream.ToArray();
    }

    private StateDict DeserializeState(byte[] data)
    {
        using var stream = new MemoryStream(data);
        using var reader = new BinaryReader(stream);
        return new StateDict();
    }

    private List<TensorMetadata> CollectTensorMetadata(StateDict state)
    {
        var metadata = new List<TensorMetadata>();
        foreach (var (name, tensor) in state)
        {
            metadata.Add(new TensorMetadata
            {
                Name = name,
                Shape = tensor.Shape,
                DataType = tensor.DataType,
                Offset = 0,
                Size = tensor.GetSizeInBytes()
            });
        }
        return metadata;
    }
}
```

### 7. DistributedCheckpointExtension (Extension Methods)
```csharp
public static class DistributedCheckpointExtension
{
    /// <summary>
    /// Save checkpoint with automatic strategy detection
    /// </summary>
    public static async Task<string> SaveDistributedAsync(
        this DistributedCheckpoint checkpoint,
        IDistributedModel model,
        IStateful optimizer,
        SaveOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        // Create appropriate helper
        var helper = CheckpointIntegrationHelperFactory.CreateHelper(
            model,
            checkpoint.GetCoordinator());  // Need to add this getter

        // Use helper-specific save logic
        return model.Strategy switch
        {
            DistributedStrategy.FullyShardedDataParallel =>
                await SaveFSDPAsync(checkpoint, model, optimizer, helper as FSDPCheckpointHelper, options, cancellationToken),
            DistributedStrategy.DataParallel =>
                await SaveDDPAsync(checkpoint, model, optimizer, helper as DDPCheckpointHelper, options, cancellationToken),
            _ => throw new NotSupportedException($"Unsupported strategy: {model.Strategy}")
        };
    }

    private static async Task<string> SaveFSDPAsync(
        DistributedCheckpoint checkpoint,
        IDistributedModel model,
        IStateful optimizer,
        FSDPCheckpointHelper? helper,
        SaveOptions? options,
        CancellationToken cancellationToken)
    {
        // FSDP-specific save logic
        return await checkpoint.SaveAsync(model, optimizer, options, cancellationToken);
    }

    private static async Task<string> SaveDDPAsync(
        DistributedCheckpoint checkpoint,
        IDistributedModel model,
        IStateful optimizer,
        DDPCheckpointHelper? helper,
        SaveOptions? options,
        CancellationToken cancellationToken)
    {
        // DDP-specific save logic
        return await checkpoint.SaveAsync(model, optimizer, options, cancellationToken);
    }
}
```

## Usage Examples

### FSDP Checkpoint
```csharp
var model = new FSDPModel();  // Implements IDistributedModel
var optimizer = new AdamOptimizer();
var coordinator = new DistributedCoordinator();

var checkpoint = new DistributedCheckpoint(coordinator, storage);

// Use extension method for strategy-aware saving
await checkpoint.SaveDistributedAsync(model, optimizer, new SaveOptions
{
    CheckpointPrefix = "fsdp_checkpoint"
});
```

### DDP Checkpoint
```csharp
var model = new DDPModel();  // Implements IDistributedModel
var optimizer = new AdamOptimizer();
var coordinator = new DistributedCoordinator();

var checkpoint = new DistributedCheckpoint(coordinator, storage);

// Same API works for DDP
await checkpoint.SaveDistributedAsync(model, optimizer, new SaveOptions
{
    CheckpointPrefix = "ddp_checkpoint"
});
```

## Integration Points
- Used by: Training loops with distributed models
- Depends on: `IDistributedModel`, `IDistributedCoordinator`, `DistributedCheckpoint`

## Testing Requirements
- Test FSDP shard collection
- Test DDP full state collection
- Test TP shard collection
- Test helper factory
- Test extension methods
- Test with different strategies

## Success Criteria
- Clean integration with FSDP, DDP, TP
- Strategy-specific optimizations
- Consistent API across strategies
- Handles different sharding schemes
- Maintains state integrity
