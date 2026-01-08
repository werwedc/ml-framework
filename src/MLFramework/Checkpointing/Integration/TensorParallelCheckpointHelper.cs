namespace MachineLearning.Checkpointing;

using Microsoft.Extensions.Logging;

/// <summary>
/// Helper class for Tensor Parallel-specific checkpoint operations
/// </summary>
public class TensorParallelCheckpointHelper
{
    private readonly IDistributedModel _model;
    private readonly IDistributedCoordinator _coordinator;
    private readonly ILogger<TensorParallelCheckpointHelper>? _logger;

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
