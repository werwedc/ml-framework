namespace MachineLearning.Checkpointing;

using Microsoft.Extensions.Logging;

/// <summary>
/// Helper class for DDP-specific checkpoint operations
/// </summary>
public class DDPCheckpointHelper
{
    private readonly IDistributedModel _model;
    private readonly IDistributedCoordinator _coordinator;
    private readonly ILogger<DDPCheckpointHelper>? _logger;

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
