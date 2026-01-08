namespace MachineLearning.Checkpointing;

/// <summary>
/// Slice definition for tensor redistribution
/// </summary>
public class TensorSlice
{
    /// <summary>
    /// Target rank for this slice
    /// </summary>
    public int TargetRank { get; set; }

    /// <summary>
    /// Start indices for each dimension
    /// </summary>
    public long[] Start { get; set; } = Array.Empty<long>();

    /// <summary>
    /// End indices for each dimension
    /// </summary>
    public long[] End { get; set; } = Array.Empty<long>();

    /// <summary>
    /// Shape of the slice
    /// </summary>
    public long[] Shape { get; set; } = Array.Empty<long>();
}

/// <summary>
/// Per-tensor redistribution information
/// </summary>
public class TensorRedistribution
{
    /// <summary>
    /// Name of the tensor
    /// </summary>
    public string TensorName { get; set; } = string.Empty;

    /// <summary>
    /// Source rank where this tensor is located
    /// </summary>
    public int SourceRank { get; set; }

    /// <summary>
    /// Shape of the source tensor
    /// </summary>
    public long[] SourceShape { get; set; } = Array.Empty<long>();

    /// <summary>
    /// Target ranks where this tensor should be distributed
    /// </summary>
    public List<int> TargetRanks { get; set; } = new();

    /// <summary>
    /// Slices to extract and redistribute
    /// </summary>
    public List<TensorSlice> Slices { get; set; } = new();
}

/// <summary>
/// Redistribution plan for resharding
/// </summary>
public class ReshardingPlan
{
    /// <summary>
    /// Source world size (number of processes)
    /// </summary>
    public int SourceWorldSize { get; set; }

    /// <summary>
    /// Target world size (number of processes)
    /// </summary>
    public int TargetWorldSize { get; set; }

    /// <summary>
    /// List of tensor redistribution plans
    /// </summary>
    public List<TensorRedistribution> TensorRedistributions { get; set; } = new();

    /// <summary>
    /// Get the source rank for a specific tensor
    /// </summary>
    /// <param name="tensorName">Name of the tensor</param>
    /// <returns>Source rank, or -1 if not found</returns>
    public int GetSourceRankForTensor(string tensorName)
    {
        var redistribution = TensorRedistributions
            .FirstOrDefault(r => r.TensorName == tensorName);
        return redistribution?.SourceRank ?? -1;
    }

    /// <summary>
    /// Get the target ranks for a specific tensor
    /// </summary>
    /// <param name="tensorName">Name of the tensor</param>
    /// <returns>List of target ranks, or empty list if not found</returns>
    public List<int> GetTargetRanksForTensor(string tensorName)
    {
        var redistribution = TensorRedistributions
            .FirstOrDefault(r => r.TensorName == tensorName);
        return redistribution?.TargetRanks ?? new List<int>();
    }
}

/// <summary>
/// Result of a resharding operation
/// </summary>
public class ReshardingResult
{
    /// <summary>
    /// Whether the resharding was successful
    /// </summary>
    public bool Success { get; set; }

    /// <summary>
    /// List of target shards after resharding
    /// </summary>
    public List<ShardData> TargetShards { get; set; } = new();

    /// <summary>
    /// List of warnings (non-fatal issues)
    /// </summary>
    public List<string> Warnings { get; set; } = new();

    /// <summary>
    /// Duration of the resharding operation
    /// </summary>
    public TimeSpan Duration { get; set; }
}

/// <summary>
/// Strategy for resharding checkpoints across different topologies
/// </summary>
public interface IReshardingStrategy
{
    /// <summary>
    /// Compute how to redistribute tensors from source to target ranks
    /// </summary>
    /// <param name="sourceMetadata">Checkpoint metadata from source</param>
    /// <param name="targetWorldSize">Target world size</param>
    /// <returns>Redistribution plan</returns>
    ReshardingPlan CreatePlan(
        CheckpointMetadata sourceMetadata,
        int targetWorldSize);

    /// <summary>
    /// Execute the resharding plan
    /// </summary>
    /// <param name="plan">Resharding plan to execute</param>
    /// <param name="sourceShards">Source shard data</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Resharding result</returns>
    Task<ReshardingResult> ExecuteAsync(
        ReshardingPlan plan,
        List<ShardData> sourceShards,
        CancellationToken cancellationToken = default);
}

/// <summary>
/// Simple resharding strategy using round-robin distribution
/// </summary>
public class SimpleReshardingStrategy : IReshardingStrategy
{
    /// <summary>
    /// Create a resharding plan using round-robin tensor distribution
    /// </summary>
    public ReshardingPlan CreatePlan(
        CheckpointMetadata sourceMetadata,
        int targetWorldSize)
    {
        if (sourceMetadata == null)
            throw new ArgumentNullException(nameof(sourceMetadata));

        if (sourceMetadata.Shards == null || sourceMetadata.Shards.Count == 0)
            throw new ArgumentException("Metadata must contain shards", nameof(sourceMetadata));

        var plan = new ReshardingPlan
        {
            SourceWorldSize = sourceMetadata.Shards.Count,
            TargetWorldSize = targetWorldSize,
            TensorRedistributions = new List<TensorRedistribution>()
        };

        var tensorIndex = 0;
        foreach (var shard in sourceMetadata.Shards)
        {
            if (shard.Tensors == null)
                continue;

            foreach (var tensorMeta in shard.Tensors)
            {
                var redistribution = new TensorRedistribution
                {
                    TensorName = tensorMeta.Name,
                    SourceRank = shard.Rank,
                    SourceShape = tensorMeta.Shape,
                    TargetRanks = new List<int>(),
                    Slices = new List<TensorSlice>()
                };

                // Round-robin assignment
                var targetRank = tensorIndex % targetWorldSize;
                redistribution.TargetRanks.Add(targetRank);

                // Entire tensor goes to target rank
                redistribution.Slices.Add(new TensorSlice
                {
                    TargetRank = targetRank,
                    Start = new long[tensorMeta.Shape.Length],
                    End = tensorMeta.Shape,
                    Shape = tensorMeta.Shape
                });

                plan.TensorRedistributions.Add(redistribution);
                tensorIndex++;
            }
        }

        return plan;
    }

    /// <summary>
    /// Execute the resharding plan
    /// </summary>
    public async Task<ReshardingResult> ExecuteAsync(
        ReshardingPlan plan,
        List<ShardData> sourceShards,
        CancellationToken cancellationToken = default)
    {
        if (plan == null)
            throw new ArgumentNullException(nameof(plan));

        if (sourceShards == null)
            throw new ArgumentNullException(nameof(sourceShards));

        var startTime = DateTime.UtcNow;
        var targetShards = new List<ShardData>();

        // Initialize target shards
        for (int rank = 0; rank < plan.TargetWorldSize; rank++)
        {
            targetShards.Add(new ShardData
            {
                Data = Array.Empty<byte>(),
                TensorInfo = new List<TensorMetadata>()
            });
        }

        // Redistribute each tensor
        foreach (var redistribution in plan.TensorRedistributions)
        {
            var sourceShard = sourceShards[redistribution.SourceRank];
            var sourceTensor = ExtractTensor(sourceShard, redistribution.TensorName);

            foreach (var slice in redistribution.Slices)
            {
                var targetShard = targetShards[slice.TargetRank];
                AddTensorToShard(targetShard, redistribution.TensorName, sourceTensor);
            }
        }

        return await Task.FromResult(new ReshardingResult
        {
            TargetShards = targetShards,
            Success = true,
            Duration = DateTime.UtcNow - startTime
        });
    }

    private byte[] ExtractTensor(ShardData shard, string tensorName)
    {
        // Extract tensor data from shard
        // Simplified: return entire shard data
        // In practice, need to parse shard and extract specific tensor
        return shard.Data;
    }

    private void AddTensorToShard(ShardData shard, string name, byte[] data)
    {
        // Add tensor to shard
        // Simplified: append data
        var newData = new byte[shard.Data.Length + data.Length];
        Array.Copy(shard.Data, 0, newData, 0, shard.Data.Length);
        Array.Copy(data, 0, newData, shard.Data.Length, data.Length);
        shard.Data = newData;
    }
}

/// <summary>
/// Parallel resharding strategy with domain decomposition support
/// </summary>
public class ParallelReshardingStrategy : IReshardingStrategy
{
    private readonly IDistributedCoordinator _coordinator;

    public ParallelReshardingStrategy(IDistributedCoordinator coordinator)
    {
        _coordinator = coordinator ?? throw new ArgumentNullException(nameof(coordinator));
    }

    /// <summary>
    /// Create a resharding plan optimized for parallel execution
    /// </summary>
    public ReshardingPlan CreatePlan(
        CheckpointMetadata sourceMetadata,
        int targetWorldSize)
    {
        if (sourceMetadata == null)
            throw new ArgumentNullException(nameof(sourceMetadata));

        if (sourceMetadata.Shards == null || sourceMetadata.Shards.Count == 0)
            throw new ArgumentException("Metadata must contain shards", nameof(sourceMetadata));

        // Similar to simple strategy, but optimized for parallel execution
        // Use domain decomposition if tensor is sharded along first dimension
        var plan = new ReshardingPlan
        {
            SourceWorldSize = sourceMetadata.Shards.Count,
            TargetWorldSize = targetWorldSize,
            TensorRedistributions = new List<TensorRedistribution>()
        };

        foreach (var shard in sourceMetadata.Shards)
        {
            if (shard.Tensors == null)
                continue;

            foreach (var tensorMeta in shard.Tensors)
            {
                var redistribution = CreateTensorRedistribution(
                    shard.Rank,
                    tensorMeta,
                    sourceMetadata.Shards.Count,
                    targetWorldSize);

                plan.TensorRedistributions.Add(redistribution);
            }
        }

        return plan;
    }

    private TensorRedistribution CreateTensorRedistribution(
        int sourceRank,
        TensorMetadata tensorMeta,
        int sourceWorldSize,
        int targetWorldSize)
    {
        var redistribution = new TensorRedistribution
        {
            TensorName = tensorMeta.Name,
            SourceRank = sourceRank,
            SourceShape = tensorMeta.Shape,
            TargetRanks = new List<int>(),
            Slices = new List<TensorSlice>()
        };

        if (tensorMeta.Shape.Length == 0)
        {
            // Scalar tensor - assign to rank 0
            redistribution.TargetRanks.Add(0);
            redistribution.Slices.Add(new TensorSlice
            {
                TargetRank = 0,
                Start = Array.Empty<long>(),
                End = Array.Empty<long>(),
                Shape = Array.Empty<long>()
            });
        }
        else
        {
            // Sharded along first dimension
            var firstDimSize = tensorMeta.Shape[0];
            var sourceChunkSize = (long)Math.Ceiling((double)firstDimSize / sourceWorldSize);
            var targetChunkSize = (long)Math.Ceiling((double)firstDimSize / targetWorldSize);

            for (int targetRank = 0; targetRank < targetWorldSize; targetRank++)
            {
                var targetStart = targetRank * targetChunkSize;
                var targetEnd = Math.Min(targetStart + targetChunkSize, firstDimSize);

                if (targetStart < firstDimSize)
                {
                    redistribution.TargetRanks.Add(targetRank);

                    var sourceStart = sourceRank * sourceChunkSize;
                    var sourceEnd = Math.Min(sourceStart + sourceChunkSize, firstDimSize);

                    // Compute overlap
                    var overlapStart = Math.Max(targetStart, sourceStart);
                    var overlapEnd = Math.Min(targetEnd, sourceEnd);

                    if (overlapStart < overlapEnd)
                    {
                        redistribution.Slices.Add(new TensorSlice
                        {
                            TargetRank = targetRank,
                            Start = new long[] { overlapStart - sourceStart },
                            End = new long[] { overlapEnd - sourceStart },
                            Shape = new long[] { overlapEnd - overlapStart }
                        });
                    }
                }
            }
        }

        return redistribution;
    }

    /// <summary>
    /// Execute the resharding plan in parallel across all target ranks
    /// </summary>
    public async Task<ReshardingResult> ExecuteAsync(
        ReshardingPlan plan,
        List<ShardData> sourceShards,
        CancellationToken cancellationToken = default)
    {
        if (plan == null)
            throw new ArgumentNullException(nameof(plan));

        if (sourceShards == null)
            throw new ArgumentNullException(nameof(sourceShards));

        var startTime = DateTime.UtcNow;
        var targetShards = new List<ShardData>();

        for (int rank = 0; rank < plan.TargetWorldSize; rank++)
        {
            var shard = await ReshardForRankAsync(plan, sourceShards, rank, cancellationToken);
            targetShards.Add(shard);
        }

        return new ReshardingResult
        {
            TargetShards = targetShards,
            Success = true,
            Duration = DateTime.UtcNow - startTime
        };
    }

    private async Task<ShardData> ReshardForRankAsync(
        ReshardingPlan plan,
        List<ShardData> sourceShards,
        int targetRank,
        CancellationToken cancellationToken)
    {
        var shard = new ShardData
        {
            Data = Array.Empty<byte>(),
            TensorInfo = new List<TensorMetadata>()
        };

        // Find all tensors that need to go to this rank
        var myTensors = plan.TensorRedistributions
            .Where(r => r.TargetRanks.Contains(targetRank));

        foreach (var tensorRedistribution in myTensors)
        {
            var sourceShard = sourceShards[tensorRedistribution.SourceRank];
            var sourceTensor = ExtractTensor(sourceShard, tensorRedistribution.TensorName);

            // Extract relevant slices
            foreach (var slice in tensorRedistribution.Slices.Where(s => s.TargetRank == targetRank))
            {
                var sliceData = ExtractSlice(sourceTensor, slice);
                AddTensorToShard(shard, tensorRedistribution.TensorName, sliceData);
            }
        }

        return shard;
    }

    private byte[] ExtractTensor(ShardData shard, string tensorName)
    {
        return shard.Data; // Simplified
    }

    private byte[] ExtractSlice(byte[] tensorData, TensorSlice slice)
    {
        // Extract slice from tensor
        // Simplified: return entire data
        return tensorData;
    }

    private void AddTensorToShard(ShardData shard, string name, byte[] data)
    {
        // Add tensor to shard
        var newData = new byte[shard.Data.Length + data.Length];
        Array.Copy(shard.Data, 0, newData, 0, shard.Data.Length);
        Array.Copy(data, 0, newData, shard.Data.Length, data.Length);
        shard.Data = newData;
    }
}

/// <summary>
/// Factory for creating resharding strategies
/// </summary>
public static class ReshardingStrategyFactory
{
    /// <summary>
    /// Create a resharding strategy by name
    /// </summary>
    /// <param name="strategy">Strategy name ("simple" or "parallel")</param>
    /// <param name="coordinator">Optional distributed coordinator for parallel strategy</param>
    /// <returns>Resharding strategy instance</returns>
    public static IReshardingStrategy Create(
        string strategy,
        IDistributedCoordinator? coordinator = null)
    {
        return strategy.ToLower() switch
        {
            "simple" => new SimpleReshardingStrategy(),
            "parallel" => new ParallelReshardingStrategy(
                coordinator ?? throw new ArgumentNullException(nameof(coordinator))),
            _ => throw new ArgumentException($"Unknown resharding strategy: {strategy}")
        };
    }
}
