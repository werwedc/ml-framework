# Spec: Checkpoint Resharding

## Overview
Implement cross-topology resharding logic to load a checkpoint saved on N GPUs and distribute it across M GPUs (where N ≠ M). This enables flexible scaling without full training restarts.

## Scope
- 45-60 minutes coding time
- Focus on redistribution algorithms
- Target: `src/MLFramework/Checkpointing/Resharding/`

## Classes

### 1. IReshardingStrategy (Interface)
```csharp
public interface IReshardingStrategy
{
    /// <summary>
    /// Compute how to redistribute tensors from source to target ranks
    /// </summary>
    ReshardingPlan CreatePlan(
        CheckpointMetadata sourceMetadata,
        int targetWorldSize);

    /// <summary>
    /// Execute the resharding plan
    /// </summary>
    Task<ReshardingResult> ExecuteAsync(
        ReshardingPlan plan,
        List<ShardData> sourceShards,
        CancellationToken cancellationToken = default);
}
```

### 2. ReshardingPlan (Redistribution Plan)
```csharp
public class ReshardingPlan
{
    public int SourceWorldSize { get; set; }
    public int TargetWorldSize { get; set; }
    public List<TensorRedistribution> TensorRedistributions { get; set; }

    public int GetSourceRankForTensor(string tensorName)
    {
        var redistribution = TensorRedistributions
            .FirstOrDefault(r => r.TensorName == tensorName);
        return redistribution?.SourceRank ?? -1;
    }

    public List<int> GetTargetRanksForTensor(string tensorName)
    {
        var redistribution = TensorRedistributions
            .FirstOrDefault(r => r.TensorName == tensorName);
        return redistribution?.TargetRanks ?? new List<int>();
    }
}
```

### 3. TensorRedistribution (Per-Tensor Redistribution Info)
```csharp
public class TensorRedistribution
{
    public string TensorName { get; set; }
    public int SourceRank { get; set; }
    public long[] SourceShape { get; set; }
    public List<int> TargetRanks { get; set; }
    public List<TensorSlice> Slices { get; set; }
}
```

### 4. TensorSlice (Slice Definition)
```csharp
public class TensorSlice
{
    public int TargetRank { get; set; }
    public long[] Start { get; set; }
    public long[] End { get; set; }
    public long[] Shape { get; set; }
}
```

### 5. SimpleReshardingStrategy (Round-Robin Distribution)
```csharp
public class SimpleReshardingStrategy : IReshardingStrategy
{
    public ReshardingPlan CreatePlan(
        CheckpointMetadata sourceMetadata,
        int targetWorldSize)
    {
        var plan = new ReshardingPlan
        {
            SourceWorldSize = sourceMetadata.Sharding.ShardCount,
            TargetWorldSize = targetWorldSize,
            TensorRedistributions = new List<TensorRedistribution>()
        };

        var tensorIndex = 0;
        foreach (var shard in sourceMetadata.Shards)
        {
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

    public async Task<ReshardingResult> ExecuteAsync(
        ReshardingPlan plan,
        List<ShardData> sourceShards,
        CancellationToken cancellationToken = default)
    {
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

        return new ReshardingResult
        {
            TargetShards = targetShards,
            Success = true
        };
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
```

### 6. ParallelReshardingStrategy (Parallel Redistribution)
```csharp
public class ParallelReshardingStrategy : IReshardingStrategy
{
    private readonly IDistributedCoordinator _coordinator;

    public ParallelReshardingStrategy(IDistributedCoordinator coordinator)
    {
        _coordinator = coordinator;
    }

    public ReshardingPlan CreatePlan(
        CheckpointMetadata sourceMetadata,
        int targetWorldSize)
    {
        // Similar to simple strategy, but optimized for parallel execution
        // Use domain decomposition if tensor is sharded along first dimension
        var plan = new ReshardingPlan
        {
            SourceWorldSize = sourceMetadata.Sharding.ShardCount,
            TargetWorldSize = targetWorldSize,
            TensorRedistributions = new List<TensorRedistribution>()
        };

        foreach (var shard in sourceMetadata.Shards)
        {
            foreach (var tensorMeta in shard.Tensors)
            {
                var redistribution = CreateTensorRedistribution(
                    shard.Rank,
                    tensorMeta,
                    sourceMetadata.Sharding.ShardCount,
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

    public async Task<ReshardingResult> ExecuteAsync(
        ReshardingPlan plan,
        List<ShardData> sourceShards,
        CancellationToken cancellationToken = default)
    {
        // Execute resharding in parallel across all target ranks
        var targetShards = new List<ShardData>();

        for (int rank = 0; rank < plan.TargetWorldSize; rank++)
        {
            var shard = await ReshardForRankAsync(plan, sourceShards, rank, cancellationToken);
            targetShards.Add(shard);
        }

        return new ReshardingResult
        {
            TargetShards = targetShards,
            Success = true
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
```

### 7. ReshardingResult (Resharding Result)
```csharp
public class ReshardingResult
{
    public List<ShardData> TargetShards { get; set; }
    public bool Success { get; set; }
    public List<string> Warnings { get; set; } = new();
    public TimeSpan Duration { get; set; }
}
```

### 8. ReshardingStrategyFactory (Strategy Selection)
```csharp
public static class ReshardingStrategyFactory
{
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
```

## Integration Points
- Used by: `CheckpointLoader`, `DistributedCheckpoint.LoadAsync()`
- Depends on: `IDistributedCoordinator`, `CheckpointMetadata`, `ShardData`

## Testing Requirements
- Test simple round-robin resharding
- Test parallel resharding with sharded tensors
- Test resharding from fewer to more GPUs
- Test resharding from more to fewer GPUs
- Test resharding with various tensor shapes
- Test performance with different strategy choices

## Success Criteria
- Can redistribute from N to M GPUs (N ≠ M)
- Handles various tensor shapes correctly
- Supports multiple resharding strategies
- Maintains data integrity during redistribution
- Performance scales with number of GPUs
