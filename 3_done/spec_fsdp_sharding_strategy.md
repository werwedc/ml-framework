# Spec: FSDP Sharding Strategy

## Overview
Define the interface and implementations for different sharding strategies (Full, LayerWise, Hybrid).

## Requirements

### 1. IShardingStrategy Interface
Create an interface for sharding strategies:

```csharp
public interface IShardingStrategy
{
    /// <summary>
    /// Calculate how to shard parameters across devices.
    /// </summary>
    /// <param name="parameters">List of parameters to shard</param>
    /// <param name="worldSize">Number of devices</param>
    /// <returns>Mapping of parameter names to shard assignments</returns>
    ShardingPlan CalculateShardingPlan(List<ParameterInfo> parameters, int worldSize);

    /// <summary>
    /// Get the name of this sharding strategy.
    /// </summary>
    string Name { get; }
}
```

### 2. ShardingPlan Class
Define a class to represent a sharding plan:

```csharp
public class ShardingPlan
{
    /// <summary>Mapping of parameter names to their shard assignments</summary>
    public Dictionary<string, ShardAssignment> Assignments { get; set; }

    /// <summary>Total number of shards</summary>
    public int TotalShards { get; set; }

    /// <summary>Parameters that are always gathered (e.g., embeddings)</summary>
    public HashSet<string> AlwaysGathered { get; set; }

    public ShardingPlan()
    {
        Assignments = new Dictionary<string, ShardAssignment>();
        AlwaysGathered = new HashSet<string>();
    }
}

public class ShardAssignment
{
    /// <summary>Which rank owns this shard</summary>
    public int OwnerRank { get; set; }

    /// <index>Shard index across all ranks</summary>
    public int ShardIndex { get; set; }

    /// <summary>Start offset in the parameter tensor</summary>
    public long StartOffset { get; set; }

    /// <summary>Size of this shard</summary>
    public long ShardSize { get; set; }
}

public class ParameterInfo
{
    /// <summary>Parameter name</summary>
    public string Name { get; set; }

    /// <summary>Parameter shape</summary>
    public long[] Shape { get; set; }

    /// <summary>Parameter size in bytes</summary>
    public long SizeBytes { get; set; }

    /// <summary>Layer name this parameter belongs to</summary>
    public string LayerName { get; set; }

    /// <summary>Whether this parameter should always be gathered (e.g., embeddings)</summary>
    public bool AlwaysGather { get; set; }
}
```

### 3. FullShardingStrategy Class
Implement full sharding strategy (shard all parameters):

```csharp
public class FullShardingStrategy : IShardingStrategy
{
    public string Name => "Full";

    public ShardingPlan CalculateShardingPlan(List<ParameterInfo> parameters, int worldSize)
    {
        if (parameters == null || parameters.Count == 0)
            throw new ArgumentException("Parameters list cannot be empty", nameof(parameters));

        var plan = new ShardingPlan { TotalShards = worldSize };

        foreach (var param in parameters)
        {
            if (param.AlwaysGather)
            {
                plan.AlwaysGathered.Add(param.Name);
                continue;
            }

            // Calculate total size
            long totalSize = 1;
            foreach (var dim in param.Shape)
                totalSize *= dim;

            // Calculate shard size
            long shardSize = (totalSize + worldSize - 1) / worldSize;

            // Assign shards across all devices
            for (int rank = 0; rank < worldSize; rank++)
            {
                var startOffset = rank * shardSize;
                var actualShardSize = Math.Min(shardSize, totalSize - startOffset);

                if (actualShardSize > 0)
                {
                    var assignment = new ShardAssignment
                    {
                        OwnerRank = rank,
                        ShardIndex = rank,
                        StartOffset = startOffset,
                        ShardSize = actualShardSize
                    };

                    plan.Assignments[$"{param.Name}_rank{rank}"] = assignment;
                }
            }
        }

        return plan;
    }
}
```

### 4. LayerWiseShardingStrategy Class
Implement layer-wise sharding strategy (shard layers sequentially):

```csharp
public class LayerWiseShardingStrategy : IShardingStrategy
{
    public string Name => "LayerWise";

    public ShardingPlan CalculateShardingPlan(List<ParameterInfo> parameters, int worldSize)
    {
        if (parameters == null || parameters.Count == 0)
            throw new ArgumentException("Parameters list cannot be empty", nameof(parameters));

        var plan = new ShardingPlan { TotalShards = worldSize };

        // Group parameters by layer
        var layers = parameters
            .Where(p => !p.AlwaysGather)
            .GroupBy(p => p.LayerName)
            .OrderBy(g => g.Key)
            .ToList();

        // Assign each layer to a different rank
        foreach (var layerGroup in layers)
        {
            var layerRank = GetLayerRank(layerGroup.Key, layers.Count, worldSize);

            foreach (var param in layerGroup)
            {
                var assignment = new ShardAssignment
                {
                    OwnerRank = layerRank,
                    ShardIndex = layerRank,
                    StartOffset = 0,
                    ShardSize = param.SizeBytes / 4 // Assume float32
                };

                plan.Assignments[param.Name] = assignment;
            }
        }

        return plan;
    }

    private int GetLayerRank(string layerName, int totalLayers, int worldSize)
    {
        // Simple hash-based assignment
        var hash = layerName.GetHashCode();
        return Math.Abs(hash) % worldSize;
    }
}
```

### 5. HybridShardingStrategy Class
Implement hybrid sharding strategy (mix of full and layer-wise):

```csharp
public class HybridShardingStrategy : IShardingStrategy
{
    private readonly List<string> _fullShardedLayers;
    private readonly List<string> _layerWiseShardedLayers;

    public string Name => "Hybrid";

    public HybridShardingStrategy(List<string> fullShardedLayers, List<string> layerWiseShardedLayers)
    {
        _fullShardedLayers = fullShardedLayers ?? new List<string>();
        _layerWiseShardedLayers = layerWiseShardedLayers ?? new List<string>();
    }

    public ShardingPlan CalculateShardingPlan(List<ParameterInfo> parameters, int worldSize)
    {
        if (parameters == null || parameters.Count == 0)
            throw new ArgumentException("Parameters list cannot be empty", nameof(parameters));

        var plan = new ShardingPlan { TotalShards = worldSize };

        // Separate parameters by strategy
        var fullShardedParams = new List<ParameterInfo>();
        var layerWiseParams = new List<ParameterInfo>();

        foreach (var param in parameters)
        {
            if (param.AlwaysGather)
            {
                plan.AlwaysGathered.Add(param.Name);
                continue;
            }

            if (_fullShardedLayers.Any(layer => param.LayerName.Contains(layer)))
            {
                fullShardedParams.Add(param);
            }
            else if (_layerWiseShardedLayers.Any(layer => param.LayerName.Contains(layer)))
            {
                layerWiseParams.Add(param);
            }
            else
            {
                // Default to full sharding
                fullShardedParams.Add(param);
            }
        }

        // Apply full sharding strategy
        var fullStrategy = new FullShardingStrategy();
        var fullPlan = fullStrategy.CalculateShardingPlan(fullShardedParams, worldSize);

        // Apply layer-wise sharding strategy
        var layerWiseStrategy = new LayerWiseShardingStrategy();
        var layerWisePlan = layerWiseStrategy.CalculateShardingPlan(layerWiseParams, worldSize);

        // Merge plans
        foreach (var kvp in fullPlan.Assignments)
            plan.Assignments[kvp.Key] = kvp.Value;

        foreach (var kvp in layerWisePlan.Assignments)
            plan.Assignments[kvp.Key] = kvp.Value;

        return plan;
    }
}
```

### 6. ShardingStrategyFactory Class
Create a factory for creating sharding strategies:

```csharp
public static class ShardingStrategyFactory
{
    /// <summary>
    /// Create a sharding strategy from the enum value.
    /// </summary>
    public static IShardingStrategy Create(ShardingStrategy strategy, object config = null)
    {
        return strategy switch
        {
            ShardingStrategy.Full => new FullShardingStrategy(),
            ShardingStrategy.LayerWise => new LayerWiseShardingStrategy(),
            ShardingStrategy.Hybrid => CreateHybridStrategy(config),
            _ => throw new ArgumentException($"Unknown sharding strategy: {strategy}", nameof(strategy))
        };
    }

    private static IShardingStrategy CreateHybridStrategy(object config)
    {
        // Parse config to get layer lists
        if (config is HybridConfig hybridConfig)
        {
            return new HybridShardingStrategy(hybridConfig.FullShardedLayers, hybridConfig.LayerWiseShardedLayers);
        }

        // Default hybrid configuration
        return new HybridShardingStrategy(
            new List<string> { "transformer", "attention" },
            new List<string> { "classifier", "head" }
        );
    }
}

public class HybridConfig
{
    public List<string> FullShardedLayers { get; set; }
    public List<string> LayerWiseShardedLayers { get; set; }
}
```

## Directory Structure
- **File**: `src/MLFramework/Distributed/FSDP/ShardingStrategy.cs`
- **Namespace**: `MLFramework.Distributed.FSDP`

## Dependencies
- `MLFramework.Distributed.FSDP.FSDPConfig` (for ShardingStrategy enum)

## Implementation Notes
1. Use strategy pattern for extensibility
2. Full sharding should provide maximum memory savings
3. Layer-wise sharding should minimize communication
4. Hybrid should balance both
5. Include proper validation and error handling

## Testing Requirements
- Test FullShardingStrategy with various parameter distributions
- Test LayerWiseShardingStrategy with different layer configurations
- Test HybridShardingStrategy with mixed layer assignments
- Test ShardingStrategyFactory
- Test edge cases (empty parameters, single rank)

## Estimated Time
60 minutes
