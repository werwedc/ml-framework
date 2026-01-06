# Spec: FSDP Sharding Unit

## Overview
Create a core data structure for managing sharded parameters, gradients, and optimizer states.

## Requirements

### 1. FSDPShardingUnit Class
Create a class that manages a single sharded parameter:

```csharp
public class FSDPShardingUnit : IDisposable
{
    private readonly IProcessGroup _processGroup;
    private readonly FSDPState _state;

    /// <summary>Local shard of the parameter (only stores 1/num_devices)</summary>
    public Tensor? ShardedParameter { get; set; }

    /// <summary>Full gathered parameter (temporarily allocated during forward/backward)</summary>
    public Tensor? GatheredParameter { get; set; }

    /// <summary>Local gradient for this shard</summary>
    public Tensor? LocalGradient { get; set; }

    /// <summary>Optimizer state for this shard (momentum, variance, etc.)</summary>
    public object? LocalOptimizerState { get; set; }

    /// <summary>Original parameter name</summary>
    public string ParameterName { get; set; }

    /// <summary>Parameter shape</summary>
    public long[] Shape { get; set; }

    /// <summary>Parameter data type</summary>
    public TensorDataType DataType { get; set; }

    /// <summary>Current state of this sharded unit</summary>
    public FSDPState State => _state;

    /// <summary>
    /// Initialize a new sharding unit for a parameter.
    /// </summary>
    /// <param name="parameterName">Name of the parameter</param>
    /// <param name="fullParameter">Full parameter tensor (will be sharded)</param>
    /// <param name="processGroup">Process group for communication</param>
    public FSDPShardingUnit(string parameterName, Tensor fullParameter, IProcessGroup processGroup)
    {
        if (string.IsNullOrEmpty(parameterName))
            throw new ArgumentException("Parameter name cannot be empty", nameof(parameterName));

        _processGroup = processGroup ?? throw new ArgumentNullException(nameof(processGroup));
        ParameterName = parameterName;
        Shape = fullParameter.Shape;
        DataType = fullParameter.DataType;

        // Calculate shard index and owner rank
        var worldSize = _processGroup.WorldSize;
        var rank = _processGroup.Rank;
        var numParameters = fullParameter.Size;
        var shardSize = (numParameters + worldSize - 1) / worldSize;

        // Determine which shard this rank owns
        var shardStart = rank * shardSize;
        var shardEnd = Math.Min(shardStart + shardSize, numParameters);
        var actualShardSize = shardEnd - shardStart;

        // Create local shard
        ShardedParameter = Tensor.Zeros(new[] { actualShardSize }, DataType);

        // Initialize state
        _state = new FSDPState
        {
            OwnerRank = rank,
            NumShards = worldSize,
            ShardIndex = rank,
            IsGathered = false,
            IsOffloaded = false
        };

        // Copy the portion of the parameter that belongs to this shard
        var flatData = fullParameter.Data;
        var shardData = ShardedParameter.Data;
        Array.Copy(flatData, shardStart, shardData, 0, actualShardSize);
    }

    /// <summary>
    /// Gather parameters from all devices.
    /// Creates a full-sized parameter tensor containing all shards.
    /// </summary>
    /// <returns>Full gathered parameter</returns>
    public Tensor GatherParameters()
    {
        if (_state.IsGathered)
            return GatheredParameter!;

        var worldSize = _processGroup.WorldSize;
        var rank = _processGroup.Rank;

        // Allocate buffer for gathered parameter
        GatheredParameter = Tensor.Zeros(Shape, DataType);

        // All-gather: combine all shards into full parameter
        // This will be implemented in a separate spec for communication primitives
        throw new NotImplementedException("Communication primitives to be implemented in spec_fsdp_all_gather.md");
    }

    /// <summary>
    /// Release gathered parameters to free memory.
    /// </summary>
    public void ReleaseGatheredParameters()
    {
        if (GatheredParameter != null)
        {
            GatheredParameter.Dispose();
            GatheredParameter = null;
        }
        _state.IsGathered = false;
    }

    /// <summary>
    /// Scatter gradients to owning devices.
    /// </summary>
    public void ScatterGradients()
    {
        if (LocalGradient == null)
            throw new InvalidOperationException("No gradient to scatter");

        // Scatter gradients to owner ranks
        // This will be implemented in a separate spec for communication primitives
        throw new NotImplementedException("Communication primitives to be implemented in spec_fsdp_reduce_scatter.md");
    }

    /// <summary>
    /// Dispose of resources.
    /// </summary>
    public void Dispose()
    {
        ShardedParameter?.Dispose();
        GatheredParameter?.Dispose();
        LocalGradient?.Dispose();

        ShardedParameter = null;
        GatheredParameter = null;
        LocalGradient = null;
    }
}
```

### 2. ShardedParameterType Enum
Define types of optimizer states:

```csharp
public enum OptimizerStateType
{
    /// <summary>No optimizer state</summary>
    None,

    /// <summary>SGD: just the parameter itself</summary>
    SGD,

    /// <summary>Adam: momentum and variance</summary>
    Adam,

    /// <summary>AdamW: Adam with weight decay</summary>,
    AdamW
}
```

## Directory Structure
- **File**: `src/MLFramework/Distributed/FSDP/FSDPShardingUnit.cs`
- **Namespace**: `MLFramework.Distributed.FSDP`

## Dependencies
- `MLFramework.Distributed.IProcessGroup`
- `RitterFramework.Core.Tensor`

## Implementation Notes
1. Implement proper disposal of tensor resources
2. Track state with `_state` object
3. Throw `NotImplementedException` for methods that depend on communication primitives
4. Calculate shard indices correctly based on rank and world size

## Testing Requirements
- Test sharding of a parameter across multiple ranks
- Test edge cases (single rank, empty parameters)
- Test that gathered parameter contains correct data from all shards
- Test disposal of resources

## Estimated Time
45 minutes
