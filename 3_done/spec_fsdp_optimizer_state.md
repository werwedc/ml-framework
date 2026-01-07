# Spec: FSDP Optimizer State Sharding

## Overview
Implement sharding of optimizer states (momentum, variance, etc.) to work with sharded parameters.

## Requirements

### 1. OptimizerState Class
Create a base class for optimizer states:

```csharp
public abstract class OptimizerState
{
    /// <summary>Type of optimizer state</summary>
    public OptimizerStateType StateType { get; protected set; }

    /// <summary>Shard index</summary>
    public int ShardIndex { get; set; }

    /// <summary>Number of shards</summary>
    public int NumShards { get; set; }

    /// <summary>Clone the optimizer state</summary>
    public abstract OptimizerState Clone();

    /// <summary>Dispose of resources</summary>
    public abstract void Dispose();
}
```

### 2. AdamOptimizerState Class
Implement Adam optimizer state with sharding:

```csharp
public class AdamOptimizerState : OptimizerState
{
    /// <summary>Momentum buffer (first moment)</summary>
    public Tensor MomentumBuffer { get; set; }

    /// <summary>Variance buffer (second moment)</summary>
    public Tensor VarianceBuffer { get; set; }

    /// <summary>Step count</summary>
    public int StepCount { get; set; }

    /// <summary>
    /// Initialize Adam optimizer state for a sharded parameter.
    /// </summary>
    /// <param name="shardedParameter">Sharded parameter tensor</param>
    /// <param name="shardIndex">Shard index</param>
    /// <param name="numShards">Total number of shards</param>
    public AdamOptimizerState(Tensor shardedParameter, int shardIndex, int numShards)
    {
        if (shardedParameter == null)
            throw new ArgumentNullException(nameof(shardedParameter));

        StateType = OptimizerStateType.Adam;
        ShardIndex = shardIndex;
        NumShards = numShards;
        StepCount = 0;

        // Initialize buffers to zeros
        MomentumBuffer = Tensor.Zeros(shardedParameter.Shape, shardedParameter.DataType);
        VarianceBuffer = Tensor.Zeros(shardedParameter.Shape, shardedParameter.DataType);
    }

    /// <summary>
    /// Clone the optimizer state.
    /// </summary>
    public override OptimizerState Clone()
    {
        var cloned = new AdamOptimizerState(MomentumBuffer, ShardIndex, NumShards)
        {
            StepCount = StepCount
        };

        // Copy buffers
        Array.Copy(MomentumBuffer.Data, cloned.MomentumBuffer.Data, MomentumBuffer.Size);
        Array.Copy(VarianceBuffer.Data, cloned.VarianceBuffer.Data, VarianceBuffer.Size);

        return cloned;
    }

    /// <summary>
    /// Dispose of resources.
    /// </summary>
    public override void Dispose()
    {
        MomentumBuffer?.Dispose();
        VarianceBuffer?.Dispose();

        MomentumBuffer = null;
        VarianceBuffer = null;
    }
}
```

### 3. SGDOptimizerState Class
Implement SGD optimizer state (minimal):

```csharp
public class SGDOptimizerState : OptimizerState
{
    /// <summary>
    /// Initialize SGD optimizer state for a sharded parameter.
    /// SGD doesn't require state buffers.
    /// </summary>
    /// <param name="shardIndex">Shard index</param>
    /// <param name="numShards">Total number of shards</param>
    public SGDOptimizerState(int shardIndex, int numShards)
    {
        StateType = OptimizerStateType.SGD;
        ShardIndex = shardIndex;
        NumShards = numShards;
    }

    /// <summary>
    /// Clone the optimizer state.
    /// </summary>
    public override OptimizerState Clone()
    {
        return new SGDOptimizerState(ShardIndex, NumShards);
    }

    /// <summary>
    /// Dispose of resources.
    /// </summary>
    public override void Dispose()
    {
        // No resources to dispose
    }
}
```

### 4. FSDPOptimizerStateManager Class
Create a manager for sharded optimizer states:

```csharp
public class FSDPOptimizerStateManager : IDisposable
{
    private readonly IProcessGroup _processGroup;
    private readonly Dictionary<string, OptimizerState> _optimizerStates;

    /// <summary>
    /// Initialize optimizer state manager.
    /// </summary>
    /// <param name="processGroup">Process group for communication</param>
    public FSDPOptimizerStateManager(IProcessGroup processGroup)
    {
        _processGroup = processGroup ?? throw new ArgumentNullException(nameof(processGroup));
        _optimizerStates = new Dictionary<string, OptimizerState>();
    }

    /// <summary>
    /// Create optimizer state for a sharded parameter.
    /// </summary>
    /// <param name="paramName">Parameter name</param>
    /// <param name="shardedParameter">Sharded parameter tensor</param>
    /// <param name="optimizerType">Type of optimizer</param>
    /// <param name="shardIndex">Shard index</param>
    /// <param name="numShards">Total number of shards</param>
    public OptimizerState CreateOptimizerState(
        string paramName,
        Tensor shardedParameter,
        OptimizerStateType optimizerType,
        int shardIndex,
        int numShards)
    {
        if (string.IsNullOrEmpty(paramName))
            throw new ArgumentException("Parameter name cannot be empty", nameof(paramName));

        if (_optimizerStates.ContainsKey(paramName))
            throw new ArgumentException($"Optimizer state already exists for {paramName}");

        OptimizerState state = optimizerType switch
        {
            OptimizerStateType.Adam => new AdamOptimizerState(shardedParameter, shardIndex, numShards),
            OptimizerStateType.AdamW => new AdamOptimizerState(shardedParameter, shardIndex, numShards),
            OptimizerStateType.SGD => new SGDOptimizerState(shardIndex, numShards),
            _ => throw new ArgumentException($"Unsupported optimizer type: {optimizerType}", nameof(optimizerType))
        };

        _optimizerStates[paramName] = state;
        return state;
    }

    /// <summary>
    /// Get optimizer state for a parameter.
    /// </summary>
    /// <param name="paramName">Parameter name</param>
    /// <returns>Optimizer state</returns>
    public OptimizerState GetOptimizerState(string paramName)
    {
        if (!_optimizerStates.TryGetValue(paramName, out var state))
            throw new ArgumentException($"No optimizer state found for {paramName}");

        return state;
    }

    /// <summary>
    /// Check if optimizer state exists for a parameter.
    /// </summary>
    /// <param name="paramName">Parameter name</param>
    /// <returns>True if state exists</returns>
    public bool HasOptimizerState(string paramName)
    {
        return _optimizerStates.ContainsKey(paramName);
    }

    /// <summary>
    /// Remove optimizer state for a parameter.
    /// </summary>
    /// <param name="paramName">Parameter name</param>
    public void RemoveOptimizerState(string paramName)
    {
        if (_optimizerStates.TryGetValue(paramName, out var state))
        {
            state.Dispose();
            _optimizerStates.Remove(paramName);
        }
    }

    /// <summary>
    /// Gather optimizer states from all devices (for checkpointing).
    /// </summary>
    /// <param name="paramName">Parameter name</param>
    /// <returns>Full optimizer state</returns>
    public OptimizerState GatherOptimizerState(string paramName)
    {
        if (!_optimizerStates.TryGetValue(paramName, out var state))
            throw new ArgumentException($"No optimizer state found for {paramName}");

        if (_processGroup.WorldSize == 1)
        {
            // Single device, no gathering needed
            return state.Clone();
        }

        // Gather all shards of optimizer state
        // This is complex because optimizer states have multiple buffers
        if (state is AdamOptimizerState adamState)
        {
            return GatherAdamState(paramName, adamState);
        }
        else if (state is SGDOptimizerState sgdState)
        {
            // SGD has no state to gather
            return sgdState.Clone();
        }

        throw new ArgumentException($"Unsupported optimizer state type: {state.StateType}");
    }

    /// <summary>
    /// Gather Adam optimizer state from all devices.
    /// </summary>
    private AdamOptimizerState GatherAdamState(string paramName, AdamOptimizerState localState)
    {
        var worldSize = _processGroup.WorldSize;
        var rank = _processGroup.Rank;

        // Gather momentum buffer
        var momentumGatherOp = new AllGatherOperation(
            _processGroup,
            localState.MomentumBuffer.Shape,
            localState.MomentumBuffer.DataType,
            rank
        );
        var fullMomentum = momentumGatherOp.AllGather(localState.MomentumBuffer);

        // Gather variance buffer
        var varianceGatherOp = new AllGatherOperation(
            _processGroup,
            localState.VarianceBuffer.Shape,
            localState.VarianceBuffer.DataType,
            rank
        );
        var fullVariance = varianceGatherOp.AllGather(localState.VarianceBuffer);

        // Create full state (only rank 0 returns the full state)
        AdamOptimizerState fullState;
        if (rank == 0)
        {
            fullState = new AdamOptimizerState(fullMomentum, 0, worldSize);
            fullState.MomentumBuffer = fullMomentum;
            fullState.VarianceBuffer = fullVariance;
            fullState.StepCount = localState.StepCount;
        }
        else
        {
            // Other ranks return null or throw
            fullState = null;
        }

        return fullState;
    }

    /// <summary>
    /// Scatter optimizer state to devices (for loading checkpoints).
    /// </summary>
    /// <param name="paramName">Parameter name</param>
    /// <param name="fullState">Full optimizer state from checkpoint</param>
    public void ScatterOptimizerState(string paramName, OptimizerState fullState)
    {
        if (fullState == null)
            throw new ArgumentNullException(nameof(fullState));

        if (!_optimizerStates.TryGetValue(paramName, out var localState))
            throw new ArgumentException($"No optimizer state found for {paramName}");

        if (_processGroup.WorldSize == 1)
        {
            // Single device, just copy
            CopyOptimizerState(fullState, localState);
            return;
        }

        // Broadcast from rank 0 and extract local shard
        if (fullState is AdamOptimizerState fullAdamState && localState is AdamOptimizerState localAdamState)
        {
            ScatterAdamState(paramName, fullAdamState, localAdamState);
        }
    }

    /// <summary>
    /// Scatter Adam optimizer state to devices.
    /// </summary>
    private void ScatterAdamState(string paramName, AdamOptimizerState fullState, AdamOptimizerState localState)
    {
        var worldSize = _processGroup.WorldSize;
        var rank = _processGroup.Rank;

        // Broadcast full momentum buffer
        _processGroup.Broadcast(fullState.MomentumBuffer, 0);

        // Broadcast full variance buffer
        _processGroup.Broadcast(fullState.VarianceBuffer, 0);

        // Extract local shard from full state
        var shardSize = localState.MomentumBuffer.Size;
        var shardOffset = rank * shardSize;

        Array.Copy(fullState.MomentumBuffer.Data, shardOffset, localState.MomentumBuffer.Data, 0, shardSize);
        Array.Copy(fullState.VarianceBuffer.Data, shardOffset, localState.VarianceBuffer.Data, 0, shardSize);

        localState.StepCount = fullState.StepCount;
    }

    /// <summary>
    /// Copy optimizer state from source to destination.
    /// </summary>
    private void CopyOptimizerState(OptimizerState source, OptimizerState destination)
    {
        if (source.StateType != destination.StateType)
            throw new ArgumentException("Optimizer state types must match");

        if (source is AdamOptimizerState sourceAdam && destination is AdamOptimizerState destAdam)
        {
            Array.Copy(sourceAdam.MomentumBuffer.Data, destAdam.MomentumBuffer.Data, sourceAdam.MomentumBuffer.Size);
            Array.Copy(sourceAdam.VarianceBuffer.Data, destAdam.VarianceBuffer.Data, sourceAdam.VarianceBuffer.Size);
            destAdam.StepCount = sourceAdam.StepCount;
        }
    }

    /// <summary>
    /// Dispose of resources.
    /// </summary>
    public void Dispose()
    {
        foreach (var state in _optimizerStates.Values)
        {
            state.Dispose();
        }
        _optimizerStates.Clear();
    }
}
```

## Directory Structure
- **File**: `src/MLFramework/Distributed/FSDP/FSDPOptimizerState.cs`
- **Namespace**: `MLFramework.Distributed.FSDP`

## Dependencies
- `MLFramework.Distributed.ProcessGroup`
- `MLFramework.Distributed.FSDP.AllGatherOperation`
- `RitterFramework.Core.Tensor`

## Implementation Notes
1. Support multiple optimizer types (Adam, SGD)
2. Sharded optimizer states only store buffers for local shards
3. Provide gathering/scattering for checkpointing
4. Implement proper disposal of tensor buffers
5. Use All-Gather/Broadcast for state synchronization

## Testing Requirements
- Test creating optimizer states for different optimizer types
- Test gathering optimizer states from multiple devices
- Test scattering optimizer states to multiple devices
- Test optimizer state cloning
- Test optimizer state disposal
- Test edge cases (single device, empty states)

## Estimated Time
60 minutes
