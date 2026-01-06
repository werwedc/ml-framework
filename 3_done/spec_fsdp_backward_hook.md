# Spec: FSDP Backward Hook

## Overview
Implement backward hooks to automatically scatter gradients after gradient computation.

## Requirements

### 1. FSDPBackwardHook Class
Create a class to manage backward hooks:

```csharp
public class FSDPBackwardHook
{
    private readonly FSDP _fsdp;
    private readonly IProcessGroup _processGroup;
    private readonly Dictionary<string, ReduceScatterOperation> _scatterOperations;

    /// <summary>
    /// Initialize backward hooks for FSDP.
    /// </summary>
    /// <param name="fsdp">FSDP wrapper instance</param>
    public FSDPBackwardHook(FSDP fsdp)
    {
        _fsdp = fsdp ?? throw new ArgumentNullException(nameof(fsdp));
        _processGroup = fsdp.ProcessGroup;
        _scatterOperations = new Dictionary<string, ReduceScatterOperation>();
    }

    /// <summary>
    /// Register backward hooks for all sharding units.
    /// </summary>
    /// <param name="model">The model to register hooks on</param>
    /// <param name="shardingUnits">Dictionary of parameter name to sharding unit</param>
    public void RegisterHooks(IModel model, Dictionary<string, FSDPShardingUnit> shardingUnits)
    {
        if (model == null)
            throw new ArgumentNullException(nameof(model));

        if (shardingUnits == null || shardingUnits.Count == 0)
            return;

        foreach (var kvp in shardingUnits)
        {
            var paramName = kvp.Key;
            var shardingUnit = kvp.Value;

            // Register backward hook to scatter gradients
            RegisterBackwardHook(model, paramName, shardingUnit);
        }
    }

    /// <summary>
    /// Register a backward hook to scatter gradients.
    /// </summary>
    private void RegisterBackwardHook(IModel model, string paramName, FSDPShardingUnit shardingUnit)
    {
        // Create Reduce-Scatter operation for this parameter
        var reduceScatterOp = new ReduceScatterOperation(
            _processGroup,
            shardingUnit.Shape,
            shardingUnit.DataType,
            shardingUnit.State.ShardIndex,
            ReduceOp.Sum
        );
        _scatterOperations[paramName] = reduceScatterOp;

        // Register hook (implementation depends on the model's hook system)
        model.RegisterBackwardHook(paramName, (gradient) =>
        {
            // Scatter gradient after backward pass
            return ScatterGradient(shardingUnit, reduceScatterOp, gradient);
        });
    }

    /// <summary>
    /// Scatter a gradient to the owning device.
    /// </summary>
    private Tensor ScatterGradient(FSDPShardingUnit shardingUnit, ReduceScatterOperation reduceScatterOp, Tensor fullGradient)
    {
        if (fullGradient == null)
            throw new InvalidOperationException($"Gradient is null for {shardingUnit.ParameterName}");

        // Perform Reduce-Scatter
        var scatteredGradient = reduceScatterOp.ReduceScatter(fullGradient);

        // Store scattered gradient in sharding unit
        shardingUnit.LocalGradient = scatteredGradient;

        // Return the scattered gradient (only the portion owned by this device)
        return scatteredGradient;
    }

    /// <summary>
    /// Scatter multiple gradients in parallel.
    /// </summary>
    /// <param name="shardingUnits">Sharding units to scatter</param>
    /// <param name="fullGradients">Full gradients from backward pass</param>
    public async Task ScatterMultipleAsync(
        Dictionary<string, FSDPShardingUnit> shardingUnits,
        Dictionary<string, Tensor> fullGradients)
    {
        if (shardingUnits == null || shardingUnits.Count == 0)
            return;

        if (fullGradients == null || fullGradients.Count == 0)
            return;

        if (shardingUnits.Count != fullGradients.Count)
            throw new ArgumentException("Sharding units and gradients must have the same count");

        var tasks = shardingUnits.Zip(fullGradients, (unitKvp, gradKvp) =>
        {
            var unit = unitKvp.Value;
            var grad = gradKvp.Value;

            if (grad == null || !_scatterOperations.TryGetValue(unit.ParameterName, out var op))
            {
                return Task.CompletedTask;
            }

            return Task.Run(() =>
            {
                var scatteredGrad = op.ReduceScatter(grad);
                unit.LocalGradient = scatteredGrad;
            });
        }).ToList();

        await Task.WhenAll(tasks);
    }

    /// <summary>
    /// Accumulate gradients from multiple micro-batches.
    /// </summary>
    /// <param name="shardingUnits">Sharding units to accumulate</param>
    /// <param name="newGradients">New gradients to accumulate</param>
    public void AccumulateGradients(
        Dictionary<string, FSDPShardingUnit> shardingUnits,
        Dictionary<string, Tensor> newGradients)
    {
        if (shardingUnits == null || shardingUnits.Count == 0)
            return;

        if (newGradients == null || newGradients.Count == 0)
            return;

        foreach (var kvp in shardingUnits)
        {
            var paramName = kvp.Key;
            var shardingUnit = kvp.Value;

            if (newGradients.TryGetValue(paramName, out var newGrad))
            {
                if (shardingUnit.LocalGradient == null)
                {
                    // First gradient, just assign
                    shardingUnit.LocalGradient = newGrad.Clone();
                }
                else
                {
                    // Accumulate
                    var localGradData = shardingUnit.LocalGradient.Data;
                    var newGradData = newGrad.Data;

                    for (int i = 0; i < Math.Min(localGradData.Length, newGradData.Length); i++)
                    {
                        localGradData[i] += newGradData[i];
                    }
                }
            }
        }
    }

    /// <summary>
    /// Clear accumulated gradients.
    /// </summary>
    /// <param name="shardingUnits">Sharding units to clear</param>
    public void ClearGradients(Dictionary<string, FSDPShardingUnit> shardingUnits)
    {
        if (shardingUnits == null || shardingUnits.Count == 0)
            return;

        foreach (var unit in shardingUnits.Values)
        {
            if (unit.LocalGradient != null)
            {
                // Zero out the gradient
                Array.Clear(unit.LocalGradient.Data, 0, unit.LocalGradient.Data.Length);
            }
        }
    }

    /// <summary>
    /// Verify that gradients are correctly scattered.
    /// </summary>
    /// <param name="shardingUnits">Sharding units to verify</param>
    /// <param name="fullGradients">Full gradients to compare against</param>
    public bool VerifyGradients(
        Dictionary<string, FSDPShardingUnit> shardingUnits,
        Dictionary<string, Tensor> fullGradients)
    {
        if (shardingUnits == null || shardingUnits.Count == 0)
            return true;

        if (fullGradients == null || fullGradients.Count == 0)
            return true;

        foreach (var kvp in shardingUnits)
        {
            var paramName = kvp.Key;
            var shardingUnit = kvp.Value;

            if (!fullGradients.TryGetValue(paramName, out var fullGrad))
                continue;

            if (shardingUnit.LocalGradient == null)
                return false;

            var worldSize = _processGroup.WorldSize;
            var shardIndex = shardingUnit.State.ShardIndex;

            // Calculate expected shard from full gradient
            var totalSize = fullGrad.Size;
            var chunkSize = (totalSize + worldSize - 1) / worldSize;
            var startOffset = shardIndex * chunkSize;
            var shardSize = Math.Min(chunkSize, totalSize - startOffset);

            // Compare
            for (int i = 0; i < shardSize; i++)
            {
                var expected = fullGrad.Data[startOffset + i];
                var actual = shardingUnit.LocalGradient.Data[i];

                if (Math.Abs(expected - actual) > 1e-5)
                    return false;
            }
        }

        return true;
    }

    /// <summary>
    /// Dispose of resources.
    /// </summary>
    public void Dispose()
    {
        foreach (var op in _scatterOperations.Values)
        {
            op.Dispose();
        }
        _scatterOperations.Clear();
    }
}
```

### 2. GradientAccumulator Class
Create a class to manage gradient accumulation:

```csharp
public class GradientAccumulator
{
    private readonly Dictionary<string, List<Tensor>> _accumulatedGradients;
    private readonly int _accumulationSteps;

    /// <summary>
    /// Initialize a gradient accumulator.
    /// </summary>
    /// <param name="accumulationSteps">Number of micro-batches to accumulate</param>
    public GradientAccumulator(int accumulationSteps = 1)
    {
        if (accumulationSteps <= 0)
            throw new ArgumentException("Accumulation steps must be positive", nameof(accumulationSteps));

        _accumulatedGradients = new Dictionary<string, List<Tensor>>();
        _accumulationSteps = accumulationSteps;
    }

    /// <summary>
    /// Add gradients from a micro-batch.
    /// </summary>
    /// <param name="gradients">Gradients to add</param>
    public void AddGradients(Dictionary<string, Tensor> gradients)
    {
        if (gradients == null || gradients.Count == 0)
            return;

        foreach (var kvp in gradients)
        {
            var paramName = kvp.Key;
            var grad = kvp.Value;

            if (!_accumulatedGradients.ContainsKey(paramName))
            {
                _accumulatedGradients[paramName] = new List<Tensor>();
            }

            _accumulatedGradients[paramName].Add(grad.Clone());
        }
    }

    /// <summary>
    /// Get accumulated gradients and clear accumulator.
    /// </summary>
    /// <returns>Accumulated gradients</returns>
    public Dictionary<string, Tensor> GetAndClearGradients()
    {
        var result = new Dictionary<string, Tensor>();

        foreach (var kvp in _accumulatedGradients)
        {
            var paramName = kvp.Key;
            var grads = kvp.Value;

            if (grads.Count == 0)
                continue;

            // Sum all accumulated gradients
            var summedGrad = grads[0].Clone();
            for (int i = 1; i < grads.Count; i++)
            {
                var gradData = grads[i].Data;
                for (int j = 0; j < gradData.Length; j++)
                {
                    summedGrad.Data[j] += gradData[j];
                }
            }

            result[paramName] = summedGrad;

            // Clear accumulated gradients
            foreach (var grad in grads)
            {
                grad.Dispose();
            }
            grads.Clear();
        }

        return result;
    }

    /// <summary>
    /// Check if accumulation is complete.
    /// </summary>
    public bool IsComplete
    {
        get
        {
            if (_accumulatedGradients.Count == 0)
                return false;

            return _accumulatedGradients.Values.All(list => list.Count >= _accumulationSteps);
        }
    }

    /// <summary>
    /// Reset the accumulator.
    /// </summary>
    public void Reset()
    {
        foreach (var list in _accumulatedGradients.Values)
        {
            foreach (var grad in list)
            {
                grad.Dispose();
            }
            list.Clear();
        }
        _accumulatedGradients.Clear();
    }
}
```

## Directory Structure
- **File**: `src/MLFramework/Distributed/FSDP/FSDPBackwardHook.cs`
- **Namespace**: `MLFramework.Distributed.FSDP`

## Dependencies
- `MLFramework.Distributed.ProcessGroup`
- `MLFramework.Distributed.FSDP.FSDP`
- `MLFramework.Distributed.FSDP.FSDPShardingUnit`
- `MLFramework.Distributed.FSDP.ReduceScatterOperation`
- `RitterFramework.Core.Tensor`

## Implementation Notes
1. Backward hooks should scatter gradients after computation
2. Support gradient accumulation for micro-batching
3. Provide verification for testing
4. Support both synchronous and asynchronous scattering
5. Clear gradients after optimizer step

## Testing Requirements
- Test single gradient scattering
- Test multiple gradient scattering in parallel
- Test gradient accumulation
- Test gradient verification
- Test hook registration and execution
- Test edge cases (null gradients, empty model)
- Test error handling (failed scattering)

## Estimated Time
45 minutes
