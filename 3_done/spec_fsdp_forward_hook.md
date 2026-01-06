# Spec: FSDP Forward Hook

## Overview
Implement forward hooks to automatically gather parameters before layer execution and release them after.

## Requirements

### 1. FSDPForwardHook Class
Create a class to manage forward hooks:

```csharp
public class FSDPForwardHook
{
    private readonly FSDP _fsdp;
    private readonly IProcessGroup _processGroup;
    private readonly Dictionary<string, AllGatherOperation> _gatherOperations;

    /// <summary>
    /// Initialize forward hooks for FSDP.
    /// </summary>
    /// <param name="fsdp">FSDP wrapper instance</param>
    public FSDPForwardHook(FSDP fsdp)
    {
        _fsdp = fsdp ?? throw new ArgumentNullException(nameof(fsdp));
        _processGroup = fsdp.ProcessGroup;
        _gatherOperations = new Dictionary<string, AllGatherOperation>();
    }

    /// <summary>
    /// Register forward hooks for all sharding units.
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

            // Register pre-forward hook to gather parameter
            RegisterPreForwardHook(model, paramName, shardingUnit);

            // Register post-forward hook to release parameter
            RegisterPostForwardHook(model, paramName, shardingUnit);
        }
    }

    /// <summary>
    /// Register a pre-forward hook to gather a parameter.
    /// </summary>
    private void RegisterPreForwardHook(IModel model, string paramName, FSDPShardingUnit shardingUnit)
    {
        // Create All-Gather operation for this parameter
        var allGatherOp = new AllGatherOperation(
            _processGroup,
            shardingUnit.Shape,
            shardingUnit.DataType,
            shardingUnit.State.ShardIndex
        );
        _gatherOperations[paramName] = allGatherOp;

        // Register hook (implementation depends on the model's hook system)
        // For now, we'll assume the model has a RegisterPreForward method
        model.RegisterPreForwardHook(paramName, () =>
        {
            // Gather parameter before forward pass
            GatherParameter(shardingUnit, allGatherOp);
        });
    }

    /// <summary>
    /// Register a post-forward hook to release a parameter.
    /// </summary>
    private void RegisterPostForwardHook(IModel model, string paramName, FSDPShardingUnit shardingUnit)
    {
        model.RegisterPostForwardHook(paramName, () =>
        {
            // Release gathered parameter after forward pass
            ReleaseParameter(shardingUnit);
        });
    }

    /// <summary>
    /// Gather a parameter from all devices.
    /// </summary>
    private void GatherParameter(FSDPShardingUnit shardingUnit, AllGatherOperation allGatherOp)
    {
        if (shardingUnit.ShardedParameter == null)
            throw new InvalidOperationException($"Sharded parameter is null for {shardingUnit.ParameterName}");

        // Perform All-Gather
        var gatheredParam = allGatherOp.AllGather(shardingUnit.ShardedParameter);

        // Store gathered parameter in sharding unit
        shardingUnit.GatheredParameter = gatheredParam;
        shardingUnit.State.IsGathered = true;

        // Replace the parameter in the model with the gathered version
        // This requires the model to support parameter replacement
        _fsdp.ReplaceParameter(shardingUnit.ParameterName, gatheredParam);
    }

    /// <summary>
    /// Release a gathered parameter to free memory.
    /// </summary>
    private void ReleaseParameter(FSDPShardingUnit shardingUnit)
    {
        if (shardingUnit.GatheredParameter == null)
            return;

        // Release gathered parameter
        shardingUnit.ReleaseGatheredParameters();

        // Restore the sharded parameter in the model
        // (This may not be necessary if the model keeps references)
    }

    /// <summary>
    /// Gather multiple parameters in parallel.
    /// </summary>
    /// <param name="shardingUnits">Sharding units to gather</param>
    public async Task GatherMultipleAsync(Dictionary<string, FSDPShardingUnit> shardingUnits)
    {
        if (shardingUnits == null || shardingUnits.Count == 0)
            return;

        var tasks = shardingUnits.Values.Select(unit =>
        {
            if (unit.ShardedParameter == null || _gatherOperations.TryGetValue(unit.ParameterName, out var op))
            {
                return Task.Run(() =>
                {
                    var gatheredParam = op.AllGather(unit.ShardedParameter!);
                    unit.GatheredParameter = gatheredParam;
                    unit.State.IsGathered = true;
                });
            }
            return Task.CompletedTask;
        }).ToList();

        await Task.WhenAll(tasks);
    }

    /// <summary>
    /// Release multiple gathered parameters.
    /// </summary>
    /// <param name="shardingUnits">Sharding units to release</param>
    public void ReleaseMultiple(Dictionary<string, FSDPShardingUnit> shardingUnits)
    {
        if (shardingUnits == null || shardingUnits.Count == 0)
            return;

        foreach (var unit in shardingUnits.Values)
        {
            unit.ReleaseGatheredParameters();
        }
    }

    /// <summary>
    /// Dispose of resources.
    /// </summary>
    public void Dispose()
    {
        foreach (var op in _gatherOperations.Values)
        {
            op.Dispose();
        }
        _gatherOperations.Clear();
    }
}
```

### 2. HookContext Class
Create a class to manage hook context:

```csharp
public class HookContext
{
    /// <summary>Layer name</summary>
    public string LayerName { get; set; }

    /// <summary>Input tensor</summary>
    public Tensor Input { get; set; }

    /// <summary>Output tensor</summary>
    public Tensor Output { get; set; }

    /// <summary>Whether this is the forward pass</summary>
    public bool IsForward { get; set; }

    /// <summary>Current step in the pass</summary>
    public int Step { get; set; }

    public HookContext(string layerName, Tensor input, bool isForward)
    {
        LayerName = layerName;
        Input = input;
        IsForward = isForward;
        Step = 0;
    }
}
```

### 3. FSDP Extensions
Add extension methods to the FSDP class:

```csharp
public static class FSDPExtensions
{
    /// <summary>
    /// Get the process group from FSDP.
    /// </summary>
    public static IProcessGroup GetProcessGroup(this FSDP fsdp)
    {
        // This will be exposed through a property in FSDP class
        throw new NotImplementedException("Add ProcessGroup property to FSDP class");
    }

    /// <summary>
    /// Replace a parameter in the FSDP-wrapped model.
    /// </summary>
    public static void ReplaceParameter(this FSDP fsdp, string paramName, Tensor newTensor)
    {
        // This requires the underlying model to support parameter replacement
        throw new NotImplementedException("Add ReplaceParameter method to FSDP class");
    }
}
```

## Directory Structure
- **File**: `src/MLFramework/Distributed/FSDP/FSDPForwardHook.cs`
- **Namespace**: `MLFramework.Distributed.FSDP`

## Dependencies
- `MLFramework.Distributed.ProcessGroup`
- `MLFramework.Distributed.FSDP.FSDP`
- `MLFramework.Distributed.FSDP.FSDPShardingUnit`
- `MLFramework.Distributed.FSDP.AllGatherOperation`
- `RitterFramework.Core.Tensor`

## Implementation Notes
1. Forward hooks should gather parameters before layer execution
2. Post-forward hooks should release parameters after execution
3. Support both synchronous and asynchronous gathering
4. Manage lifecycle of All-Gather operations
5. Handle errors gracefully (e.g., if gathering fails)

## Testing Requirements
- Test single parameter gathering
- Test multiple parameter gathering in parallel
- Test parameter release after forward pass
- Test hook registration and execution
- Test edge cases (null parameters, empty model)
- Test error handling (failed gathering)

## Estimated Time
45 minutes
