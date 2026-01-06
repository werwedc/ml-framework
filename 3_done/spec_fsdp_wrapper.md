# Spec: FSDP Wrapper

## Overview
Create the main FSDP wrapper class that wraps a model and applies sharding during training.

## Requirements

### 1. FSDP Class
Create the main wrapper class:

```csharp
public class FSDP : IModel, IDisposable
{
    private readonly IModel _model;
    private readonly IProcessGroup _processGroup;
    private readonly FSDPConfig _config;
    private readonly IShardingStrategy _shardingStrategy;
    private readonly Dictionary<string, FSDPShardingUnit> _shardingUnits;
    private readonly ShardingPlan _shardingPlan;
    private bool _disposed;

    /// <summary>
    /// Wrap a model with FSDP for distributed training.
    /// </summary>
    /// <param name="model">The model to wrap</param>
    /// <param name="config">FSDP configuration</param>
    /// <param name="processGroup">Process group for communication</param>
    public FSDP(IModel model, FSDPConfig config, IProcessGroup processGroup = null)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _config = config ?? throw new ArgumentNullException(nameof(config));

        // Validate configuration
        _config.Validate();

        // Use default process group if not provided
        _processGroup = processGroup ?? ProcessGroup.Default;
        if (_processGroup == null)
            throw new InvalidOperationException("No process group available. Call ProcessGroup.Init() first.");

        // Validate world size
        if (_processGroup.WorldSize == 1 && _config.ShardingStrategy == ShardingStrategy.Full)
        {
            // Warn but allow for single-device testing
            // In production, FSDP requires multiple devices
        }

        // Create sharding strategy
        _shardingStrategy = ShardingStrategyFactory.Create(_config.ShardingStrategy);

        // Collect parameter information
        var parameters = CollectParameterInfo(_model);

        // Calculate sharding plan
        _shardingPlan = _shardingStrategy.CalculateShardingPlan(parameters, _processGroup.WorldSize);

        // Create sharding units
        _shardingUnits = new Dictionary<string, FSDPShardingUnit>();
        foreach (var param in parameters)
        {
            if (!_shardingPlan.AlwaysGathered.Contains(param.Name))
            {
                var shardingUnit = CreateShardingUnit(param, _model);
                _shardingUnits[param.Name] = shardingUnit;
            }
        }

        // Register hooks
        RegisterForwardHooks();
        RegisterBackwardHooks();
    }

    /// <summary>
    /// Forward pass through the model.
    /// </summary>
    /// <param name="input">Input tensor</param>
    /// <returns>Output tensor</returns>
    public Tensor Forward(Tensor input)
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(FSDP));

        // Forward pass will trigger hooks that gather parameters
        return _model.Forward(input);
    }

    /// <summary>
    /// Backward pass (computes gradients).
    /// </summary>
    public void Backward()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(FSDP));

        // Backward pass will trigger hooks that scatter gradients
        _model.Backward();
    }

    /// <summary>
    /// Get model parameters.
    /// </summary>
    /// <returns>List of parameter tensors</returns>
    public List<Tensor> GetParameters()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(FSDP));

        // Return sharded parameters
        var parameters = new List<Tensor>();
        foreach (var unit in _shardingUnits.Values)
        {
            parameters.Add(unit.ShardedParameter!);
        }
        return parameters;
    }

    /// <summary>
    /// Get model gradients.
    /// </summary>
    /// <returns>List of gradient tensors</returns>
    public List<Tensor> GetGradients()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(FSDP));

        // Return sharded gradients
        var gradients = new List<Tensor>();
        foreach (var unit in _shardingUnits.Values)
        {
            if (unit.LocalGradient != null)
                gradients.Add(unit.LocalGradient);
        }
        return gradients;
    }

    /// <summary>
    /// Collect parameter information from the model.
    /// </summary>
    private List<ParameterInfo> CollectParameterInfo(IModel model)
    {
        var parameters = new List<ParameterInfo>();
        var modelParameters = model.GetParameters();

        foreach (var param in modelParameters)
        {
            var paramInfo = new ParameterInfo
            {
                Name = param.Name ?? $"param_{parameters.Count}",
                Shape = param.Shape,
                SizeBytes = param.Size * 4, // Assume float32
                LayerName = InferLayerName(param.Name),
                AlwaysGather = ShouldAlwaysGather(param.Name)
            };
            parameters.Add(paramInfo);
        }

        return parameters;
    }

    /// <summary>
    /// Infer layer name from parameter name.
    /// </summary>
    private string InferLayerName(string paramName)
    {
        if (string.IsNullOrEmpty(paramName))
            return "layer_0";

        // Simple heuristic: extract layer name from parameter name
        // e.g., "transformer.layer1.weight" -> "transformer.layer1"
        var parts = paramName.Split('.');
        if (parts.Length >= 2)
        {
            return string.Join(".", parts.SkipLast(1));
        }
        return parts[0];
    }

    /// <summary>
    /// Determine if a parameter should always be gathered.
    /// </summary>
    private bool ShouldAlwaysGather(string paramName)
    {
        if (string.IsNullOrEmpty(paramName))
            return false;

        // Embeddings should always be gathered for simplicity
        if (paramName.Contains("embedding", StringComparison.OrdinalIgnoreCase))
            return true;

        return false;
    }

    /// <summary>
    /// Create a sharding unit for a parameter.
    /// </summary>
    private FSDPShardingUnit CreateShardingUnit(ParameterInfo paramInfo, IModel model)
    {
        var modelParams = model.GetParameters();
        var param = modelParams.FirstOrDefault(p => p.Name == paramInfo.Name);

        if (param == null)
            throw new ArgumentException($"Parameter {paramInfo.Name} not found in model");

        return new FSDPShardingUnit(paramInfo.Name, param.Tensor, _processGroup);
    }

    /// <summary>
    /// Register forward hooks to gather parameters.
    /// </summary>
    private void RegisterForwardHooks()
    {
        // This will be implemented in spec_fsdp_forward_hook.md
        // For now, just mark as not implemented
    }

    /// <summary>
    /// Register backward hooks to scatter gradients.
    /// </summary>
    private void RegisterBackwardHooks()
    {
        // This will be implemented in spec_fsdp_backward_hook.md
        // For now, just mark as not implemented
    }

    /// <summary>
    /// Dispose of resources.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
            return;

        foreach (var unit in _shardingUnits.Values)
        {
            unit.Dispose();
        }

        _shardingUnits.Clear();
        _disposed = true;
    }
}
```

### 2. IModel Interface (if not exists)
Define a simple model interface:

```csharp
public interface IModel
{
    /// <summary>Forward pass</summary>
    Tensor Forward(Tensor input);

    /// <summary>Backward pass</summary>
    void Backward();

    /// <summary>Get model parameters</summary>
    List<NamedTensor> GetParameters();
}

public class NamedTensor
{
    public string Name { get; set; }
    public Tensor Tensor { get; set; }
}
```

## Directory Structure
- **File**: `src/MLFramework/Distributed/FSDP/FSDP.cs`
- **Namespace**: `MLFramework.Distributed.FSDP`

## Dependencies
- `MLFramework.Distributed.ProcessGroup`
- `MLFramework.Distributed.FSDP.FSDPConfig`
- `MLFramework.Distributed.FSDP.FSDPShardingUnit`
- `MLFramework.Distributed.FSDP.ShardingStrategy`
- `RitterFramework.Core.Tensor`

## Implementation Notes
1. FSDP should act as a wrapper/decorator around the model
2. Maintain backward compatibility with existing model interface
3. Validate inputs and configuration
4. Manage lifecycle of sharding units
5. Hook registration will be implemented in separate specs

## Testing Requirements
- Test wrapping a simple model
- Test wrapping a model with multiple layers
- Test parameter collection
- Test layer name inference
- Test AlwaysGather logic
- Test disposal
- Test edge cases (empty model, single device)

## Estimated Time
60 minutes
