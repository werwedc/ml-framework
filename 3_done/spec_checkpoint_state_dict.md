# Spec: State Dict and Model State Management

## Overview
Implement the state dictionary abstraction for collecting and distributing model parameters, optimizer states, and other training components in a distributed environment.

## Scope
- 30-45 minutes coding time
- Focus on state management abstractions
- Target: `src/MLFramework/Checkpointing/State/`

## Classes

### 1. IStateful (Interface for Stateful Components)
```csharp
public interface IStateful
{
    /// <summary>
    /// Get current state as a dictionary
    /// </summary>
    StateDict GetStateDict();

    /// <summary>
    /// Load state from a dictionary
    /// </summary>
    void LoadStateDict(StateDict state);
}
```

### 2. StateDict (State Dictionary)
```csharp
public class StateDict : Dictionary<string, Tensor>
{
    /// <summary>
    /// Get a tensor by key, throw if not found
    /// </summary>
    public Tensor GetTensor(string key)
    {
        if (!TryGetValue(key, out var tensor))
        {
            throw new KeyNotFoundException($"Tensor '{key}' not found in state dict");
        }
        return tensor;
    }

    /// <summary>
    /// Get a tensor by key, return null if not found
    /// </summary>
    public Tensor? GetTensorOrNull(string key)
    {
        return TryGetValue(key, out var tensor) ? tensor : null;
    }

    /// <summary>
    /// Check if tensor exists
    /// </summary>
    public bool HasTensor(string key) => ContainsKey(key);

    /// <summary>
    /// Remove a tensor from state
    /// </summary>
    public void RemoveTensor(string key) => Remove(key);
}
```

### 3. OptimizerStateDict (Optimizer-Specific State)
```csharp
public class OptimizerStateDict : StateDict
{
    public OptimizerType OptimizerType { get; set; }
    public long Step { get; set; }
    public float LearningRate { get; set; }

    /// <summary>
    /// Create empty state dict for specific optimizer
    /// </summary>
    public static OptimizerStateDict Create(OptimizerType type)
    {
        return new OptimizerStateDict { OptimizerType = type };
    }
}
```

### 4. ModelStateDict (Model-Specific State)
```csharp
public class ModelStateDict : StateDict
{
    public string ModelType { get; set; }
    public int LayerCount { get; set; }

    /// <summary>
    /// Get state for a specific layer
    /// </summary>
    public StateDict GetLayerState(string layerName)
    {
        var layerState = new StateDict();
        foreach (var (key, value) in this)
        {
            if (key.StartsWith($"{layerName}."))
            {
                var newKey = key.Substring(layerName.Length + 1);
                layerState[newKey] = value;
            }
        }
        return layerState;
    }

    /// <summary>
    /// Set state for a specific layer
    /// </summary>
    public void SetLayerState(string layerName, StateDict layerState)
    {
        foreach (var (key, value) in layerState)
        {
            this[$"{layerName}.{key}"] = value;
        }
    }
}
```

### 5. DistributedStateCollector (Collect State Across Ranks)
```csharp
public class DistributedStateCollector
{
    private readonly IDistributedCoordinator _coordinator;

    public DistributedStateCollector(IDistributedCoordinator coordinator)
    {
        _coordinator = coordinator;
    }

    /// <summary>
    /// Collect local sharded state from a model
    /// </summary>
    public StateDict CollectLocalState(IStateful model)
    {
        // For FSDP, model only has local shard
        // For DDP, model has full state duplicated
        return model.GetStateDict();
    }

    /// <summary>
    /// Merge states from multiple ranks (for load)
    /// </summary>
    public StateDict MergeStates(StateDict[] states)
    {
        var merged = new StateDict();

        foreach (var state in states)
        {
            foreach (var (key, tensor) in state)
            {
                if (merged.ContainsKey(key))
                {
                    // DDP case: all ranks have same tensors
                    // Keep first occurrence
                    continue;
                }
                merged[key] = tensor;
            }
        }

        return merged;
    }
}
```

### 6. StateUtils (Utility Functions)
```csharp
public static class StateUtils
{
    /// <summary>
    /// Check if two state dicts have matching keys
    /// </summary>
    public static bool KeysMatch(StateDict state1, StateDict state2)
    {
        var keys1 = new HashSet<string>(state1.Keys);
        var keys2 = new HashSet<string>(state2.Keys);
        return keys1.SetEquals(keys2);
    }

    /// <summary>
    /// Get missing keys from state2 compared to state1
    /// </summary>
    public static HashSet<string> GetMissingKeys(StateDict state1, StateDict state2)
    {
        var missing = new HashSet<string>(state1.Keys);
        missing.ExceptWith(state2.Keys);
        return missing;
    }

    /// <summary>
    /// Get extra keys in state2 compared to state1
    /// </summary>
    public static HashSet<string> GetUnexpectedKeys(StateDict state1, StateDict state2)
    {
        var unexpected = new HashSet<string>(state2.Keys);
        unexpected.ExceptWith(state1.Keys);
        return unexpected;
    }

    /// <summary>
    /// Verify tensor shapes match between two state dicts
    /// </summary>
    public static bool ShapesMatch(StateDict state1, StateDict state2)
    {
        foreach (var (key, tensor1) in state1)
        {
            if (!state2.TryGetValue(key, out var tensor2))
            {
                return false;
            }

            if (!tensor1.Shape.SequenceEqual(tensor2.Shape))
            {
                return false;
            }
        }
        return true;
    }
}
```

### 7. StateCompatibilityChecker (Version and Compatibility)
```csharp
public class StateCompatibilityChecker
{
    public CompatibilityResult CheckCompatibility(
        StateDict sourceState,
        StateDict targetState)
    {
        var result = new CompatibilityResult();

        // Check key match
        var missingKeys = StateUtils.GetMissingKeys(sourceState, targetState);
        var unexpectedKeys = StateUtils.GetUnexpectedKeys(sourceState, targetState);

        if (missingKeys.Count > 0)
        {
            result.AddWarning($"Missing keys: {string.Join(", ", missingKeys)}");
        }

        if (unexpectedKeys.Count > 0)
        {
            result.AddWarning($"Unexpected keys: {string.Join(", ", unexpectedKeys)}");
        }

        // Check shapes
        if (!StateUtils.ShapesMatch(sourceState, targetState))
        {
            result.AddError("Tensor shapes do not match");
        }

        return result;
    }
}
```

### 8. CompatibilityResult (Compatibility Check Result)
```csharp
public class CompatibilityResult
{
    public List<string> Errors { get; } = new();
    public List<string> Warnings { get; } = new();

    public bool IsCompatible => Errors.Count == 0;
    public bool HasWarnings => Warnings.Count > 0;

    public void AddError(string error) => Errors.Add(error);
    public void AddWarning(string warning) => Warnings.Add(warning);
}
```

## Integration Points
- Used by: `DistributedCheckpointCoordinator`, `Model`, `Optimizer`
- Depends on: `Tensor`, `IDistributedCoordinator`

## Testing Requirements
- Test state dict creation and manipulation
- Test layer state extraction/setting
- Test state merging from multiple ranks
- Test key matching and compatibility checking
- Test shape validation

## Success Criteria
- Clean abstraction for model and optimizer state
- Supports partial state updates
- Handles distributed state collection
- Provides clear compatibility checking
- Easy to use for both FSDP and DDP scenarios
