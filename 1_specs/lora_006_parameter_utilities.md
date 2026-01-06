# Spec: Parameter Freezing and Counting Utilities

## Overview
Implement utilities for managing parameter states (frozen/trainable) and counting parameters in models with LoRA adapters. These utilities are essential for monitoring memory usage and optimizing training configurations.

## Implementation Details

### 1. ParameterStats Class
**File**: `src/LoRA/ParameterStats.cs`

```csharp
/// <summary>
/// Statistics about model parameters
/// </summary>
public class ParameterStats
{
    /// <summary>
    /// Total number of parameters in the model
    /// </summary>
    public long TotalParameters { get; set; }

    /// <summary>
    /// Number of trainable parameters
    /// </summary>
    public long TrainableParameters { get; set; }

    /// <summary>
    /// Number of frozen parameters
    /// </summary>
    public long FrozenParameters { get; set; }

    /// <summary>
    /// Number of LoRA adapter parameters
    /// </summary>
    public long AdapterParameters { get; set; }

    /// <summary>
    /// Total parameter count in millions
    /// </summary>
    public double TotalParametersMillions => TotalParameters / 1_000_000.0;

    /// <summary>
    /// Trainable parameter count in millions
    /// </summary>
    public double TrainableParametersMillions => TrainableParameters / 1_000_000.0;

    /// <summary>
    /// Percentage of parameters that are trainable
    /// </summary>
    public double TrainablePercentage => TotalParameters > 0
        ? (TrainableParameters * 100.0) / TotalParameters
        : 0.0;

    /// <summary>
    /// Percentage reduction in trainable parameters (compared to full fine-tuning)
    /// </summary>
    public double ReductionPercentage => TotalParameters > 0
        ? ((TotalParameters - TrainableParameters) * 100.0) / TotalParameters
        : 0.0;

    /// <summary>
    /// Estimated memory usage in MB (assuming float32)
    /// </summary>
    public double EstimatedMemoryMB => TrainableParameters * 4.0 / (1024.0 * 1024.0);

    public override string ToString()
    {
        return $"Parameters: {TotalParametersMillions:F2}M | " +
               $"Trainable: {TrainableParametersMillions:F2}M ({TrainablePercentage:F1}%) | " +
               $"Reduction: {ReductionPercentage:F1}%";
    }
}
```

### 2. ParameterManager Class
**File**: `src/LoRA/ParameterManager.cs`

```csharp
public class ParameterManager
{
    private readonly IModule _model;

    public ParameterManager(IModule model)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
    }

    /// <summary>
    /// Gets parameter statistics for the model
    /// </summary>
    public ParameterStats GetParameterStats()
    {
        var stats = new ParameterStats();
        var processedTensors = new HashSet<ITensor>();

        void ProcessModule(IModule module)
        {
            // Process LoRA adapters specially
            if (module is ILoRAAdapter adapter)
            {
                foreach (var tensor in adapter.TrainableParameters)
                {
                    if (!processedTensors.Contains(tensor))
                    {
                        stats.TrainableParameters += tensor.NumElements;
                        stats.TotalParameters += tensor.NumElements;
                        processedTensors.Add(tensor);

                        // Check if this is an adapter parameter
                        var (matrixA, matrixB) = adapter.GetAdapterWeights();
                        if (tensor == matrixA || tensor == matrixB)
                        {
                            stats.AdapterParameters += tensor.NumElements;
                        }
                    }
                }

                foreach (var tensor in adapter.FrozenParameters)
                {
                    if (!processedTensors.Contains(tensor))
                    {
                        stats.FrozenParameters += tensor.NumElements;
                        stats.TotalParameters += tensor.NumElements;
                        processedTensors.Add(tensor);
                    }
                }
            }
            else if (module is IHasParameters hasParams)
            {
                // Regular module with parameters
                foreach (var tensor in hasParams.Parameters)
                {
                    if (!processedTensors.Contains(tensor))
                    {
                        if (tensor.RequiresGrad)
                        {
                            stats.TrainableParameters += tensor.NumElements;
                        }
                        else
                        {
                            stats.FrozenParameters += tensor.NumElements;
                        }
                        stats.TotalParameters += tensor.NumElements;
                        processedTensors.Add(tensor);
                    }
                }
            }

            // Recursively process submodules
            if (module is IHasSubmodules hasSubmodules)
            {
                foreach (var (_, subModule) in hasSubmodules.NamedChildren())
                {
                    ProcessModule(subModule);
                }
            }
        }

        ProcessModule(_model);
        return stats;
    }

    /// <summary>
    /// Gets all trainable parameters in the model
    /// </summary>
    public IEnumerable<ITensor> GetTrainableParameters()
    {
        var parameters = new List<ITensor>();
        var processedTensors = new HashSet<ITensor>();

        void CollectParameters(IModule module)
        {
            if (module is ILoRAAdapter adapter)
            {
                foreach (var tensor in adapter.TrainableParameters)
                {
                    if (!processedTensors.Contains(tensor))
                    {
                        parameters.Add(tensor);
                        processedTensors.Add(tensor);
                    }
                }
            }
            else if (module is IHasParameters hasParams)
            {
                foreach (var tensor in hasParams.Parameters)
                {
                    if (tensor.RequiresGrad && !processedTensors.Contains(tensor))
                    {
                        parameters.Add(tensor);
                        processedTensors.Add(tensor);
                    }
                }
            }

            if (module is IHasSubmodules hasSubmodules)
            {
                foreach (var (_, subModule) in hasSubmodules.NamedChildren())
                {
                    CollectParameters(subModule);
                }
            }
        }

        CollectParameters(_model);
        return parameters;
    }

    /// <summary>
    /// Gets all frozen parameters in the model
    /// </summary>
    public IEnumerable<ITensor> GetFrozenParameters()
    {
        var parameters = new List<ITensor>();
        var processedTensors = new HashSet<ITensor>();

        void CollectParameters(IModule module)
        {
            if (module is ILoRAAdapter adapter)
            {
                foreach (var tensor in adapter.FrozenParameters)
                {
                    if (!processedTensors.Contains(tensor))
                    {
                        parameters.Add(tensor);
                        processedTensors.Add(tensor);
                    }
                }
            }
            else if (module is IHasParameters hasParams)
            {
                foreach (var tensor in hasParams.Parameters)
                {
                    if (!tensor.RequiresGrad && !processedTensors.Contains(tensor))
                    {
                        parameters.Add(tensor);
                        processedTensors.Add(tensor);
                    }
                }
            }

            if (module is IHasSubmodules hasSubmodules)
            {
                foreach (var (_, subModule) in hasSubmodules.NamedChildren())
                {
                    CollectParameters(subModule);
                }
            }
        }

        CollectParameters(_model);
        return parameters;
    }

    /// <summary>
    /// Freezes all parameters in the model
    /// </summary>
    public void FreezeAll()
    {
        void FreezeModule(IModule module)
        {
            if (module is ILoRAAdapter adapter)
            {
                adapter.FreezeBaseLayer();
            }
            else if (module is IHasParameters hasParams)
            {
                foreach (var tensor in hasParams.Parameters)
                {
                    tensor.RequiresGrad = false;
                }
            }

            if (module is IHasSubmodules hasSubmodules)
            {
                foreach (var (_, subModule) in hasSubmodules.NamedChildren())
                {
                    FreezeModule(subModule);
                }
            }
        }

        FreezeModule(_model);
    }

    /// <summary>
    /// Unfreezes all parameters in the model
    /// </summary>
    public void UnfreezeAll()
    {
        void UnfreezeModule(IModule module)
        {
            if (module is ILoRAAdapter adapter)
            {
                adapter.UnfreezeBaseLayer();
            }
            else if (module is IHasParameters hasParams)
            {
                foreach (var tensor in hasParams.Parameters)
                {
                    tensor.RequiresGrad = true;
                }
            }

            if (module is IHasSubmodules hasSubmodules)
            {
                foreach (var (_, subModule) in hasSubmodules.NamedChildren())
                {
                    UnfreezeModule(subModule);
                }
            }
        }

        UnfreezeModule(_model);
    }

    /// <summary>
    /// Freezes parameters matching a module name pattern
    /// </summary>
    public void FreezeByPattern(string pattern)
    {
        void ProcessModule(IModule module, string name)
        {
            if (module is ILoRAAdapter adapter && Regex.IsMatch(name, pattern))
            {
                adapter.FreezeBaseLayer();
            }
            else if (module is IHasParameters hasParams && Regex.IsMatch(name, pattern))
            {
                foreach (var tensor in hasParams.Parameters)
                {
                    tensor.RequiresGrad = false;
                }
            }

            if (module is IHasSubmodules hasSubmodules)
            {
                foreach (var (subName, subModule) in hasSubmodules.NamedChildren())
                {
                    var fullName = string.IsNullOrEmpty(name) ? subName : $"{name}.{subName}";
                    ProcessModule(subModule, fullName);
                }
            }
        }

        ProcessModule(_model, "");
    }

    /// <summary>
    /// Gets parameters grouped by learning rate
    /// </summary>
    /// <param name="baseLayerLR">Learning rate for base layers</param>
    /// <param name="adapterLR">Learning rate for adapter layers</param>
    /// <returns>Dictionary mapping learning rates to parameter groups</returns>
    public Dictionary<float, List<ITensor>> GetParameterGroups(
        float baseLayerLR = 1e-4f,
        float adapterLR = 1e-3f)
    {
        var groups = new Dictionary<float, List<ITensor>>
        {
            [baseLayerLR] = new List<ITensor>(),
            [adapterLR] = new List<ITensor>()
        };

        var processedTensors = new HashSet<ITensor>();

        void ProcessModule(IModule module)
        {
            if (module is ILoRAAdapter adapter)
            {
                // Adapter parameters use adapterLR
                var (matrixA, matrixB) = adapter.GetAdapterWeights();
                if (matrixA != null && !processedTensors.Contains(matrixA))
                {
                    groups[adapterLR].Add(matrixA);
                    processedTensors.Add(matrixA);
                }
                if (matrixB != null && !processedTensors.Contains(matrixB))
                {
                    groups[adapterLR].Add(matrixB);
                    processedTensors.Add(matrixB);
                }

                // Base layer parameters use baseLayerLR (if trainable)
                foreach (var tensor in adapter.TrainableParameters)
                {
                    if (!processedTensors.Contains(tensor) && tensor != matrixA && tensor != matrixB)
                    {
                        groups[baseLayerLR].Add(tensor);
                        processedTensors.Add(tensor);
                    }
                }
            }

            if (module is IHasSubmodules hasSubmodules)
            {
                foreach (var (_, subModule) in hasSubmodules.NamedChildren())
                {
                    ProcessModule(subModule);
                }
            }
        }

        ProcessModule(_model);
        return groups;
    }
}
```

### 3. IHasParameters Interface
**File**: `src/LoRA/IHasParameters.cs`

```csharp
/// <summary>
/// Interface for modules that have parameters
/// </summary>
public interface IHasParameters
{
    /// <summary>
    /// Gets all parameters in the module
    /// </summary>
    IEnumerable<ITensor> Parameters { get; }
}
```

### 4. Extension Methods
**File**: `src/LoRA/ParameterExtensions.cs`

```csharp
public static class ParameterExtensions
{
    /// <summary>
    /// Gets parameter statistics for a model
    /// </summary>
    public static ParameterStats GetParameterStats(this IModule model)
    {
        var manager = new ParameterManager(model);
        return manager.GetParameterStats();
    }

    /// <summary>
    /// Prints parameter statistics to console
    /// </summary>
    public static void PrintParameterStats(this IModule model, string modelPrefix = "")
    {
        var stats = model.GetParameterStats();
        var prefix = string.IsNullOrEmpty(modelPrefix) ? "Model" : modelPrefix;
        Console.WriteLine($"{prefix} {stats}");
    }

    /// <summary>
    /// Freezes all parameters in a model
    /// </summary>
    public static void FreezeParameters(this IModule model)
    {
        var manager = new ParameterManager(model);
        manager.FreezeAll();
    }

    /// <summary>
    /// Unfreezes all parameters in a model
    /// </summary>
    public static void UnfreezeParameters(this IModule model)
    {
        var manager = new ParameterManager(model);
        manager.UnfreezeAll();
    }
}
```

## Testing Requirements

**File**: `tests/LoRA/ParameterManagerTests.cs`

1. **Parameter Counting Tests**
   - Test counting parameters in simple model
   - Test counting parameters in model with LoRA adapters
   - Test correct separation of trainable/frozen parameters
   - Test adapter parameter counting

2. **Statistics Tests**
   - Test ParameterStats calculations
   - Test percentage calculations
   - Test memory estimation
   - Test ToString formatting

3. **Freeze/Unfreeze Tests**
   - Test FreezeAll freezes all parameters
   - Test UnfreezeAll unfreezes all parameters
   - Test FreezeByPattern with regex patterns

4. **Parameter Groups Tests**
   - Test GetParameterGroups with different learning rates
   - Test correct grouping of adapter vs base parameters
   - Test different learning rate configurations

5. **Extension Method Tests**
   - Test GetParameterStats extension
   - Test PrintParameterStats extension
   - Test FreezeParameters/UnfreezeParameters extensions

## Dependencies
- IModule interface (existing)
- ITensor interface (existing)
- ILoRAAdapter interface (from spec 001)
- Linear, Conv2d, Embedding layers (existing)

## Success Criteria
- ParameterManager correctly counts all parameters
- Stats calculations are accurate
- Freeze/unfreeze operations work correctly
- Parameter grouping for different learning rates works
- All unit tests pass

## Estimated Time
45 minutes

## Notes
- Ensure no double-counting of shared parameters
- Consider adding parameter type tracking (weights, biases, embeddings)
- Memory estimation assumes float32; could add FP16/BF16 support
- Consider adding parameter shape information for debugging
