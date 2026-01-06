# Spec: Model Conversion API and LoRA Injection

## Overview
Implement the model conversion API that automatically injects LoRA adapters into existing models. This provides a high-level, user-friendly interface for applying LoRA to pretrained models without manual layer wrapping.

## Implementation Details

### 1. LoRAInjector Class
**File**: `src/LoRA/LoRAInjector.cs`

```csharp
public class LoRAInjector
{
    private readonly LoRAConfig _config;
    private readonly List<ILoRAAdapter> _injectedAdapters;

    public LoRAInjector(LoRAConfig config)
    {
        _config = config ?? throw new ArgumentNullException(nameof(config));
        _injectedAdapters = new List<ILoRAAdapter>();
    }

    /// <summary>
    /// Injects LoRA adapters into a model based on configuration
    /// </summary>
    /// <returns>The number of adapters injected</returns>
    public int ApplyLoRA(IModule model)
    {
        _injectedAdapters.Clear();
        int count = 0;

        // Traverse model hierarchy and inject adapters
        count += InjectIntoModule(model, "");

        return count;
    }

    private int InjectIntoModule(IModule module, string name)
    {
        int count = 0;

        // Handle named submodules (Sequential, ModuleDict, etc.)
        if (module is IHasSubmodules hasSubmodules)
        {
            foreach (var (subName, subModule) in hasSubmodules.NamedChildren())
            {
                var fullName = string.IsNullOrEmpty(name) ? subName : $"{name}.{subName}";
                count += InjectIntoModule(subModule, fullName);
            }
        }

        // Check if this module should be wrapped with LoRA
        if (ShouldInject(module, name))
        {
            var adapter = CreateAdapter(module, name);
            ReplaceModule(module, adapter, name);
            _injectedAdapters.Add(adapter);
            count++;
        }

        return count;
    }

    private bool ShouldInject(IModule module, string name)
    {
        // Check if module is a supported layer type
        bool isSupportedType = module is Linear || module is Conv2d || module is Embedding;
        if (!isSupportedType)
            return false;

        // Check if target layer types are specified
        if (_config.TargetLayerTypes != null && _config.TargetLayerTypes.Length > 0)
        {
            string moduleType = module.GetType().Name;
            if (!_config.TargetLayerTypes.Contains(moduleType))
                return false;
        }

        // Check if target modules are specified
        if (_config.TargetModules != null && _config.TargetModules.Length > 0)
        {
            // Check if module name matches any target pattern
            bool matchesTarget = _config.TargetModules.Any(target =>
                name == target || name.EndsWith("." + target));

            if (!matchesTarget)
                return false;
        }

        // Don't wrap already wrapped modules
        if (module is ILoRAAdapter)
            return false;

        return true;
    }

    private ILoRAAdapter CreateAdapter(IModule module, string name)
    {
        switch (module)
        {
            case Linear linear:
                return new LoRALinear(
                    linear,
                    _config.Rank,
                    _config.Alpha,
                    _config.Initialization,
                    _config.Dropout,
                    _config.UseBias
                );

            case Conv2d conv:
                return new LoRAConv2d(
                    conv,
                    _config.Rank,
                    _config.Alpha,
                    _config.Initialization,
                    _config.Dropout,
                    _config.UseBias
                );

            case Embedding embedding:
                return new LoRAEmbedding(
                    embedding,
                    _config.Rank,
                    _config.Alpha,
                    _config.Initialization,
                    _config.Dropout
                );

            default:
                throw new ArgumentException($"Unsupported module type: {module.GetType().Name}");
        }
    }

    private void ReplaceModule(IModule original, ILoRAAdapter adapter, string name)
    {
        // This implementation depends on the framework's module hierarchy
        // For models with IHasSubmodules, we need to replace in the parent

        if (original.Parent is IHasSubmodules parent)
        {
            // Find the name of this module in its parent
            var childName = parent.NamedChildren()
                .FirstOrDefault(x => x.Item2 == original).Item1;

            if (childName != null)
            {
                parent.SetModule(childName, adapter);
            }
        }
    }

    /// <summary>
    /// Gets all injected LoRA adapters
    /// </summary>
    public IReadOnlyList<ILoRAAdapter> InjectedAdapters => _injectedAdapters.AsReadOnly();

    /// <summary>
    /// Freezes all base layers in injected adapters
    /// </summary>
    public void FreezeAllBaseLayers()
    {
        foreach (var adapter in _injectedAdapters)
        {
            adapter.FreezeBaseLayer();
        }
    }

    /// <summary>
    /// Unfreezes all base layers in injected adapters
    /// </summary>
    public void UnfreezeAllBaseLayers()
    {
        foreach (var adapter in _injectedAdapters)
        {
            adapter.UnfreezeBaseLayer();
        }
    }

    /// <summary>
    /// Gets all trainable parameters (adapter + unfrozen base)
    /// </summary>
    public IEnumerable<ITensor> GetTrainableParameters()
    {
        return _injectedAdapters.SelectMany(adapter => adapter.TrainableParameters);
    }

    /// <summary>
    /// Gets all frozen parameters
    /// </summary>
    public IEnumerable<ITensor> GetFrozenParameters()
    {
        return _injectedAdapters.SelectMany(adapter => adapter.FrozenParameters);
    }

    /// <summary>
    /// Enables all LoRA adapters
    /// </summary>
    public void EnableAllAdapters()
    {
        foreach (var adapter in _injectedAdapters)
        {
            adapter.IsEnabled = true;
        }
    }

    /// <summary>
    /// Disables all LoRA adapters
    /// </summary>
    public void DisableAllAdapters()
    {
        foreach (var adapter in _injectedAdapters)
        {
            adapter.IsEnabled = false;
        }
    }
}
```

### 2. IHasSubmodules Interface
**File**: `src/LoRA/IHasSubmodules.cs`

```csharp
/// <summary>
/// Interface for modules that contain submodules
/// </summary>
public interface IHasSubmodules
{
    /// <summary>
    /// Gets all named children modules
    /// </summary>
    IEnumerable<(string Name, IModule Module)> NamedChildren();

    /// <summary>
    /// Sets a named child module
    /// </summary>
    void SetModule(string name, IModule module);
}
```

### 3. Extension Methods for Models
**File**: `src/LoRA/ModelExtensions.cs`

```csharp
public static class ModelExtensions
{
    /// <summary>
    /// Applies LoRA to a model with the given configuration
    /// </summary>
    public static LoRAInjector ApplyLoRA(this IModule model, LoRAConfig config)
    {
        var injector = new LoRAInjector(config);
        injector.ApplyLoRA(model);
        return injector;
    }

    /// <summary>
    /// Applies LoRA to a model with simple configuration
    /// </summary>
    public static LoRAInjector ApplyLoRA(this IModule model, int rank = 8, float alpha = 16.0f)
    {
        var config = new LoRAConfig(rank, alpha);
        return model.ApplyLoRA(config);
    }

    /// <summary>
    /// Applies LoRA to specific modules only
    /// </summary>
    public static LoRAInjector ApplyLoRAToModules(
        this IModule model,
        int rank,
        float alpha,
        string[] targetModules)
    {
        var config = new LoRAConfig(rank, alpha)
        {
            TargetModules = targetModules
        };
        return model.ApplyLoRA(config);
    }

    /// <summary>
    /// Applies LoRA to specific layer types only
    /// </summary>
    public static LoRAInjector ApplyLoRAToLayerTypes(
        this IModule model,
        int rank,
        float alpha,
        string[] targetLayerTypes)
    {
        var config = new LoRAConfig(rank, alpha)
        {
            TargetLayerTypes = targetLayerTypes
        };
        return model.ApplyLoRA(config);
    }

    /// <summary>
    /// Gets all LoRA adapters in a model
    /// </summary>
    public static IEnumerable<ILoRAAdapter> GetLoRAAdapters(this IModule model)
    {
        var adapters = new List<ILoRAAdapter>();

        void Traverse(IModule module)
        {
            if (module is ILoRAAdapter adapter)
            {
                adapters.Add(adapter);
            }

            if (module is IHasSubmodules hasSubmodules)
            {
                foreach (var (_, subModule) in hasSubmodules.NamedChildren())
                {
                    Traverse(subModule);
                }
            }
        }

        Traverse(model);
        return adapters;
    }

    /// <summary>
    /// Freezes all LoRA base layers in a model
    /// </summary>
    public static void FreezeLoRABaseLayers(this IModule model)
    {
        foreach (var adapter in model.GetLoRAAdapters())
        {
            adapter.FreezeBaseLayer();
        }
    }

    /// <summary>
    /// Unfreezes all LoRA base layers in a model
    /// </summary>
    public static void UnfreezeLoRABaseLayers(this IModule model)
    {
        foreach (var adapter in model.GetLoRAAdapters())
        {
            adapter.UnfreezeBaseLayer();
        }
    }
}
```

### 4. Module Parent Property
**File**: `src/LoRA/IModule.cs` (extension to existing interface)

Add to existing IModule interface:

```csharp
public interface IModule
{
    // ... existing members ...

    /// <summary>
    /// Gets the parent module (null for root)
    /// </summary>
    IModule? Parent { get; set; }
}
```

## Testing Requirements

**File**: `tests/LoRA/LoRAInjectorTests.cs`

1. **Injection Tests**
   - Test injection into simple Linear-only model
   - Test injection into complex hierarchical model
   - Test selective module targeting (by name)
   - Test selective layer type targeting

2. **Configuration Tests**
   - Test with various rank/alpha values
   - Test with dropout enabled
   - Test with bias enabled
   - Test with different initialization strategies

3. **Parameter Management Tests**
   - Test GetTrainableParameters returns correct tensors
   - Test GetFrozenParameters returns correct tensors
   - Test FreezeAllBaseLayers/UnfreezeAllBaseLayers

4. **Adapter Control Tests**
   - Test EnableAllAdapters/DisableAllAdapters
   - Test adapter state persists across calls

5. **Model Extension Tests**
   - Test ApplyLoRA extension methods
   - Test GetLoRAAdapters extension method
   - Test FreezeLoRABaseLayers extension method

## Dependencies
- All LoRA adapter implementations (from specs 002-004)
- LoRAConfig (from spec 001)
- IModule interface (existing)
- Linear, Conv2d, Embedding layers (existing)

## Success Criteria
- LoRAInjector correctly traverses model hierarchy
- Adapters are injected into correct layers based on configuration
- Module replacement works correctly in parent-child relationships
- Extension methods provide convenient API
- All unit tests pass

## Estimated Time
60 minutes

## Notes
- Implementation depends on framework's module hierarchy (Sequential, ModuleDict, etc.)
- Consider supporting common model architectures (GPT, BERT, ViT) with preset configurations
- Ensure parent-child relationships are maintained after injection
- Consider thread safety if used in multi-threaded environments
- Add logging/debugging options for troubleshooting injection
