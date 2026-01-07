# Spec: Adapter Manager

## Overview
Implement `AdapterManager` to load, save, switch, and manage multiple LoRA adapters on a model.

## Requirements
- Load adapters from disk onto a model
- Save adapters to disk separately from base model
- Switch between active adapters at runtime
- Support multiple adapters loaded simultaneously
- List and manage available adapters

## Classes to Implement

### 1. AdapterManager
```csharp
public class AdapterManager
{
    private readonly IModule _baseModel;
    private readonly Dictionary<string, LoraAdapter> _loadedAdapters;
    private readonly HashSet<string> _activeAdapters;
    private readonly LoraConfig _defaultConfig;

    public AdapterManager(IModule baseModel, LoraConfig config = null)
    {
        _baseModel = baseModel;
        _defaultConfig = config ?? new LoraConfig();
        _loadedAdapters = new Dictionary<string, LoraAdapter>();
        _activeAdapters = new HashSet<string>();
    }

    /// <summary>Load adapter from file</summary>
    public void LoadAdapter(string name, string path)
    {
        var adapter = AdapterSerializer.Load(path);
        _loadedAdapters[name] = adapter;
    }

    /// <summary>Load adapter from LoraAdapter object</summary>
    public void LoadAdapter(LoraAdapter adapter)
    {
        _loadedAdapters[adapter.Name] = adapter;
    }

    /// <summary>Save adapter to disk</summary>
    public void SaveAdapter(string name, string path)
    {
        if (!_loadedAdapters.TryGetValue(name, out var adapter))
        {
            throw new ArgumentException($"Adapter '{name}' not loaded");
        }

        // Extract current weights from model
        var currentWeights = ExtractAdapterWeights(name);
        adapter.Weights = currentWeights;

        AdapterSerializer.Save(adapter, path);
    }

    /// <summary>Set active adapter(s)</summary>
    public void SetActiveAdapter(params string[] names)
    {
        _activeAdapters.Clear();
        foreach (var name in names)
        {
            if (!_loadedAdapters.ContainsKey(name))
            {
                throw new ArgumentException($"Adapter '{name}' not loaded");
            }
            _activeAdapters.Add(name);
        }

        ApplyActiveAdapters();
    }

    /// <summary>Add adapter to active set (multi-adapter)</summary>
    public void ActivateAdapter(string name)
    {
        if (!_loadedAdapters.ContainsKey(name))
        {
            throw new ArgumentException($"Adapter '{name}' not loaded");
        }
        _activeAdapters.Add(name);
        ApplyActiveAdapters();
    }

    /// <summary>Remove adapter from active set</summary>
    public void DeactivateAdapter(string name)
    {
        _activeAdapters.Remove(name);
        ApplyActiveAdapters();
    }

    /// <summary>List all loaded adapter names</summary>
    public IReadOnlyList<string> ListAdapters()
    {
        return _loadedAdapters.Keys.ToList();
    }

    /// <summary>List active adapter names</summary>
    public IReadOnlyList<string> ListActiveAdapters()
    {
        return _activeAdapters.ToList();
    }

    /// <summary>Unload adapter</summary>
    public void UnloadAdapter(string name)
    {
        _activeAdapters.Remove(name);
        _loadedAdapters.Remove(name);
        ApplyActiveAdapters();
    }

    /// <summary>Get adapter by name</summary>
    public LoraAdapter GetAdapter(string name)
    {
        return _loadedAdapters[name];
    }

    private void ApplyActiveAdapters()
    {
        // Apply active adapter weights to model's LoRA layers
        foreach (var loraLayer in FindLoRALayers(_baseModel))
        {
            var moduleName = GetModuleName(loraLayer);

            // Reset to zeros first
            loraLayer.ResetLoRA();

            // Accumulate weights from all active adapters
            foreach (var adapterName in _activeAdapters)
            {
                var adapter = _loadedAdapters[adapterName];
                if (adapter.TryGetModuleWeights(moduleName, out var weights))
                {
                    loraLayer.AddLoRAWeights(weights.LoraA, weights.LoraB);
                }
            }
        }
    }

    private IEnumerable<LoraLinear> FindLoRALayers(IModule model)
    {
        // Recursively find all LoraLinear layers in the model
        // Implementation depends on module traversal API
    }

    private string GetModuleName(LoraLinear layer)
    {
        // Get the module path/name for the layer
        // Implementation depends on module naming API
    }

    private Dictionary<string, LoraModuleWeights> ExtractAdapterWeights(string name)
    {
        // Extract current LoRA weights from model for a specific adapter
        var weights = new Dictionary<string, LoraModuleWeights>();

        foreach (var loraLayer in FindLoRALayers(_baseModel))
        {
            weights[GetModuleName(loraLayer)] = loraLayer.GetLoRAWeights();
        }

        return weights;
    }
}
```

## Implementation Details
- Support both single and multiple active adapters (for style transfer, etc.)
- Thread-safe adapter switching
- Handle adapter compatibility (rank, target modules must match)
- Provide clear error messages for missing/invalid adapters
- Support hot-swapping without model recompilation

## Deliverables
- `AdapterManager.cs` in `src/Core/LoRA/`
- Unit tests in `tests/Core/LoRA/`
