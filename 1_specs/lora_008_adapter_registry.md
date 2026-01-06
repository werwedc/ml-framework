# Spec: Adapter Registry Implementation

## Overview
Implement the adapter registry for managing multiple LoRA adapters. This allows users to load, save, switch between, and manage different adapters for the same base model - essential for multi-tenancy and task switching.

## Implementation Details

### 1. AdapterMetadata Class
**File**: `src/LoRA/AdapterMetadata.cs`

```csharp
/// <summary>
/// Metadata about a LoRA adapter
/// </summary>
public class AdapterMetadata
{
    /// <summary>
    /// Unique identifier for the adapter
    /// </summary>
    public string Id { get; set; } = string.Empty;

    /// <summary>
    /// Human-readable name
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Description of what the adapter does
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Rank of the adapter
    /// </summary>
    public int Rank { get; set; }

    /// <summary>
    /// Alpha scaling factor
    /// </summary>
    public float Alpha { get; set; }

    /// <summary>
    /// Base model architecture
    /// </summary>
    public string BaseModel { get; set; } = string.Empty;

    /// <summary>
    /// Task type (e.g., "text-generation", "classification", "translation")
    /// </summary>
    public string TaskType { get; set; } = string.Empty;

    /// <summary>
    /// Training data used (optional)
    /// </summary>
    public string TrainingData { get; set; } = string.Empty;

    /// <summary>
    /// Creation timestamp
    /// </summary>
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Last modified timestamp
    /// </summary>
    public DateTime ModifiedAt { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Number of training steps
    /// </summary>
    public int TrainingSteps { get; set; }

    /// <summary>
    /// Performance metrics (e.g., accuracy, loss)
    /// </summary>
    public Dictionary<string, float> Metrics { get; set; } = new();

    /// <summary>
    /// Additional custom metadata
    /// </summary>
    public Dictionary<string, string> Custom { get; set; } = new();
}
```

### 2. AdapterState Class
**File**: `src/LoRA/AdapterState.cs`

```csharp
/// <summary>
/// Represents the complete state of a LoRA adapter
/// </summary>
public class AdapterState
{
    /// <summary>
    /// Adapter metadata
    /// </summary>
    public AdapterMetadata Metadata { get; set; } = new();

    /// <summary>
    /// LoRA configuration used
    /// </summary>
    public LoRAConfig Config { get; set; } = new();

    /// <summary>
    /// Adapter weights indexed by module name
    /// Key: module name, Value: (MatrixA, MatrixB, Bias)
    /// </summary>
    public Dictionary<string, AdapterWeights> Weights { get; set; } = new();

    /// <summary>
    /// Optional optimizer state for resuming training
    /// </summary>
    public Dictionary<string, ITensor>? OptimizerState { get; set; }
}

/// <summary>
/// Adapter weights for a single layer
/// </summary>
public class AdapterWeights
{
    /// <summary>
    /// Matrix A (low-rank decomposition)
    /// </summary>
    public ITensor MatrixA { get; set; } = null!;

    /// <summary>
    /// Matrix B (low-rank decomposition)
    /// </summary>
    public ITensor MatrixB { get; set; } = null!;

    /// <summary>
    /// Optional bias adapter
    /// </summary>
    public ITensor? Bias { get; set; }

    /// <summary>
    /// Layer type (Linear, Conv2d, Embedding)
    /// </summary>
    public string LayerType { get; set; } = string.Empty;
}
```

### 3. LoRAAdapterRegistry Class
**File**: `src/LoRA/LoRAAdapterRegistry.cs`

```csharp
/// <summary>
/// Registry for managing multiple LoRA adapters
/// </summary>
public class LoRAAdapterRegistry
{
    private readonly Dictionary<string, AdapterState> _adapters;
    private readonly string _basePath;
    private IModule? _model;

    public LoRAAdapterRegistry(string basePath = "./adapters")
    {
        _basePath = basePath ?? throw new ArgumentNullException(nameof(basePath));
        _adapters = new Dictionary<string, AdapterState>();

        // Ensure directory exists
        Directory.CreateDirectory(_basePath);
    }

    /// <summary>
    /// Sets the model for this registry
    /// </summary>
    public void SetModel(IModule model)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
    }

    /// <summary>
    /// Saves the current adapter state
    /// </summary>
    /// <param name="adapterId">Unique identifier for the adapter</param>
    /// <param name="metadata">Optional metadata</param>
    public void SaveAdapter(string adapterId, AdapterMetadata? metadata = null)
    {
        if (_model == null)
            throw new InvalidOperationException("Model not set. Call SetModel() first.");

        // Extract current adapter state
        var state = ExtractAdapterState(_model, adapterId, metadata);

        // Save to memory registry
        _adapters[adapterId] = state;

        // Save to disk
        SaveToDisk(adapterId, state);
    }

    /// <summary>
    /// Loads an adapter into the model
    /// </summary>
    /// <param name="adapterId">ID of the adapter to load</param>
    public void LoadAdapter(string adapterId)
    {
        if (_model == null)
            throw new InvalidOperationException("Model not set. Call SetModel() first.");

        // Load from memory or disk
        AdapterState state;
        if (_adapters.TryGetValue(adapterId, out var cachedState))
        {
            state = cachedState;
        }
        else
        {
            state = LoadFromDisk(adapterId);
            _adapters[adapterId] = state;
        }

        // Apply to model
        ApplyAdapterState(_model, state);
    }

    /// <summary>
    /// Gets metadata for an adapter
    /// </summary>
    public AdapterMetadata? GetAdapterMetadata(string adapterId)
    {
        if (_adapters.TryGetValue(adapterId, out var state))
        {
            return state.Metadata;
        }

        // Try loading from disk
        var diskState = LoadFromDisk(adapterId);
        _adapters[adapterId] = diskState;
        return diskState.Metadata;
    }

    /// <summary>
    /// Lists all available adapters
    /// </summary>
    public List<AdapterMetadata> ListAdapters()
    {
        var allMetadata = new List<AdapterMetadata>();

        // Add from memory
        foreach (var (_, state) in _adapters)
        {
            allMetadata.Add(state.Metadata);
        }

        // Add from disk (not in memory)
        if (Directory.Exists(_basePath))
        {
            var adapterDirs = Directory.GetDirectories(_basePath);
            foreach (var dir in adapterDirs)
            {
                var adapterId = Path.GetFileName(dir);
                if (!_adapters.ContainsKey(adapterId))
                {
                    try
                    {
                        var state = LoadFromDisk(adapterId);
                        allMetadata.Add(state.Metadata);
                    }
                    catch
                    {
                        // Skip invalid adapters
                        continue;
                    }
                }
            }
        }

        return allMetadata;
    }

    /// <summary>
    /// Deletes an adapter from registry and disk
    /// </summary>
    public void DeleteAdapter(string adapterId)
    {
        // Remove from memory
        _adapters.Remove(adapterId);

        // Remove from disk
        var adapterPath = Path.Combine(_basePath, adapterId);
        if (Directory.Exists(adapterPath))
        {
            Directory.Delete(adapterPath, recursive: true);
        }
    }

    /// <summary>
    /// Exports an adapter to a file
    /// </summary>
    public void ExportAdapter(string adapterId, string outputPath)
    {
        var state = _adapters.TryGetValue(adapterId, out var cachedState)
            ? cachedState
            : LoadFromDisk(adapterId);

        // Serialize and save
        using var stream = File.OpenWrite(outputPath);
        SerializeAdapter(state, stream);
    }

    /// <summary>
    /// Imports an adapter from a file
    /// </summary>
    public void ImportAdapter(string adapterId, string inputPath)
    {
        using var stream = File.OpenRead(inputPath);
        var state = DeserializeAdapter(stream);

        // Save to registry
        _adapters[adapterId] = state;
        SaveToDisk(adapterId, state);
    }

    private AdapterState ExtractAdapterState(IModule model, string adapterId, AdapterMetadata? metadata)
    {
        var state = new AdapterState
        {
            Metadata = metadata ?? new AdapterMetadata
            {
                Id = adapterId,
                Name = adapterId,
                CreatedAt = DateTime.UtcNow,
                ModifiedAt = DateTime.UtcNow
            }
        };

        state.Metadata.Rank = GetAdapterRank(model);
        state.Metadata.Alpha = GetAdapterAlpha(model);

        void ExtractFromModule(IModule module, string name)
        {
            if (module is ILoRAAdapter adapter)
            {
                var (matrixA, matrixB) = adapter.GetAdapterWeights();

                var weights = new AdapterWeights
                {
                    MatrixA = matrixA!.Clone(),
                    MatrixB = matrixB!.Clone(),
                    LayerType = module.GetType().Name
                };

                // Extract bias if present
                if (adapter is LoRALinear linearAdapter)
                {
                    weights.Bias = linearAdapter.GetBias()?.Clone();
                }

                state.Weights[name] = weights;
            }

            if (module is IHasSubmodules hasSubmodules)
            {
                foreach (var (subName, subModule) in hasSubmodules.NamedChildren())
                {
                    var fullName = string.IsNullOrEmpty(name) ? subName : $"{name}.{subName}";
                    ExtractFromModule(subModule, fullName);
                }
            }
        }

        ExtractFromModule(model, "");
        return state;
    }

    private void ApplyAdapterState(IModule model, AdapterState state)
    {
        void ApplyToModule(IModule module, string name)
        {
            if (module is ILoRAAdapter adapter && state.Weights.TryGetValue(name, out var weights))
            {
                adapter.SetAdapterWeights(weights.MatrixA, weights.MatrixB);

                if (weights.Bias != null && adapter is LoRALinear linearAdapter)
                {
                    linearAdapter.SetBias(weights.Bias);
                }
            }

            if (module is IHasSubmodules hasSubmodules)
            {
                foreach (var (subName, subModule) in hasSubmodules.NamedChildren())
                {
                    var fullName = string.IsNullOrEmpty(name) ? subName : $"{name}.{subName}";
                    ApplyToModule(subModule, fullName);
                }
            }
        }

        ApplyToModule(model, "");
    }

    private void SaveToDisk(string adapterId, AdapterState state)
    {
        var adapterPath = Path.Combine(_basePath, adapterId);
        Directory.CreateDirectory(adapterPath);

        // Save metadata
        var metadataPath = Path.Combine(adapterPath, "metadata.json");
        var metadataJson = JsonSerializer.Serialize(state.Metadata, new JsonSerializerOptions
        {
            WriteIndented = true
        });
        File.WriteAllText(metadataPath, metadataJson);

        // Save weights
        var weightsPath = Path.Combine(adapterPath, "weights.bin");
        using var stream = File.OpenWrite(weightsPath);
        SerializeAdapter(state, stream);
    }

    private AdapterState LoadFromDisk(string adapterId)
    {
        var adapterPath = Path.Combine(_basePath, adapterId);

        if (!Directory.Exists(adapterPath))
            throw new FileNotFoundException($"Adapter not found: {adapterId}");

        var state = new AdapterState();

        // Load metadata
        var metadataPath = Path.Combine(adapterPath, "metadata.json");
        var metadataJson = File.ReadAllText(metadataPath);
        state.Metadata = JsonSerializer.Deserialize<AdapterMetadata>(metadataJson)
            ?? throw new InvalidOperationException("Failed to load metadata");

        // Load weights
        var weightsPath = Path.Combine(adapterPath, "weights.bin");
        using var stream = File.OpenRead(weightsPath);
        state = DeserializeAdapter(stream);
        state.Metadata = state.Metadata; // Preserve loaded metadata

        return state;
    }

    private void SerializeAdapter(AdapterState state, Stream stream)
    {
        // Serialize weights to binary format
        // Implementation depends on framework's serialization utilities
        // This is a placeholder - actual implementation varies by framework

        using var writer = new BinaryWriter(stream);
        writer.Write(state.Weights.Count);

        foreach (var (name, weights) in state.Weights)
        {
            writer.Write(name);
            writer.Write(weights.LayerType);
            // Serialize tensors (framework-specific)
            // ...
        }
    }

    private AdapterState DeserializeAdapter(Stream stream)
    {
        // Deserialize weights from binary format
        // Implementation depends on framework's deserialization utilities
        // This is a placeholder - actual implementation varies by framework

        var state = new AdapterState();
        using var reader = new BinaryReader(stream);

        int count = reader.ReadInt32();
        for (int i = 0; i < count; i++)
        {
            var name = reader.ReadString();
            var layerType = reader.ReadString();
            // Deserialize tensors (framework-specific)
            // ...
        }

        return state;
    }

    private int GetAdapterRank(IModule model)
    {
        var adapter = model.GetLoRAAdapters().FirstOrDefault();
        return adapter?.Rank ?? 0;
    }

    private float GetAdapterAlpha(IModule model)
    {
        var adapter = model.GetLoRAAdapters().FirstOrDefault();
        return adapter?.ScalingFactor * (adapter?.Rank ?? 1) ?? 0;
    }
}
```

## Testing Requirements

**File**: `tests/LoRA/AdapterRegistryTests.cs`

1. **Save/Load Tests**
   - Test SaveAdapter stores adapter correctly
   - Test LoadAdapter applies weights to model
   - Test persistence across registry instances

2. **Metadata Tests**
   - Test GetAdapterMetadata returns correct info
   - Test metadata serialization/deserialization
   - Test custom metadata storage

3. **List/Delete Tests**
   - Test ListAdapters returns all adapters
   - Test DeleteAdapter removes from registry and disk
   - Test import/export functionality

4. **Multi-Adapter Tests**
   - Test managing multiple adapters
   - Test switching between adapters
   - Test adapter ID uniqueness

## Dependencies
- IModule interface (existing)
- ILoRAAdapter interface (from spec 001)
- Tensor serialization utilities (framework-specific)
- System.Text.Json (for metadata)

## Success Criteria
- AdapterRegistry correctly saves and loads adapter states
- Metadata is properly persisted
- Multiple adapters can be managed
- Import/export works correctly
- All unit tests pass

## Estimated Time
60 minutes

## Notes
- Serialization implementation depends on framework capabilities
- Consider adding compression for large adapters
- Add validation to ensure adapter matches model architecture
- Consider adding adapter versioning for compatibility
