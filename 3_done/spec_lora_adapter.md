# Spec: LoRA Adapter Data Structure

## Overview
Define the `LoraAdapter` class to represent a single LoRA adapter (fine-tuned task).

## Requirements
- Store LoRA weights and configuration for a single adapter
- Support multiple LoRA modules per adapter
- Enable saving/loading adapter metadata
- Track adapter name and version

## Classes to Implement

### 1. LoraAdapter
```csharp
public class LoraAdapter
{
    /// <summary>Adapter name/identifier</summary>
    public string Name { get; set; }

    /// <summary>LoRA configuration</summary>
    public LoraConfig Config { get; set; }

    /// <summary>LoRA module weights keyed by module name</summary>
    public Dictionary<string, LoraModuleWeights> Weights { get; set; }

    /// <summary>Adapter metadata (training info, date, etc.)</summary>
    public AdapterMetadata Metadata { get; set; }

    public LoraAdapter(string name, LoraConfig config)
    {
        Name = name;
        Config = config;
        Weights = new Dictionary<string, LoraModuleWeights>();
        Metadata = new AdapterMetadata();
    }

    /// <summary>Add LoRA weights for a module</summary>
    public void AddModuleWeights(string moduleName, Tensor loraA, Tensor loraB)
    {
        Weights[moduleName] = new LoraModuleWeights
        {
            LoraA = loraA.Clone(),
            LoraB = loraB.Clone()
        };
    }

    /// <summary>Get LoRA weights for a module</summary>
    public bool TryGetModuleWeights(string moduleName, out LoraModuleWeights weights)
    {
        return Weights.TryGetValue(moduleName, out weights);
    }

    /// <summary>Get total number of parameters in this adapter</summary>
    public long GetParameterCount()
    {
        return Weights.Values.Sum(w => w.LoraA.NumElements + w.LoraB.NumElements);
    }

    /// <summary>Calculate memory size in bytes</summary>
    public long GetMemorySize()
    {
        return GetParameterCount() * sizeof(float);
    }
}

public class LoraModuleWeights
{
    public Tensor LoraA { get; set; }  // [out_features, rank]
    public Tensor LoraB { get; set; }  // [rank, in_features]
}

public class AdapterMetadata
{
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public DateTime? UpdatedAt { get; set; }
    public string BaseModel { get; set; }
    public int? TrainingEpochs { get; set; }
    public float? FinalLoss { get; set; }
    public Dictionary<string, string> CustomFields { get; set; } = new();
}
```

## Implementation Details
- Use deep cloning when adding weights (don't store references)
- Provide thread-safe access to weights dictionary
- Support serialization-friendly data structures
- Include version information for compatibility checking

## Deliverables
- `LoraAdapter.cs` in `src/Core/LoRA/`
- Unit tests in `tests/Core/LoRA/`
