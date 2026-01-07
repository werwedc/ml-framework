# Spec: LoRA Configuration System

## Overview
Define configuration classes and data structures for LoRA hyperparameters and module targeting.

## Requirements
- Create `LoraConfig` class to hold LoRA hyperparameters
- Support common LoRA parameters (rank, alpha, dropout, target_modules)
- Provide validation for parameter values
- Enable configuration inheritance for different adapters

## Classes to Implement

### 1. LoraConfig
```csharp
public class LoraConfig
{
    /// <summary>Rank of low-rank matrices (default: 8)</summary>
    public int Rank { get; set; } = 8;

    /// <summary>LoRA scaling factor (default: 16)</summary>
    public int Alpha { get; set; } = 16;

    /// <summary>Dropout probability for LoRA layers (default: 0.0)</summary>
    public float Dropout { get; set; } = 0.0f;

    /// <summary>Target module names/patterns to inject LoRA into</summary>
    public string[] TargetModules { get; set; } = new[] { "q_proj", "v_proj" };

    /// <summary>Whether to bias LoRA layers (none, all, lora_only)</summary>
    public string Bias { get; set; } = "none";

    /// <summary>LoRA module type (default, scaled)</summary>
    public string LoraType { get; set; } = "default";

    /// <summary>Initialize with default values for common model types</summary>
    public static LoraConfig ForLLaMA() { }
    public static LoraConfig ForGPT() { }
    public static LoraConfig ForBERT() { }
}
```

### 2. ModuleTargetPattern
```csharp
public class ModuleTargetPattern
{
    /// <summary>Exact module name match</summary>
    public string ExactName { get; set; }

    /// <summary>Regex pattern for module matching</summary>
    public Regex Pattern { get; set; }

    /// <summary>Check if module name matches this pattern</summary>
    public bool Matches(string moduleName) { }
}
```

## Implementation Details
- Validate that `Rank > 0`
- Validate that `Alpha > 0`
- Validate that `0.0 <= Dropout <= 1.0`
- Support both exact names and regex patterns in `TargetModules`
- Default presets for common architectures (LLaMA, GPT, BERT)

## Deliverables
- `LoraConfig.cs` in `src/Core/LoRA/`
- Unit tests in `tests/Core/LoRA/`
