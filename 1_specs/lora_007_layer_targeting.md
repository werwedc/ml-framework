# Spec: Layer Targeting Utilities

## Overview
Implement utilities for targeting specific layers in a model by name, type, or pattern. This provides flexible control over which layers receive LoRA adapters, supporting common patterns like "only attention Q/K/V matrices" or "all linear layers except output."

## Implementation Details

### 1. LayerTargetSelector Class
**File**: `src/LoRA/LayerTargetSelector.cs`

```csharp
/// <summary>
/// Provides utilities for selecting target layers in a model
/// </summary>
public class LayerTargetSelector
{
    private readonly IModule _model;

    public LayerTargetSelector(IModule model)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
    }

    /// <summary>
    /// Gets all module names matching a pattern
    /// </summary>
    public List<string> FindModulesByPattern(string pattern)
    {
        var matches = new List<string>();

        void SearchModule(IModule module, string name)
        {
            if (Regex.IsMatch(name, pattern))
            {
                matches.Add(name);
            }

            if (module is IHasSubmodules hasSubmodules)
            {
                foreach (var (subName, subModule) in hasSubmodules.NamedChildren())
                {
                    var fullName = string.IsNullOrEmpty(name) ? subName : $"{name}.{subName}";
                    SearchModule(subModule, fullName);
                }
            }
        }

        SearchModule(_model, "");
        return matches;
    }

    /// <summary>
    /// Gets all modules of a specific type
    /// </summary>
    public List<(string Name, IModule Module)> FindModulesByType<T>()
        where T : IModule
    {
        var matches = new List<(string Name, IModule Module)>();

        void SearchModule(IModule module, string name)
        {
            if (module is T)
            {
                matches.Add((name, module));
            }

            if (module is IHasSubmodules hasSubmodules)
            {
                foreach (var (subName, subModule) in hasSubmodules.NamedChildren())
                {
                    var fullName = string.IsNullOrEmpty(name) ? subName : $"{name}.{subName}";
                    SearchModule(subModule, fullName);
                }
            }
        }

        SearchModule(_model, "");
        return matches;
    }

    /// <summary>
    /// Gets all Linear layers in the model
    /// </summary>
    public List<(string Name, Linear Module)> FindLinearLayers()
    {
        var linearLayers = new List<(string Name, Linear Module)>();

        void SearchModule(IModule module, string name)
        {
            if (module is Linear linear)
            {
                linearLayers.Add((name, linear));
            }

            if (module is IHasSubmodules hasSubmodules)
            {
                foreach (var (subName, subModule) in hasSubmodules.NamedChildren())
                {
                    var fullName = string.IsNullOrEmpty(name) ? subName : $"{name}.{subName}";
                    SearchModule(subModule, fullName);
                }
            }
        }

        SearchModule(_model, "");
        return linearLayers;
    }

    /// <summary>
    /// Gets all Conv2d layers in the model
    /// </summary>
    public List<(string Name, Conv2d Module)> FindConv2dLayers()
    {
        var convLayers = new List<(string Name, Conv2d Module)>();

        void SearchModule(IModule module, string name)
        {
            if (module is Conv2d conv)
            {
                convLayers.Add((name, conv));
            }

            if (module is IHasSubmodules hasSubmodules)
            {
                foreach (var (subName, subModule) in hasSubmodules.NamedChildren())
                {
                    var fullName = string.IsNullOrEmpty(name) ? subName : $"{name}.{subName}";
                    SearchModule(subModule, fullName);
                }
            }
        }

        SearchModule(_model, "");
        return convLayers;
    }

    /// <summary>
    /// Gets all Embedding layers in the model
    /// </summary>
    public List<(string Name, Embedding Module)> FindEmbeddingLayers()
    {
        var embLayers = new List<(string Name, Embedding Module)>();

        void SearchModule(IModule module, string name)
        {
            if (module is Embedding emb)
            {
                embLayers.Add((name, emb));
            }

            if (module is IHasSubmodules hasSubmodules)
            {
                foreach (var (subName, subModule) in hasSubmodules.NamedChildren())
                {
                    var fullName = string.IsNullOrEmpty(name) ? subName : $"{name}.{subName}";
                    SearchModule(subModule, fullName);
                }
            }
        }

        SearchModule(_model, "");
        return embLayers;
    }
}
```

### 2. Preset Targeting Strategies
**File**: `src/LoRA/LoRA targetingPresets.cs`

```csharp
/// <summary>
/// Preset targeting strategies for common model architectures
/// </summary>
public static class LoRATargetingPresets
{
    /// <summary>
    /// Target all linear layers (common for GPT-style models)
    /// </summary>
    public static string[] AllLinear { get; } = Array.Empty<string>();

    /// <summary>
    /// Target only attention Q, K, V projection layers
    /// </summary>
    public static string[] AttentionProjections { get; } = new[]
    {
        "attn.q_proj",
        "attn.k_proj",
        "attn.v_proj"
    };

    /// <summary>
    /// Target attention Q, K, V, and O projections
    /// </summary>
    public static string[] AllAttention { get; } = new[]
    {
        "attn.q_proj",
        "attn.k_proj",
        "attn.v_proj",
        "attn.o_proj"
    };

    /// <summary>
    /// Target all linear layers in transformer blocks
    /// </summary>
    public static string[] TransformerLinear { get; } = new[]
    {
        "attn.q_proj",
        "attn.k_proj",
        "attn.v_proj",
        "attn.o_proj",
        "mlp.fc1",
        "mlp.fc2"
    };

    /// <summary>
    /// Target only MLP layers (not attention)
    /// </summary>
    public static string[] MLPLayers { get; } = new[]
    {
        "mlp.fc1",
        "mlp.fc2"
    };

    /// <summary>
    /// Target layers matching a regex pattern
    /// </summary>
    public static string[] Pattern(string pattern)
    {
        return new[] { pattern };
    }

    /// <summary>
    /// Exclude specific layer names
    /// </summary>
    public static string[] ExcludeFrom(string[] baseTargets, string[] excludeNames)
    {
        // This is for documentation - actual exclusion logic happens in LoRAInjector
        return baseTargets.Where(t => !excludeNames.Contains(t)).ToArray();
    }
}
```

### 3. Model-Specific Presets
**File**: `src/LoRA/ModelPresets.cs`

```csharp
/// <summary>
/// LoRA targeting presets for specific model architectures
/// </summary>
public static class ModelPresets
{
    /// <summary>
    /// Default targeting for GPT-style models (GPT, GPT-2, GPT-3, etc.)
    /// </summary>
    public static string[] GPT { get; } = new[]
    {
        "attn.c_attn",  // Combined QKV in GPT-2
        "attn.c_proj",  // Output projection
        "mlp.c_fc",     // First MLP layer
        "mlp.c_proj"    // Second MLP layer
    };

    /// <summary>
    /// Targeting for LLaMA-style models
    /// </summary>
    public static string[] LLaMA { get; } = new[]
    {
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj"
    };

    /// <summary>
    /// Targeting for BERT-style models
    /// </summary>
    public static string[] BERT { get; } = new[]
    {
        "attention.self.query",
        "attention.self.key",
        "attention.self.value",
        "attention.output.dense",
        "intermediate.dense",
        "output.dense"
    };

    /// <summary>
    /// Targeting for ViT (Vision Transformer) models
    /// </summary>
    public static string[] ViT { get; } = new[]
    {
        "attn.qkv",
        "attn.proj",
        "mlp.fc1",
        "mlp.fc2"
    };

    /// <summary>
    /// Targeting for ResNet-style CNNs (Conv2d layers only)
    /// </summary>
    public static string[] ResNet { get; } = new[]
    {
        "layer.*.*.conv1",
        "layer.*.*.conv2",
        "layer.*.*.conv3"
    };
}
```

### 4. Extension Methods
**File**: `src/LoRA/TargetingExtensions.cs`

```csharp
public static class TargetingExtensions
{
    /// <summary>
    /// Gets a selector for the model
    /// </summary>
    public static LayerTargetSelector GetLayerSelector(this IModule model)
    {
        return new LayerTargetSelector(model);
    }

    /// <summary>
    /// Finds all modules matching a pattern
    /// </summary>
    public static List<string> FindModules(this IModule model, string pattern)
    {
        var selector = new LayerTargetSelector(model);
        return selector.FindModulesByPattern(pattern);
    }

    /// <summary>
    /// Finds all modules of a specific type
    /// </summary>
    public static List<(string Name, T Module)> FindModules<T>(this IModule model)
        where T : IModule
    {
        var selector = new LayerTargetSelector(model);
        return selector.FindModulesByType<T>();
    }

    /// <summary>
    /// Prints all linear layer names in the model
    /// </summary>
    public static void PrintLinearLayers(this IModule model)
    {
        var selector = new LayerTargetSelector(model);
        var layers = selector.FindLinearLayers();

        Console.WriteLine("Linear Layers:");
        foreach (var (name, layer) in layers)
        {
            Console.WriteLine($"  {name}: In={layer.InFeatures}, Out={layer.OutFeatures}");
        }
    }

    /// <summary>
    /// Prints all Conv2d layer names in the model
    /// </summary>
    public static void PrintConv2dLayers(this IModule model)
    {
        var selector = new LayerTargetSelector(model);
        var layers = selector.FindConv2dLayers();

        Console.WriteLine("Conv2d Layers:");
        foreach (var (name, layer) in layers)
        {
            Console.WriteLine($"  {name}: In={layer.InChannels}, Out={layer.OutChannels}, K={layer.KernelSize}");
        }
    }

    /// <summary>
    /// Prints all module names in the model
    /// </summary>
    public static void PrintModuleStructure(this IModule model)
    {
        void PrintModule(IModule module, string name, int indent)
        {
            var prefix = new string(' ', indent * 2);
            Console.WriteLine($"{prefix}{name} ({module.GetType().Name})");

            if (module is IHasSubmodules hasSubmodules)
            {
                foreach (var (subName, subModule) in hasSubmodules.NamedChildren())
                {
                    var fullName = string.IsNullOrEmpty(name) ? subName : $"{name}.{subName}";
                    PrintModule(subModule, subName, indent + 1);
                }
            }
        }

        Console.WriteLine("Model Structure:");
        PrintModule(model, "", 0);
    }
}
```

## Testing Requirements

**File**: `tests/LoRA/LayerTargetSelectorTests.cs`

1. **Pattern Matching Tests**
   - Test FindModulesByPattern with various patterns
   - Test regex pattern matching
   - Test wildcard patterns

2. **Type-Based Selection Tests**
   - Test FindModulesByType<Linear>()
   - Test FindModulesByType<Conv2d>()
   - Test FindModulesByType<Embedding>()

3. **Preset Tests**
   - Test AttentionProjections preset
   - Test AllAttention preset
   - Test TransformerLinear preset

4. **Model-Specific Presets Tests**
   - Test GPT preset targets correct layers
   - Test LLaMA preset targets correct layers
   - Test BERT preset targets correct layers

5. **Extension Method Tests**
   - Test PrintLinearLayers output
   - Test PrintConv2dLayers output
   - Test PrintModuleStructure output

## Dependencies
- IModule interface (existing)
- IHasSubmodules interface (from spec 005)
- Linear, Conv2d, Embedding layers (existing)

## Success Criteria
- LayerTargetSelector correctly finds all matching modules
- Pattern matching works with regex
- Type-based selection works correctly
- Presets match known model architectures
- Extension methods provide convenient API
- All unit tests pass

## Estimated Time
45 minutes

## Notes
- Pattern matching should support standard regex syntax
- Consider caching selector results for performance
- Add validation for preset names
- Consider supporting hierarchical targeting (e.g., "layers.0.*.attn")
- Print methods are for debugging/diagnostic purposes
