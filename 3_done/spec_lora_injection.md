# Spec: LoRA Injection System

## Overview
Implement automatic LoRA layer injection into existing models without modifying base model code.

## Requirements
- Automatically replace target linear layers with LoraLinear wrappers
- Support configurable target module patterns
- Preserve original model structure
- Handle nested module hierarchies
- Support injection into transformer architectures (QKV, output projections)

## Classes to Implement

### 1. LoraInjector
```csharp
public class LoraInjector
{
    /// <summary>Inject LoRA layers into model</summary>
    public static void Inject(IModule model, LoraConfig config)
    {
        var targetPatterns = ParseTargetModules(config.TargetModules);
        InjectRecursive(model, config, targetPatterns);
    }

    /// <summary>Remove LoRA layers, restore original model</summary>
    public static void Remove(IModule model)
    {
        RemoveRecursive(model);
    }

    /// <summary>Check if module has LoRA injected</summary>
    public static bool HasLoRA(IModule model)
    {
        return HasLoRARecursive(model);
    }

    /// <summary>Get all LoRA-injected modules</summary>
    public static List<LoraLinear> GetLoRALayers(IModule model)
    {
        var layers = new List<LoraLinear>();
        FindLoRALayersRecursive(model, layers);
        return layers;
    }

    private static void InjectRecursive(IModule module, LoraConfig config, List<ModuleTargetPattern> patterns)
    {
        if (module is Linear linear && ShouldInject(module.Name, patterns))
        {
            // Wrap Linear with LoraLinear
            var loraLayer = new LoraLinear(linear, config.Rank, config.Alpha, config.Dropout);
            ReplaceModule(module, loraLayer);
        }
        else
        {
            // Recursively process child modules
            foreach (var child in module.Children())
            {
                InjectRecursive(child, config, patterns);
            }
        }
    }

    private static void RemoveRecursive(IModule module)
    {
        if (module is LoraLinear loraLinear)
        {
            // Restore original Linear layer
            var baseLinear = loraLinear.BaseLinear;
            ReplaceModule(module, baseLinear);
        }
        else
        {
            foreach (var child in module.Children())
            {
                RemoveRecursive(child);
            }
        }
    }

    private static bool ShouldInject(string moduleName, List<ModuleTargetPattern> patterns)
    {
        return patterns.Any(p => p.Matches(moduleName));
    }

    private static List<ModuleTargetPattern> ParseTargetModules(string[] targetModules)
    {
        var patterns = new List<ModuleTargetPattern>();

        foreach (var target in targetModules)
        {
            if (target.Contains("*") || target.Contains("?"))
            {
                // Wildcard pattern
                var regexPattern = Regex.Escape(target)
                    .Replace(@"\*", ".*")
                    .Replace(@"\?", ".");
                patterns.Add(new ModuleTargetPattern
                {
                    Pattern = new Regex($"^{regexPattern}$")
                });
            }
            else
            {
                // Exact match
                patterns.Add(new ModuleTargetPattern
                {
                    ExactName = target
                });
            }
        }

        return patterns;
    }

    private static void ReplaceModule(IModule oldModule, IModule newModule)
    {
        // Replace module in parent's child collection
        // Implementation depends on module container API
        var parent = oldModule.Parent;
        parent.ReplaceChild(oldModule.Name, newModule);
    }
}

### 2. Model Extension Methods
```csharp
public static class LoRAExtensions
{
    public static void ApplyLoRA(this IModule model, LoraConfig config = null)
    {
        config ??= new LoraConfig();
        LoraInjector.Inject(model, config);
    }

    public static void RemoveLoRA(this IModule model)
    {
        LoraInjector.Remove(model);
    }

    public static List<Parameter> GetLoRAParameters(this IModule model)
    {
        return LoraInjector.GetLoRALayers(model)
            .SelectMany(l => l.TrainableParameters())
            .ToList();
    }

    public static void FreezeBase(this IModule model)
    {
        // Freeze all non-LoRA parameters
        foreach (var param in model.Parameters())
        {
            if (!IsLoRAParameter(param))
            {
                param.RequiresGrad = false;
            }
        }
    }

    public static void UnfreezeAll(this IModule model)
    {
        foreach (var param in model.Parameters())
        {
            param.RequiresGrad = true;
        }
    }
}
```

## Implementation Details
- Support module path patterns like "encoder.layers.*.self_attn.q_proj"
- Handle both exact names and wildcard patterns
- Preserve original module naming and structure
- Support injection into multi-GPU models (DDP, FSDP)
- Ensure gradients flow correctly through injected layers

## Common Target Modules for Transformers
```csharp
// LLaMA
new[] { "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj" }

// GPT
new[] { "c_attn", "c_fc", "c_proj" }

// BERT
new[] { "query", "key", "value", "output", "intermediate" }
```

## Deliverables
- `LoraInjector.cs` in `src/Core/LoRA/`
- `LoRAExtensions.cs` in `src/Core/LoRA/`
- Unit tests in `tests/Core/LoRA/`
