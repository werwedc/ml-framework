# Spec: Adapter Merging Implementation

## Overview
Implement utilities for merging LoRA adapters into base model weights. This is essential for deployment, as it allows saving a single merged model instead of base + adapters, improving inference efficiency.

## Implementation Details

### 1. AdapterMerger Class
**File**: `src/LoRA/AdapterMerger.cs`

```csharp
/// <summary>
/// Provides utilities for merging LoRA adapters into base model weights
/// </summary>
public class AdapterMerger
{
    private readonly IModule _model;

    public AdapterMerger(IModule model)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
    }

    /// <summary>
    /// Merges all LoRA adapters into their base layers
    /// </summary>
    /// <returns>Number of adapters merged</returns>
    public int MergeAllAdapters()
    {
        int count = 0;

        void MergeInModule(IModule module, string name)
        {
            if (module is ILoRAAdapter adapter)
            {
                adapter.MergeAdapter();
                count++;
            }

            if (module is IHasSubmodules hasSubmodules)
            {
                foreach (var (_, subModule) in hasSubmodules.NamedChildren())
                {
                    MergeInModule(subModule, "");
                }
            }
        }

        MergeInModule(_model, "");
        return count;
    }

    /// <summary>
    /// Merges specific adapters by module name pattern
    /// </summary>
    /// <param name="pattern">Regex pattern to match module names</param>
    /// <returns>Number of adapters merged</returns>
    public int MergeAdaptersByPattern(string pattern)
    {
        int count = 0;

        void MergeInModule(IModule module, string name)
        {
            if (module is ILoRAAdapter adapter && Regex.IsMatch(name, pattern))
            {
                adapter.MergeAdapter();
                count++;
            }

            if (module is IHasSubmodules hasSubmodules)
            {
                foreach (var (subName, subModule) in hasSubmodules.NamedChildren())
                {
                    var fullName = string.IsNullOrEmpty(name) ? subName : $"{name}.{subName}";
                    MergeInModule(subModule, fullName);
                }
            }
        }

        MergeInModule(_model, "");
        return count;
    }

    /// <summary>
    /// Resets all merged adapters to their original state
    /// </summary>
    /// <returns>Number of adapters reset</returns>
    public int ResetAllAdapters()
    {
        int count = 0;

        void ResetInModule(IModule module)
        {
            if (module is ILoRAAdapter adapter)
            {
                adapter.ResetBaseLayer();
                count++;
            }

            if (module is IHasSubmodules hasSubmodules)
            {
                foreach (var (_, subModule) in hasSubmodules.NamedChildren())
                {
                    ResetInModule(subModule);
                }
            }
        }

        ResetInModule(_model, "");
        return count;
    }

    /// <summary>
    /// Resets specific adapters by module name pattern
    /// </summary>
    public int ResetAdaptersByPattern(string pattern)
    {
        int count = 0;

        void ResetInModule(IModule module, string name)
        {
            if (module is ILoRAAdapter adapter && Regex.IsMatch(name, pattern))
            {
                adapter.ResetBaseLayer();
                count++;
            }

            if (module is IHasSubmodules hasSubmodules)
            {
                foreach (var (subName, subModule) in hasSubmodules.NamedChildren())
                {
                    var fullName = string.IsNullOrEmpty(name) ? subName : $"{name}.{subName}";
                    ResetInModule(subModule, fullName);
                }
            }
        }

        ResetInModule(_model, "");
        return count;
    }

    /// <summary>
    /// Checks if all adapters are merged
    /// </summary>
    public bool AreAllAdaptersMerged()
    {
        bool allMerged = true;

        void CheckModule(IModule module)
        {
            if (module is ILoRAAdapter adapter)
            {
                // Check if adapter is still enabled (disabled = merged)
                if (adapter.IsEnabled)
                {
                    allMerged = false;
                }
            }

            if (module is IHasSubmodules hasSubmodules && allMerged)
            {
                foreach (var (_, subModule) in hasSubmodules.NamedChildren())
                {
                    CheckModule(subModule);
                }
            }
        }

        CheckModule(_model);
        return allMerged;
    }

    /// <summary>
    /// Gets a summary of adapter merge status
    /// </summary>
    public AdapterMergeSummary GetMergeSummary()
    {
        var summary = new AdapterMergeSummary
        {
            TotalAdapters = 0,
            MergedAdapters = 0,
            UnmergedAdapters = 0
        };

        void CheckModule(IModule module)
        {
            if (module is ILoRAAdapter adapter)
            {
                summary.TotalAdapters++;
                if (adapter.IsEnabled)
                {
                    summary.UnmergedAdapters++;
                }
                else
                {
                    summary.MergedAdapters++;
                }
            }

            if (module is IHasSubmodules hasSubmodules)
            {
                foreach (var (_, subModule) in hasSubmodules.NamedChildren())
                {
                    CheckModule(subModule);
                }
            }
        }

        CheckModule(_model);
        return summary;
    }

    /// <summary>
    /// Exports the model with merged adapters
    /// </summary>
    /// <param name="outputPath">Path to save the merged model</param>
    public void ExportMergedModel(string outputPath)
    {
        // Ensure all adapters are merged
        MergeAllAdapters();

        // Save the model (framework-specific)
        SaveModel(_model, outputPath);
    }

    /// <summary>
    /// Exports the model with merged adapters (asynchronous)
    /// </summary>
    public async Task ExportMergedModelAsync(string outputPath)
    {
        await Task.Run(() => ExportMergedModel(outputPath));
    }

    /// <summary>
    /// Compares model outputs before and after merge to verify correctness
    /// </summary>
    public MergeVerificationResult VerifyMerge(ITensor testInput, float tolerance = 1e-5f)
    {
        // Get output before merge
        var beforeMerge = _model.Forward(testInput);

        // Merge all adapters
        MergeAllAdapters();

        // Get output after merge
        var afterMerge = _model.Forward(testInput);

        // Compare outputs
        var maxDiff = Math.Abs(beforeMerge - afterMerge).Max().ToScalar<float>();
        var meanDiff = Math.Abs(beforeMerge - afterMerge).Mean().ToScalar<float>();

        // Reset adapters
        ResetAllAdapters();

        var isCorrect = maxDiff < tolerance;

        return new MergeVerificationResult
        {
            IsCorrect = isCorrect,
            MaxDifference = maxDiff,
            MeanDifference = meanDiff,
            Tolerance = tolerance,
            Passed = isCorrect
        };
    }

    /// <summary>
    /// Partially merges adapters (useful for fine-grained control)
    /// </summary>
    /// <param name="mergeFactor">Factor to scale adapter weights by (0.0 = no merge, 1.0 = full merge)</param>
    /// <returns>Number of adapters partially merged</returns>
    public int PartialMergeAll(float mergeFactor)
    {
        if (mergeFactor < 0.0f || mergeFactor > 1.0f)
            throw new ArgumentException("Merge factor must be in [0, 1]", nameof(mergeFactor));

        int count = 0;

        void PartialMergeInModule(IModule module)
        {
            if (module is ILoRAAdapter adapter)
            {
                var (matrixA, matrixB) = adapter.GetAdapterWeights();
                var scaling = adapter.ScalingFactor * mergeFactor;

                // Get base weights
                var baseWeights = adapter.BaseLayer switch
                {
                    Linear linear => linear.Weight,
                    Conv2d conv => conv.Weight,
                    Embedding emb => emb.Weight,
                    _ => throw new NotSupportedException("Unsupported layer type")
                };

                // Calculate delta
                var delta = Tensor.MatMul(matrixB!, matrixA!).Mul(scaling);

                // Partially merge
                baseWeights = baseWeights.Add(delta);

                // Update base layer weights
                if (adapter.BaseLayer is Linear linear)
                {
                    linear.Weight = baseWeights;
                }
                else if (adapter.BaseLayer is Conv2d conv)
                {
                    conv.Weight = baseWeights;
                }
                else if (adapter.BaseLayer is Embedding emb)
                {
                    emb.Weight = baseWeights;
                }

                // Disable adapter to prevent double-application
                if (mergeFactor >= 1.0f)
                {
                    adapter.IsEnabled = false;
                }

                count++;
            }

            if (module is IHasSubmodules hasSubmodules)
            {
                foreach (var (_, subModule) in hasSubmodules.NamedChildren())
                {
                    PartialMergeInModule(subModule);
                }
            }
        }

        PartialMergeInModule(_model);
        return count;
    }

    private void SaveModel(IModule model, string outputPath)
    {
        // Framework-specific model saving
        // This is a placeholder - actual implementation varies by framework
        throw new NotImplementedException("Model saving is framework-specific");
    }
}

/// <summary>
/// Summary of adapter merge status
/// </summary>
public class AdapterMergeSummary
{
    public int TotalAdapters { get; set; }
    public int MergedAdapters { get; set; }
    public int UnmergedAdapters { get; set; }
    public double MergePercentage => TotalAdapters > 0
        ? (MergedAdapters * 100.0) / TotalAdapters
        : 0.0;

    public override string ToString()
    {
        return $"Merged: {MergedAdapters}/{TotalAdapters} ({MergePercentage:F1}%)";
    }
}

/// <summary>
/// Result of merge verification
/// </summary>
public class MergeVerificationResult
{
    public bool IsCorrect { get; set; }
    public float MaxDifference { get; set; }
    public float MeanDifference { get; set; }
    public float Tolerance { get; set; }
    public bool Passed { get; set; }

    public override string ToString()
    {
        return $"Merge Verification: {(Passed ? "PASSED" : "FAILED")}\n" +
               $"  Max Diff: {MaxDifference:E3} (tolerance: {Tolerance:E3})\n" +
               $"  Mean Diff: {MeanDifference:E3}";
    }
}
```

### 2. Extension Methods
**File**: `src/LoRA/MergeExtensions.cs`

```csharp
public static class MergeExtensions
{
    /// <summary>
    /// Merges all LoRA adapters in a model
    /// </summary>
    public static int MergeLoRAAdapters(this IModule model)
    {
        var merger = new AdapterMerger(model);
        return merger.MergeAllAdapters();
    }

    /// <summary>
    /// Resets all merged LoRA adapters
    /// </summary>
    public static int ResetLoRAAdapters(this IModule model)
    {
        var merger = new AdapterMerger(model);
        return merger.ResetAllAdapters();
    }

    /// <summary>
    /// Gets the merge status of adapters
    /// </summary>
    public static AdapterMergeSummary GetLoRAMergeSummary(this IModule model)
    {
        var merger = new AdapterMerger(model);
        return merger.GetMergeSummary();
    }

    /// <summary>
    /// Exports model with merged adapters
    /// </summary>
    public static void ExportMergedLoRAModel(this IModule model, string outputPath)
    {
        var merger = new AdapterMerger(model);
        merger.ExportMergedModel(outputPath);
    }

    /// <summary>
    /// Verifies that merge was done correctly
    /// </summary>
    public static MergeVerificationResult VerifyLoRAMerge(
        this IModule model,
        ITensor testInput,
        float tolerance = 1e-5f)
    {
        var merger = new AdapterMerger(model);
        return merger.VerifyMerge(testInput, tolerance);
    }

    /// <summary>
    /// Partially merges LoRA adapters
    /// </summary>
    public static int PartialMergeLoRA(this IModule model, float mergeFactor)
    {
        var merger = new AdapterMerger(model);
        return merger.PartialMergeAll(mergeFactor);
    }
}
```

## Testing Requirements

**File**: `tests/LoRA/AdapterMergerTests.cs`

1. **Merge Tests**
   - Test MergeAllAdapters correctly merges all adapters
   - Test MergeAdaptersByPattern with different patterns
   - Test PartialMergeAll with different merge factors

2. **Reset Tests**
   - Test ResetAllAdapters restores original weights
   - Test ResetAdaptersByPattern with patterns
   - Test merge-reset cycle works correctly

3. **Verification Tests**
   - Test VerifyMerge detects incorrect merges
   - Test VerifyMerge passes for correct merges
   - Test tolerance parameter works correctly

4. **Summary Tests**
   - Test GetMergeSummary returns correct counts
   - Test AreAllAdaptersMerged detection
   - Test summary after partial merges

5. **Export Tests**
   - Test ExportMergedModel saves correctly
   - Test async export works
   - Test exported model can be loaded

## Dependencies
- IModule interface (existing)
- ILoRAAdapter interface (from spec 001)
- Tensor arithmetic operations (existing)

## Success Criteria
- Adapter merging produces correct weight updates
- Reset functionality restores original weights
- Verification correctly detects merge errors
- Partial merging works with different factors
- Export creates valid model files
- All unit tests pass

## Estimated Time
45 minutes

## Notes
- Merging is destructive - always provide reset capability
- Partial merging can be useful for adapter interpolation
- Consider adding support for merging specific layers only
- Verification should be run after merge to ensure correctness
