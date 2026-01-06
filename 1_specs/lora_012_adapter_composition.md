# Spec: Adapter Composition Utilities

## Overview
Implement utilities for composing multiple LoRA adapters. This enables combining different adapter effects (e.g., task + style adapters) without training from scratch.

## Implementation Details

### 1. AdapterComposer Class
**File**: `src/LoRA/AdapterComposer.cs`

```csharp
/// <summary>
/// Provides utilities for composing multiple LoRA adapters
/// </summary>
public class AdapterComposer
{
    private readonly IModule _model;
    private readonly LoRAAdapterRegistry _registry;

    public AdapterComposer(IModule model, LoRAAdapterRegistry registry)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _registry = registry ?? throw new ArgumentNullException(nameof(registry));
        _registry.SetModel(model);
    }

    /// <summary>
    /// Adds multiple adapters together
    /// </summary>
    /// <param name="adapterIds">IDs of adapters to compose</param>
    /// <param name="weights">Optional weights for each adapter (default: equal)</param>
    /// <param name="outputAdapterId">ID for the composed adapter</param>
    public void AddAdapters(
        string[] adapterIds,
        float[]? weights = null,
        string? outputAdapterId = null)
    {
        if (adapterIds == null || adapterIds.Length == 0)
            throw new ArgumentException("At least one adapter ID required", nameof(adapterIds));

        if (weights != null && weights.Length != adapterIds.Length)
            throw new ArgumentException("Weights must match adapter IDs length", nameof(weights));

        // Normalize weights if not provided
        if (weights == null)
        {
            weights = Enumerable.Repeat(1.0f / adapterIds.Length, adapterIds.Length).ToArray();
        }

        // Load all adapter states
        var states = adapterIds.Select(id => LoadAdapterState(id)).ToList();

        // Compose adapters
        var composedState = ComposeStates(states, weights, AdapterCompositionMode.Add);

        // Save composed adapter
        outputAdapterId ??= $"composed_{DateTime.UtcNow:yyyyMMdd_HHmmss}";
        _registry.SetModel(_model);
        ApplyAdapterState(composedState);
        _registry.SaveAdapter(outputAdapterId);
    }

    /// <summary>
    /// Interpolates between two adapters
    /// </summary>
    /// <param name="adapterId1">First adapter ID</param>
    /// <param name="adapterId2">Second adapter ID</param>
    /// <param name="alpha">Interpolation factor (0.0 = adapter1, 1.0 = adapter2)</param>
    /// <param name="outputAdapterId">ID for the composed adapter</param>
    public void InterpolateAdapters(
        string adapterId1,
        string adapterId2,
        float alpha,
        string? outputAdapterId = null)
    {
        if (alpha < 0.0f || alpha > 1.0f)
            throw new ArgumentException("Alpha must be in [0, 1]", nameof(alpha));

        var state1 = LoadAdapterState(adapterId1);
        var state2 = LoadAdapterState(adapterId2);

        var weights = new[] { 1.0f - alpha, alpha };
        var composedState = ComposeStates(new[] { state1, state2 }, weights, AdapterCompositionMode.Add);

        outputAdapterId ??= $"interpolated_{adapterId1}_{adapterId2}_{alpha:F2}";
        _registry.SetModel(_model);
        ApplyAdapterState(composedState);
        _registry.SaveAdapter(outputAdapterId);
    }

    /// <summary>
    /// Averages multiple adapters
    /// </summary>
    /// <param name="adapterIds">IDs of adapters to average</param>
    /// <param name="outputAdapterId">ID for the composed adapter</param>
    public void AverageAdapters(string[] adapterIds, string? outputAdapterId = null)
    {
        AddAdapters(adapterIds, null, outputAdapterId);
    }

    /// <summary>
    /// Selects best adapter from ensemble based on validation
    /// </summary>
    /// <param name="adapterIds">IDs of adapters to evaluate</param>
    /// <param name="validationData">Validation dataset</param>
    /// <param name="metric">Metric to optimize (e.g., "accuracy", "loss")</param>
    /// <returns>ID of best adapter</returns>
    public string SelectBestAdapter(
        string[] adapterIds,
        IEnumerable<(ITensor Input, ITensor Target)> validationData,
        string metric = "loss")
    {
        var scores = new Dictionary<string, float>();

        foreach (var adapterId in adapterIds)
        {
            // Load adapter
            _registry.LoadAdapter(adapterId);

            // Evaluate on validation data
            float score = 0;
            int count = 0;

            foreach (var (input, target) in validationData)
            {
                var output = _model.Forward(input);

                if (metric == "loss")
                {
                    var loss = ComputeLoss(output, target);
                    score += loss.ToScalar<float>();
                }
                else if (metric == "accuracy")
                {
                    var accuracy = ComputeAccuracy(output, target);
                    score += accuracy;
                }

                count++;
            }

            scores[adapterId] = score / count;
        }

        // Select best (lowest for loss, highest for accuracy)
        if (metric == "loss")
        {
            return scores.OrderBy(kvp => kvp.Value).First().Key;
        }
        else
        {
            return scores.OrderByDescending(kvp => kvp.Value).First().Key;
        }
    }

    /// <summary>
    /// Creates a task-specific adapter by subtracting a base task adapter
    /// </summary>
    /// <param name="fullAdapterId">Adapter with task+base effect</param>
    /// <param name="baseAdapterId">Adapter with base effect</param>
    /// <param name="outputAdapterId">ID for the composed adapter</param>
    public void CreateTaskSpecificAdapter(
        string fullAdapterId,
        string baseAdapterId,
        string? outputAdapterId = null)
    {
        var state1 = LoadAdapterState(fullAdapterId);
        var state2 = LoadAdapterState(baseAdapterId);

        var weights = new[] { 1.0f, -1.0f };
        var composedState = ComposeStates(new[] { state1, state2 }, weights, AdapterCompositionMode.Add);

        outputAdapterId ??= $"task_specific_{fullAdapterId}_minus_{baseAdapterId}";
        _registry.SetModel(_model);
        ApplyAdapterState(composedState);
        _registry.SaveAdapter(outputAdapterId);
    }

    /// <summary>
    /// Ties multiple adapter weights together for joint training
    /// </summary>
    /// <param name="adapterIds">IDs of adapters to tie</param>
    /// <param name="tieMode">How to tie weights</param>
    public void TieAdapters(string[] adapterIds, AdapterTieMode tieMode)
    {
        // Load all adapter states
        var states = adapterIds.Select(id => LoadAdapterState(id)).ToList();

        switch (tieMode)
        {
            case AdapterTieMode.Shared:
                // All adapters share the same weights
                TieSharedWeights(states);
                break;

            case AdapterTieMode.Layerwise:
                // Same layers share weights, different layers are independent
                TieLayerwiseWeights(states);
                break;

            case AdapterTieMode.None:
                // No tying
                break;

            default:
                throw new ArgumentException($"Unknown tie mode: {tieMode}");
        }

        // Save updated adapters
        foreach (var (adapterId, state) in adapterIds.Zip(states))
        {
            _registry.SetModel(_model);
            ApplyAdapterState(state);
            _registry.SaveAdapter(adapterId);
        }
    }

    private AdapterState LoadAdapterState(string adapterId)
    {
        _registry.LoadAdapter(adapterId);

        var state = new AdapterState
        {
            Metadata = _registry.GetAdapterMetadata(adapterId) ?? new AdapterMetadata { Id = adapterId }
        };

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

        ExtractFromModule(_model, "");
        return state;
    }

    private AdapterState ComposeStates(
        List<AdapterState> states,
        float[] weights,
        AdapterCompositionMode mode)
    {
        var composedState = new AdapterState
        {
            Metadata = new AdapterMetadata
            {
                Id = "composed",
                Name = "Composed Adapter",
                CreatedAt = DateTime.UtcNow
            }
        };

        // Get all module names from first state
        var moduleNames = states[0].Weights.Keys.ToList();

        foreach (var moduleName in moduleNames)
        {
            var composedWeights = ComposeWeights(
                states.Select(s => s.Weights[moduleName]).ToList(),
                weights,
                mode
            );

            composedState.Weights[moduleName] = composedWeights;
        }

        return composedState;
    }

    private AdapterWeights ComposeWeights(
        List<AdapterWeights> weightsList,
        float[] weights,
        AdapterCompositionMode mode)
    {
        if (weightsList.Count == 0)
            throw new ArgumentException("No weights to compose", nameof(weightsList));

        var first = weightsList[0];
        var composed = new AdapterWeights
        {
            LayerType = first.LayerType
        };

        // Initialize with first weights
        composed.MatrixA = first.MatrixA.Clone();
        composed.MatrixB = first.MatrixB.Clone();
        composed.Bias = first.Bias?.Clone();

        // Compose remaining weights
        for (int i = 1; i < weightsList.Count; i++)
        {
            var current = weightsList[i];
            var weight = weights[i];

            switch (mode)
            {
                case AdapterCompositionMode.Add:
                    composed.MatrixA = composed.MatrixA.Add(current.MatrixA.Mul(weight));
                    composed.MatrixB = composed.MatrixB.Add(current.MatrixB.Mul(weight));
                    if (composed.Bias != null && current.Bias != null)
                    {
                        composed.Bias = composed.Bias.Add(current.Bias.Mul(weight));
                    }
                    break;

                case AdapterCompositionMode.Multiply:
                    // Element-wise multiplication
                    composed.MatrixA = composed.MatrixA.Mul(current.MatrixA.Pow(weight));
                    composed.MatrixB = composed.MatrixB.Mul(current.MatrixB.Pow(weight));
                    if (composed.Bias != null && current.Bias != null)
                    {
                        composed.Bias = composed.Bias.Mul(current.Bias.Pow(weight));
                    }
                    break;

                default:
                    throw new ArgumentException($"Unknown composition mode: {mode}");
            }
        }

        return composed;
    }

    private void ApplyAdapterState(AdapterState state)
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

        ApplyToModule(_model, "");
    }

    private void TieSharedWeights(List<AdapterState> states)
    {
        // Make all adapters use the first adapter's weights
        var firstWeights = states[0].Weights;

        for (int i = 1; i < states.Count; i++)
        {
            foreach (var (name, weights) in states[i].Weights)
            {
                if (firstWeights.TryGetValue(name, out var first))
                {
                    weights.MatrixA = first.MatrixA;
                    weights.MatrixB = first.MatrixB;
                    weights.Bias = first.Bias;
                }
            }
        }
    }

    private void TieLayerwiseWeights(List<AdapterState> states)
    {
        // Tie weights for same layers across adapters
        var layerNames = states[0].Weights.Keys.ToList();

        foreach (var layerName in layerNames)
        {
            var firstLayerWeights = states[0].Weights[layerName];

            for (int i = 1; i < states.Count; i++)
            {
                if (states[i].Weights.TryGetValue(layerName, out var current))
                {
                    current.MatrixA = firstLayerWeights.MatrixA;
                    current.MatrixB = firstLayerWeights.MatrixB;
                    current.Bias = firstLayerWeights.Bias;
                }
            }
        }
    }

    private ITensor ComputeLoss(ITensor output, ITensor target)
    {
        // Simplified loss computation (MSE)
        return Tensor.Mean((output - target).Pow(2));
    }

    private float ComputeAccuracy(ITensor output, ITensor target)
    {
        // Simplified accuracy computation
        var predictions = Tensor.ArgMax(output, 1);
        var labels = Tensor.ArgMax(target, 1);
        return Tensor.Mean(predictions == labels).ToScalar<float>();
    }
}

/// <summary>
/// Mode for composing adapters
/// </summary>
public enum AdapterCompositionMode
{
    Add,
    Multiply
}

/// <summary>
/// Mode for tying adapter weights
/// </summary>
public enum AdapterTieMode
{
    None,
    Shared,
    Layerwise
}
```

## Testing Requirements

**File**: `tests/LoRA/AdapterComposerTests.cs`

1. **Composition Tests**
   - Test AddAdapters with equal weights
   - Test AddAdapters with custom weights
   - Test InterpolateAdapters with different alpha values

2. **Selection Tests**
   - Test SelectBestAdapter with loss metric
   - Test SelectBestAdapter with accuracy metric
   - Test selection on multiple adapters

3. **Task-Specific Tests**
   - Test CreateTaskSpecificAdapter subtracts correctly
   - Test task-specific adapter produces expected behavior

4. **Tying Tests**
   - Test TieAdapters with Shared mode
   - Test TieAdapters with Layerwise mode
   - Test tied weights are shared correctly

## Dependencies
- IModule interface (existing)
- ILoRAAdapter interface (from spec 001)
- LoRAAdapterRegistry (from spec 008)
- Tensor arithmetic operations (existing)

## Success Criteria
- Adapter composition produces correct weight combinations
- Interpolation works smoothly between adapters
- Best adapter selection chooses correctly
- Weight tying works as expected
- All unit tests pass

## Estimated Time
45 minutes

## Notes
- Composition assumes adapters have compatible architectures
- Consider adding more sophisticated ensemble methods
- Task-specific decomposition can be useful for multi-task learning
