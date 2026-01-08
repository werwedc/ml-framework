# Spec: Advanced Scheduler Features

## Overview
Implement advanced learning rate scheduler features including polynomial decay, layer-wise scheduling, and discriminative learning rates. These are specialized schedulers for advanced training scenarios.

## Dependencies
- `spec_scheduler_base_interface.md` (must be completed first)
- Target namespace: `MLFramework.Schedulers`

## Files to Create
- `src/Schedulers/PolynomialDecayScheduler.cs`
- `src/Schedulers/LayerWiseLRDecayScheduler.cs`
- `src/Schedulers/DiscriminativeLRScheduler.cs`

## Technical Specifications

### 1. Polynomial Decay Scheduler

**Purpose**: Decays learning rate polynomially over the training duration. Similar to TensorFlow's polynomial decay.

**Formula**:
```
if step < totalSteps:
    LR = initialLR + (finalLR - initialLR) * (1 - step/totalSteps)^power
else:
    LR = finalLR
```

**Implementation**:

```csharp
namespace MLFramework.Schedulers;

/// <summary>
/// Polynomial decay learning rate scheduler.
/// Decays LR polynomially from initialLR to finalLR over totalSteps.
/// </summary>
public sealed class PolynomialDecayScheduler : BaseScheduler, IStepScheduler
{
    private readonly float _initialLearningRate;
    private readonly float _finalLearningRate;
    private readonly float _totalSteps;
    private readonly float _power;

    public PolynomialDecayScheduler(
        float initialLearningRate,
        float finalLearningRate,
        float totalSteps,
        float power = 1.0f)
    {
        if (initialLearningRate <= 0)
            throw new ArgumentException("initialLearningRate must be positive", nameof(initialLearningRate));
        if (finalLearningRate < 0)
            throw new ArgumentException("finalLearningRate must be non-negative", nameof(finalLearningRate));
        if (totalSteps <= 0)
            throw new ArgumentException("totalSteps must be positive", nameof(totalSteps));
        if (power <= 0)
            throw new ArgumentException("power must be positive", nameof(power));

        _initialLearningRate = initialLearningRate;
        _finalLearningRate = finalLearningRate;
        _totalSteps = totalSteps;
        _power = power;
    }

    public override float GetLearningRate(int step, float baseLearningRate)
    {
        // Use the initialLearningRate if provided, otherwise use baseLR
        float initialLR = _initialLearningRate;
        float finalLR = _finalLearningRate;

        if (step >= _totalSteps)
        {
            return finalLR;
        }

        float progress = (float)step / _totalSteps;
        float decayFactor = (float)Math.Pow(1.0 - progress, _power);

        return initialLR + (finalLR - initialLR) * decayFactor;
    }

    public override StateDict GetState()
    {
        var state = new StateDict();
        state.Set("initial_lr", _initialLearningRate);
        state.Set("final_lr", _finalLearningRate);
        state.Set("total_steps", _totalSteps);
        state.Set("power", _power);
        state.Set("step_count", _stepCount);
        state.Set("epoch_count", _epochCount);
        return state;
    }

    public override void LoadState(StateDict state)
    {
        _stepCount = state.Get<int>("step_count", 0);
        _epochCount = state.Get<int>("epoch_count", 0);
    }
}
```

**Constructor Parameters**:
- `initialLearningRate` (float): Starting learning rate
- `finalLearningRate` (float): Final learning rate
- `totalSteps` (float): Total number of decay steps
- `power` (float): Polynomial power (default: 1.0 for linear decay)

**Example Usage**:
```csharp
// Linear decay from 0.01 to 0.0001 over 10000 steps
var scheduler = new PolynomialDecayScheduler(
    initialLearningRate: 0.01f,
    finalLearningRate: 0.0001f,
    totalSteps: 10000f,
    power: 1.0f
);

// Quadratic decay
var scheduler = new PolynomialDecayScheduler(
    initialLearningRate: 0.1f,
    finalLearningRate: 0f,
    totalSteps: 5000f,
    power: 2.0f
);
```

### 2. Layer-wise Learning Rate Decay Scheduler

**Purpose**: Applies different learning rates to different layers of a model, typically with higher learning rates for later layers and lower for earlier layers.

**Implementation**:

```csharp
namespace MLFramework.Schedulers;

/// <summary>
/// Applies layer-wise learning rate decay.
/// Later layers get higher learning rates, earlier layers get lower.
/// </summary>
public sealed class LayerWiseLRDecayScheduler : BaseScheduler, IStepScheduler
{
    private readonly float _decayFactor;
    private readonly string[] _excludedLayers;

    public LayerWiseLRDecayScheduler(
        float decayFactor = 0.8f,
        string[] excludedLayers = null)
    {
        if (decayFactor <= 0 || decayFactor >= 1)
            throw new ArgumentException("decayFactor must be in (0, 1)", nameof(decayFactor));

        _decayFactor = decayFactor;
        _excludedLayers = excludedLayers ?? Array.Empty<string>();
    }

    /// <summary>
    /// Gets the learning rate multiplier for a specific layer.
    /// </summary>
    /// <param name="layerIndex">Index of the layer (0 = input layer).</param>
    /// <param name="totalLayers">Total number of layers.</param>
    /// <param name="layerName">Name of the layer (optional).</param>
    /// <returns>Learning rate multiplier for this layer.</returns>
    public float GetLayerMultiplier(int layerIndex, int totalLayers, string layerName = null)
    {
        // Check if layer is excluded
        if (layerName != null && _excludedLayers.Contains(layerName))
        {
            return 1.0f;  // No decay for excluded layers
        }

        // Calculate decay based on layer position
        // Layer 0 (earliest) has lowest multiplier
        // Layer N-1 (latest) has multiplier = 1.0
        int positionFromEnd = totalLayers - 1 - layerIndex;
        float multiplier = (float)Math.Pow(_decayFactor, positionFromEnd);

        return multiplier;
    }

    public override float GetLearningRate(int step, float baseLearningRate)
    {
        // This is a special scheduler - it doesn't change over time
        // But must implement the interface
        // Actual layer-specific LR is obtained via GetLayerMultiplier
        return baseLearningRate;
    }

    public override StateDict GetState()
    {
        var state = new StateDict();
        state.Set("decay_factor", _decayFactor);
        state.Set("excluded_layers", _excludedLayers);
        state.Set("step_count", _stepCount);
        state.Set("epoch_count", _epochCount);
        return state;
    }

    public override void LoadState(StateDict state)
    {
        _stepCount = state.Get<int>("step_count", 0);
        _epochCount = state.Get<int>("epoch_count", 0);
    }
}
```

**Constructor Parameters**:
- `decayFactor` (float): Factor for decay between adjacent layers (default: 0.8)
- `excludedLayers` (string[]): Layer names to exclude from decay (default: null)

**Example Usage**:

```csharp
var model = new MyModel();
var optimizer = new SGD(model.Parameters(), learningRate: 0.1f);
var layerScheduler = new LayerWiseLRDecayScheduler(decayFactor: 0.8f);

// Apply layer-wise LR to model parameters
var layers = model.GetLayers();  // Assume this returns layers in order
for (int i = 0; i < layers.Count; i++)
{
    float multiplier = layerScheduler.GetLayerMultiplier(i, layers.Count, layers[i].Name);
    layers[i].LearningRateMultiplier = multiplier;
}

// Combine with time-based scheduler
var timeScheduler = new CosineAnnealingScheduler(tMax: 1000f);
var finalLR = layerScheduler.GetLearningRate(step, timeScheduler.GetLearningRate(step, baseLR));
```

### 3. Discriminative Learning Rate Scheduler

**Purpose**: Applies different learning rates to different parameter groups, commonly used in transfer learning where earlier layers should change slower than later layers.

**Implementation**:

```csharp
namespace MLFramework.Schedulers;

/// <summary>
/// Applies different learning rates to different parameter groups.
/// Commonly used in transfer learning for fine-tuning.
/// </summary>
public sealed class DiscriminativeLRScheduler : BaseScheduler, IStepScheduler
{
    private readonly float _baseLearningRate;
    private readonly float[] _layerMultipliers;
    private readonly string[] _layerNames;

    public DiscriminativeLRScheduler(
        float baseLearningRate,
        float[] layerMultipliers,
        string[] layerNames = null)
    {
        if (baseLearningRate <= 0)
            throw new ArgumentException("baseLearningRate must be positive", nameof(baseLearningRate));
        if (layerMultipliers == null || layerMultipliers.Length == 0)
            throw new ArgumentException("layerMultipliers must not be empty", nameof(layerMultipliers));

        _baseLearningRate = baseLearningRate;
        _layerMultipliers = layerMultipliers;
        _layerNames = layerNames ?? Array.Empty<string>();
    }

    /// <summary>
    /// Gets the learning rate for a specific parameter group.
    /// </summary>
    /// <param name="groupIndex">Index of the parameter group.</param>
    /// <param name="layerName">Name of the layer (optional, for named lookup).</param>
    /// <returns>Learning rate for this parameter group.</returns>
    public float GetGroupLearningRate(int groupIndex, string layerName = null)
    {
        float multiplier;

        if (layerName != null && _layerNames.Contains(layerName))
        {
            // Find by name
            int nameIndex = Array.IndexOf(_layerNames, layerName);
            if (nameIndex >= 0 && nameIndex < _layerMultipliers.Length)
            {
                multiplier = _layerMultipliers[nameIndex];
            }
            else
            {
                multiplier = 1.0f;  // Default if name not found
            }
        }
        else
        {
            // Find by index
            if (groupIndex >= 0 && groupIndex < _layerMultipliers.Length)
            {
                multiplier = _layerMultipliers[groupIndex];
            }
            else
            {
                multiplier = _layerMultipliers[_layerMultipliers.Length - 1];  // Use last multiplier
            }
        }

        return _baseLearningRate * multiplier;
    }

    public override float GetLearningRate(int step, float baseLearningRate)
    {
        // Returns the base learning rate (default group)
        // Use GetGroupLearningRate for specific groups
        return _baseLearningRate;
    }

    public override StateDict GetState()
    {
        var state = new StateDict();
        state.Set("base_lr", _baseLearningRate);
        state.Set("layer_multipliers", _layerMultipliers);
        state.Set("layer_names", _layerNames);
        state.Set("step_count", _stepCount);
        state.Set("epoch_count", _epochCount);
        return state;
    }

    public override void LoadState(StateDict state)
    {
        _stepCount = state.Get<int>("step_count", 0);
        _epochCount = state.Get<int>("epoch_count", 0);
    }
}
```

**Constructor Parameters**:
- `baseLearningRate` (float): Base learning rate for reference
- `layerMultipliers` (float[]): Array of multipliers for each layer/group
- `layerNames` (string[]): Optional layer names for named lookup

**Example Usage**:

```csharp
// Transfer learning scenario: freeze encoder, fine-tune decoder
var model = LoadPretrainedModel();
var layerMultipliers = new float[] { 0.1f, 0.2f, 0.5f, 1.0f };  // Early layers: lower LR
var layerNames = new[] { "encoder.1", "encoder.2", "decoder.1", "decoder.2" };

var scheduler = new DiscriminativeLRScheduler(
    baseLearningRate: 1e-3f,
    layerMultipliers: layerMultipliers,
    layerNames: layerNames
);

// Apply to optimizer parameter groups
var paramGroups = new[]
{
    new ParamGroup(model.GetParameters("encoder.1"), scheduler.GetGroupLearningRate(0)),
    new ParamGroup(model.GetParameters("encoder.2"), scheduler.GetGroupLearningRate(1)),
    new ParamGroup(model.GetParameters("decoder.1"), scheduler.GetGroupLearningRate(2)),
    new ParamGroup(model.GetParameters("decoder.2"), scheduler.GetGroupLearningRate(3))
};

var optimizer = new SGD(paramGroups);
```

## Implementation Notes

### Design Decisions

1. **PolynomialDecayScheduler**:
   - Ignores `baseLearningRate` parameter and uses its own `initialLearningRate`
   - Supports polynomial powers other than 1.0 (e.g., quadratic decay)
   - Final LR is used after totalSteps are exceeded

2. **LayerWiseLRDecayScheduler**:
   - Doesn't change over time (step-based decay only)
   - Uses layer position to determine multiplier
   - Supports layer name exclusions
   - Requires model integration to apply multipliers

3. **DiscriminativeLRScheduler**:
   - Designed for transfer learning scenarios
   - Supports both index-based and name-based lookups
   - Can be combined with time-based schedulers

### Edge Cases

- **PolynomialDecay**: When finalLR > initialLR, LR increases over time
- **LayerWiseLRDecay**: Single layer returns multiplier = 1.0 (no decay)
- **DiscriminativeLRScheduler**: Group index out of range returns last multiplier

### Performance Considerations

- All schedulers are O(1) per call
- Layer-wise schedulers require model integration for applying multipliers
- DiscriminativeLR may need parameter group support in optimizer

## Usage Patterns

### Pattern 1: Polynomial Decay for Training
```csharp
var scheduler = new PolynomialDecayScheduler(
    initialLearningRate: 0.01f,
    finalLearningRate: 1e-5f,
    totalSteps: 50000f,
    power: 0.9f
);

optimizer.SetScheduler(scheduler);
```

### Pattern 2: Layer-wise Decay + Cosine Annealing
```csharp
var layerScheduler = new LayerWiseLRDecayScheduler(decayFactor: 0.8f);
var timeScheduler = new CosineAnnealingScheduler(tMax: 1000f);

// Apply to each layer
for (int i = 0; i < model.LayerCount; i++)
{
    float layerMultiplier = layerScheduler.GetLayerMultiplier(i, model.LayerCount);
    float timeBasedLR = timeScheduler.GetLearningRate(step, baseLR);
    model.Layers[i].LearningRate = timeBasedLR * layerMultiplier;
}
```

### Pattern 3: Transfer Learning Fine-Tuning
```csharp
var scheduler = new DiscriminativeLRScheduler(
    baseLearningRate: 1e-3f,
    layerMultipliers: new[] { 0.1f, 0.2f, 0.5f, 1.0f }
);

// Create parameter groups with different LRs
// (Assuming optimizer supports parameter groups)
```

## Testing Requirements

### Unit Tests for PolynomialDecayScheduler

- Test at step 0: should return initialLR
- Test at step totalSteps/2: should be intermediate value
- Test at step totalSteps: should return finalLR
- Test beyond totalSteps: should return finalLR
- Test with different power values (linear, quadratic, cubic)
- Test state serialization and deserialization

### Unit Tests for LayerWiseLRDecayScheduler

- Test layer multipliers at different positions
- Test excluded layers (should have multiplier = 1.0)
- Test single layer (should have multiplier = 1.0)
- Test with different decay factors
- Test state serialization

### Unit Tests for DiscriminativeLRScheduler

- Test group learning rates by index
- Test group learning rates by name
- Test out-of-range group indices
- Test state serialization

## Estimated Implementation Time
50-60 minutes
