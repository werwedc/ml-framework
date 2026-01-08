# Spec: Simple Decay Schedulers

## Overview
Implement three fundamental learning rate decay schedulers: Step Decay, Multi-Step Decay, and Exponential Decay. These are the most commonly used schedulers in practice and serve as building blocks for more complex scheduling strategies.

## Dependencies
- `spec_scheduler_base_interface.md` (must be completed first)
- Target namespace: `MLFramework.Schedulers`

## Files to Create
- `src/Schedulers/StepDecayScheduler.cs`
- `src/Schedulers/MultiStepDecayScheduler.cs`
- `src/Schedulers/ExponentialDecayScheduler.cs`

## Technical Specifications

### 1. Step Decay Scheduler

**Purpose**: Decays learning rate by a fixed factor (`gamma`) at regular intervals (`stepSize`).

**Formula**:
```
LR = baseLR * gamma^(step / stepSize)
```

**Implementation**:

```csharp
namespace MLFramework.Schedulers;

/// <summary>
/// Decays learning rate by gamma every step_size steps.
/// Example: LR * 0.1 every 30 epochs.
/// </summary>
public sealed class StepDecayScheduler : BaseScheduler, IStepScheduler
{
    private readonly int _stepSize;
    private readonly float _gamma;

    public StepDecayScheduler(int stepSize, float gamma = 0.1f)
    {
        _stepSize = stepSize;
        _gamma = gamma;
    }

    public override float GetLearningRate(int step, float baseLearningRate)
    {
        int decayCount = step / _stepSize;
        float decayFactor = (float)Math.Pow(_gamma, decayCount);
        return baseLearningRate * decayFactor;
    }

    public override StateDict GetState()
    {
        var state = new StateDict();
        state.Set("step_size", _stepSize);
        state.Set("gamma", _gamma);
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
- `stepSize` (int): Number of steps between decays
- `gamma` (float): Multiplicative factor of decay (default: 0.1)

**Example Usage**:
```csharp
// LR * 0.1 every 30 epochs
var scheduler = new StepDecayScheduler(stepSize: 30, gamma: 0.1f);
```

### 2. Multi-Step Decay Scheduler

**Purpose**: Decays learning rate by a fixed factor at specific milestone steps.

**Formula**:
```
LR = baseLR * gamma^k
where k = number of milestones passed
```

**Implementation**:

```csharp
namespace MLFramework.Schedulers;

/// <summary>
/// Decays learning rate by gamma at specific milestone steps.
/// Example: LR * 0.1 at epochs 30, 60, 90.
/// </summary>
public sealed class MultiStepDecayScheduler : BaseScheduler, IStepScheduler
{
    private readonly int[] _milestones;
    private readonly float _gamma;

    public MultiStepDecayScheduler(int[] milestones, float gamma = 0.1f)
    {
        _milestones = milestones ?? Array.Empty<int>();
        _gamma = gamma;
    }

    public override float GetLearningRate(int step, float baseLearningRate)
    {
        int decayCount = _milestones.Count(m => step >= m);
        float decayFactor = (float)Math.Pow(_gamma, decayCount);
        return baseLearningRate * decayFactor;
    }

    public override StateDict GetState()
    {
        var state = new StateDict();
        state.Set("milestones", _milestones);
        state.Set("gamma", _gamma);
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
- `milestones` (int[]): Array of step indices where decay should occur
- `gamma` (float): Multiplicative factor of decay (default: 0.1)

**Example Usage**:
```csharp
// LR * 0.1 at epochs 30, 60, 90
var scheduler = new MultiStepDecayScheduler(
    milestones: new[] { 30, 60, 90 },
    gamma: 0.1f
);
```

### 3. Exponential Decay Scheduler

**Purpose**: Decays learning rate exponentially at every step.

**Formula**:
```
LR = baseLR * gamma^step
```

**Implementation**:

```csharp
namespace MLFramework.Schedulers;

/// <summary>
/// Decays learning rate exponentially every step.
/// Formula: LR = baseLR * gamma^step
/// </summary>
public sealed class ExponentialDecayScheduler : BaseScheduler, IStepScheduler
{
    private readonly float _gamma;

    public ExponentialDecayScheduler(float gamma = 0.95f)
    {
        _gamma = gamma;
    }

    public override float GetLearningRate(int step, float baseLearningRate)
    {
        float decayFactor = (float)Math.Pow(_gamma, step);
        return baseLearningRate * decayFactor;
    }

    public override StateDict GetState()
    {
        var state = new StateDict();
        state.Set("gamma", _gamma);
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
- `gamma` (float): Exponential decay factor (default: 0.95)

**Example Usage**:
```csharp
// Exponential decay with factor 0.95 per step
var scheduler = new ExponentialDecayScheduler(gamma: 0.95f);
```

## Implementation Notes

### Design Decisions
1. **Sealed Classes**: All three schedulers are marked `sealed` as they are not intended to be inherited.

2. **Step-Based Scheduling**: All implement `IStepScheduler` interface to indicate they operate on step-wise basis, though they can be called with epoch counts as well.

3. **Math.Pow**: Uses `Math.Pow` for consistent floating-point calculation across different schedulers.

4. **Immutable Parameters**: Configuration parameters (stepSize, gamma, milestones) are stored as readonly fields.

### Edge Cases
- **StepDecay**: When `stepSize <= 0`, will cause division by zero in calculation. Constructor should validate and throw `ArgumentException`.

- **MultiStepDecay**: Empty milestones array is valid (no decay). Milestones should be sorted but not strictly required (Count works regardless).

- **ExponentialDecay**: When `gamma >= 1.0`, learning rate will not decay. This is valid but unusual - consider adding validation warning.

### Performance Considerations
- `MultiStepDecayScheduler.GetLearningRate` uses LINQ's `Count`, which iterates through milestones array each call. For large milestone arrays, consider optimizing with binary search.

- All calculations are O(1) except MultiStepDecay which is O(n) where n = number of milestones.

## Testing Requirements

### Unit Tests for StepDecayScheduler
- Test decay at exact step boundaries (e.g., step 0, step 29, step 30, step 60)
- Test state serialization and deserialization
- Test reset functionality
- Test with various gamma values

### Unit Tests for MultiStepDecayScheduler
- Test decay at each milestone
- Test that decay accumulates correctly (gamma^2 after second milestone)
- Test with empty milestones array (should not decay)
- Test with unsorted milestones array
- Test state serialization and deserialization

### Unit Tests for ExponentialDecayScheduler
- Test exponential decay formula at steps 0, 10, 100
- Test state serialization and deserialization
- Test reset functionality
- Test with gamma = 1.0 (no decay)
- Test with gamma < 1.0 (decay)

## Mathematical Verification

All schedulers should be verified against known PyTorch behavior:

```csharp
// Test: StepDecayScheduler(stepSize=30, gamma=0.1)
// At step=0:  LR = baseLR * 0.1^0 = baseLR
// At step=29: LR = baseLR * 0.1^0 = baseLR
// At step=30: LR = baseLR * 0.1^1 = baseLR * 0.1
// At step=60: LR = baseLR * 0.1^2 = baseLR * 0.01
```

## Estimated Implementation Time
45-60 minutes
