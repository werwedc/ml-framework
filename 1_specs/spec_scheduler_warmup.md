# Spec: Warmup Learning Rate Schedulers

## Overview
Implement learning rate warmup schedulers that gradually increase the learning rate during the initial phase of training. Warmup is critical for stabilizing training, especially with large batch sizes and transformer models.

## Dependencies
- `spec_scheduler_base_interface.md` (must be completed first)
- Target namespace: `MLFramework.Schedulers`

## Files to Create
- `src/Schedulers/LinearWarmupScheduler.cs`
- `src/Schedulers/ConstantWarmupScheduler.cs`
- `src/Schedulers/WarmupSchedulerBase.cs` (helper base class)

## Technical Specifications

### 1. Warmup Scheduler Base Class

**Purpose**: Provides common functionality for warmup schedulers that wrap another scheduler.

```csharp
namespace MLFramework.Schedulers;

/// <summary>
/// Abstract base class for warmup schedulers.
/// Wraps another scheduler and applies warmup during the initial warmupSteps.
/// </summary>
public abstract class WarmupSchedulerBase : BaseScheduler, IStepScheduler
{
    protected readonly ILearningRateScheduler _baseScheduler;
    protected readonly int _warmupSteps;

    public WarmupSchedulerBase(ILearningRateScheduler baseScheduler, int warmupSteps)
    {
        _baseScheduler = baseScheduler ?? throw new ArgumentNullException(nameof(baseScheduler));
        _warmupSteps = warmupSteps;

        if (warmupSteps < 0)
        {
            throw new ArgumentException("warmupSteps must be non-negative", nameof(warmupSteps));
        }
    }

    protected abstract float GetWarmupLearningRate(int step, float baseLearningRate);

    public override float GetLearningRate(int step, float baseLearningRate)
    {
        if (step < _warmupSteps)
        {
            return GetWarmupLearningRate(step, baseLearningRate);
        }
        else
        {
            // After warmup, delegate to base scheduler
            return _baseScheduler.GetLearningRate(step - _warmupSteps, baseLearningRate);
        }
    }

    public override void Step()
    {
        base.Step();
        _baseScheduler?.Step();
    }

    public override void Reset()
    {
        base.Reset();
        _baseScheduler?.Reset();
    }

    public override StateDict GetState()
    {
        var state = new StateDict();
        state.Set("warmup_steps", _warmupSteps);
        state.Set("step_count", _stepCount);
        state.Set("epoch_count", _epochCount);
        state.Set("base_scheduler_state", _baseScheduler?.GetState());
        return state;
    }

    public override void LoadState(StateDict state)
    {
        _stepCount = state.Get<int>("step_count", 0);
        _epochCount = state.Get<int>("epoch_count", 0);

        var baseState = state.Get<StateDict>("base_scheduler_state");
        if (baseState != null && _baseScheduler != null)
        {
            _baseScheduler.LoadState(baseState);
        }
    }
}
```

### 2. Linear Warmup Scheduler

**Purpose**: Linearly increases learning rate from 0 (or a small initial value) to the base learning rate over the warmup period.

**Formula**:
```
if step < warmupSteps:
    LR = baseLR * (step / warmupSteps)
else:
    LR = baseScheduler.GetLearningRate(step - warmupSteps, baseLR)
```

**Implementation**:

```csharp
namespace MLFramework.Schedulers;

/// <summary>
/// Linearly increases learning rate from 0 to baseLR over warmupSteps.
/// After warmup, delegates to the base scheduler.
/// </summary>
public sealed class LinearWarmupScheduler : WarmupSchedulerBase
{
    private readonly float _startLearningRate;

    public LinearWarmupScheduler(
        ILearningRateScheduler baseScheduler,
        int warmupSteps,
        float startLearningRate = 0f) : base(baseScheduler, warmupSteps)
    {
        _startLearningRate = startLearningRate;

        if (startLearningRate < 0)
        {
            throw new ArgumentException("startLearningRate must be non-negative", nameof(startLearningRate));
        }
    }

    protected override float GetWarmupLearningRate(int step, float baseLearningRate)
    {
        if (_warmupSteps == 0)
        {
            return baseLearningRate;
        }

        float progress = (float)step / _warmupSteps;
        return _startLearningRate + (baseLearningRate - _startLearningRate) * progress;
    }

    public override StateDict GetState()
    {
        var state = base.GetState();
        state.Set("start_lr", _startLearningRate);
        return state;
    }
}
```

**Constructor Parameters**:
- `baseScheduler` (ILearningRateScheduler): The scheduler to use after warmup
- `warmupSteps` (int): Number of warmup steps
- `startLearningRate` (float): Starting learning rate (default: 0)

**Example Usage**:
```csharp
var baseScheduler = new CosineAnnealingScheduler(tMax: 9000f);
var scheduler = new LinearWarmupScheduler(
    baseScheduler: baseScheduler,
    warmupSteps: 1000
);
```

### 3. Constant Warmup Scheduler

**Purpose**: Uses a constant warmup learning rate during the warmup period, then transitions to the base scheduler.

**Formula**:
```
if step < warmupSteps:
    LR = warmupLR
else:
    LR = baseScheduler.GetLearningRate(step - warmupSteps, baseLR)
```

**Implementation**:

```csharp
namespace MLFramework.Schedulers;

/// <summary>
/// Uses a constant warmup learning rate during warmupSteps.
/// After warmup, delegates to the base scheduler.
/// </summary>
public sealed class ConstantWarmupScheduler : WarmupSchedulerBase
{
    private readonly float _warmupLearningRate;

    public ConstantWarmupScheduler(
        ILearningRateScheduler baseScheduler,
        int warmupSteps,
        float warmupLearningRate) : base(baseScheduler, warmupSteps)
    {
        if (warmupLearningRate <= 0)
        {
            throw new ArgumentException("warmupLearningRate must be positive", nameof(warmupLearningRate));
        }

        _warmupLearningRate = warmupLearningRate;
    }

    protected override float GetWarmupLearningRate(int step, float baseLearningRate)
    {
        return _warmupLearningRate;
    }

    public override StateDict GetState()
    {
        var state = base.GetState();
        state.Set("warmup_lr", _warmupLearningRate);
        return state;
    }
}
```

**Constructor Parameters**:
- `baseScheduler` (ILearningRateScheduler): The scheduler to use after warmup
- `warmupSteps` (int): Number of warmup steps
- `warmupLearningRate` (float): Constant learning rate during warmup

**Example Usage**:
```csharp
var baseScheduler = new CosineAnnealingScheduler(tMax: 9000f);
var scheduler = new ConstantWarmupScheduler(
    baseScheduler: baseScheduler,
    warmupSteps: 500,
    warmupLearningRate: 1e-6f
);
```

## Implementation Notes

### Design Decisions

1. **Base Class Pattern**:
   - `WarmupSchedulerBase` provides common delegation logic
   - Concrete implementations only need to define the warmup strategy
   - Reduces code duplication between different warmup types

2. **Base Scheduler Handling**:
   - Warmup schedulers wrap another scheduler
   - During warmup: use warmup strategy
   - After warmup: delegate to base scheduler with adjusted step count

3. **Step Adjustment**:
   - When calling base scheduler, subtract warmupSteps from the step count
   - This ensures the base scheduler starts at step 0 after warmup

4. **State Management**:
   - Both warmup state and base scheduler state are preserved
   - Enables proper checkpointing and resumption

### Edge Cases

- **warmupSteps = 0**: Should immediately delegate to base scheduler (no warmup)
- **warmupSteps < 0**: Constructor throws ArgumentException
- **baseScheduler is null**: Constructor throws ArgumentNullException
- **step during warmup**: Always returns warmup LR (ignores baseScheduler)
- **step after warmup**: Delegates to baseScheduler with adjusted step

### Performance Considerations

- Warmup schedulers add minimal overhead
- During warmup: one extra method call and arithmetic operation
- After warmup: delegates to base scheduler (same cost as using base scheduler directly)

## Usage Patterns

### Pattern 1: Warmup + Cosine Annealing
```csharp
// Common pattern for transformer training
var baseScheduler = new CosineAnnealingScheduler(tMax: 9000f, etaMin: 1e-6f);
var scheduler = new LinearWarmupScheduler(
    baseScheduler: baseScheduler,
    warmupSteps: 1000
);
```

### Pattern 2: Warmup + Multi-Step Decay
```csharp
// Warmup + milestones for vision models
var baseScheduler = new MultiStepDecayScheduler(
    milestones: new[] { 30, 60, 90 },
    gamma: 0.1f
);
var scheduler = new LinearWarmupScheduler(
    baseScheduler: baseScheduler,
    warmupSteps: 1000
);
```

### Pattern 3: Two-Stage Warmup
```csharp
// Complex pattern: constant then linear warmup
var innerScheduler = new CosineAnnealingScheduler(tMax: 9000f);
var linearWarmup = new LinearWarmupScheduler(innerScheduler, 500);
var constantWarmup = new ConstantWarmupScheduler(linearWarmup, 500, 1e-7f);
```

## Testing Requirements

### Unit Tests for LinearWarmupScheduler

- Test warmup behavior:
  - At step 0: should return startLR
  - At step warmupSteps/2: should return (startLR + baseLR) / 2
  - At step warmupSteps - 1: should be close to baseLR

- Test post-warmup behavior:
  - At step warmupSteps: should delegate to base scheduler with step=0
  - Verify base scheduler is called with adjusted step count

- Test with different startLR values:
  - startLR = 0 (default)
  - startLR > 0

- Test state serialization and deserialization

- Test reset functionality (should reset both warmup and base scheduler)

### Unit Tests for ConstantWarmupScheduler

- Test warmup behavior:
  - During warmup: should always return warmupLR
  - Verify it doesn't depend on step or baseLearningRate

- Test post-warmup behavior:
  - At step warmupSteps: should delegate to base scheduler with step=0
  - Verify base scheduler is called with adjusted step count

- Test state serialization and deserialization

- Test reset functionality

### Unit Tests for WarmupSchedulerBase

- Test that warmupSteps = 0 works correctly (delegates immediately)
- Test Step() and Reset() delegation to base scheduler
- Test null base scheduler handling

## Mathematical Verification

```csharp
// LinearWarmupScheduler: warmupSteps=1000, startLR=0
// step=0:    LR = 0 + (baseLR - 0) * 0/1000 = 0
// step=500:  LR = 0 + (baseLR - 0) * 500/1000 = baseLR/2
// step=999:  LR = 0 + (baseLR - 0) * 999/1000 ≈ baseLR
// step=1000: LR = baseScheduler.GetLearningRate(0, baseLR)

// ConstantWarmupScheduler: warmupSteps=500, warmupLR=1e-6
// step=0-499:   LR = 1e-6
// step=500:     LR = baseScheduler.GetLearningRate(0, baseLR)
```

## Estimated Implementation Time
40-50 minutes
