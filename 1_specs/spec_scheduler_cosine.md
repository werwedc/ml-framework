# Spec: Cosine-Based Learning Rate Schedulers

## Overview
Implement two cosine-based learning rate schedulers: Cosine Annealing and Cosine Annealing with Warm Restarts (SGDR). These schedulers are widely used in modern deep learning for their smooth decay properties and ability to escape local minima.

## Dependencies
- `spec_scheduler_base_interface.md` (must be completed first)
- Target namespace: `MLFramework.Schedulers`

## Files to Create
- `src/Schedulers/CosineAnnealingScheduler.cs`
- `src/Schedulers/CosineAnnealingWarmRestartsScheduler.cs`

## Technical Specifications

### 1. Cosine Annealing Scheduler

**Purpose**: Implements cosine annealing schedule that smoothly decays learning rate from baseLR to a minimum value over T_max steps.

**Formula**:
```
LR = eta_min + 0.5 * (baseLR - eta_min) * (1 + cos(pi * step / T_max))
```

**Implementation**:

```csharp
namespace MLFramework.Schedulers;

/// <summary>
/// Cosine annealing learning rate scheduler.
/// Smoothly decays LR from baseLR to eta_min over T_max steps.
/// Formula: LR = eta_min + 0.5 * (baseLR - eta_min) * (1 + cos(pi * step / T_max))
/// </summary>
public sealed class CosineAnnealingScheduler : BaseScheduler, IStepScheduler
{
    private readonly float _tMax;
    private readonly float _etaMin;
    private readonly float _tMaxInversePi;

    public CosineAnnealingScheduler(float tMax, float etaMin = 0f)
    {
        if (tMax <= 0)
        {
            throw new ArgumentException("T_max must be positive", nameof(tMax));
        }

        _tMax = tMax;
        _etaMin = etaMin;

        // Precompute constant for performance
        _tMaxInversePi = (float)Math.PI / tMax;
    }

    public override float GetLearningRate(int step, float baseLearningRate)
    {
        float cosinePhase = _tMaxInversePi * step;
        float cosineValue = (float)Math.Cos(cosinePhase);
        float decayFactor = 0.5f * (1.0f + cosineValue);

        return _etaMin + (baseLearningRate - _etaMin) * decayFactor;
    }

    public override StateDict GetState()
    {
        var state = new StateDict();
        state.Set("t_max", _tMax);
        state.Set("eta_min", _etaMin);
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
- `tMax` (float): Maximum number of iterations in the schedule
- `etaMin` (float): Minimum learning rate (default: 0)

**Example Usage**:
```csharp
// Decay LR from 0.1 to 0 over 100 steps
var scheduler = new CosineAnnealingScheduler(tMax: 100f, etaMin: 0f);

// Decay LR from 0.1 to 1e-6 over 90 epochs
var scheduler = new CosineAnnealingScheduler(tMax: 90f, etaMin: 1e-6f);
```

**Behavior**:
- At step 0: LR = baseLR (cos(0) = 1, decayFactor = 1)
- At step T_max/2: LR = eta_min + 0.5 * (baseLR - eta_min)
- At step T_max: LR = eta_min (cos(π) = -1, decayFactor = 0)

### 2. Cosine Annealing with Warm Restarts Scheduler (SGDR)

**Purpose**: Implements Stochastic Gradient Descent with Warm Restarts. Periodically resets the cosine schedule with potentially increasing cycle lengths.

**Formula**:
```
T_cur = step % T_i
LR = eta_min + 0.5 * (baseLR - eta_min) * (1 + cos(pi * T_cur / T_i))

Where T_i = T_0 * (T_mult^i) and i is the current restart cycle
```

**Implementation**:

```csharp
namespace MLFramework.Schedulers;

/// <summary>
/// Cosine annealing scheduler with warm restarts (SGDR).
/// Resets the schedule every T_0 * (T_mult^i) steps.
/// Formula: LR = eta_min + 0.5 * (baseLR - eta_min) * (1 + cos(pi * T_cur / T_i))
/// </summary>
public sealed class CosineAnnealingWarmRestartsScheduler : BaseScheduler, IStepScheduler
{
    private readonly float _t0;
    private readonly float _tMult;
    private readonly float _etaMin;

    // Derived state
    private float _currentCycleLength;
    private int _cycleCount;

    public CosineAnnealingWarmRestartsScheduler(
        float t0,
        float tMult = 2f,
        float etaMin = 1e-6f)
    {
        if (t0 <= 0)
        {
            throw new ArgumentException("T_0 must be positive", nameof(t0));
        }

        if (tMult < 1f)
        {
            throw new ArgumentException("T_mult must be >= 1", nameof(tMult));
        }

        _t0 = t0;
        _tMult = tMult;
        _etaMin = etaMin;

        _currentCycleLength = _t0;
        _cycleCount = 0;
    }

    public override float GetLearningRate(int step, float baseLearningRate)
    {
        // Calculate position within current cycle
        float stepsInCycle = step % _currentCycleLength;

        // If we've completed a cycle, update cycle length and count
        if (step >= (int)_currentCycleLength * (_cycleCount + 1))
        {
            _cycleCount++;
            _currentCycleLength = _t0 * (float)Math.Pow(_tMult, _cycleCount);
        }

        // Cosine annealing within current cycle
        float cycleProgress = stepsInCycle / _currentCycleLength;
        float cosineValue = (float)Math.Cos(cycleProgress * Math.PI);
        float decayFactor = 0.5f * (1.0f + cosineValue);

        return _etaMin + (baseLearningRate - _etaMin) * decayFactor;
    }

    public override void Reset()
    {
        base.Reset();
        _currentCycleLength = _t0;
        _cycleCount = 0;
    }

    public override StateDict GetState()
    {
        var state = new StateDict();
        state.Set("t_0", _t0);
        state.Set("t_mult", _tMult);
        state.Set("eta_min", _etaMin);
        state.Set("step_count", _stepCount);
        state.Set("epoch_count", _epochCount);
        state.Set("cycle_count", _cycleCount);
        state.Set("current_cycle_length", _currentCycleLength);
        return state;
    }

    public override void LoadState(StateDict state)
    {
        _stepCount = state.Get<int>("step_count", 0);
        _epochCount = state.Get<int>("epoch_count", 0);
        _cycleCount = state.Get<int>("cycle_count", 0);
        _currentCycleLength = state.Get<float>("current_cycle_length", _t0);
    }
}
```

**Constructor Parameters**:
- `t0` (float): Length of first restart cycle
- `tMult` (float): Multiplier for cycle length after each restart (default: 2)
- `etaMin` (float): Minimum learning rate (default: 1e-6)

**Example Usage**:
```csharp
// First cycle 10 steps, then 20, 40, 80, ...
var scheduler = new CosineAnnealingWarmRestartsScheduler(
    t0: 10f,
    tMult: 2f,
    etaMin: 1e-6f
);

// First cycle 50 steps, constant cycle length
var scheduler = new CosineAnnealingWarmRestartsScheduler(
    t0: 50f,
    tMult: 1f,
    etaMin: 0f
);
```

**Behavior**:
- Each cycle follows a cosine annealing pattern
- At the end of each cycle, learning rate jumps back to baseLR
- Cycle lengths grow multiplicatively if T_mult > 1
- When T_mult = 1, cycles have constant length

## Implementation Notes

### Design Decisions

1. **Performance Optimization**:
   - `CosineAnnealingScheduler` precomputes `π / T_max` to avoid division in `GetLearningRate`
   - Both schedulers use direct float operations for optimal performance

2. **Cycle Tracking**:
   - `CosineAnnealingWarmRestartsScheduler` tracks cycle count and current cycle length as derived state
   - These are included in state serialization for proper checkpointing

3. **Validation**:
   - `T_max` and `T_0` must be positive
   - `T_mult` must be >= 1 to ensure non-decreasing cycle lengths

### Edge Cases

- **T_max/step boundaries**: When step == T_max, cosine value is exactly -1, so LR = eta_min
- **Cycle transitions**: At exact cycle boundaries, the scheduler should transition smoothly
- **T_mult = 1**: Creates constant-length cycles, useful for periodic LR restarts
- **eta_min >= baseLR**: Scheduler will increase learning rate (unusual but valid)

### Performance Considerations

- Both schedulers are O(1) per call
- `CosineAnnealingWarmRestartsScheduler.GetLearningRate` requires additional state tracking but still O(1)
- Math.Cos and Math.Pow are the only non-trivial operations

## Testing Requirements

### Unit Tests for CosineAnnealingScheduler

- Test boundary values:
  - At step 0: should return baseLR
  - At step T_max/2: should return eta_min + 0.5 * (baseLR - eta_min)
  - At step T_max: should return eta_min

- Test with different eta_min values:
  - eta_min = 0 (decay to zero)
  - eta_min > 0 (decay to non-zero minimum)

- Test state serialization and deserialization

- Test reset functionality

### Unit Tests for CosineAnnealingWarmRestartsScheduler

- Test first cycle behavior:
  - At step 0: should return baseLR
  - At step T_0/2: should return eta_min + 0.5 * (baseLR - eta_min)
  - At step T_0: should return eta_min (end of first cycle)

- Test cycle transitions:
  - At step T_0 + 1: should jump back toward baseLR (start of second cycle)
  - Verify cycle length increases correctly for T_mult > 1

- Test with T_mult = 1 (constant cycle lengths)

- Test state serialization and deserialization including cycle tracking

- Test reset functionality (should reset cycle count and length)

## Mathematical Verification

All schedulers should be verified against known PyTorch behavior:

```csharp
// Test: CosineAnnealingScheduler(T_max=100, eta_min=0)
// At step=0:   cos(0) = 1,    LR = 0 + 0.5 * (baseLR - 0) * (1 + 1) = baseLR
// At step=50:  cos(π/2) = 0,  LR = 0 + 0.5 * (baseLR - 0) * (1 + 0) = baseLR/2
// At step=100: cos(π) = -1,   LR = 0 + 0.5 * (baseLR - 0) * (1 - 1) = 0

// Test: CosineAnnealingWarmRestartsScheduler(T_0=10, T_mult=2)
// First cycle (0-9): T_i = 10
// Second cycle (10-29): T_i = 20
// Third cycle (30-69): T_i = 40
```

## Estimated Implementation Time
50-60 minutes
