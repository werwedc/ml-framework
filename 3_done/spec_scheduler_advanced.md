# Spec: Advanced Learning Rate Schedulers

## Overview
Implement three advanced learning rate schedulers: One Cycle Policy, Cyclic Learning Rate, and Reduce LR on Plateau. These schedulers are used in state-of-the-art training workflows for achieving fast convergence and avoiding local minima.

## Dependencies
- `spec_scheduler_base_interface.md` (must be completed first)
- Target namespace: `MLFramework.Schedulers`

## Files to Create
- `src/Schedulers/OneCycleScheduler.cs`
- `src/Schedulers/CyclicLRScheduler.cs`
- `src/Schedulers/ReduceLROnPlateauScheduler.cs`

## Technical Specifications

### 1. One Cycle Scheduler

**Purpose**: Popularized by fast.ai, implements a learning rate policy that increases from a small initial value to a maximum, then decreases to a final minimum over a single cycle.

**Formula**:
```
if step < pctStart * totalSteps:
    LR = initialLR + (maxLR - initialLR) * pct
else:
    if annealStrategy == "cos":
        LR = finalLR + (maxLR - finalLR) * (1 + cos(pi * pct)) / 2
    else:
        LR = maxLR - (maxLR - finalLR) * pct

where pct is the fraction of totalSteps in the current phase
```

**Implementation**:

```csharp
namespace MLFramework.Schedulers;

/// <summary>
/// One cycle learning rate scheduler from fast.ai.
/// Increases LR from initialLR to maxLR, then decreases to finalLR over one cycle.
/// </summary>
public sealed class OneCycleScheduler : BaseScheduler, IStepScheduler
{
    private readonly float _maxLearningRate;
    private readonly float _totalSteps;
    private readonly float _pctStart;
    private readonly string _annealStrategy;
    private readonly float _divFactor;
    private readonly float _finalDivFactor;

    // Derived parameters
    private readonly float _initialLearningRate;
    private readonly float _finalLearningRate;

    public OneCycleScheduler(
        float maxLearningRate,
        float totalSteps,
        float pctStart = 0.3f,
        string annealStrategy = "cos",
        float divFactor = 25f,
        float finalDivFactor = 1e4f)
    {
        if (maxLearningRate <= 0)
            throw new ArgumentException("maxLearningRate must be positive", nameof(maxLearningRate));
        if (totalSteps <= 0)
            throw new ArgumentException("totalSteps must be positive", nameof(totalSteps));
        if (pctStart <= 0 || pctStart >= 1)
            throw new ArgumentException("pctStart must be in (0, 1)", nameof(pctStart));
        if (annealStrategy != "cos" && annealStrategy != "linear")
            throw new ArgumentException("annealStrategy must be 'cos' or 'linear'", nameof(annealStrategy));

        _maxLearningRate = maxLearningRate;
        _totalSteps = totalSteps;
        _pctStart = pctStart;
        _annealStrategy = annealStrategy;
        _divFactor = divFactor;
        _finalDivFactor = finalDivFactor;

        _initialLearningRate = maxLearningRate / divFactor;
        _finalLearningRate = maxLearningRate / finalDivFactor;
    }

    public override float GetLearningRate(int step, float baseLearningRate)
    {
        if (step >= _totalSteps)
        {
            step = _totalSteps - 1;
        }

        float pctStartSteps = _pctStart * _totalSteps;

        if (step < pctStartSteps)
        {
            // Increasing phase: initialLR -> maxLR
            float pct = step / pctStartSteps;
            return _initialLearningRate + (_maxLearningRate - _initialLearningRate) * pct;
        }
        else
        {
            // Decreasing phase: maxLR -> finalLR
            float pct = (step - pctStartSteps) / (_totalSteps - pctStartSteps);

            if (_annealStrategy == "cos")
            {
                float cosineValue = (float)Math.Cos(pct * Math.PI);
                return _finalLearningRate + (_maxLearningRate - _finalLearningRate) * (1 + cosineValue) / 2;
            }
            else // linear
            {
                return _maxLearningRate - (_maxLearningRate - _finalLearningRate) * pct;
            }
        }
    }

    public override StateDict GetState()
    {
        var state = new StateDict();
        state.Set("max_lr", _maxLearningRate);
        state.Set("total_steps", _totalSteps);
        state.Set("pct_start", _pctStart);
        state.Set("anneal_strategy", _annealStrategy);
        state.Set("div_factor", _divFactor);
        state.Set("final_div_factor", _finalDivFactor);
        state.Set("initial_lr", _initialLearningRate);
        state.Set("final_lr", _finalLearningRate);
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
- `maxLearningRate` (float): Peak learning rate
- `totalSteps` (float): Total number of steps in the cycle
- `pctStart` (float): Percentage of cycle for increasing phase (default: 0.3)
- `annealStrategy` (string): "cos" or "linear" for decreasing phase (default: "cos")
- `divFactor` (float): Factor to determine initialLR = maxLR / divFactor (default: 25)
- `finalDivFactor` (float): Factor to determine finalLR = maxLR / finalDivFactor (default: 1e4)

**Example Usage**:
```csharp
// One cycle over 1000 steps, 30% increasing phase
var scheduler = new OneCycleScheduler(
    maxLearningRate: 0.1f,
    totalSteps: 1000f,
    pctStart: 0.3f
);
```

### 2. Cyclic Learning Rate Scheduler

**Purpose**: Implements cyclical learning rate policy that oscillates between baseLR and maxLR. Helps escape saddle points and find better local minima.

**Formula**:
```
cycle = floor(1 + step / (2 * stepSizeUp))
x = abs(step / stepSizeUp - 2 * cycle + 1)

if mode == "triangular":
    LR = baseLR + (maxLR - baseLR) * max(0, 1 - x)
elif mode == "triangular2":
    LR = baseLR + (maxLR - baseLR) * max(0, 1 - x) / (2^(cycle - 1))
elif mode == "exp_range":
    LR = baseLR + (maxLR - baseLR) * max(0, 1 - x) * gamma^step
```

**Implementation**:

```csharp
namespace MLFramework.Schedulers;

/// <summary>
/// Cyclic learning rate scheduler.
/// Oscillates LR between baseLR and maxLR using triangular or exponential policy.
/// </summary>
public sealed class CyclicLRScheduler : BaseScheduler, IStepScheduler
{
    private readonly float _baseLearningRate;
    private readonly float _maxLearningRate;
    private readonly float _stepSizeUp;
    private readonly string _mode;
    private readonly float _gamma;
    private readonly float _stepSizeDown;

    public CyclicLRScheduler(
        float baseLearningRate,
        float maxLearningRate,
        float stepSizeUp,
        string mode = "triangular",
        float gamma = 0.99994f)
    {
        if (baseLearningRate <= 0)
            throw new ArgumentException("baseLearningRate must be positive", nameof(baseLearningRate));
        if (maxLearningRate <= 0)
            throw new ArgumentException("maxLearningRate must be positive", nameof(maxLearningRate));
        if (maxLearningRate <= baseLearningRate)
            throw new ArgumentException("maxLearningRate must be > baseLearningRate", nameof(maxLearningRate));
        if (stepSizeUp <= 0)
            throw new ArgumentException("stepSizeUp must be positive", nameof(stepSizeUp));
        if (mode != "triangular" && mode != "triangular2" && mode != "exp_range")
            throw new ArgumentException("mode must be 'triangular', 'triangular2', or 'exp_range'", nameof(mode));
        if (gamma <= 0 || gamma >= 1)
            throw new ArgumentException("gamma must be in (0, 1)", nameof(gamma));

        _baseLearningRate = baseLearningRate;
        _maxLearningRate = maxLearningRate;
        _stepSizeUp = stepSizeUp;
        _mode = mode;
        _gamma = gamma;
        _stepSizeDown = stepSizeUp; // Default: symmetric cycle
    }

    public override float GetLearningRate(int step, float baseLearningRate)
    {
        float cycle = (float)Math.Floor(1 + step / (2 * _stepSizeUp));
        float x = Math.Abs(step / _stepSizeUp - 2 * cycle + 1);
        float baseHeight = Math.Max(0, 1 - x);

        float scale;
        if (_mode == "triangular")
        {
            scale = 1;
        }
        else if (_mode == "triangular2")
        {
            scale = (float)Math.Pow(2, -(cycle - 1));
        }
        else // exp_range
        {
            scale = (float)Math.Pow(_gamma, step);
        }

        return _baseLearningRate + (_maxLearningRate - _baseLearningRate) * baseHeight * scale;
    }

    public override StateDict GetState()
    {
        var state = new StateDict();
        state.Set("base_lr", _baseLearningRate);
        state.Set("max_lr", _maxLearningRate);
        state.Set("step_size_up", _stepSizeUp);
        state.Set("step_size_down", _stepSizeDown);
        state.Set("mode", _mode);
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
- `baseLearningRate` (float): Minimum learning rate
- `maxLearningRate` (float): Maximum learning rate
- `stepSizeUp` (float): Number of iterations in the increasing half of a cycle
- `mode` (string): "triangular", "triangular2", or "exp_range" (default: "triangular")
- `gamma` (float): Decay rate for exp_range mode (default: 0.99994)

**Example Usage**:
```csharp
// Triangular policy: cycle every 4000 steps (2000 up, 2000 down)
var scheduler = new CyclicLRScheduler(
    baseLearningRate: 1e-4f,
    maxLearningRate: 1e-2f,
    stepSizeUp: 2000f,
    mode: "triangular"
);
```

### 3. Reduce LR on Plateau Scheduler

**Purpose**: Reduces learning rate when a metric has stopped improving. Useful for adaptive training based on validation loss or accuracy.

**Implementation**:

```csharp
namespace MLFramework.Schedulers;

/// <summary>
/// Reduces learning rate when a metric has stopped improving.
/// </summary>
public sealed class ReduceLROnPlateauScheduler : BaseScheduler, IMetricBasedScheduler
{
    private readonly string _mode;
    private readonly float _factor;
    private readonly int _patience;
    private readonly float _threshold;
    private readonly int _cooldown;
    private readonly float _minLearningRate;

    // State
    private int _wait;
    private int _cooldownCounter;
    private float _bestMetric;
    private float _currentLearningRate;

    public ReduceLROnPlateauScheduler(
        string mode = "min",
        float factor = 0.1f,
        int patience = 10,
        float threshold = 1e-4f,
        int cooldown = 0,
        float minLearningRate = 1e-6f)
    {
        if (mode != "min" && mode != "max")
            throw new ArgumentException("mode must be 'min' or 'max'", nameof(mode));
        if (factor <= 0 || factor >= 1)
            throw new ArgumentException("factor must be in (0, 1)", nameof(factor));
        if (patience <= 0)
            throw new ArgumentException("patience must be positive", nameof(patience));
        if (threshold < 0)
            throw new ArgumentException("threshold must be non-negative", nameof(threshold));
        if (cooldown < 0)
            throw new ArgumentException("cooldown must be non-negative", nameof(cooldown));
        if (minLearningRate <= 0)
            throw new ArgumentException("minLearningRate must be positive", nameof(minLearningRate));

        _mode = mode;
        _factor = factor;
        _patience = patience;
        _threshold = threshold;
        _cooldown = cooldown;
        _minLearningRate = minLearningRate;

        Reset();
    }

    public override float GetLearningRate(int step, float baseLearningRate)
    {
        return _currentLearningRate;
    }

    public void UpdateMetric(string metricName, float value)
    {
        _stepCount++; // Update step count on metric update

        // Check if we're in cooldown
        if (_cooldownCounter > 0)
        {
            _cooldownCounter--;
            return;
        }

        bool isBetter;
        if (_mode == "min")
        {
            // For 'min' mode, lower metric is better (e.g., loss)
            isBetter = value < _bestMetric * (1 - _threshold);
        }
        else // 'max' mode
        {
            // For 'max' mode, higher metric is better (e.g., accuracy)
            isBetter = value > _bestMetric * (1 + _threshold);
        }

        if (isBetter)
        {
            _bestMetric = value;
            _wait = 0;
        }
        else
        {
            _wait++;
            if (_wait >= _patience)
            {
                // Reduce learning rate
                float newLR = _currentLearningRate * _factor;
                _currentLearningRate = Math.Max(newLR, _minLearningRate);
                _wait = 0;
                _cooldownCounter = _cooldown;
            }
        }
    }

    public override void Reset()
    {
        base.Reset();
        _wait = 0;
        _cooldownCounter = 0;
        _bestMetric = _mode == "min" ? float.MaxValue : float.MinValue;
        _currentLearningRate = float.MaxValue; // Will be set on first GetLearningRate call
    }

    public override StateDict GetState()
    {
        var state = new StateDict();
        state.Set("mode", _mode);
        state.Set("factor", _factor);
        state.Set("patience", _patience);
        state.Set("threshold", _threshold);
        state.Set("cooldown", _cooldown);
        state.Set("min_lr", _minLearningRate);
        state.Set("wait", _wait);
        state.Set("cooldown_counter", _cooldownCounter);
        state.Set("best_metric", _bestMetric);
        state.Set("current_lr", _currentLearningRate);
        state.Set("step_count", _stepCount);
        state.Set("epoch_count", _epochCount);
        return state;
    }

    public override void LoadState(StateDict state)
    {
        _stepCount = state.Get<int>("step_count", 0);
        _epochCount = state.Get<int>("epoch_count", 0);
        _wait = state.Get<int>("wait", 0);
        _cooldownCounter = state.Get<int>("cooldown_counter", 0);
        _bestMetric = state.Get<float>("best_metric", _mode == "min" ? float.MaxValue : float.MinValue);
        _currentLearningRate = state.Get<float>("current_lr", float.MaxValue);
    }
}
```

**Constructor Parameters**:
- `mode` (string): "min" for loss, "max" for accuracy (default: "min")
- `factor` (float): Factor to multiply learning rate by (default: 0.1)
- `patience` (int): Number of bad epochs before reducing LR (default: 10)
- `threshold` (float): Threshold for measuring new optimum (default: 1e-4)
- `cooldown` (int): Number of epochs to wait after LR reduction (default: 0)
- `minLearningRate` (float): Lower bound on LR (default: 1e-6)

**Example Usage**:
```csharp
// Reduce LR by 0.1 when validation loss doesn't improve for 10 epochs
var scheduler = new ReduceLROnPlateauScheduler(
    mode: "min",
    factor: 0.1f,
    patience: 10,
    minLearningRate: 1e-6f
);

// During training:
scheduler.UpdateMetric("val_loss", currentValLoss);
float currentLR = scheduler.GetLearningRate(step, baseLR);
```

## Implementation Notes

### Design Decisions

1. **OneCycleScheduler**:
   - Ignores `baseLearningRate` parameter and uses internal LR calculations
   - Validates that pctStart is in (0, 1) to avoid division by zero
   - Caps step to totalSteps - 1 to handle out-of-range queries

2. **CyclicLRScheduler**:
   - Implements symmetric cycles (stepSizeUp = stepSizeDown) for simplicity
   - Uses max(0, 1 - x) to ensure non-negative scaling factor
   - Three modes provide different decay strategies

3. **ReduceLROnPlateauScheduler**:
   - Implements both 'min' (for loss) and 'max' (for accuracy) modes
   - Tracks best metric value and wait counter as state
   - Uses cooldown to prevent multiple LR reductions in quick succession

### Edge Cases

- **OneCycleScheduler**: When step >= totalSteps, returns the final LR value
- **CyclicLRScheduler**: When mode="triangular2", amplitude decays to zero over many cycles
- **ReduceLROnPlateau**: If metric doesn't improve initially, may never reduce LR (consider setting a reasonable initial bestMetric)

### Performance Considerations

- All schedulers are O(1) per call
- ReduceLROnPlateau requires stateful metric tracking, which adds minimal overhead
- OneCycle and CyclicLR use only basic arithmetic operations

## Testing Requirements

### Unit Tests for OneCycleScheduler

- Test boundary conditions at step 0, pctStart * totalSteps, and totalSteps
- Test both "cos" and "linear" anneal strategies
- Test with different divFactor and finalDivFactor values
- Test state serialization and deserialization

### Unit Tests for CyclicLRScheduler

- Test one complete cycle (2 * stepSizeUp steps)
- Test all three modes: triangular, triangular2, exp_range
- Verify amplitude decay in triangular2 mode
- Verify exponential decay in exp_range mode
- Test state serialization and deserialization

### Unit Tests for ReduceLROnPlateauScheduler

- Test with mode="min" (improving metric decreases)
- Test with mode="max" (improving metric increases)
- Test patience mechanism (should not reduce until patience steps)
- Test threshold mechanism (small changes should not trigger)
- Test cooldown mechanism
- Test minLearningRate floor
- Test state serialization including all tracking variables

## Mathematical Verification

Verify against PyTorch implementations:

```csharp
// OneCycleScheduler: maxLR=0.1, totalSteps=10, pctStart=0.3
// step=0:   initialLR = 0.1/25 = 0.004
// step=2:   (2/3) of increasing phase -> LR ≈ 0.068
// step=3:   start of decreasing phase -> LR = maxLR = 0.1
// step=9:   near end of cycle -> LR ≈ finalLR = 0.1/1e4

// CyclicLRScheduler: baseLR=0.001, maxLR=0.1, stepSizeUp=5
// step=0:   x=1, LR = baseLR = 0.001
// step=2:   x=0.6, LR = baseLR + (maxLR - baseLR) * 0.4
// step=5:   x=1, LR = baseLR = 0.001 (cycle complete)
```

## Estimated Implementation Time
60-75 minutes
