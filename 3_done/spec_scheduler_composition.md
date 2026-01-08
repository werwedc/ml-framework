# Spec: Scheduler Composition Patterns

## Overview
Implement scheduler composition utilities that allow combining or sequencing multiple schedulers. This enables complex learning rate schedules by composing simple building blocks.

## Dependencies
- `spec_scheduler_base_interface.md` (must be completed first)
- Target namespace: `MLFramework.Schedulers`

## Files to Create
- `src/Schedulers/ChainedScheduler.cs`
- `src/Schedulers/SequentialScheduler.cs`
- `src/Schedulers/ConstantLR.cs` (utility scheduler)

## Technical Specifications

### 1. Constant LR Scheduler

**Purpose**: A simple utility scheduler that returns a constant learning rate. Useful for building composite schedules and as a placeholder.

**Implementation**:

```csharp
namespace MLFramework.Schedulers;

/// <summary>
/// Simple scheduler that always returns a constant learning rate.
/// </summary>
public sealed class ConstantLR : BaseScheduler, IStepScheduler
{
    private readonly float _learningRate;

    public ConstantLR(float learningRate)
    {
        _learningRate = learningRate;
    }

    public override float GetLearningRate(int step, float baseLearningRate)
    {
        return _learningRate;
    }

    public override StateDict GetState()
    {
        var state = new StateDict();
        state.Set("lr", _learningRate);
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
- `learningRate` (float): Constant learning rate to return

**Example Usage**:
```csharp
// Constant LR of 1e-3
var scheduler = new ConstantLR(1e-3f);
```

### 2. Chained Scheduler

**Purpose**: Combines multiple schedulers by multiplying their outputs. This allows composing different scheduling strategies (e.g., warmup * cosine annealing).

**Formula**:
```
LR = baseLR * product(scheduler_i.GetLearningRate(step, 1.0))
```

**Implementation**:

```csharp
namespace MLFramework.Schedulers;

/// <summary>
/// Chains multiple schedulers together by multiplying their outputs.
/// Useful for composing warmup with decay schedules.
/// </summary>
public sealed class ChainedScheduler : BaseScheduler, IStepScheduler
{
    private readonly ILearningRateScheduler[] _schedulers;

    public ChainedScheduler(params ILearningRateScheduler[] schedulers)
    {
        _schedulers = schedulers ?? throw new ArgumentNullException(nameof(schedulers));
    }

    public override float GetLearningRate(int step, float baseLearningRate)
    {
        float combinedLR = baseLearningRate;

        foreach (var scheduler in _schedulers)
        {
            // Each scheduler operates on the combined LR with normalized input
            combinedLR = scheduler.GetLearningRate(step, combinedLR);
        }

        return combinedLR;
    }

    public override void Step()
    {
        base.Step();
        foreach (var scheduler in _schedulers)
        {
            scheduler.Step();
        }
    }

    public override void Reset()
    {
        base.Reset();
        foreach (var scheduler in _schedulers)
        {
            scheduler.Reset();
        }
    }

    public override StateDict GetState()
    {
        var state = new StateDict();
        state.Set("scheduler_count", _schedulers.Length);
        state.Set("step_count", _stepCount);
        state.Set("epoch_count", _epochCount);

        // Store states for all chained schedulers
        var schedulerStates = new List<StateDict>();
        foreach (var scheduler in _schedulers)
        {
            schedulerStates.Add(scheduler.GetState());
        }
        state.Set("scheduler_states", schedulerStates);

        return state;
    }

    public override void LoadState(StateDict state)
    {
        _stepCount = state.Get<int>("step_count", 0);
        _epochCount = state.Get<int>("epoch_count", 0);

        var schedulerStates = state.Get<List<StateDict>>("scheduler_states");
        if (schedulerStates != null && schedulerStates.Count == _schedulers.Length)
        {
            for (int i = 0; i < _schedulers.Length; i++)
            {
                _schedulers[i].LoadState(schedulerStates[i]);
            }
        }
    }
}
```

**Constructor Parameters**:
- `schedulers` (params ILearningRateScheduler[]): Array of schedulers to chain

**Example Usage**:
```csharp
// Combine warmup and cosine annealing
var warmup = new ConstantWarmupScheduler(
    new CosineAnnealingScheduler(tMax: 9000f),
    warmupSteps: 1000,
    warmupLearningRate: 1e-6f
);

// Alternative using ChainedScheduler
var baseScheduler = new CosineAnnealingScheduler(tMax: 10000f);
var warmupScheduler = new LinearWarmupScheduler(
    new ConstantLR(1.0f),
    warmupSteps: 1000
);
var scheduler = new ChainedScheduler(baseScheduler, warmupScheduler);
```

### 3. Sequential Scheduler

**Purpose**: Runs schedulers sequentially. Each scheduler is used for a specified number of steps, then the next scheduler takes over.

**Implementation**:

```csharp
namespace MLFramework.Schedulers;

/// <summary>
/// Runs schedulers sequentially. Each scheduler runs for a specified duration,
/// then the next scheduler in the sequence takes over.
/// </summary>
public sealed class SequentialScheduler : BaseScheduler, IStepScheduler
{
    private readonly List<(ILearningRateScheduler scheduler, int duration)> _schedulerSequence;
    private int _currentSchedulerIndex;

    public SequentialScheduler(params (ILearningRateScheduler scheduler, int duration)[] sequence)
    {
        if (sequence == null || sequence.Length == 0)
        {
            throw new ArgumentException("Sequence must contain at least one scheduler", nameof(sequence));
        }

        _schedulerSequence = new List<(ILearningRateScheduler, int)>();
        int cumulativeSteps = 0;

        foreach (var (scheduler, duration) in sequence)
        {
            if (scheduler == null)
            {
                throw new ArgumentException("Scheduler cannot be null", nameof(sequence));
            }

            if (duration <= 0)
            {
                throw new ArgumentException($"Duration must be positive for scheduler at index {_schedulerSequence.Count}", nameof(sequence));
            }

            _schedulerSequence.Add((scheduler, duration));
            cumulativeSteps += duration;
        }
    }

    public override float GetLearningRate(int step, float baseLearningRate)
    {
        // Find which scheduler is active at the current step
        int cumulativeSteps = 0;
        int schedulerIndex = -1;
        int schedulerStartStep = 0;

        for (int i = 0; i < _schedulerSequence.Count; i++)
        {
            var (scheduler, duration) = _schedulerSequence[i];

            if (step < cumulativeSteps + duration)
            {
                schedulerIndex = i;
                schedulerStartStep = cumulativeSteps;
                break;
            }

            cumulativeSteps += duration;
        }

        // If step is beyond all schedulers, use the last one
        if (schedulerIndex == -1)
        {
            schedulerIndex = _schedulerSequence.Count - 1;
            schedulerStartStep = cumulativeSteps - _schedulerSequence[schedulerIndex].duration;
        }

        var (activeScheduler, _) = _schedulerSequence[schedulerIndex];

        // Call the active scheduler with relative step count
        int relativeStep = step - schedulerStartStep;
        return activeScheduler.GetLearningRate(relativeStep, baseLearningRate);
    }

    public override void Step()
    {
        base.Step();
        _currentSchedulerIndex = DetermineActiveScheduler(_stepCount);
        _schedulerSequence[_currentSchedulerIndex].scheduler.Step();
    }

    private int DetermineActiveScheduler(int step)
    {
        int cumulativeSteps = 0;
        for (int i = 0; i < _schedulerSequence.Count; i++)
        {
            var (_, duration) = _schedulerSequence[i];
            if (step < cumulativeSteps + duration)
            {
                return i;
            }
            cumulativeSteps += duration;
        }
        return _schedulerSequence.Count - 1; // Return last scheduler
    }

    public override void Reset()
    {
        base.Reset();
        _currentSchedulerIndex = 0;
        foreach (var (scheduler, _) in _schedulerSequence)
        {
            scheduler.Reset();
        }
    }

    public override StateDict GetState()
    {
        var state = new StateDict();
        state.Set("scheduler_count", _schedulerSequence.Count);
        state.Set("current_scheduler_index", _currentSchedulerIndex);
        state.Set("step_count", _stepCount);
        state.Set("epoch_count", _epochCount);

        // Store durations and states
        var durations = _schedulerSequence.Select(s => s.duration).ToList();
        state.Set("durations", durations);

        var schedulerStates = _schedulerSequence.Select(s => s.scheduler.GetState()).ToList();
        state.Set("scheduler_states", schedulerStates);

        return state;
    }

    public override void LoadState(StateDict state)
    {
        _stepCount = state.Get<int>("step_count", 0);
        _epochCount = state.Get<int>("epoch_count", 0);
        _currentSchedulerIndex = state.Get<int>("current_scheduler_index", 0);

        var schedulerStates = state.Get<List<StateDict>>("scheduler_states");
        if (schedulerStates != null && schedulerStates.Count == _schedulerSequence.Count)
        {
            for (int i = 0; i < _schedulerSequence.Count; i++)
            {
                _schedulerSequence[i].scheduler.LoadState(schedulerStates[i]);
            }
        }
    }
}
```

**Constructor Parameters**:
- `sequence` (params (ILearningRateScheduler scheduler, int duration)[]): Array of tuples (scheduler, duration)

**Example Usage**:
```csharp
// Constant LR for 1000 steps, then cosine annealing for 9000 steps
var scheduler = new SequentialScheduler(
    (new ConstantLR(1e-3f), 1000),
    (new CosineAnnealingScheduler(tMax: 9000f), 9000)
);

// Three-phase training
var scheduler = new SequentialScheduler(
    (new ConstantLR(1e-4f), 500),
    (new StepDecayScheduler(30, 0.1f), 2000),
    (new CosineAnnealingScheduler(5000f), 5000)
);
```

## Implementation Notes

### Design Decisions

1. **ChainedScheduler**:
   - Applies schedulers sequentially by passing the output of one to the input of the next
   - Each scheduler operates on the modified learning rate
   - Useful for combining independent scheduling strategies

2. **SequentialScheduler**:
   - Runs each scheduler for a fixed duration, then switches to the next
   - Each scheduler's step count is relative to its start in the sequence
   - Last scheduler is used indefinitely if step exceeds total duration

3. **ConstantLR**:
   - Simple utility scheduler
   - Implements the interface but ignores step and baseLR parameters

### Edge Cases

- **ChainedScheduler with no schedulers**: Constructor throws ArgumentException
- **SequentialScheduler with invalid durations**: Constructor throws ArgumentException for non-positive durations
- **SequentialScheduler beyond total duration**: Uses the last scheduler with extended step count
- **ChainedScheduler step counting**: All schedulers step together (each sees the same step count)

### Performance Considerations

- **ChainedScheduler**: O(n) per GetLearningRate call, where n = number of schedulers
- **SequentialScheduler**: O(n) per GetLearningRate call for finding active scheduler
- Both add minimal overhead for typical use cases (2-3 schedulers)

## Usage Patterns

### Pattern 1: Warmup + Decay (Chained)
```csharp
// Equivalent to LinearWarmupScheduler
var decay = new CosineAnnealingScheduler(tMax: 10000f);
var warmup = new LinearWarmupScheduler(new ConstantLR(1.0f), 1000);
var scheduler = new ChainedScheduler(decay, warmup);
```

### Pattern 2: Phased Training (Sequential)
```csharp
// Phase 1: Freeze encoder, train decoder
// Phase 2: Unfreeze encoder, train entire model
var scheduler = new SequentialScheduler(
    (new ConstantLR(1e-3f), 1000),
    (new MultiStepDecayScheduler(new[] { 30, 60 }, 0.1f), 5000)
);
```

### Pattern 3: Multi-Stage Composition
```csharp
// Complex schedule: warmup -> constant -> decay -> cosine
var scheduler = new SequentialScheduler(
    (new ConstantLR(1e-6f), 500),          // Warmup
    (new ConstantLR(1e-3f), 500),          // Constant
    (new ChainedScheduler(                 // Phase 3: warmup + step decay
        new StepDecayScheduler(30, 0.1f),
        new LinearWarmupScheduler(new ConstantLR(1.0f), 1000)
    ), 3000),
    (new CosineAnnealingScheduler(5000f), 5000)  // Phase 4: cosine
);
```

## Testing Requirements

### Unit Tests for ConstantLR

- Test that it always returns the specified learning rate
- Test with different learning rates
- Test state serialization and deserialization

### Unit Tests for ChainedScheduler

- Test with two schedulers:
  - Verify both schedulers' GetLearningRate is called
  - Verify final LR is the product of all scheduler outputs

- Test with multiple schedulers
- Test Step() and Reset() delegation to all schedulers
- Test state serialization with multiple scheduler states

- Test edge case: single scheduler (should behave identically to the scheduler)

### Unit Tests for SequentialScheduler

- Test scheduler switching at duration boundaries
- Test with two schedulers:
  - During first scheduler's duration
  - At exact boundary
  - During second scheduler's duration

- Test with more than two schedulers
- Test beyond total duration (should use last scheduler)
- Test Step() only steps the active scheduler
- Test state serialization with scheduler sequences and durations

## Estimated Implementation Time
50-60 minutes
