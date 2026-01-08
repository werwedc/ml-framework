# Spec: Optimizer Integration for Learning Rate Schedulers

## Overview
Integrate learning rate schedulers into the Optimizer class. This spec defines how schedulers interact with optimizers during training, including automatic learning rate updates and state management.

## Dependencies
- `spec_scheduler_base_interface.md` (must be completed first)
- Existing Optimizer class (will be modified)
- Target namespace: `MLFramework.Optimizers`

## Files to Modify
- `src/Optimizers/Optimizer.cs` (add scheduler support)
- May need to modify other optimizer classes if they inherit from Optimizer

## Technical Specifications

### 1. Optimizer Class Modifications

**Purpose**: Add scheduler support to the base Optimizer class without breaking existing functionality.

**Additions to Optimizer.cs**:

```csharp
namespace MLFramework.Optimizers;

public abstract class Optimizer
{
    // Existing properties and methods...

    // New properties
    private ILearningRateScheduler _scheduler;
    private float _baseLearningRate;

    // Existing base learning rate property (may already exist)
    public abstract float BaseLearningRate { get; }

    // New property for scheduler access
    public ILearningRateScheduler Scheduler
    {
        get => _scheduler;
    }

    /// <summary>
    /// Sets a learning rate scheduler for this optimizer.
    /// </summary>
    /// <param name="scheduler">The scheduler to use, or null to disable scheduling.</param>
    public void SetScheduler(ILearningRateScheduler scheduler)
    {
        _scheduler = scheduler;
        if (_scheduler != null)
        {
            // Store the current learning rate as the base for the scheduler
            _baseLearningRate = BaseLearningRate;
        }
    }

    /// <summary>
    /// Gets the current learning rate from the scheduler or base learning rate.
    /// </summary>
    /// <returns>Current learning rate.</returns>
    protected float GetCurrentLearningRate()
    {
        if (_scheduler != null)
        {
            return _scheduler.GetLearningRate(StepCount, _baseLearningRate);
        }
        return BaseLearningRate;
    }

    /// <summary>
    /// Updates the learning rate based on the scheduler (if any) before applying gradients.
    /// This should be called at the start of Step().
    /// </summary>
    protected virtual void UpdateLearningRate()
    {
        if (_scheduler != null)
        {
            float newLR = _scheduler.GetLearningRate(StepCount, _baseLearningRate);
            SetLearningRate(newLR);
        }
    }

    /// <summary>
    /// Steps the scheduler forward (if any) after applying gradients.
    /// This should be called at the end of Step().
    /// </summary>
    protected virtual void StepScheduler()
    {
        _scheduler?.Step();
    }

    // Modify the existing Step() method or ensure these are called in concrete implementations
    public abstract void Step();
}
```

### 2. Concrete Optimizer Example (SGD)

**Purpose**: Show how concrete optimizer implementations should integrate scheduler support.

**Modifications to SGD.cs**:

```csharp
namespace MLFramework.Optimizers;

public class SGD : Optimizer
{
    // Existing code...

    public override void Step()
    {
        // Update learning rate from scheduler
        UpdateLearningRate();

        // Apply gradients (existing logic)
        // ...

        // Step the scheduler forward
        StepScheduler();

        // Increment step count (existing)
        _stepCount++;
    }
}
```

### 3. Example Optimizer Without Scheduler Support

If an optimizer currently has a Step() method like:

```csharp
// Before
public override void Step()
{
    // Gradient computation
    // Parameter update
    _stepCount++;
}
```

It should be modified to:

```csharp
// After
public override void Step()
{
    UpdateLearningRate();  // New: update LR from scheduler

    // Gradient computation (existing)
    // ...

    // Parameter update (existing)
    // ...

    StepScheduler();        // New: step scheduler

    _stepCount++;           // Existing
}
```

### 4. Optimizer State Management

**Purpose**: Ensure optimizer state includes scheduler state for checkpointing.

```csharp
// Add to Optimizer.cs
public override StateDict GetState()
{
    var state = base.GetState();  // Existing optimizer state

    if (_scheduler != null)
    {
        state.Set("scheduler_state", _scheduler.GetState());
        state.Set("base_lr", _baseLearningRate);
    }

    return state;
}

public override void LoadState(StateDict state)
{
    base.LoadState(state);  // Load existing optimizer state

    var schedulerState = state.Get<StateDict>("scheduler_state");
    if (schedulerState != null && _scheduler != null)
    {
        _scheduler.LoadState(schedulerState);
        _baseLearningRate = state.Get<float>("base_lr", _baseLearningRate);
    }
}
```

## Implementation Notes

### Design Decisions

1. **Non-Breaking Changes**:
   - All scheduler integration is additive (no breaking changes)
   - Optimizers work the same way when no scheduler is set
   - `_scheduler` is nullable, defaulting to null

2. **Base Learning Rate Storage**:
   - When a scheduler is set, store the current learning rate as `_baseLearningRate`
   - Scheduler uses this as reference for all calculations
   - Prevents confusion when optimizer's learning rate changes during training

3. **Step Integration Points**:
   - `UpdateLearningRate()`: Called at start of Step() to apply scheduled LR
   - `StepScheduler()`: Called at end of Step() to advance scheduler state
   - Both are `protected virtual` to allow customization

4. **Scheduler Interface Compatibility**:
   - All schedulers implement `ILearningRateScheduler`
   - No special handling needed for different scheduler types

### Edge Cases

- **Scheduler is null**: Optimizer behaves normally without scheduling
- **Scheduler is set mid-training**: Scheduler uses current step count from optimizer
- **Scheduler is replaced**: New scheduler starts from current step count
- **Step counting**: Scheduler and optimizer step counts may differ slightly; use optimizer's StepCount for scheduler queries

### Performance Considerations

- Minimal overhead: two method calls per optimizer step
- `UpdateLearningRate`: One scheduler.GetLearningRate() call
- `StepScheduler`: One scheduler.Step() call
- No additional allocations in the hot path

## Usage Examples

### Example 1: Basic Scheduler Usage

```csharp
// Create optimizer and scheduler
var optimizer = new SGD(parameters, learningRate: 0.1f);
var scheduler = new CosineAnnealingScheduler(tMax: 1000f);

// Attach scheduler to optimizer
optimizer.SetScheduler(scheduler);

// Training loop
for (int epoch = 0; epoch < epochs; epoch++)
{
    foreach (var batch in trainData)
    {
        // Forward and backward pass
        var loss = model.Forward(batch.Inputs);
        loss.Backward();

        // Optimizer step (automatically applies scheduler LR)
        optimizer.Step();
        optimizer.ZeroGrad();
    }
}
```

### Example 2: Warmup + Decay

```csharp
var optimizer = new Adam(parameters, learningRate: 1e-3f);

// Create warmup + decay schedule
var baseScheduler = new CosineAnnealingScheduler(tMax: 9000f);
var scheduler = new LinearWarmupScheduler(baseScheduler, warmupSteps: 1000);

optimizer.SetScheduler(scheduler);

// Train...
```

### Example 3: Changing Scheduler Mid-Training

```csharp
var optimizer = new SGD(parameters, learningRate: 0.1f);

// Phase 1: Constant learning rate
var scheduler1 = new ConstantLR(0.1f);
optimizer.SetScheduler(scheduler1);

TrainFor(optimizer, model, data, 10);

// Phase 2: Switch to cosine decay
var scheduler2 = new CosineAnnealingScheduler(tMax: 1000f);
optimizer.SetScheduler(scheduler2);

TrainFor(optimizer, model, data, 10);
```

### Example 4: Checkpointing and Resuming

```csharp
// Save checkpoint
var optimizerState = optimizer.GetState();
SaveCheckpoint("checkpoint.pt", model.GetState(), optimizerState);

// Load checkpoint
var optimizerState = LoadCheckpoint("checkpoint.pt");
optimizer.LoadState(optimizerState);

// Training continues with correct scheduler state
```

## Testing Requirements

### Unit Tests for Optimizer Integration

- Test optimizer without scheduler (should use base LR)
- Test optimizer with scheduler:
  - Verify learning rate changes according to scheduler
  - Verify Step() calls both UpdateLearningRate and StepScheduler

- Test scheduler replacement:
  - Set scheduler, train for some steps, replace scheduler
  - Verify new scheduler is used

- Test state management:
  - Save optimizer state with scheduler
  - Load state and verify scheduler state is restored
  - Verify learning rate continues correctly after load

### Integration Tests

- Full training loop with scheduler:
  - Train model for multiple epochs
  - Verify learning rate changes correctly
  - Verify model converges appropriately

- Multiple optimizers with schedulers:
  - Test with SGD, Adam, etc.
  - Verify scheduler works consistently

## Migration Guide

### For Existing Optimizer Implementations

If you have custom optimizers, update them to support schedulers:

1. **Inherit from Optimizer**: Ensure your optimizer inherits from the base `Optimizer` class

2. **Modify Step() method**:
   ```csharp
   public override void Step()
   {
       UpdateLearningRate();  // Add this at the start

       // Existing gradient computation and update logic
       // ...

       StepScheduler();       // Add this at the end

       // Existing step count increment
       _stepCount++;
   }
   ```

3. **No other changes needed**: The base class handles everything else

### For Existing Training Loops

No changes needed to training loops. Just set the scheduler on the optimizer:

```csharp
// Before: manual LR updates
optimizer.LearningRate = ComputeLR(step);

// After: use scheduler
optimizer.SetScheduler(scheduler);
// No changes to training loop needed
```

## Estimated Implementation Time
40-50 minutes
