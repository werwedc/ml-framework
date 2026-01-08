# Spec: Base Scheduler Interface and Infrastructure

## Overview
Create the foundational interfaces and base classes for the learning rate scheduling system. This establishes the contract that all schedulers must implement and provides common functionality.

## Dependencies
- None (this is the foundation)
- Target namespace: `MLFramework.Schedulers`

## Files to Create
- `src/Schedulers/ILearningRateScheduler.cs` - Core interface
- `src/Schedulers/BaseScheduler.cs` - Abstract base class
- `src/Schedulers/ISchedulerState.cs` - State management interface
- `src/Schedulers/SchedulerState.cs` - State implementation
- `src/Schedulers/ISchedulerStep.cs` - Step interface marker
- `src/Schedulers/IEpochScheduler.cs` - Epoch-based scheduler marker

## Technical Specifications

### 1. Core Interface: `ILearningRateScheduler`

```csharp
namespace MLFramework.Schedulers;

/// <summary>
/// Defines the contract for learning rate schedulers.
/// </summary>
public interface ILearningRateScheduler
{
    /// <summary>
    /// Gets the learning rate for the current step.
    /// </summary>
    /// <param name="step">Current training step/iteration.</param>
    /// <param name="baseLearningRate">Base learning rate provided by the optimizer.</param>
    /// <returns>Learning rate to use for this step.</returns>
    float GetLearningRate(int step, float baseLearningRate);

    /// <summary>
    /// Advances the scheduler state by one step.
    /// Called after each optimizer step.
    /// </summary>
    void Step();

    /// <summary>
    /// Resets the scheduler to its initial state.
    /// </summary>
    void Reset();

    /// <summary>
    /// Gets the current state of the scheduler for checkpointing.
    /// </summary>
    StateDict GetState();

    /// <summary>
    /// Loads the scheduler state from a checkpoint.
    /// </summary>
    /// <param name="state">State dictionary to load from.</param>
    void LoadState(StateDict state);
}
```

### 2. Step-Based Interface Marker

```csharp
namespace MLFramework.Schedulers;

/// <summary>
/// Marker interface for schedulers that step on each batch/iteration.
/// </summary>
public interface IStepScheduler : ILearningRateScheduler
{
    // Marker interface - no additional methods needed
}
```

### 3. Epoch-Based Interface Marker

```csharp
namespace MLFramework.Schedulers;

/// <summary>
/// Marker interface for schedulers that step on each epoch.
/// </summary>
public interface IEpochScheduler : ILearningRateScheduler
{
    /// <summary>
    /// Advances the scheduler by one epoch.
    /// Called at the end of each epoch.
    /// </summary>
    void StepEpoch();
}
```

### 4. Metric-Based Interface Marker

```csharp
namespace MLFramework.Schedulers;

/// <summary>
/// Marker interface for schedulers that react to training metrics.
/// </summary>
public interface IMetricBasedScheduler : ILearningRateScheduler
{
    /// <summary>
    /// Updates the scheduler with a new metric value.
    /// </summary>
    /// <param name="metricName">Name of the metric (e.g., "val_loss").</param>
    /// <param name="value">Current metric value.</param>
    void UpdateMetric(string metricName, float value);
}
```

### 5. Abstract Base Class: `BaseScheduler`

```csharp
namespace MLFramework.Schedulers;

/// <summary>
/// Abstract base class providing common functionality for all schedulers.
/// </summary>
public abstract class BaseScheduler : ILearningRateScheduler
{
    protected int _stepCount;
    protected int _epochCount;

    /// <summary>
    /// Gets the current step count.
    /// </summary>
    public int StepCount => _stepCount;

    /// <summary>
    /// Gets the current epoch count.
    /// </summary>
    public int EpochCount => _epochCount;

    public abstract float GetLearningRate(int step, float baseLearningRate);

    public virtual void Step()
    {
        _stepCount++;
    }

    public virtual void StepEpoch()
    {
        _epochCount++;
    }

    public virtual void Reset()
    {
        _stepCount = 0;
        _epochCount = 0;
    }

    public abstract StateDict GetState();

    public abstract void LoadState(StateDict state);
}
```

### 6. State Management Interfaces

```csharp
namespace MLFramework.Schedulers;

/// <summary>
/// Represents a dictionary of scheduler state for checkpointing.
/// Uses key-value pairs for serializable state.
/// </summary>
public class StateDict
{
    private Dictionary<string, object> _state;

    public StateDict()
    {
        _state = new Dictionary<string, object>();
    }

    public StateDict(Dictionary<string, object> state)
    {
        _state = state;
    }

    public T Get<T>(string key, T defaultValue = default)
    {
        if (_state.TryGetValue(key, out var value))
        {
            return (T)value;
        }
        return defaultValue;
    }

    public void Set<T>(string key, T value)
    {
        _state[key] = value;
    }

    public bool ContainsKey(string key)
    {
        return _state.ContainsKey(key);
    }

    public Dictionary<string, object> ToDictionary()
    {
        return new Dictionary<string, object>(_state);
    }
}
```

## Implementation Notes

### Design Decisions
1. **Marker Interfaces**: Use interfaces like `IStepScheduler` and `IEpochScheduler` as markers to enable type checking without requiring additional method implementations.

2. **Base Class Pattern**: The abstract `BaseScheduler` class provides common tracking for step/epoch counts, reducing boilerplate in concrete implementations.

3. **State Dictionary**: The `StateDict` class provides a simple, flexible way to serialize scheduler state without requiring complex serialization frameworks.

4. **Flexibility**: The interface allows schedulers to ignore the base learning rate parameter if they have their own logic (e.g., `ConstantLR`).

### Thread Safety
- All schedulers are designed for single-threaded training loop usage.
- State management should be made thread-safe in a future update if distributed training requires it.

### Performance Considerations
- `GetLearningRate` should have minimal overhead (avoid allocations)
- `Step()` should be a simple counter increment
- State serialization should only be called during checkpointing

## Testing Requirements
- Test that `BaseScheduler` correctly tracks step and epoch counts
- Test that `Reset()` clears all counters
- Test `StateDict` get/set operations with various types
- Test marker interface type checking

## Estimated Implementation Time
45-60 minutes
