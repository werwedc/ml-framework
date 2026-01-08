# Spec: Training Loop Hooks

## Overview
Implement hooks that integrate visualization and profiling into training loops, enabling automatic logging without manual intervention.

## Objectives
- Provide automatic metric logging during training
- Enable profiling of training phases (forward, backward, optimizer)
- Support customizable hooks for different training scenarios
- Minimize code changes required for visualization

## API Design

```csharp
// Training phase
public enum TrainingPhase
{
    EpochStart,
    EpochEnd,
    BatchStart,
    BatchEnd,
    ForwardPassStart,
    ForwardPassEnd,
    BackwardPassStart,
    BackwardPassEnd,
    OptimizerStep,
    ValidationStart,
    ValidationEnd,
    CheckpointSave
}

// Training context
public class TrainingContext
{
    public long CurrentStep { get; set; }
    public int CurrentEpoch { get; set; }
    public int CurrentBatch { get; set; }
    public int TotalBatches { get; set; }
    public int TotalEpochs { get; set; }

    public float Loss { get; set; }
    public Dictionary<string, float> Metrics { get; set; }
    public float LearningRate { get; set; }

    public string Phase { get; set; } // "train", "validation", "test"
}

// Training hook interface
public interface ITrainingHook
{
    void OnPhaseStart(TrainingPhase phase, TrainingContext context);
    void OnPhaseEnd(TrainingPhase phase, TrainingContext context);
    void OnMetricUpdate(string metricName, float value, TrainingContext context);
    void OnException(Exception exception, TrainingContext context);
}

// Visualization training hook
public class VisualizationTrainingHook : ITrainingHook
{
    public VisualizationTrainingHook(TensorBoardVisualizer visualizer);

    // Configuration
    public bool LogLoss { get; set; } = true;
    public bool LogMetrics { get; set; } = true;
    public bool LogLearningRate { get; set; } = true;
    public bool ProfileForwardPass { get; set; } = true;
    public bool ProfileBackwardPass { get; set; } = true;
    public bool ProfileOptimizerStep { get; set; } = true;

    public string LogPrefix { get; set; } = "train/";
    public int LogFrequencyBatches { get; set; } = 1;
}

// Training loop with hooks
public interface ITrainingLoop
{
    void AddHook(ITrainingHook hook);
    void RemoveHook(ITrainingHook hook);
    void Train(int epochs, IModel model, IDataLoader dataLoader);
    void Validate(IModel model, IDataLoader dataLoader);
}

public class TrainingLoop : ITrainingLoop
{
    public TrainingLoop(IOptimizer optimizer, ILossFunction lossFunction);

    // Configuration
    public IDevice Device { get; set; }
    public int LogFrequency { get; set; } = 10;
    public bool EnableProfiling { get; set; } = true;
}

// Usage example
var visualizer = new TensorBoardVisualizer("./logs");
var hook = new VisualizationTrainingHook(visualizer)
{
    LogLoss = true,
    LogMetrics = true,
    ProfileForwardPass = true
};

var trainingLoop = new TrainingLoop(optimizer, lossFunction);
trainingLoop.AddHook(hook);
trainingLoop.Train(epochs: 10, model, trainLoader);
```

## Implementation Requirements

### 1. TrainingContext and TrainingPhase (20-30 min)
- Implement `TrainingContext` with training state:
  - Current step, epoch, batch
  - Loss and metrics
  - Learning rate
  - Phase (train/validation/test)
- Implement `TrainingPhase` enum with all training phases
- Add validation for context values (e.g., non-negative steps)

### 2. ITrainingHook Interface (20-30 min)
- Define `ITrainingHook` interface with callback methods:
  - `OnPhaseStart` - Called when a phase starts
  - `OnPhaseEnd` - Called when a phase ends
  - `OnMetricUpdate` - Called when a metric is updated
  - `OnException` - Called when an exception occurs
- Ensure hooks can modify context if needed
- Support multiple hooks per training loop

### 3. VisualizationTrainingHook (45-60 min)
- Implement `VisualizationTrainingHook` class:
  - Accept `TensorBoardVisualizer` in constructor
  - Log metrics based on configuration:
    - Loss: `LogScalar("train/loss", loss, step)`
    - Metrics: `LogScalar("train/accuracy", accuracy, step)`
    - Learning rate: `LogScalar("train/lr", lr, step)`
  - Profile training phases:
    - Use `StartProfile` to profile forward/backward passes
    - Profile optimizer step
  - Support logging frequency:
    - Only log every N batches
    - Always log epoch boundaries
- Implement context tracking:
  - Update step number automatically
  - Track phase changes
- Handle exceptions gracefully
- Support custom log prefixes for different scenarios

### 4. TrainingLoop Integration (45-60 min)
- Implement `ITrainingLoop` interface (if not already exists):
  - Add hooks management (`AddHook`, `RemoveHook`)
  - Store list of active hooks
- Integrate hooks into training loop:
  - Call `OnPhaseStart` at start of each phase
  - Call `OnPhaseEnd` at end of each phase
  - Call `OnMetricUpdate` when metrics are computed
  - Call `OnException` when exceptions occur
- Ensure hooks are called in order they were added
- Support removing hooks during training
- Provide default training loop implementation with hooks

### 5. Built-in Hooks (30-45 min)
- Implement additional useful hooks:
  - `CheckpointHook` - Save model checkpoints periodically
  - `EarlyStoppingHook` - Stop training when metrics plateau
  - `LearningRateSchedulerHook` - Adjust learning rate based on metrics
  - `GradientClippingHook` - Clip gradients to prevent explosion
- Make each hook configurable
- Support combining multiple hooks

## File Structure
```
src/
  MLFramework.Visualization/
    Hooks/
      TrainingPhase.cs
      TrainingContext.cs
      ITrainingHook.cs
      VisualizationTrainingHook.cs
      BuiltIn/
        CheckpointHook.cs
        EarlyStoppingHook.cs
        LearningRateSchedulerHook.cs
        GradientClippingHook.cs

tests/
  MLFramework.Visualization.Tests/
    Hooks/
      VisualizationTrainingHookTests.cs
      TrainingLoopIntegrationTests.cs
```

## Dependencies
- `MLFramework.Visualization` (TensorBoardVisualizer)
- `MLFramework.Training` (Training loop components)

## Integration Points
- Integrated with training loops
- Automatically logs metrics without manual intervention
- Enables profiling of entire training workflow

## Success Criteria
- Adding visualization to training loop requires <5 lines of code
- Hooks don't significantly slow down training (<1% overhead)
- All training phases are correctly identified and profiled
- Metrics are logged at correct frequency
- Multiple hooks can be combined without conflicts
- Unit tests verify hook execution order and timing
- Integration tests verify end-to-end training with visualization
