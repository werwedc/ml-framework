# Spec: Callback Integration for Learning Rate Schedulers

## Overview
Implement callback infrastructure for learning rate schedulers. This allows schedulers to be stepped automatically during training without manually calling Step() after each batch or epoch.

## Dependencies
- `spec_scheduler_base_interface.md` (must be completed first)
- Callback system (assumes existing callback infrastructure)
- Target namespace: `MLFramework.Training`

## Files to Create
- `src/Training/LRSchedulerCallback.cs`

## Technical Specifications

### 1. LRSchedulerCallback Class

**Purpose**: Automatically steps learning rate schedulers at appropriate points in the training loop.

**Assumptions**:
- A base `Callback` class exists with virtual methods like:
  - `OnBatchEnd(int batch, Dictionary<string, float> metrics)`
  - `OnEpochEnd(int epoch, Dictionary<string, float> metrics)`
  - `OnValidationEnd(Dictionary<string, float> metrics)`

**Implementation**:

```csharp
namespace MLFramework.Training;

/// <summary>
/// Callback that automatically steps learning rate schedulers during training.
/// Handles step-based, epoch-based, and metric-based schedulers.
/// </summary>
public class LRSchedulerCallback : Callback
{
    private readonly ILearningRateScheduler _scheduler;
    private readonly bool _stepOnBatch;
    private readonly bool _stepOnEpoch;
    private readonly string _metricName;

    /// <summary>
    /// Creates a callback for the given scheduler.
    /// Automatically detects scheduler type and steps appropriately.
    /// </summary>
    /// <param name="scheduler">The scheduler to step.</param>
    public LRSchedulerCallback(ILearningRateScheduler scheduler)
    {
        _scheduler = scheduler ?? throw new ArgumentNullException(nameof(scheduler));

        // Auto-detect stepping behavior
        _stepOnBatch = scheduler is IStepScheduler;
        _stepOnEpoch = scheduler is IEpochScheduler;
        _metricName = null;

        if (scheduler is IMetricBasedScheduler metricScheduler)
        {
            // For metric-based schedulers, we need the metric name
            _metricName = null; // User should set this manually or use alternative constructor
        }
    }

    /// <summary>
    /// Creates a callback with explicit stepping behavior.
    /// </summary>
    /// <param name="scheduler">The scheduler to step.</param>
    /// <param name="stepOnBatch">Whether to step on each batch.</param>
    /// <param name="stepOnEpoch">Whether to step on each epoch.</param>
    /// <param name="metricName">Metric name for metric-based schedulers.</param>
    public LRSchedulerCallback(
        ILearningRateScheduler scheduler,
        bool stepOnBatch = false,
        bool stepOnEpoch = false,
        string metricName = null)
    {
        _scheduler = scheduler ?? throw new ArgumentNullException(nameof(scheduler));
        _stepOnBatch = stepOnBatch;
        _stepOnEpoch = stepOnEpoch;
        _metricName = metricName;
    }

    public override void OnBatchEnd(int batch, Dictionary<string, float> metrics)
    {
        if (_stepOnBatch && _scheduler is IStepScheduler)
        {
            _scheduler.Step();
        }
    }

    public override void OnEpochEnd(int epoch, Dictionary<string, float> metrics)
    {
        if (_stepOnEpoch && _scheduler is IEpochScheduler epochScheduler)
        {
            epochScheduler.StepEpoch();
        }

        // Handle metric-based schedulers
        if (_scheduler is IMetricBasedScheduler metricScheduler && !string.IsNullOrEmpty(_metricName))
        {
            if (metrics.TryGetValue(_metricName, out float value))
            {
                metricScheduler.UpdateMetric(_metricName, value);
            }
        }
    }

    public override void OnValidationEnd(Dictionary<string, float> metrics)
    {
        // Handle metric-based schedulers with validation metrics
        if (_scheduler is IMetricBasedScheduler metricScheduler && !string.IsNullOrEmpty(_metricName))
        {
            if (metrics.TryGetValue(_metricName, out float value))
            {
                metricScheduler.UpdateMetric(_metricName, value);
            }
        }
    }

    /// <summary>
    /// Gets the scheduler associated with this callback.
    /// </summary>
    public ILearningRateScheduler Scheduler => _scheduler;
}
```

## Implementation Notes

### Design Decisions

1. **Auto-Detection**:
   - Default constructor automatically detects scheduler type using interface checks
   - `IStepScheduler` → steps on batch end
   - `IEpochScheduler` → steps on epoch end
   - `IMetricBasedScheduler` → requires metric name to be set separately

2. **Manual Override**:
   - Alternative constructor allows manual control of stepping behavior
   - Useful for custom training loops or non-standard behavior

3. **Metric Handling**:
   - Metric-based schedulers (like ReduceLROnPlateau) need metric values
   - Can use training metrics (OnEpochEnd) or validation metrics (OnValidationEnd)
   - Metric name must be specified for metric-based schedulers

4. **Flexible Integration**:
   - Works with any ILearningRateScheduler implementation
   - Can be added to any callback list or trainer

### Edge Cases

- **Multiple interfaces**: Scheduler can implement both IStepScheduler and IEpochScheduler (uncommon but possible)
- **Metric not found**: Silently skips updating if metricName is not in the metrics dictionary
- **Null scheduler**: Constructor throws ArgumentNullException

### Performance Considerations

- Minimal overhead: one interface check per batch/epoch
- No allocations in the hot path
- Dictionary lookup for metric-based schedulers only

## Usage Examples

### Example 1: Step-Based Scheduler (e.g., CosineAnnealing)

```csharp
var model = new MyModel();
var optimizer = new SGD(model.Parameters(), learningRate: 0.1f);
var scheduler = new CosineAnnealingScheduler(tMax: 1000f);

// Attach scheduler to optimizer
optimizer.SetScheduler(scheduler);

// Create callback to step scheduler
var lrCallback = new LRSchedulerCallback(scheduler);

// Add to trainer (assuming trainer has callback list)
trainer.AddCallback(lrCallback);

// Train (callback automatically steps scheduler)
trainer.Train(model, trainData, epochs: 10);
```

### Example 2: Epoch-Based Scheduler

```csharp
var scheduler = new MultiStepDecayScheduler(
    milestones: new[] { 30, 60, 90 },
    gamma: 0.1f
);

// Auto-detection: will step on epoch end
var lrCallback = new LRSchedulerCallback(scheduler);
trainer.AddCallback(lrCallback);
```

### Example 3: Metric-Based Scheduler (ReduceLROnPlateau)

```csharp
var scheduler = new ReduceLROnPlateauScheduler(
    mode: "min",
    factor: 0.1f,
    patience: 10
);

// Must specify metric name for metric-based schedulers
var lrCallback = new LRSchedulerCallback(
    scheduler,
    stepOnEpoch: false,  // Don't step on epoch
    metricName: "val_loss"
);

// Or use property to set metric name
trainer.AddCallback(lrCallback);
```

### Example 4: Manual Control

```csharp
var scheduler = new CyclicLRScheduler(
    baseLearningRate: 1e-4f,
    maxLearningRate: 1e-2f,
    stepSizeUp: 2000f
);

// Manual override: explicitly set stepping behavior
var lrCallback = new LRSchedulerCallback(
    scheduler,
    stepOnBatch: true,
    stepOnEpoch: false
);
```

### Example 5: Multiple Schedulers

```csharp
// Different schedulers for different optimizers
var optimizer1 = new SGD(param1, 0.1f);
var optimizer2 = new Adam(param2, 0.001f);

var scheduler1 = new CosineAnnealingScheduler(1000f);
var scheduler2 = new StepDecayScheduler(30, 0.1f);

optimizer1.SetScheduler(scheduler1);
optimizer2.SetScheduler(scheduler2);

// Create callbacks for both
var callback1 = new LRSchedulerCallback(scheduler1);
var callback2 = new LRSchedulerCallback(scheduler2);

trainer.AddCallback(callback1);
trainer.AddCallback(callback2);
```

### Example 6: Custom Training Loop

```csharp
var scheduler = new LinearWarmupScheduler(
    new CosineAnnealingScheduler(9000f),
    1000
);

var callback = new LRSchedulerCallback(scheduler, stepOnBatch: true);

// Custom training loop
for (int epoch = 0; epoch < epochs; epoch++)
{
    foreach (var batch in data)
    {
        // Forward/backward/step
        // ...

        // Manually trigger callback
        callback.OnBatchEnd(batchIndex, metrics);
    }

    callback.OnEpochEnd(epoch, epochMetrics);
}
```

## Integration with Training Loop

The callback assumes the following trainer interface pattern:

```csharp
public class Trainer
{
    private List<Callback> _callbacks = new List<Callback>();

    public void AddCallback(Callback callback)
    {
        _callbacks.Add(callback);
    }

    public void Train(Model model, DataLoader trainData, int epochs)
    {
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            int batchIndex = 0;
            foreach (var batch in trainData)
            {
                // Training step
                // ...

                // Notify callbacks
                foreach (var callback in _callbacks)
                {
                    callback.OnBatchEnd(batchIndex, batchMetrics);
                }
                batchIndex++;
            }

            // Notify callbacks after epoch
            foreach (var callback in _callbacks)
            {
                callback.OnEpochEnd(epoch, epochMetrics);
            }
        }
    }
}
```

## Testing Requirements

### Unit Tests

- Test with step-based scheduler:
  - Verify Step() is called on each batch
  - Verify learning rate changes correctly

- Test with epoch-based scheduler:
  - Verify StepEpoch() is called on each epoch
  - Verify learning rate changes correctly

- Test with metric-based scheduler:
  - Verify UpdateMetric() is called with correct metric
  - Verify metric not found handling (graceful skip)

- Test manual control:
  - Verify stepOnBatch and stepOnEpoch flags work correctly

- Test with scheduler implementing multiple interfaces

### Integration Tests

- Test in full training loop with callback system
- Test checkpointing: save/load with callback
- Test multiple callbacks (multiple schedulers)
- Test with validation metrics

## Estimated Implementation Time
35-45 minutes
