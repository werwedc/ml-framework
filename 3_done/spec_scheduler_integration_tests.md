# Spec: Scheduler Integration Tests and Benchmarking

## Overview
Create integration tests and benchmarks for the learning rate scheduler system. Integration tests verify schedulers work correctly in real training scenarios, while benchmarks measure performance overhead.

## Dependencies
- All scheduler specs and implementations (must be completed first)
- Basic model and optimizer implementations
- Target namespace: `MLFramework.Tests.Integration`

## Files to Create
- `tests/Integration/SchedulerIntegrationTests.cs`
- `tests/Integration/SchedulerBenchmark.cs`

## Technical Specifications

### 1. Integration Tests

**Purpose**: Verify schedulers work correctly in full training scenarios with optimizers, models, and data.

**Tests**:

```csharp
using Xunit;
using MLFramework.Schedulers;
using MLFramework.Optimizers;
using MLFramework.Models;
using MLFramework.Data;

namespace MLFramework.Tests.Integration;

public class SchedulerIntegrationTests
{
    // Helper: Simple linear model for testing
    private class SimpleModel : Model
    {
        public SimpleModel()
        {
            // Initialize a simple linear layer
            // Assume Parameter class exists
            var weight = new Parameter(new float[10, 5]);
            var bias = new Parameter(new float[5]);
            RegisterParameters(weight, bias);
        }
    }

    #region Optimizer Integration Tests

    [Fact]
    public void Optimizer_WithScheduler_UpdatesLearningRateCorrectly()
    {
        var model = new SimpleModel();
        var optimizer = new SGD(model.Parameters(), learningRate: 0.1f);
        var scheduler = new StepDecayScheduler(stepSize: 3, gamma: 0.1f);

        optimizer.SetScheduler(scheduler);

        // Step 0: LR = 0.1
        optimizer.Step();
        Assert.Equal(0.1f, optimizer.LearningRate);

        // Step 1: LR = 0.1
        optimizer.Step();
        Assert.Equal(0.1f, optimizer.LearningRate);

        // Step 3: LR = 0.1 * 0.1 = 0.01
        optimizer.Step();
        Assert.Equal(0.01f, optimizer.LearningRate);
    }

    [Fact]
    public void Optimizer_WithoutScheduler_UsesBaseLearningRate()
    {
        var model = new SimpleModel();
        var optimizer = new SGD(model.Parameters(), learningRate: 0.1f);

        // No scheduler set
        optimizer.Step();
        optimizer.Step();
        optimizer.Step();

        Assert.Equal(0.1f, optimizer.LearningRate);
    }

    [Fact]
    public void Optimizer_SchedulerReplacement_UpdatesCorrectly()
    {
        var model = new SimpleModel();
        var optimizer = new SGD(model.Parameters(), learningRate: 0.1f);

        var scheduler1 = new ConstantLR(0.05f);
        optimizer.SetScheduler(scheduler1);

        optimizer.Step();
        Assert.Equal(0.05f, optimizer.LearningRate);

        // Replace scheduler
        var scheduler2 = new ConstantLR(0.001f);
        optimizer.SetScheduler(scheduler2);

        optimizer.Step();
        Assert.Equal(0.001f, optimizer.LearningRate);
    }

    [Fact]
    public void Optimizer_SchedulerState_SaveAndLoad_RestoresCorrectly()
    {
        var model = new SimpleModel();
        var optimizer = new SGD(model.Parameters(), learningRate: 0.1f);
        var scheduler = new CosineAnnealingScheduler(tMax: 100f, etaMin: 0f);

        optimizer.SetScheduler(scheduler);

        // Train for some steps
        for (int i = 0; i < 50; i++)
        {
            optimizer.Step();
        }

        // Save state
        var optimizerState = optimizer.GetState();

        // Create new optimizer with new scheduler
        var newOptimizer = new SGD(model.Parameters(), learningRate: 0.1f);
        var newScheduler = new CosineAnnealingScheduler(tMax: 100f, etaMin: 0f);
        newOptimizer.SetScheduler(newScheduler);

        // Load state
        newOptimizer.LoadState(optimizerState);

        // Verify scheduler state is restored
        Assert.Equal(50, newScheduler.StepCount);
    }

    #endregion

    #region Full Training Loop Tests

    [Fact]
    public void FullTrainingLoop_WithCosineScheduler_CompletesSuccessfully()
    {
        var model = new SimpleModel();
        var optimizer = new SGD(model.Parameters(), learningRate: 0.1f);
        var scheduler = new CosineAnnealingScheduler(tMax: 100f, etaMin: 0f);

        optimizer.SetScheduler(scheduler);

        // Simulate training for 1 epoch with 10 batches
        for (int batch = 0; batch < 10; batch++)
        {
            // Simulate forward/backward
            // (In real test, would compute actual gradients)

            optimizer.Step();
            optimizer.ZeroGrad();
        }

        // Verify scheduler stepped 10 times
        Assert.Equal(10, scheduler.StepCount);

        // Verify LR changed according to cosine schedule
        float lr = scheduler.GetLearningRate(10, 0.1f);
        Assert.NotEqual(0.1f, lr);  // Should have decayed
    }

    [Fact]
    public void FullTrainingLoop_WithWarmupScheduler_CompletesSuccessfully()
    {
        var model = new SimpleModel();
        var optimizer = new SGD(model.Parameters(), learningRate: 0.1f);

        var baseScheduler = new CosineAnnealingScheduler(tMax: 9000f, etaMin: 0f);
        var scheduler = new LinearWarmupScheduler(baseScheduler, 1000);

        optimizer.SetScheduler(scheduler);

        // Train through warmup
        for (int step = 0; step < 1500; step++)
        {
            optimizer.Step();
            optimizer.ZeroGrad();
        }

        // Verify LR schedule is correct
        // After 1000 steps, should have started cosine decay
        float lr = scheduler.GetLearningRate(1500, 0.1f);
        Assert.True(lr > 0);  // Should still be positive
        Assert.True(lr < 0.1f);  // Should have changed from initial
    }

    [Fact]
    public void FullTrainingLoop_WithChainedScheduler_CompletesSuccessfully()
    {
        var model = new SimpleModel();
        var optimizer = new SGD(model.Parameters(), learningRate: 0.1f);

        var s1 = new CosineAnnealingScheduler(tMax: 100f);
        var s2 = new LinearWarmupScheduler(new ConstantLR(1.0f), 10);
        var scheduler = new ChainedScheduler(s1, s2);

        optimizer.SetScheduler(scheduler);

        // Train
        for (int step = 0; step < 50; step++)
        {
            optimizer.Step();
            optimizer.ZeroGrad();
        }

        // Verify both schedulers stepped
        Assert.Equal(50, s1.StepCount);
        Assert.Equal(50, s2.StepCount);
    }

    #endregion

    #region Callback Integration Tests

    [Fact]
    public void Callback_WithStepScheduler_StepsOnBatchEnd()
    {
        var model = new SimpleModel();
        var optimizer = new SGD(model.Parameters(), learningRate: 0.1f);
        var scheduler = new StepDecayScheduler(stepSize: 3, gamma: 0.1f);
        var callback = new LRSchedulerCallback(scheduler);

        // Simulate batch callbacks
        for (int batch = 0; batch < 10; batch++)
        {
            // Simulate forward/backward/step
            optimizer.Step();

            // Call callback
            var metrics = new Dictionary<string, float> { { "loss", 0.5f } };
            callback.OnBatchEnd(batch, metrics);
        }

        // Verify scheduler stepped 10 times
        Assert.Equal(10, scheduler.StepCount);
    }

    [Fact]
    public void Callback_WithEpochScheduler_StepsOnEpochEnd()
    {
        var model = new SimpleModel();
        var optimizer = new SGD(model.Parameters(), learningRate: 0.1f);
        var scheduler = new MultiStepDecayScheduler(
            milestones: new[] { 2, 4, 6 },
            gamma: 0.1f
        );
        var callback = new LRSchedulerCallback(scheduler);

        // Simulate epoch callbacks
        for (int epoch = 0; epoch < 10; epoch++)
        {
            // Simulate epoch training
            for (int batch = 0; batch < 5; batch++)
            {
                optimizer.Step();
            }

            // Call epoch callback
            var metrics = new Dictionary<string, float> { { "val_loss", 0.3f } };
            callback.OnEpochEnd(epoch, metrics);
        }

        // Verify epoch count matches
        Assert.Equal(10, scheduler.EpochCount);
    }

    [Fact]
    public void Callback_WithMetricBasedScheduler_UpdatesMetric()
    {
        var model = new SimpleModel();
        var optimizer = new SGD(model.Parameters(), learningRate: 0.1f);

        var scheduler = new ReduceLROnPlateauScheduler(
            mode: "min",
            factor: 0.5f,
            patience: 2
        );
        var callback = new LRSchedulerCallback(
            scheduler,
            stepOnEpoch: false,
            metricName: "val_loss"
        );

        // Simulate training with improving then stagnating loss
        float[] losses = { 1.0f, 0.8f, 0.6f, 0.6f, 0.6f, 0.6f };

        for (int i = 0; i < losses.Length; i++)
        {
            optimizer.Step();

            var metrics = new Dictionary<string, float> { { "val_loss", losses[i] } };
            callback.OnEpochEnd(i, metrics);
        }

        // After 3 stagnant updates, LR should be reduced
        float lr = scheduler.GetLearningRate(0, 0.1f);
        Assert.Equal(0.05f, lr);  // 0.1 * 0.5
    }

    #endregion

    #region Composition Integration Tests

    [Fact]
    public void SequentialScheduler_InTraining_SwitchesCorrectly()
    {
        var model = new SimpleModel();
        var optimizer = new SGD(model.Parameters(), learningRate: 0.1f);

        var s1 = new ConstantLR(0.001f);
        var s2 = new CosineAnnealingScheduler(tMax: 100f);
        var scheduler = new SequentialScheduler(
            (s1, 100),
            (s2, 200)
        );

        optimizer.SetScheduler(scheduler);

        // Train through first scheduler
        for (int step = 0; step < 150; step++)
        {
            optimizer.Step();
        }

        // At step 150, should be using second scheduler
        float lr = scheduler.GetLearningRate(150, 0.1f);
        float expectedLR = s2.GetLearningRate(50, 0.1f);  // 150 - 100 = 50
        Assert.Equal(expectedLR, lr);
    }

    #endregion

    #region State Management Integration Tests

    [Fact]
    public void Checkpoint_SaveAndLoad_RestoresFullTrainingState()
    {
        var model = new SimpleModel();
        var optimizer = new SGD(model.Parameters(), learningRate: 0.1f);
        var scheduler = new CosineAnnealingWarmRestartsScheduler(t0: 10f, tMult: 2f);

        optimizer.SetScheduler(scheduler);

        // Train for some steps
        for (int step = 0; step < 25; step++)
        {
            optimizer.Step();
            optimizer.ZeroGrad();
        }

        // Save checkpoint
        var modelState = model.GetState();
        var optimizerState = optimizer.GetState();

        // Create new model, optimizer, scheduler
        var newModel = new SimpleModel();
        var newOptimizer = new SGD(newModel.Parameters(), learningRate: 0.1f);
        var newScheduler = new CosineAnnealingWarmRestartsScheduler(t0: 10f, tMult: 2f);
        newOptimizer.SetScheduler(newScheduler);

        // Load checkpoint
        newModel.LoadState(modelState);
        newOptimizer.LoadState(optimizerState);

        // Continue training
        for (int step = 0; step < 10; step++)
        {
            newOptimizer.Step();
            newOptimizer.ZeroGrad();
        }

        // Verify scheduler state continued correctly
        Assert.Equal(35, newScheduler.StepCount);
    }

    #endregion

    #region Edge Case Integration Tests

    [Fact]
    public void TrainingWithMultipleOptimizers_WithSchedulers_WorksCorrectly()
    {
        var model1 = new SimpleModel();
        var model2 = new SimpleModel();

        var optimizer1 = new SGD(model1.Parameters(), learningRate: 0.1f);
        var optimizer2 = new Adam(model2.Parameters(), learningRate: 0.01f);

        var scheduler1 = new CosineAnnealingScheduler(tMax: 100f);
        var scheduler2 = new StepDecayScheduler(stepSize: 30, gamma: 0.1f);

        optimizer1.SetScheduler(scheduler1);
        optimizer2.SetScheduler(scheduler2);

        // Train with both optimizers
        for (int step = 0; step < 50; step++)
        {
            optimizer1.Step();
            optimizer2.Step();
        }

        // Verify both schedulers stepped correctly
        Assert.Equal(50, scheduler1.StepCount);
        Assert.Equal(50, scheduler2.StepCount);

        // Verify different LRs
        float lr1 = scheduler1.GetLearningRate(50, 0.1f);
        float lr2 = scheduler2.GetLearningRate(50, 0.01f);

        Assert.NotEqual(lr1, lr2);
    }

    [Fact]
    public void TrainingWithVeryLongSchedule_NoMemoryLeak()
    {
        var model = new SimpleModel();
        var optimizer = new SGD(model.Parameters(), learningRate: 0.1f);
        var scheduler = new CosineAnnealingScheduler(tMax: 1_000_000f);

        optimizer.SetScheduler(scheduler);

        // Train for many steps
        for (int step = 0; step < 10_000; step++)
        {
            optimizer.Step();
            optimizer.ZeroGrad();
        }

        // Verify scheduler still works
        float lr = scheduler.GetLearningRate(10_000, 0.1f);
        Assert.True(lr > 0 && lr <= 0.1f);
    }

    #endregion
}
```

### 2. Performance Benchmarks

**Purpose**: Measure overhead of scheduler integration and identify performance bottlenecks.

**Benchmarks**:

```csharp
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using MLFramework.Schedulers;
using MLFramework.Optimizers;
using MLFramework.Models;

namespace MLFramework.Tests.Integration;

[MemoryDiagnoser]
public class SchedulerBenchmark
{
    private SimpleModel _model;
    private SGD _optimizer;
    private ILearningRateScheduler[] _schedulers;

    [GlobalSetup]
    public void Setup()
    {
        _model = new SimpleModel();
        _optimizer = new SGD(_model.Parameters(), learningRate: 0.1f);

        _schedulers = new ILearningRateScheduler[]
        {
            new StepDecayScheduler(stepSize: 100, gamma: 0.1f),
            new CosineAnnealingScheduler(tMax: 1000f),
            new OneCycleScheduler(0.1f, 1000f),
            new CyclicLRScheduler(1e-4f, 1e-2f, 200f),
            new LinearWarmupScheduler(new CosineAnnealingScheduler(9000f), 1000),
            new ChainedScheduler(
                new CosineAnnealingScheduler(1000f),
                new LinearWarmupScheduler(new ConstantLR(1.0f), 100)
            )
        };
    }

    [Benchmark(Baseline = true)]
    public void OptimizerWithoutScheduler()
    {
        _optimizer.Step();
        _optimizer.ZeroGrad();
    }

    [Benchmark]
    public void OptimizerWithStepDecayScheduler()
    {
        _optimizer.SetScheduler(_schedulers[0]);
        _optimizer.Step();
        _optimizer.ZeroGrad();
    }

    [Benchmark]
    public void OptimizerWithCosineScheduler()
    {
        _optimizer.SetScheduler(_schedulers[1]);
        _optimizer.Step();
        _optimizer.ZeroGrad();
    }

    [Benchmark]
    public void OptimizerWithOneCycleScheduler()
    {
        _optimizer.SetScheduler(_schedulers[2]);
        _optimizer.Step();
        _optimizer.ZeroGrad();
    }

    [Benchmark]
    public void OptimizerWithCyclicLRScheduler()
    {
        _optimizer.SetScheduler(_schedulers[3]);
        _optimizer.Step();
        _optimizer.ZeroGrad();
    }

    [Benchmark]
    public void OptimizerWithWarmupScheduler()
    {
        _optimizer.SetScheduler(_schedulers[4]);
        _optimizer.Step();
        _optimizer.ZeroGrad();
    }

    [Benchmark]
    public void OptimizerWithChainedScheduler()
    {
        _optimizer.SetScheduler(_schedulers[5]);
        _optimizer.Step();
        _optimizer.ZeroGrad();
    }

    [Benchmark]
    public void GetLearningRate_StepDecay()
    {
        var scheduler = _schedulers[0];
        for (int i = 0; i < 1000; i++)
        {
            scheduler.GetLearningRate(i, 0.1f);
        }
    }

    [Benchmark]
    public void GetLearningRate_CosineAnnealing()
    {
        var scheduler = _schedulers[1];
        for (int i = 0; i < 1000; i++)
        {
            scheduler.GetLearningRate(i, 0.1f);
        }
    }

    [Benchmark]
    public void GetLearningRate_OneCycle()
    {
        var scheduler = _schedulers[2];
        for (int i = 0; i < 1000; i++)
        {
            scheduler.GetLearningRate(i, 0.1f);
        }
    }

    [Benchmark]
    public void SaveAndLoadState_CosineScheduler()
    {
        var scheduler = _schedulers[1];
        for (int i = 0; i < 100; i++)
        {
            scheduler.Step();
        }

        var state = scheduler.GetState();
        var newScheduler = new CosineAnnealingScheduler(1000f);
        newScheduler.LoadState(state);
    }

    [Benchmark]
    public void SaveAndLoadState_ComplexScheduler()
    {
        var scheduler = _schedulers[5];  // ChainedScheduler
        for (int i = 0; i < 100; i++)
        {
            scheduler.Step();
        }

        var state = scheduler.GetState();
        var innerScheduler = new CosineAnnealingScheduler(1000f);
        var warmup = new LinearWarmupScheduler(new ConstantLR(1.0f), 100);
        var newScheduler = new ChainedScheduler(innerScheduler, warmup);
        newScheduler.LoadState(state);
    }
}

// Benchmark runner (optional)
public class Program
{
    public static void Main(string[] args)
    {
        var summary = BenchmarkRunner.Run<SchedulerBenchmark>();
    }
}
```

## Testing Requirements

### Integration Test Coverage

1. **Optimizer Integration**:
   - Scheduler set/get operations
   - Learning rate updates during training
   - Scheduler replacement
   - State save/load with scheduler

2. **Full Training Loop**:
   - Train with various schedulers
   - Verify LR schedules over time
   - Test with warmup + decay combinations
   - Test with chained schedulers

3. **Callback Integration**:
   - Step-based scheduler stepping on batch end
   - Epoch-based scheduler stepping on epoch end
   - Metric-based scheduler updating metrics

4. **Composition Integration**:
   - Sequential scheduler switching
   - Chained scheduler behavior
   - Complex compositions

5. **State Management**:
   - Checkpoint save/load
   - Resume training from checkpoint
   - Verify scheduler continuity

### Benchmark Requirements

1. **Baseline**:
   - Measure optimizer step without scheduler
   - Measure GetLearningRate calls

2. **Overhead Measurement**:
   - Compare each scheduler vs baseline
   - Report time overhead in percentage
   - Measure memory allocations

3. **Performance Goals**:
   - Scheduler overhead < 5% of optimizer step time
   - No memory allocations in hot path
   - GetLearningRate call < 1 microsecond

### Test Data

- Use simple synthetic models
- Small parameter counts for fast tests
- Use random data for training simulations
- Avoid dependencies on actual datasets

## Estimated Implementation Time
75-90 minutes
