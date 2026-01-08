# Spec: Scheduler Unit Tests

## Overview
Create comprehensive unit tests for all learning rate schedulers. Tests should verify mathematical correctness, state management, edge cases, and interface compliance.

## Dependencies
- All scheduler specs (must be completed first)
- Testing framework (xUnit, NUnit, or similar)
- Target namespace: `MLFramework.Tests.Schedulers`

## Files to Create
- `tests/Schedulers/BaseSchedulerTests.cs`
- `tests/Schedulers/SimpleDecaySchedulerTests.cs`
- `tests/Schedulers/CosineSchedulerTests.cs`
- `tests/Schedulers/AdvancedSchedulerTests.cs`
- `tests/Schedulers/WarmupSchedulerTests.cs`
- `tests/Schedulers/CompositionSchedulerTests.cs`
- `tests/Schedulers/AdvancedFeatureSchedulerTests.cs`

## Technical Specifications

### 1. Base Scheduler Tests

**Purpose**: Test base class, interfaces, and StateDict functionality.

**Tests**:

```csharp
using Xunit;
using MLFramework.Schedulers;

namespace MLFramework.Tests.Schedulers;

public class BaseSchedulerTests
{
    [Fact]
    public void StateDict_SetAndGet_ReturnsCorrectValue()
    {
        var state = new StateDict();
        state.Set("key1", 42);
        state.Set("key2", 3.14f);
        state.Set("key3", "test");

        Assert.Equal(42, state.Get<int>("key1"));
        Assert.Equal(3.14f, state.Get<float>("key2"));
        Assert.Equal("test", state.Get<string>("key3"));
    }

    [Fact]
    public void StateDict_GetWithDefault_ReturnsDefaultWhenKeyNotFound()
    {
        var state = new StateDict();

        Assert.Equal(100, state.Get<int>("nonexistent", 100));
        Assert.Equal(0.5f, state.Get<float>("nonexistent", 0.5f));
    }

    [Fact]
    public void StateDict_ContainsKey_ReturnsCorrectValue()
    {
        var state = new StateDict();
        state.Set("exists", "value");

        Assert.True(state.ContainsKey("exists"));
        Assert.False(state.ContainsKey("does_not_exist"));
    }

    [Fact]
    public void StateDict_ToDictionary_ReturnsCorrectDictionary()
    {
        var state = new StateDict();
        state.Set("key1", 42);
        state.Set("key2", "value");

        var dict = state.ToDictionary();

        Assert.Equal(2, dict.Count);
        Assert.Equal(42, dict["key1"]);
        Assert.Equal("value", dict["key2"]);
    }
}
```

### 2. Simple Decay Scheduler Tests

**Purpose**: Test StepDecayScheduler, MultiStepDecayScheduler, ExponentialDecayScheduler.

**Tests**:

```csharp
using Xunit;
using MLFramework.Schedulers;

namespace MLFramework.Tests.Schedulers;

public class SimpleDecaySchedulerTests
{
    #region StepDecayScheduler Tests

    [Fact]
    public void StepDecayScheduler_BeforeFirstDecay_ReturnsBaseLR()
    {
        var scheduler = new StepDecayScheduler(stepSize: 30, gamma: 0.1f);
        float baseLR = 0.1f;

        float lr = scheduler.GetLearningRate(0, baseLR);

        Assert.Equal(0.1f, lr);
        Assert.Equal(0.1f, scheduler.GetLearningRate(29, baseLR));
    }

    [Fact]
    public void StepDecayScheduler_AfterFirstDecay_ReturnsDecayedLR()
    {
        var scheduler = new StepDecayScheduler(stepSize: 30, gamma: 0.1f);
        float baseLR = 0.1f;

        float lr = scheduler.GetLearningRate(30, baseLR);

        Assert.Equal(0.01f, lr);  // 0.1 * 0.1
    }

    [Fact]
    public void StepDecayScheduler_AfterMultipleDecays_ReturnsCorrectLR()
    {
        var scheduler = new StepDecayScheduler(stepSize: 30, gamma: 0.1f);
        float baseLR = 0.1f;

        Assert.Equal(0.01f, scheduler.GetLearningRate(30, baseLR));   // 1 decay
        Assert.Equal(0.001f, scheduler.GetLearningRate(60, baseLR));  // 2 decays
        Assert.Equal(0.0001f, scheduler.GetLearningRate(90, baseLR)); // 3 decays
    }

    [Fact]
    public void StepDecayScheduler_StateSaveLoad_RestoresCorrectState()
    {
        var scheduler = new StepDecayScheduler(stepSize: 30, gamma: 0.1f);
        scheduler.Step();
        scheduler.Step();

        var state = scheduler.GetState();
        var newScheduler = new StepDecayScheduler(stepSize: 30, gamma: 0.1f);
        newScheduler.LoadState(state);

        Assert.Equal(2, newScheduler.StepCount);
    }

    [Fact]
    public void StepDecayScheduler_Reset_ClearsStepCount()
    {
        var scheduler = new StepDecayScheduler(stepSize: 30, gamma: 0.1f);
        scheduler.Step();
        scheduler.Step();

        scheduler.Reset();

        Assert.Equal(0, scheduler.StepCount);
    }

    #endregion

    #region MultiStepDecayScheduler Tests

    [Fact]
    public void MultiStepDecayScheduler_BeforeFirstMilestone_ReturnsBaseLR()
    {
        var scheduler = new MultiStepDecayScheduler(
            milestones: new[] { 30, 60, 90 },
            gamma: 0.1f
        );
        float baseLR = 0.1f;

        Assert.Equal(0.1f, scheduler.GetLearningRate(0, baseLR));
        Assert.Equal(0.1f, scheduler.GetLearningRate(29, baseLR));
    }

    [Fact]
    public void MultiStepDecayScheduler_AfterMilestones_ReturnsCorrectLR()
    {
        var scheduler = new MultiStepDecayScheduler(
            milestones: new[] { 30, 60, 90 },
            gamma: 0.1f
        );
        float baseLR = 0.1f;

        Assert.Equal(0.01f, scheduler.GetLearningRate(30, baseLR));   // 1 milestone
        Assert.Equal(0.001f, scheduler.GetLearningRate(60, baseLR));  // 2 milestones
        Assert.Equal(0.0001f, scheduler.GetLearningRate(90, baseLR)); // 3 milestones
    }

    [Fact]
    public void MultiStepDecayScheduler_EmptyMilestones_NoDecay()
    {
        var scheduler = new MultiStepDecayScheduler(
            milestones: Array.Empty<int>(),
            gamma: 0.1f
        );
        float baseLR = 0.1f;

        Assert.Equal(0.1f, scheduler.GetLearningRate(0, baseLR));
        Assert.Equal(0.1f, scheduler.GetLearningRate(1000, baseLR));
    }

    [Fact]
    public void MultiStepDecayScheduler_UnsortedMilestones_WorksCorrectly()
    {
        var scheduler = new MultiStepDecayScheduler(
            milestones: new[] { 90, 30, 60 },  // Unsorted
            gamma: 0.1f
        );
        float baseLR = 0.1f;

        // Should still work based on count, not order
        Assert.Equal(0.001f, scheduler.GetLearningRate(100, baseLR)); // 3 milestones passed
    }

    #endregion

    #region ExponentialDecayScheduler Tests

    [Fact]
    public void ExponentialDecayScheduler_AtZero_ReturnsBaseLR()
    {
        var scheduler = new ExponentialDecayScheduler(gamma: 0.95f);
        float baseLR = 0.1f;

        Assert.Equal(0.1f, scheduler.GetLearningRate(0, baseLR));
    }

    [Fact]
    public void ExponentialDecayScheduler_AfterSteps_ReturnsDecayedLR()
    {
        var scheduler = new ExponentialDecayScheduler(gamma: 0.5f);
        float baseLR = 0.1f;

        Assert.Equal(0.05f, scheduler.GetLearningRate(1, baseLR));  // 0.1 * 0.5^1
        Assert.Equal(0.025f, scheduler.GetLearningRate(2, baseLR)); // 0.1 * 0.5^2
    }

    [Fact]
    public void ExponentialDecayScheduler_GammaEqualsOne_NoDecay()
    {
        var scheduler = new ExponentialDecayScheduler(gamma: 1.0f);
        float baseLR = 0.1f;

        Assert.Equal(0.1f, scheduler.GetLearningRate(0, baseLR));
        Assert.Equal(0.1f, scheduler.GetLearningRate(100, baseLR));
    }

    [Fact]
    public void ExponentialDecayScheduler_StateSaveLoad_RestoresCorrectState()
    {
        var scheduler = new ExponentialDecayScheduler(gamma: 0.95f);
        scheduler.Step();
        scheduler.Step();

        var state = scheduler.GetState();
        var newScheduler = new ExponentialDecayScheduler(gamma: 0.95f);
        newScheduler.LoadState(state);

        Assert.Equal(2, newScheduler.StepCount);
    }

    #endregion
}
```

### 3. Cosine Scheduler Tests

**Purpose**: Test CosineAnnealingScheduler and CosineAnnealingWarmRestartsScheduler.

**Tests**:

```csharp
using Xunit;
using MLFramework.Schedulers;

namespace MLFramework.Tests.Schedulers;

public class CosineSchedulerTests
{
    #region CosineAnnealingScheduler Tests

    [Fact]
    public void CosineAnnealingScheduler_AtZero_ReturnsBaseLR()
    {
        var scheduler = new CosineAnnealingScheduler(tMax: 100f, etaMin: 0f);
        float baseLR = 0.1f;

        Assert.Equal(0.1f, scheduler.GetLearningRate(0, baseLR));
    }

    [Fact]
    public void CosineAnnealingScheduler_AtHalfway_ReturnsHalfLR()
    {
        var scheduler = new CosineAnnealingScheduler(tMax: 100f, etaMin: 0f);
        float baseLR = 0.1f;

        float lr = scheduler.GetLearningRate(50, baseLR);

        Assert.Equal(0.05f, lr);  // cos(π/2) = 0, so LR = 0 + 0.5 * (0.1 - 0) * 1 = 0.05
    }

    [Fact]
    public void CosineAnnealingScheduler_AtTMax_ReturnsEtaMin()
    {
        var scheduler = new CosineAnnealingScheduler(tMax: 100f, etaMin: 0f);
        float baseLR = 0.1f;

        Assert.Equal(0f, scheduler.GetLearningRate(100, baseLR));
    }

    [Fact]
    public void CosineAnnealingScheduler_WithNonZeroEtaMin_ReturnsCorrectLR()
    {
        var scheduler = new CosineAnnealingScheduler(tMax: 100f, etaMin: 1e-6f);
        float baseLR = 0.1f;

        float lr = scheduler.GetLearningRate(100, baseLR);

        Assert.Equal(1e-6f, lr);
    }

    [Fact]
    public void CosineAnnealingScheduler_InvalidTMax_ThrowsException()
    {
        Assert.Throws<ArgumentException>(() =>
            new CosineAnnealingScheduler(tMax: -1f, etaMin: 0f));
    }

    #endregion

    #region CosineAnnealingWarmRestartsScheduler Tests

    [Fact]
    public void CosineAnnealingWarmRestarts_FirstCycle_WorksCorrectly()
    {
        var scheduler = new CosineAnnealingWarmRestartsScheduler(
            t0: 10f,
            tMult: 2f,
            etaMin: 1e-6f
        );
        float baseLR = 0.1f;

        Assert.Equal(0.1f, scheduler.GetLearningRate(0, baseLR));
        Assert.Equal(0.05f, scheduler.GetLearningRate(5, baseLR));
    }

    [Fact]
    public void CosineAnnealingWarmRestarts_CycleTransition_ResetsToBaseLR()
    {
        var scheduler = new CosineAnnealingWarmRestartsScheduler(
            t0: 10f,
            tMult: 2f,
            etaMin: 1e-6f
        );
        float baseLR = 0.1f;

        // End of first cycle
        float lrEndOfCycle = scheduler.GetLearningRate(10, baseLR);
        // Start of second cycle (close to baseLR)
        float lrStartOfNext = scheduler.GetLearningRate(11, baseLR);

        // Should jump back toward baseLR
        Assert.True(lrStartOfNext > lrEndOfCycle);
    }

    [Fact]
    public void CosineAnnealingWarmRestarts_Reset_RestartsCycle()
    {
        var scheduler = new CosineAnnealingWarmRestartsScheduler(
            t0: 10f,
            tMult: 2f,
            etaMin: 1e-6f
        );
        float baseLR = 0.1f;

        // Advance into second cycle
        scheduler.GetLearningRate(20, baseLR);

        scheduler.Reset();

        // Should be back at start
        Assert.Equal(0.1f, scheduler.GetLearningRate(0, baseLR));
    }

    [Fact]
    public void CosineAnnealingWarmRestarts_StateSaveLoad_RestoresCorrectState()
    {
        var scheduler = new CosineAnnealingWarmRestartsScheduler(
            t0: 10f,
            tMult: 2f,
            etaMin: 1e-6f
        );
        float baseLR = 0.1f;

        // Get some LR updates to trigger state
        scheduler.GetLearningRate(20, baseLR);

        var state = scheduler.GetState();
        var newScheduler = new CosineAnnealingWarmRestartsScheduler(
            t0: 10f,
            tMult: 2f,
            etaMin: 1e-6f
        );
        newScheduler.LoadState(state);

        Assert.Equal(0, newScheduler.StepCount);
    }

    #endregion
}
```

### 4. Advanced Scheduler Tests

**Purpose**: Test OneCycleScheduler, CyclicLRScheduler, ReduceLROnPlateauScheduler.

**Tests**:

```csharp
using Xunit;
using MLFramework.Schedulers;

namespace MLFramework.Tests.Schedulers;

public class AdvancedSchedulerTests
{
    #region OneCycleScheduler Tests

    [Fact]
    public void OneCycleScheduler_AtZero_ReturnsInitialLR()
    {
        var scheduler = new OneCycleScheduler(
            maxLearningRate: 0.1f,
            totalSteps: 100f,
            pctStart: 0.3f
        );

        float lr = scheduler.GetLearningRate(0, 0.1f);

        Assert.Equal(0.004f, lr);  // 0.1 / 25
    }

    [Fact]
    public void OneCycleScheduler_AtPeak_ReturnsMaxLR()
    {
        var scheduler = new OneCycleScheduler(
            maxLearningRate: 0.1f,
            totalSteps: 100f,
            pctStart: 0.3f
        );

        float lr = scheduler.GetLearningRate(30, 0.1f);

        Assert.Equal(0.1f, lr);  // At pctStart * totalSteps
    }

    [Fact]
    public void OneCycleScheduler_AtEnd_ReturnsFinalLR()
    {
        var scheduler = new OneCycleScheduler(
            maxLearningRate: 0.1f,
            totalSteps: 100f,
            pctStart: 0.3f
        );

        float lr = scheduler.GetLearningRate(99, 0.1f);

        Assert.Equal(1e-5f, lr);  // 0.1 / 1e4
    }

    [Fact]
    public void OneCycleScheduler_BeyondTotalSteps_CapsAtFinalLR()
    {
        var scheduler = new OneCycleScheduler(
            maxLearningRate: 0.1f,
            totalSteps: 100f,
            pctStart: 0.3f
        );

        float lr = scheduler.GetLearningRate(150, 0.1f);

        Assert.Equal(1e-5f, lr);
    }

    #endregion

    #region CyclicLRScheduler Tests

    [Fact]
    public void CyclicLRScheduler_AtZero_ReturnsBaseLR()
    {
        var scheduler = new CyclicLRScheduler(
            baseLearningRate: 0.001f,
            maxLearningRate: 0.1f,
            stepSizeUp: 100f
        );

        float lr = scheduler.GetLearningRate(0, 0.1f);

        Assert.Equal(0.001f, lr);
    }

    [Fact]
    public void CyclicLRScheduler_AtHalfCycle_ReturnsMidLR()
    {
        var scheduler = new CyclicLRScheduler(
            baseLearningRate: 0.001f,
            maxLearningRate: 0.1f,
            stepSizeUp: 100f
        );

        float lr = scheduler.GetLearningRate(50, 0.1f);

        Assert.Equal(0.0505f, lr);  // Midpoint between base and max
    }

    [Fact]
    public void CyclicLRScheduler_CompletesCycle_ReturnsBaseLR()
    {
        var scheduler = new CyclicLRScheduler(
            baseLearningRate: 0.001f,
            maxLearningRate: 0.1f,
            stepSizeUp: 100f
        );

        float lr = scheduler.GetLearningRate(200, 0.1f);

        Assert.Equal(0.001f, lr);
    }

    #endregion

    #region ReduceLROnPlateauScheduler Tests

    [Fact]
    public void ReduceLROnPlateau_ImprovingMetric_NoReduction()
    {
        var scheduler = new ReduceLROnPlateauScheduler(
            mode: "min",
            factor: 0.1f,
            patience: 3
        );

        // Metric improves each time
        scheduler.UpdateMetric("loss", 1.0f);
        scheduler.UpdateMetric("loss", 0.9f);
        scheduler.UpdateMetric("loss", 0.8f);

        float lr = scheduler.GetLearningRate(0, 0.1f);

        Assert.Equal(float.MaxValue, lr);  // Still at initial (uninitialized)
    }

    [Fact]
    public void ReduceLROnPlateau_NoImprovement_ReducesLR()
    {
        var scheduler = new ReduceLROnPlateauScheduler(
            mode: "min",
            factor: 0.1f,
            patience: 2
        );

        // Set best metric
        scheduler.UpdateMetric("loss", 1.0f);

        // No improvement
        scheduler.UpdateMetric("loss", 1.0f);
        scheduler.UpdateMetric("loss", 1.0f);
        scheduler.UpdateMetric("loss", 1.0f);

        float lr = scheduler.GetLearningRate(0, 0.1f);

        Assert.Equal(0.01f, lr);  // 0.1 * 0.1
    }

    [Fact]
    public void ReduceLROnPlateau_MaxMode_WorksCorrectly()
    {
        var scheduler = new ReduceLROnPlateauScheduler(
            mode: "max",
            factor: 0.5f,
            patience: 2
        );

        scheduler.UpdateMetric("accuracy", 0.9f);

        // Decreasing accuracy (bad for max mode)
        scheduler.UpdateMetric("accuracy", 0.9f);
        scheduler.UpdateMetric("accuracy", 0.9f);
        scheduler.UpdateMetric("accuracy", 0.9f);

        float lr = scheduler.GetLearningRate(0, 0.1f);

        Assert.Equal(0.05f, lr);  // 0.1 * 0.5
    }

    [Fact]
    public void ReduceLROnPlateau_MinLR_FloorWorks()
    {
        var scheduler = new ReduceLROnPlateauScheduler(
            mode: "min",
            factor: 0.1f,
            patience: 1,
            minLearningRate: 1e-5f
        );

        scheduler.UpdateMetric("loss", 1.0f);

        // Multiple reductions
        scheduler.UpdateMetric("loss", 1.0f);  // Reduces to 0.01
        scheduler.UpdateMetric("loss", 0.01f);
        scheduler.UpdateMetric("loss", 0.01f);  // Reduces to 0.001
        scheduler.UpdateMetric("loss", 0.001f);
        scheduler.UpdateMetric("loss", 0.001f);  // Reduces to 0.0001
        scheduler.UpdateMetric("loss", 0.0001f);
        scheduler.UpdateMetric("loss", 0.0001f);  // Would be 0.00001, but capped at 1e-5

        float lr = scheduler.GetLearningRate(0, 0.1f);

        Assert.Equal(1e-5f, lr);
    }

    #endregion
}
```

### 5. Warmup Scheduler Tests

**Purpose**: Test LinearWarmupScheduler and ConstantWarmupScheduler.

**Tests**:

```csharp
using Xunit;
using MLFramework.Schedulers;

namespace MLFramework.Tests.Schedulers;

public class WarmupSchedulerTests
{
    #region LinearWarmupScheduler Tests

    [Fact]
    public void LinearWarmupScheduler_AtZero_ReturnsStartLR()
    {
        var baseScheduler = new CosineAnnealingScheduler(tMax: 9000f);
        var scheduler = new LinearWarmupScheduler(baseScheduler, 1000);

        float lr = scheduler.GetLearningRate(0, 0.1f);

        Assert.Equal(0f, lr);
    }

    [Fact]
    public void LinearWarmupScheduler_DuringWarmup_ReturnsCorrectLR()
    {
        var baseScheduler = new CosineAnnealingScheduler(tMax: 9000f);
        var scheduler = new LinearWarmupScheduler(baseScheduler, 1000);

        float lr = scheduler.GetLearningRate(500, 0.1f);

        Assert.Equal(0.05f, lr);  // Halfway through warmup
    }

    [Fact]
    public void LinearWarmupScheduler_AfterWarmup_DelegatesToBase()
    {
        var baseScheduler = new CosineAnnealingScheduler(tMax: 100f, etaMin: 0f);
        var scheduler = new LinearWarmupScheduler(baseScheduler, 1000);

        float lr = scheduler.GetLearningRate(1500, 0.1f);

        // Should call base with step = 1500 - 1000 = 500
        // Base scheduler at step 500 with tMax=100
        float expectedLR = baseScheduler.GetLearningRate(500, 0.1f);
        Assert.Equal(expectedLR, lr);
    }

    #endregion

    #region ConstantWarmupScheduler Tests

    [Fact]
    public void ConstantWarmupScheduler_DuringWarmup_ReturnsWarmupLR()
    {
        var baseScheduler = new CosineAnnealingScheduler(tMax: 9000f);
        var scheduler = new ConstantWarmupScheduler(baseScheduler, 500, 1e-6f);

        float lr = scheduler.GetLearningRate(0, 0.1f);
        Assert.Equal(1e-6f, lr);

        lr = scheduler.GetLearningRate(499, 0.1f);
        Assert.Equal(1e-6f, lr);
    }

    [Fact]
    public void ConstantWarmupScheduler_AfterWarmup_DelegatesToBase()
    {
        var baseScheduler = new CosineAnnealingScheduler(tMax: 100f, etaMin: 0f);
        var scheduler = new ConstantWarmupScheduler(baseScheduler, 500, 1e-6f);

        float lr = scheduler.GetLearningRate(600, 0.1f);

        // Should call base with step = 600 - 500 = 100
        float expectedLR = baseScheduler.GetLearningRate(100, 0.1f);
        Assert.Equal(expectedLR, lr);
    }

    #endregion
}
```

### 6. Composition Scheduler Tests

**Purpose**: Test ChainedScheduler and SequentialScheduler.

**Tests**:

```csharp
using Xunit;
using MLFramework.Schedulers;

namespace MLFramework.Tests.Schedulers;

public class CompositionSchedulerTests
{
    #region ChainedScheduler Tests

    [Fact]
    public void ChainedScheduler_TwoSchedulers_MultipliesOutputs()
    {
        var s1 = new StepDecayScheduler(stepSize: 30, gamma: 0.1f);
        var s2 = new CosineAnnealingScheduler(tMax: 100f, etaMin: 0f);
        var scheduler = new ChainedScheduler(s1, s2);

        float lr = scheduler.GetLearningRate(35, 0.1f);

        // s1 at step 35: 0.1 * 0.1 = 0.01
        // s2 at step 35 with input 0.01: should give some value
        float s1LR = s1.GetLearningRate(35, 0.1f);
        float expectedLR = s2.GetLearningRate(35, s1LR);

        Assert.Equal(expectedLR, lr);
    }

    [Fact]
    public void ChainedScheduler_StepsAllSchedulers()
    {
        var s1 = new CosineAnnealingScheduler(tMax: 100f);
        var s2 = new StepDecayScheduler(stepSize: 30, gamma: 0.1f);
        var scheduler = new ChainedScheduler(s1, s2);

        scheduler.Step();

        Assert.Equal(1, s1.StepCount);
        Assert.Equal(1, s2.StepCount);
    }

    #endregion

    #region SequentialScheduler Tests

    [Fact]
    public void SequentialScheduler_FirstScheduler_ActiveForDuration()
    {
        var s1 = new ConstantLR(0.001f);
        var s2 = new CosineAnnealingScheduler(tMax: 100f);
        var scheduler = new SequentialScheduler(
            (s1, 1000),
            (s2, 2000)
        );

        float lr = scheduler.GetLearningRate(500, 0.1f);

        Assert.Equal(0.001f, lr);
    }

    [Fact]
    public void SequentialScheduler_SecondScheduler_ActiveAfterFirst()
    {
        var s1 = new ConstantLR(0.001f);
        var s2 = new CosineAnnealingScheduler(tMax: 100f, etaMin: 0f);
        var scheduler = new SequentialScheduler(
            (s1, 1000),
            (s2, 2000)
        );

        float lr = scheduler.GetLearningRate(1500, 0.1f);

        // s2 with step = 1500 - 1000 = 500
        float expectedLR = s2.GetLearningRate(500, 0.1f);
        Assert.Equal(expectedLR, lr);
    }

    [Fact]
    public void SequentialScheduler_BeyondTotalDuration_UsesLastScheduler()
    {
        var s1 = new ConstantLR(0.001f);
        var s2 = new ConstantLR(0.0001f);
        var scheduler = new SequentialScheduler(
            (s1, 1000),
            (s2, 2000)
        );

        float lr = scheduler.GetLearningRate(4000, 0.1f);

        Assert.Equal(0.0001f, lr);
    }

    #endregion
}
```

### 7. Advanced Feature Scheduler Tests

**Purpose**: Test PolynomialDecayScheduler, LayerWiseLRDecayScheduler, DiscriminativeLRScheduler.

**Tests**:

```csharp
using Xunit;
using MLFramework.Schedulers;

namespace MLFramework.Tests.Schedulers;

public class AdvancedFeatureSchedulerTests
{
    #region PolynomialDecayScheduler Tests

    [Fact]
    public void PolynomialDecayScheduler_AtZero_ReturnsInitialLR()
    {
        var scheduler = new PolynomialDecayScheduler(
            initialLearningRate: 0.01f,
            finalLearningRate: 0.0001f,
            totalSteps: 1000f
        );

        float lr = scheduler.GetLearningRate(0, 0.1f);

        Assert.Equal(0.01f, lr);
    }

    [Fact]
    public void PolynomialDecayScheduler_AtTotalSteps_ReturnsFinalLR()
    {
        var scheduler = new PolynomialDecayScheduler(
            initialLearningRate: 0.01f,
            finalLearningRate: 0.0001f,
            totalSteps: 1000f
        );

        float lr = scheduler.GetLearningRate(1000, 0.1f);

        Assert.Equal(0.0001f, lr);
    }

    [Fact]
    public void PolynomialDecayScheduler_DifferentPower_WorksCorrectly()
    {
        var linear = new PolynomialDecayScheduler(0.1f, 0f, 100f, 1.0f);
        var quadratic = new PolynomialDecayScheduler(0.1f, 0f, 100f, 2.0f);

        float lrLinear = linear.GetLearningRate(50, 0.1f);
        float lrQuadratic = quadratic.GetLearningRate(50, 0.1f);

        // Quadratic decays faster
        Assert.True(lrQuadratic < lrLinear);
    }

    #endregion

    #region LayerWiseLRDecayScheduler Tests

    [Fact]
    public void LayerWiseLRDecayScheduler_LastLayer_HasMultiplierOne()
    {
        var scheduler = new LayerWiseLRDecayScheduler(decayFactor: 0.8f);

        float multiplier = scheduler.GetLayerMultiplier(
            layerIndex: 3,
            totalLayers: 4
        );

        Assert.Equal(1.0f, multiplier);
    }

    [Fact]
    public void LayerWiseLRDecayScheduler_EarlierLayers_HaveLowerMultipliers()
    {
        var scheduler = new LayerWiseLRDecayScheduler(decayFactor: 0.8f);

        float m1 = scheduler.GetLayerMultiplier(0, 4);
        float m2 = scheduler.GetLayerMultiplier(1, 4);
        float m3 = scheduler.GetLayerMultiplier(2, 4);
        float m4 = scheduler.GetLayerMultiplier(3, 4);

        Assert.Equal(0.512f, m1);  // 0.8^3
        Assert.Equal(0.64f, m2);   // 0.8^2
        Assert.Equal(0.8f, m3);    // 0.8^1
        Assert.Equal(1.0f, m4);    // 0.8^0
    }

    [Fact]
    public void LayerWiseLRDecayScheduler_ExcludedLayer_HasMultiplierOne()
    {
        var scheduler = new LayerWiseLRDecayScheduler(
            decayFactor: 0.8f,
            excludedLayers: new[] { "embedding" }
        );

        float multiplier = scheduler.GetLayerMultiplier(
            layerIndex: 0,
            totalLayers: 4,
            layerName: "embedding"
        );

        Assert.Equal(1.0f, multiplier);
    }

    #endregion

    #region DiscriminativeLRScheduler Tests

    [Fact]
    public void DiscriminativeLRScheduler_ByIndex_ReturnsCorrectLR()
    {
        var scheduler = new DiscriminativeLRScheduler(
            baseLearningRate: 0.01f,
            layerMultipliers: new[] { 0.1f, 0.2f, 0.5f, 1.0f }
        );

        Assert.Equal(0.001f, scheduler.GetGroupLearningRate(0));   // 0.01 * 0.1
        Assert.Equal(0.002f, scheduler.GetGroupLearningRate(1));   // 0.01 * 0.2
        Assert.Equal(0.005f, scheduler.GetGroupLearningRate(2));   // 0.01 * 0.5
        Assert.Equal(0.01f, scheduler.GetGroupLearningRate(3));    // 0.01 * 1.0
    }

    [Fact]
    public void DiscriminativeLRScheduler_ByName_ReturnsCorrectLR()
    {
        var scheduler = new DiscriminativeLRScheduler(
            baseLearningRate: 0.01f,
            layerMultipliers: new[] { 0.1f, 1.0f },
            layerNames: new[] { "encoder", "decoder" }
        );

        Assert.Equal(0.001f, scheduler.GetGroupLearningRate(0, "encoder"));
        Assert.Equal(0.01f, scheduler.GetGroupLearningRate(1, "decoder"));
    }

    #endregion
}
```

## Testing Requirements

### Test Framework
- Use xUnit, NUnit, or similar framework
- Organize tests by scheduler type
- Use descriptive test names

### Test Coverage Goals
- Line coverage: > 90% for all schedulers
- Branch coverage: > 85%
- Edge case coverage: All documented edge cases tested

### Mathematical Accuracy
- Verify scheduler outputs match expected formulas
- Test against known PyTorch/TensorFlow behavior
- Use small epsilon for floating point comparisons

## Estimated Implementation Time
90-120 minutes
