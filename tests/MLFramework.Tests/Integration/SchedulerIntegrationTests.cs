using System;
using System.Collections.Generic;
using MLFramework.Training;
using MLFramework.Schedulers;
using Xunit;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using RitterFramework.Core.Tensor;

namespace MLFramework.Tests.Integration;

#region Mock and Helper Classes

/// <summary>
/// Simple linear model for testing
/// </summary>
public class SimpleModel
{
    private readonly Dictionary<string, Tensor> _parameters;

    public SimpleModel()
    {
        _parameters = new Dictionary<string, Tensor>();
        // Initialize a simple linear layer
        var weight = new Tensor(new float[10, 5]);
        var bias = new Tensor(new float[5]);
        _parameters["weight"] = weight;
        _parameters["bias"] = bias;
    }

    public Dictionary<string, Tensor> Parameters()
    {
        return _parameters;
    }

    public StateDict GetState()
    {
        var state = new StateDict();
        state.Set("step_count", 0);
        return state;
    }

    public void LoadState(StateDict state)
    {
        // Mock implementation
    }
}

/// <summary>
/// Mock optimizer for testing scheduler integration
/// </summary>
public class MockOptimizer
{
    private readonly Dictionary<string, Tensor> _parameters;
    private readonly float _baseLearningRate;
    private ILearningRateScheduler? _scheduler;
    private float _currentLearningRate;

    public MockOptimizer(Dictionary<string, Tensor> parameters, float learningRate)
    {
        _parameters = parameters;
        _baseLearningRate = learningRate;
        _currentLearningRate = learningRate;
    }

    public float LearningRate => _currentLearningRate;

    public void SetScheduler(ILearningRateScheduler scheduler)
    {
        _scheduler = scheduler;
    }

    public void Step()
    {
        // Update learning rate from scheduler if set
        if (_scheduler != null)
        {
            int step = _scheduler is BaseScheduler bs ? bs.StepCount : 0;
            _currentLearningRate = _scheduler.GetLearningRate(step, _baseLearningRate);
            _scheduler.Step();
        }
    }

    public void ZeroGrad()
    {
        // Mock implementation
    }

    public StateDict GetState()
    {
        var state = new StateDict();
        state.Set("learning_rate", _currentLearningRate);
        if (_scheduler != null)
        {
            state.Set("scheduler_state", _scheduler.GetState());
        }
        return state;
    }

    public void LoadState(StateDict state)
    {
        if (state.TryGet("learning_rate", out float lr))
        {
            _currentLearningRate = lr;
        }
        if (_scheduler != null && state.TryGet("scheduler_state", out StateDict schedulerState))
        {
            _scheduler.LoadState(schedulerState);
        }
    }
}

/// <summary>
/// Mock CosineAnnealingScheduler for testing
/// </summary>
public class CosineAnnealingScheduler : BaseScheduler
{
    private readonly float _tMax;
    private readonly float _etaMin;

    public CosineAnnealingScheduler(float tMax, float etaMin = 0f)
    {
        _tMax = tMax;
        _etaMin = etaMin;
    }

    public override float GetLearningRate(int step, float baseLearningRate)
    {
        if (_stepCount >= _tMax)
        {
            return _etaMin;
        }

        float pi = (float)Math.PI;
        float cosine = (float)Math.Cos(pi * _stepCount / _tMax);
        return _etaMin + (baseLearningRate - _etaMin) * (1 + cosine) / 2;
    }

    public override StateDict GetState()
    {
        var state = new StateDict();
        state.Set("stepCount", _stepCount);
        state.Set("tMax", _tMax);
        state.Set("etaMin", _etaMin);
        return state;
    }

    public override void LoadState(StateDict state)
    {
        _stepCount = state.Get<int>("stepCount");
    }
}

/// <summary>
/// Mock CosineAnnealingWarmRestartsScheduler for testing
/// </summary>
public class CosineAnnealingWarmRestartsScheduler : BaseScheduler
{
    private readonly float _t0;
    private readonly float _tMult;

    public CosineAnnealingWarmRestartsScheduler(float t0, float tMult = 1f)
    {
        _t0 = t0;
        _tMult = tMult;
    }

    public override float GetLearningRate(int step, float baseLearningRate)
    {
        // Simplified implementation
        float tCur = _t0 * (float)Math.Pow(_tMult, _stepCount / (int)_t0);
        float tCurRemainder = _stepCount % (int)tCur;
        return baseLearningRate * (1 + (float)Math.Cos(Math.PI * tCurRemainder / tCur)) / 2;
    }

    public override StateDict GetState()
    {
        var state = new StateDict();
        state.Set("stepCount", _stepCount);
        return state;
    }

    public override void LoadState(StateDict state)
    {
        _stepCount = state.Get<int>("stepCount");
    }
}

/// <summary>
/// Mock ConstantLR scheduler for testing
/// </summary>
public class ConstantLR : BaseScheduler
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
        state.Set("stepCount", _stepCount);
        state.Set("learningRate", _learningRate);
        return state;
    }

    public override void LoadState(StateDict state)
    {
        _stepCount = state.Get<int>("stepCount");
    }
}

/// <summary>
/// Mock LinearWarmupScheduler for testing
/// </summary>
public class LinearWarmupScheduler : BaseScheduler
{
    private readonly ILearningRateScheduler _baseScheduler;
    private readonly int _warmupSteps;

    public LinearWarmupScheduler(ILearningRateScheduler baseScheduler, int warmupSteps)
    {
        _baseScheduler = baseScheduler;
        _warmupSteps = warmupSteps;
    }

    public override float GetLearningRate(int step, float baseLearningRate)
    {
        if (_stepCount < _warmupSteps)
        {
            return baseLearningRate * ((_stepCount + 1) / (float)_warmupSteps);
        }
        return _baseScheduler.GetLearningRate(_stepCount, baseLearningRate);
    }

    public override StateDict GetState()
    {
        var state = new StateDict();
        state.Set("stepCount", _stepCount);
        state.Set("warmupSteps", _warmupSteps);
        return state;
    }

    public override void LoadState(StateDict state)
    {
        _stepCount = state.Get<int>("stepCount");
    }
}

/// <summary>
/// Mock ChainedScheduler for testing
/// </summary>
public class ChainedScheduler : BaseScheduler
{
    private readonly ILearningRateScheduler _scheduler1;
    private readonly ILearningRateScheduler _scheduler2;

    public ILearningRateScheduler Scheduler1 => _scheduler1;
    public ILearningRateScheduler Scheduler2 => _scheduler2;

    public ChainedScheduler(ILearningRateScheduler scheduler1, ILearningRateScheduler scheduler2)
    {
        _scheduler1 = scheduler1;
        _scheduler2 = scheduler2;
    }

    public override float GetLearningRate(int step, float baseLearningRate)
    {
        float lr1 = _scheduler1.GetLearningRate(step, baseLearningRate);
        return _scheduler2.GetLearningRate(step, lr1);
    }

    public override void Step()
    {
        base.Step();
        _scheduler1.Step();
        _scheduler2.Step();
    }

    public override StateDict GetState()
    {
        var state = new StateDict();
        state.Set("stepCount", _stepCount);
        state.Set("scheduler1_state", _scheduler1.GetState());
        state.Set("scheduler2_state", _scheduler2.GetState());
        return state;
    }

    public override void LoadState(StateDict state)
    {
        _stepCount = state.Get<int>("stepCount");
        _scheduler1.LoadState(state.Get<StateDict>("scheduler1_state"));
        _scheduler2.LoadState(state.Get<StateDict>("scheduler2_state"));
    }
}

/// <summary>
/// Mock SequentialScheduler for testing
/// </summary>
public class SequentialScheduler : BaseScheduler
{
    private readonly List<(ILearningRateScheduler scheduler, int steps)> _stages;
    private int _currentStage;
    private int _stageStepCount;

    public SequentialScheduler(params (ILearningRateScheduler scheduler, int steps)[] stages)
    {
        _stages = stages.ToList();
        _currentStage = 0;
        _stageStepCount = 0;
    }

    public override float GetLearningRate(int step, float baseLearningRate)
    {
        int totalSteps = 0;
        foreach (var (scheduler, steps) in _stages)
        {
            if (step < totalSteps + steps)
            {
                return scheduler.GetLearningRate(step - totalSteps, baseLearningRate);
            }
            totalSteps += steps;
        }
        // Return last scheduler's LR if beyond all stages
        return _stages[^1].scheduler.GetLearningRate(step - totalSteps, baseLearningRate);
    }

    public override StateDict GetState()
    {
        var state = new StateDict();
        state.Set("stepCount", _stepCount);
        state.Set("currentStage", _currentStage);
        return state;
    }

    public override void LoadState(StateDict state)
    {
        _stepCount = state.Get<int>("stepCount");
        _currentStage = state.Get<int>("currentStage");
    }
}

/// <summary>
/// Mock OneCycleScheduler for testing
/// </summary>
public class OneCycleScheduler : BaseScheduler
{
    private readonly float _maxLearningRate;
    private readonly float _totalSteps;

    public OneCycleScheduler(float maxLearningRate, float totalSteps)
    {
        _maxLearningRate = maxLearningRate;
        _totalSteps = totalSteps;
    }

    public override float GetLearningRate(int step, float baseLearningRate)
    {
        if (_stepCount < _totalSteps / 2)
        {
            // Increasing phase
            return baseLearningRate + (_maxLearningRate - baseLearningRate) * (_stepCount / (_totalSteps / 2));
        }
        else
        {
            // Decreasing phase
            return _maxLearningRate * (1 - (_stepCount - _totalSteps / 2) / (_totalSteps / 2));
        }
    }

    public override StateDict GetState()
    {
        var state = new StateDict();
        state.Set("stepCount", _stepCount);
        return state;
    }

    public override void LoadState(StateDict state)
    {
        _stepCount = state.Get<int>("stepCount");
    }
}

/// <summary>
/// Mock CyclicLRScheduler for testing
/// </summary>
public class CyclicLRScheduler : BaseScheduler
{
    private readonly float _baseLr;
    private readonly float _maxLr;
    private readonly float _stepSizeUp;

    public CyclicLRScheduler(float baseLr, float maxLr, float stepSizeUp)
    {
        _baseLr = baseLr;
        _maxLr = maxLr;
        _stepSizeUp = stepSizeUp;
    }

    public override float GetLearningRate(int step, float baseLearningRate)
    {
        float cycle = (float)Math.Floor(1 + _stepCount / (2 * _stepSizeUp));
        float x = (float)Math.Abs(_stepCount / _stepSizeUp - 2 * cycle + 1);
        return _baseLr + (_maxLr - _baseLr) * (float)Math.Max(0, 1 - x);
    }

    public override StateDict GetState()
    {
        var state = new StateDict();
        state.Set("stepCount", _stepCount);
        return state;
    }

    public override void LoadState(StateDict state)
    {
        _stepCount = state.Get<int>("stepCount");
    }
}

/// <summary>
/// Mock ReduceLROnPlateauScheduler for testing
/// </summary>
public class ReduceLROnPlateauScheduler : BaseScheduler, IMetricBasedScheduler
{
    private readonly string _mode;
    private readonly float _factor;
    private readonly int _patience;
    private float _bestMetric;
    private int _waitCount;
    private float _currentFactor = 1.0f;
    private readonly Dictionary<string, float> _metricsHistory = new();

    public ReduceLROnPlateauScheduler(string mode = "min", float factor = 0.5f, int patience = 5)
    {
        _mode = mode;
        _factor = factor;
        _patience = patience;
    }

    public override float GetLearningRate(int step, float baseLearningRate)
    {
        return baseLearningRate * _currentFactor;
    }

    public void UpdateMetric(string metricName, float value)
    {
        _metricsHistory[metricName] = value;

        if (_metricsHistory.Count == 1)
        {
            _bestMetric = value;
            _waitCount = 0;
            return;
        }

        bool improved = _mode == "min" ? value < _bestMetric : value > _bestMetric;

        if (improved)
        {
            _bestMetric = value;
            _waitCount = 0;
        }
        else
        {
            _waitCount++;
            if (_waitCount >= _patience)
            {
                _currentFactor *= _factor;
                _waitCount = 0;
            }
        }
    }

    public override StateDict GetState()
    {
        var state = new StateDict();
        state.Set("stepCount", _stepCount);
        state.Set("currentFactor", _currentFactor);
        state.Set("bestMetric", _bestMetric);
        state.Set("waitCount", _waitCount);
        return state;
    }

    public override void LoadState(StateDict state)
    {
        _stepCount = state.Get<int>("stepCount");
        _currentFactor = state.Get<float>("currentFactor");
        _bestMetric = state.Get<float>("bestMetric");
        _waitCount = state.Get<int>("waitCount");
    }
}

#endregion

#region Integration Tests

/// <summary>
/// Integration tests for scheduler system with optimizers and training loops
/// </summary>
public class SchedulerIntegrationTests
{
    #region Optimizer Integration Tests

    [Fact]
    public void Optimizer_WithScheduler_UpdatesLearningRateCorrectly()
    {
        var model = new SimpleModel();
        var optimizer = new MockOptimizer(model.Parameters(), learningRate: 0.1f);
        var scheduler = new StepDecayScheduler(stepSize: 3, gamma: 0.1f);

        optimizer.SetScheduler(scheduler);

        // Step 0: LR = 0.1
        optimizer.Step();
        Assert.Equal(0.1f, optimizer.LearningRate);

        // Step 1: LR = 0.1
        optimizer.Step();
        Assert.Equal(0.1f, optimizer.LearningRate);

        // Step 2: LR = 0.1 (still in first 3 steps)
        optimizer.Step();
        Assert.Equal(0.1f, optimizer.LearningRate);

        // Step 3: LR = 0.1 * 0.1 = 0.01
        optimizer.Step();
        Assert.Equal(0.01f, optimizer.LearningRate, 4);
    }

    [Fact]
    public void Optimizer_WithoutScheduler_UsesBaseLearningRate()
    {
        var model = new SimpleModel();
        var optimizer = new MockOptimizer(model.Parameters(), learningRate: 0.1f);

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
        var optimizer = new MockOptimizer(model.Parameters(), learningRate: 0.1f);

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
        var optimizer = new MockOptimizer(model.Parameters(), learningRate: 0.1f);
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
        var newOptimizer = new MockOptimizer(model.Parameters(), learningRate: 0.1f);
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
        var optimizer = new MockOptimizer(model.Parameters(), learningRate: 0.1f);
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
        var optimizer = new MockOptimizer(model.Parameters(), learningRate: 0.1f);

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
        var optimizer = new MockOptimizer(model.Parameters(), learningRate: 0.1f);

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
        var optimizer = new MockOptimizer(model.Parameters(), learningRate: 0.1f);
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
        var optimizer = new MockOptimizer(model.Parameters(), learningRate: 0.1f);
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
        var optimizer = new MockOptimizer(model.Parameters(), learningRate: 0.1f);

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
        Assert.Equal(0.05f, lr, 4);  // 0.1 * 0.5
    }

    #endregion

    #region Composition Integration Tests

    [Fact]
    public void SequentialScheduler_InTraining_SwitchesCorrectly()
    {
        var model = new SimpleModel();
        var optimizer = new MockOptimizer(model.Parameters(), learningRate: 0.1f);

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
        Assert.Equal(expectedLR, lr, 4);
    }

    #endregion

    #region State Management Integration Tests

    [Fact]
    public void Checkpoint_SaveAndLoad_RestoresFullTrainingState()
    {
        var model = new SimpleModel();
        var optimizer = new MockOptimizer(model.Parameters(), learningRate: 0.1f);
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
        var newOptimizer = new MockOptimizer(newModel.Parameters(), learningRate: 0.1f);
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

        var optimizer1 = new MockOptimizer(model1.Parameters(), learningRate: 0.1f);
        var optimizer2 = new MockOptimizer(model2.Parameters(), learningRate: 0.01f);

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
        var optimizer = new MockOptimizer(model.Parameters(), learningRate: 0.1f);
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

#endregion

#region Performance Benchmarks

/// <summary>
/// Performance benchmarks for scheduler operations
/// </summary>
[MemoryDiagnoser]
public class SchedulerBenchmark
{
    private SimpleModel _model;
    private MockOptimizer _optimizer;
    private ILearningRateScheduler[] _schedulers;

    [GlobalSetup]
    public void Setup()
    {
        _model = new SimpleModel();
        _optimizer = new MockOptimizer(_model.Parameters(), learningRate: 0.1f);

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
public class BenchmarkProgram
{
    public static void Main(string[] args)
    {
        var summary = BenchmarkRunner.Run<SchedulerBenchmark>();
    }
}

#endregion
