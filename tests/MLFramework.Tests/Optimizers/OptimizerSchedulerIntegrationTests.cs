using System;
using System.Collections.Generic;
using RitterFramework.Core.Tensor;
using MLFramework.Schedulers;
using Xunit;

namespace MLFramework.Tests.Optimizers;

/// <summary>
/// Tests for the Optimizer base class scheduler integration.
/// </summary>
public class OptimizerSchedulerIntegrationTests
{
    #region Mock Optimizer

    /// <summary>
    /// Mock optimizer for testing the base class functionality.
    /// </summary>
    private class MockOptimizer : Optimizer
    {
        private float _learningRate;
        public int UpdateLearningRateCallCount { get; private set; }
        public int StepSchedulerCallCount { get; private set; }
        public List<float> LearningRatesUsed { get; } = new List<float>();

        public MockOptimizer(Dictionary<string, Tensor> parameters, float learningRate)
            : base(parameters)
        {
            _learningRate = learningRate;
            _baseLearningRate = learningRate;
        }

        public override float BaseLearningRate => _learningRate;

        public override void Step(Dictionary<string, Tensor> gradients)
        {
            UpdateLearningRateCallCount++;
            UpdateLearningRate();

            // Simulate gradient application
            LearningRatesUsed.Add(LearningRate);

            StepScheduler();
            StepSchedulerCallCount++;

            _stepCount++;
        }

        public override void StepParameter(string parameterName, Tensor gradient)
        {
            UpdateLearningRateCallCount++;
            UpdateLearningRate();
            LearningRatesUsed.Add(LearningRate);
            StepScheduler();
            StepSchedulerCallCount++;
        }

        public override void ZeroGrad()
        {
            // Mock implementation
        }

        public override void SetLearningRate(float lr)
        {
            _learningRate = lr;
        }

        // Helper methods for testing
        public void ResetCallCounts()
        {
            UpdateLearningRateCallCount = 0;
            StepSchedulerCallCount = 0;
        }
    }

    /// <summary>
    /// Mock scheduler for testing.
    /// </summary>
    private class MockScheduler : BaseScheduler, IStepScheduler
    {
        private readonly Func<int, float, float> _lrFunction;

        public MockScheduler(Func<int, float, float> lrFunction)
        {
            _lrFunction = lrFunction;
        }

        public override float GetLearningRate(int step, float baseLearningRate)
        {
            return _lrFunction(step, baseLearningRate);
        }

        public override StateDict GetState()
        {
            var state = new StateDict();
            state.Set("step_count", _stepCount);
            return state;
        }

        public override void LoadState(StateDict state)
        {
            _stepCount = state.Get<int>("step_count", 0);
        }
    }

    #endregion

    #region Constructor and Initial State Tests

    [Fact]
    public void Constructor_NullParameters_DoesNotThrow()
    {
        var optimizer = new MockOptimizer(null, 0.1f);
        Assert.NotNull(optimizer.Parameters);
        Assert.Empty(optimizer.Parameters);
    }

    [Fact]
    public void Constructor_WithParameters_InitializesCorrectly()
    {
        var parameters = new Dictionary<string, Tensor>
        {
            { "param1", new Tensor(new float[10]) },
            { "param2", new Tensor(new float[20]) }
        };

        var optimizer = new MockOptimizer(parameters, 0.1f);

        Assert.Equal(2, optimizer.Parameters.Count);
        Assert.Equal(0.1f, optimizer.BaseLearningRate);
        Assert.Equal(0, optimizer.StepCount);
    }

    [Fact]
    public void InitialState_NoScheduler_SchedulerPropertyIsNull()
    {
        var optimizer = new MockOptimizer(null, 0.1f);
        Assert.Null(optimizer.Scheduler);
    }

    [Fact]
    public void InitialState_NoScheduler_LearningRateReturnsBaseLR()
    {
        var optimizer = new MockOptimizer(null, 0.1f);
        Assert.Equal(0.1f, optimizer.LearningRate);
    }

    #endregion

    #region SetScheduler Tests

    [Fact]
    public void SetScheduler_ValidScheduler_SetsSchedulerProperty()
    {
        var optimizer = new MockOptimizer(null, 0.1f);
        var scheduler = new MockScheduler((step, baseLR) => baseLR * 0.5f);

        optimizer.SetScheduler(scheduler);

        Assert.NotNull(optimizer.Scheduler);
        Assert.Same(scheduler, optimizer.Scheduler);
    }

    [Fact]
    public void SetScheduler_SetsBaseLearningRate()
    {
        var optimizer = new MockOptimizer(null, 0.1f);
        var scheduler = new MockScheduler((step, baseLR) => baseLR * 0.5f);

        optimizer.SetScheduler(scheduler);

        // Base learning rate should be set to current LR
        Assert.Equal(0.1f, optimizer.LearningRate);
    }

    [Fact]
    public void SetScheduler_NullScheduler_DisablesScheduling()
    {
        var optimizer = new MockOptimizer(null, 0.1f);
        var scheduler = new MockScheduler((step, baseLR) => baseLR * 0.5f);

        optimizer.SetScheduler(scheduler);
        Assert.NotNull(optimizer.Scheduler);

        optimizer.SetScheduler(null);
        Assert.Null(optimizer.Scheduler);
    }

    [Fact]
    public void SetScheduler_ReplaceScheduler_ReplacesSuccessfully()
    {
        var optimizer = new MockOptimizer(null, 0.1f);
        var scheduler1 = new MockScheduler((step, baseLR) => baseLR * 0.5f);
        var scheduler2 = new MockScheduler((step, baseLR) => baseLR * 0.8f);

        optimizer.SetScheduler(scheduler1);
        Assert.Same(scheduler1, optimizer.Scheduler);

        optimizer.SetScheduler(scheduler2);
        Assert.Same(scheduler2, optimizer.Scheduler);
    }

    #endregion

    #region GetCurrentLearningRate Tests

    [Fact]
    public void GetCurrentLearningRate_NoScheduler_ReturnsBaseLR()
    {
        var optimizer = new MockOptimizer(null, 0.1f);

        Assert.Equal(0.1f, optimizer.LearningRate);
    }

    [Fact]
    public void GetCurrentLearningRate_WithScheduler_ReturnsSchedulerValue()
    {
        var optimizer = new MockOptimizer(null, 0.1f);
        var scheduler = new MockScheduler((step, baseLR) => baseLR * 0.5f);

        optimizer.SetScheduler(scheduler);

        Assert.Equal(0.05f, optimizer.LearningRate);
    }

    [Fact]
    public void GetCurrentLearningRate_StepDependentScheduler_UpdatesWithStep()
    {
        var optimizer = new MockOptimizer(null, 1.0f);
        var scheduler = new MockScheduler((step, baseLR) => baseLR / (1 + step));

        optimizer.SetScheduler(scheduler);

        Assert.Equal(1.0f, optimizer.LearningRate);

        // Simulate a step
        optimizer.Step(new Dictionary<string, Tensor>());
        Assert.Equal(0.5f, optimizer.LearningRate);

        optimizer.Step(new Dictionary<string, Tensor>());
        Assert.Equal(0.333f, optimizer.LearningRate, precision: 3);
    }

    #endregion

    #region Step Integration Tests

    [Fact]
    public void Step_NoScheduler_CallsUpdateLearningRateAndStepScheduler()
    {
        var optimizer = new MockOptimizer(null, 0.1f);
        var gradients = new Dictionary<string, Tensor>();

        optimizer.Step(gradients);

        Assert.Equal(1, optimizer.UpdateLearningRateCallCount);
        Assert.Equal(1, optimizer.StepSchedulerCallCount);
    }

    [Fact]
    public void Step_WithScheduler_CallsUpdateLearningRateAndStepScheduler()
    {
        var optimizer = new MockOptimizer(null, 0.1f);
        var scheduler = new MockScheduler((step, baseLR) => baseLR * 0.5f);
        optimizer.SetScheduler(scheduler);

        var gradients = new Dictionary<string, Tensor>();
        optimizer.Step(gradients);

        Assert.Equal(1, optimizer.UpdateLearningRateCallCount);
        Assert.Equal(1, optimizer.StepSchedulerCallCount);
    }

    [Fact]
    public void Step_MultipleSteps_IncrementsStepCount()
    {
        var optimizer = new MockOptimizer(null, 0.1f);
        var gradients = new Dictionary<string, Tensor>();

        optimizer.Step(gradients);
        Assert.Equal(1, optimizer.StepCount);

        optimizer.Step(gradients);
        Assert.Equal(2, optimizer.StepCount);

        optimizer.Step(gradients);
        Assert.Equal(3, optimizer.StepCount);
    }

    [Fact]
    public void Step_WithScheduler_UsesScheduledLR()
    {
        var optimizer = new MockOptimizer(null, 1.0f);
        var scheduler = new MockScheduler((step, baseLR) => baseLR * 0.9f);
        optimizer.SetScheduler(scheduler);

        var gradients = new Dictionary<string, Tensor>();

        optimizer.Step(gradients);
        Assert.Equal(0.9f, optimizer.LearningRatesUsed[0]);

        optimizer.Step(gradients);
        Assert.Equal(0.81f, optimizer.LearningRatesUsed[1]);
    }

    #endregion

    #region Scheduler Replacement Mid-Training Tests

    [Fact]
    public void SchedulerReplacement_TrainThenReplace_NewSchedulerUsed()
    {
        var optimizer = new MockOptimizer(null, 1.0f);
        var scheduler1 = new MockScheduler((step, baseLR) => baseLR * 0.5f);
        var scheduler2 = new MockScheduler((step, baseLR) => baseLR * 0.8f);

        optimizer.SetScheduler(scheduler1);

        var gradients = new Dictionary<string, Tensor>();
        optimizer.Step(gradients);
        Assert.Equal(0.5f, optimizer.LearningRatesUsed[0]);

        optimizer.SetScheduler(scheduler2);
        optimizer.Step(gradients);

        // New scheduler should be used, continuing from current step
        Assert.Equal(0.8f, optimizer.LearningRatesUsed[1]);
    }

    [Fact]
    public void SchedulerReplacement_NoSchedulerToScheduler_TransitionsCorrectly()
    {
        var optimizer = new MockOptimizer(null, 1.0f);
        var gradients = new Dictionary<string, Tensor>();

        // Train without scheduler
        optimizer.Step(gradients);
        Assert.Equal(1.0f, optimizer.LearningRatesUsed[0]);

        // Add scheduler
        var scheduler = new MockScheduler((step, baseLR) => baseLR * 0.5f);
        optimizer.SetScheduler(scheduler);

        optimizer.Step(gradients);
        Assert.Equal(0.5f, optimizer.LearningRatesUsed[1]);
    }

    #endregion

    #region State Management Tests

    [Fact]
    public void GetState_NoScheduler_IncludesBasicState()
    {
        var optimizer = new MockOptimizer(null, 0.1f);

        var state = optimizer.GetState();

        Assert.Equal(0, state.Get<int>("step_count", -1));
        Assert.Equal(0.1f, state.Get<float>("base_lr", 0f));
        Assert.False(state.ContainsKey("scheduler_state"));
    }

    [Fact]
    public void GetState_WithScheduler_IncludesSchedulerState()
    {
        var optimizer = new MockOptimizer(null, 0.1f);
        var scheduler = new MockScheduler((step, baseLR) => baseLR * 0.5f);
        optimizer.SetScheduler(scheduler);

        // Step a few times
        for (int i = 0; i < 5; i++)
        {
            optimizer.Step(new Dictionary<string, Tensor>());
        }

        var state = optimizer.GetState();

        Assert.Equal(5, state.Get<int>("step_count", -1));
        Assert.Equal(0.1f, state.Get<float>("base_lr", 0f));
        Assert.True(state.ContainsKey("scheduler_state"));

        var schedulerState = state.Get<StateDict>("scheduler_state");
        Assert.NotNull(schedulerState);
        Assert.Equal(5, schedulerState.Get<int>("step_count", -1));
    }

    [Fact]
    public void LoadState_NoScheduler_RestoresBasicState()
    {
        var optimizer = new MockOptimizer(null, 0.1f);

        // Simulate some steps
        for (int i = 0; i < 10; i++)
        {
            optimizer.Step(new Dictionary<string, Tensor>());
        }

        Assert.Equal(10, optimizer.StepCount);

        // Create new optimizer and load state
        var newOptimizer = new MockOptimizer(null, 0.1f);
        newOptimizer.LoadState(optimizer.GetState());

        Assert.Equal(10, newOptimizer.StepCount);
    }

    [Fact]
    public void LoadState_WithScheduler_RestoresSchedulerState()
    {
        var optimizer = new MockOptimizer(null, 0.1f);
        var scheduler = new MockScheduler((step, baseLR) => baseLR * 0.5f);
        optimizer.SetScheduler(scheduler);

        // Step a few times
        for (int i = 0; i < 5; i++)
        {
            optimizer.Step(new Dictionary<string, Tensor>());
        }

        Assert.Equal(5, optimizer.StepCount);

        // Create new optimizer with same scheduler and load state
        var newScheduler = new MockScheduler((step, baseLR) => baseLR * 0.5f);
        var newOptimizer = new MockOptimizer(null, 0.1f);
        newOptimizer.SetScheduler(newScheduler);
        newOptimizer.LoadState(optimizer.GetState());

        Assert.Equal(5, newOptimizer.StepCount);
    }

    [Fact]
    public void LoadState_TrainingContinuesCorrectly()
    {
        var optimizer = new MockOptimizer(null, 1.0f);
        var scheduler = new MockScheduler((step, baseLR) => baseLR * 0.9f);
        optimizer.SetScheduler(scheduler);

        // Step 3 times
        var gradients = new Dictionary<string, Tensor>();
        optimizer.Step(gradients);
        optimizer.Step(gradients);
        optimizer.Step(gradients);

        Assert.Equal(3, optimizer.StepCount);
        float lastLR = optimizer.LearningRatesUsed[2];

        // Save and load state
        var newOptimizer = new MockOptimizer(null, 1.0f);
        var newScheduler = new MockScheduler((step, baseLR) => baseLR * 0.9f);
        newOptimizer.SetScheduler(newScheduler);
        newOptimizer.LoadState(optimizer.GetState());

        // Continue training
        newOptimizer.Step(gradients);

        // LR should continue decaying from where it left off
        Assert.Equal(lastLR * 0.9f, newOptimizer.LearningRatesUsed[0], precision: 4);
    }

    #endregion

    #region SetParameters Tests

    [Fact]
    public void SetParameters_NullParameters_SetsEmptyDictionary()
    {
        var parameters = new Dictionary<string, Tensor>
        {
            { "param1", new Tensor(new float[10]) }
        };

        var optimizer = new MockOptimizer(parameters, 0.1f);
        Assert.Single(optimizer.Parameters);

        optimizer.SetParameters(null);
        Assert.Empty(optimizer.Parameters);
    }

    [Fact]
    public void SetParameters_ValidParameters_ReplacesParameters()
    {
        var parameters1 = new Dictionary<string, Tensor>
        {
            { "param1", new Tensor(new float[10]) }
        };

        var parameters2 = new Dictionary<string, Tensor>
        {
            { "param2", new Tensor(new float[20]) }
        };

        var optimizer = new MockOptimizer(parameters1, 0.1f);
        Assert.Single(optimizer.Parameters);
        Assert.True(optimizer.Parameters.ContainsKey("param1"));

        optimizer.SetParameters(parameters2);
        Assert.Single(optimizer.Parameters);
        Assert.True(optimizer.Parameters.ContainsKey("param2"));
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void StepParameter_NoScheduler_UsesBaseLR()
    {
        var optimizer = new MockOptimizer(null, 0.1f);
        var gradient = new Tensor(new float[10]);

        optimizer.StepParameter("param1", gradient);

        Assert.Single(optimizer.LearningRatesUsed);
        Assert.Equal(0.1f, optimizer.LearningRatesUsed[0]);
    }

    [Fact]
    public void StepParameter_WithScheduler_UsesScheduledLR()
    {
        var optimizer = new MockOptimizer(null, 0.1f);
        var scheduler = new MockScheduler((step, baseLR) => baseLR * 0.5f);
        optimizer.SetScheduler(scheduler);

        var gradient = new Tensor(new float[10]);
        optimizer.StepParameter("param1", gradient);

        Assert.Single(optimizer.LearningRatesUsed);
        Assert.Equal(0.05f, optimizer.LearningRatesUsed[0]);
    }

    [Fact]
    public void Scheduler_SetAfterSteps_ResetsBaseLR()
    {
        var optimizer = new MockOptimizer(null, 0.1f);

        // Take some steps without scheduler
        for (int i = 0; i < 5; i++)
        {
            optimizer.Step(new Dictionary<string, Tensor>());
        }

        // Now set scheduler
        var scheduler = new MockScheduler((step, baseLR) => baseLR * 0.5f);
        optimizer.SetScheduler(scheduler);

        // Should use step count from optimizer
        optimizer.Step(new Dictionary<string, Tensor>());

        Assert.Equal(0.05f, optimizer.LearningRatesUsed[5]);
    }

    #endregion
}
