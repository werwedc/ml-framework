using System;
using System.Collections.Generic;
using MLFramework.Training;
using MLFramework.Schedulers;
using Xunit;

namespace MLFramework.Tests.Training;

/// <summary>
/// Tests for the LRSchedulerCallback class.
/// </summary>
public class LRSchedulerCallbackTests
{
    #region Mock Schedulers

    /// <summary>
    /// Mock step-based scheduler for testing.
    /// </summary>
    private class MockStepScheduler : IStepScheduler
    {
        public int StepCallCount { get; private set; }
        public float CurrentLearningRate { get; private set; } = 0.1f;
        private readonly float _decay;

        public MockStepScheduler(float decay = 0.9f)
        {
            _decay = decay;
        }

        public float GetLearningRate(int step, float baseLearningRate)
        {
            return baseLearningRate * (float)Math.Pow(_decay, step);
        }

        public void Step()
        {
            StepCallCount++;
            CurrentLearningRate *= _decay;
        }

        public void Reset()
        {
            StepCallCount = 0;
            CurrentLearningRate = 0.1f;
        }

        public StateDict GetState()
        {
            return new StateDict();
        }

        public void LoadState(StateDict state)
        {
            // Mock implementation
        }
    }

    /// <summary>
    /// Mock epoch-based scheduler for testing.
    /// </summary>
    private class MockEpochScheduler : IEpochScheduler
    {
        public int StepEpochCallCount { get; private set; }
        public int StepCallCount { get; private set; }
        public float CurrentLearningRate { get; private set; } = 0.1f;
        private readonly float _decay;

        public MockEpochScheduler(float decay = 0.8f)
        {
            _decay = decay;
        }

        public float GetLearningRate(int step, float baseLearningRate)
        {
            return baseLearningRate * (float)Math.Pow(_decay, StepEpochCallCount);
        }

        public void Step()
        {
            StepCallCount++;
        }

        public void Reset()
        {
            StepCallCount = 0;
            StepEpochCallCount = 0;
            CurrentLearningRate = 0.1f;
        }

        public StateDict GetState()
        {
            return new StateDict();
        }

        public void LoadState(StateDict state)
        {
            // Mock implementation
        }

        public void StepEpoch()
        {
            StepEpochCallCount++;
            CurrentLearningRate *= _decay;
        }
    }

    /// <summary>
    /// Mock metric-based scheduler for testing.
    /// </summary>
    private class MockMetricScheduler : IMetricBasedScheduler
    {
        public int StepCallCount { get; private set; }
        public Dictionary<string, float> MetricsReceived { get; } = new Dictionary<string, float>();

        public float GetLearningRate(int step, float baseLearningRate)
        {
            return baseLearningRate;
        }

        public void Step()
        {
            StepCallCount++;
        }

        public void Reset()
        {
            StepCallCount = 0;
            MetricsReceived.Clear();
        }

        public StateDict GetState()
        {
            return new StateDict();
        }

        public void LoadState(StateDict state)
        {
            // Mock implementation
        }

        public void UpdateMetric(string metricName, float value)
        {
            MetricsReceived[metricName] = value;
        }
    }

    /// <summary>
    /// Mock scheduler that implements both IStepScheduler and IEpochScheduler.
    /// </summary>
    private class MockMultiInterfaceScheduler : IStepScheduler, IEpochScheduler
    {
        public int StepCallCount { get; private set; }
        public int StepEpochCallCount { get; private set; }

        public float GetLearningRate(int step, float baseLearningRate)
        {
            return baseLearningRate;
        }

        public void Step()
        {
            StepCallCount++;
        }

        public void Reset()
        {
            StepCallCount = 0;
            StepEpochCallCount = 0;
        }

        public StateDict GetState()
        {
            return new StateDict();
        }

        public void LoadState(StateDict state)
        {
            // Mock implementation
        }

        public void StepEpoch()
        {
            StepEpochCallCount++;
        }
    }

    #endregion

    #region Constructor Tests

    [Fact]
    public void Constructor_NullScheduler_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() => new LRSchedulerCallback(null));
    }

    [Fact]
    public void Constructor_StepScheduler_AutoDetectsStepOnBatch()
    {
        var scheduler = new MockStepScheduler();
        var callback = new LRSchedulerCallback(scheduler);

        var metrics = new Dictionary<string, float>();
        callback.OnBatchEnd(0, metrics);
        callback.OnEpochEnd(0, metrics);

        // Should step on batch but not on epoch
        Assert.Equal(1, scheduler.StepCallCount);
    }

    [Fact]
    public void Constructor_EpochScheduler_AutoDetectsStepOnEpoch()
    {
        var scheduler = new MockEpochScheduler();
        var callback = new LRSchedulerCallback(scheduler);

        var metrics = new Dictionary<string, float>();
        callback.OnBatchEnd(0, metrics);
        callback.OnEpochEnd(0, metrics);

        // Should step on epoch but not on batch
        Assert.Equal(1, scheduler.StepEpochCallCount);
        Assert.Equal(0, scheduler.StepCallCount);
    }

    [Fact]
    public void Constructor_WithManualControl_RespectsFlags()
    {
        var scheduler = new MockStepScheduler();
        var callback = new LRSchedulerCallback(scheduler, stepOnBatch: true, stepOnEpoch: true);

        var metrics = new Dictionary<string, float>();
        callback.OnBatchEnd(0, metrics);
        callback.OnEpochEnd(0, metrics);

        // Should step on both
        Assert.Equal(1, scheduler.StepCallCount);
    }

    [Fact]
    public void SchedulerProperty_ReturnsScheduler()
    {
        var scheduler = new MockStepScheduler();
        var callback = new LRSchedulerCallback(scheduler);

        Assert.Same(scheduler, callback.Scheduler);
    }

    #endregion

    #region Step-Based Scheduler Tests

    [Fact]
    public void OnBatchEnd_StepScheduler_StepsOnEachBatch()
    {
        var scheduler = new MockStepScheduler();
        var callback = new LRSchedulerCallback(scheduler);
        var metrics = new Dictionary<string, float>();

        // Simulate 5 batches
        for (int i = 0; i < 5; i++)
        {
            callback.OnBatchEnd(i, metrics);
        }

        Assert.Equal(5, scheduler.StepCallCount);
    }

    [Fact]
    public void OnBatchEnd_StepScheduler_WithManualFalse_DoesNotStep()
    {
        var scheduler = new MockStepScheduler();
        var callback = new LRSchedulerCallback(scheduler, stepOnBatch: false);
        var metrics = new Dictionary<string, float>();

        for (int i = 0; i < 5; i++)
        {
            callback.OnBatchEnd(i, metrics);
        }

        Assert.Equal(0, scheduler.StepCallCount);
    }

    [Fact]
    public void OnEpochEnd_StepScheduler_DoesNotStep()
    {
        var scheduler = new MockStepScheduler();
        var callback = new LRSchedulerCallback(scheduler);
        var metrics = new Dictionary<string, float>();

        callback.OnEpochEnd(0, metrics);
        callback.OnEpochEnd(1, metrics);

        // StepScheduler should not step on epoch end
        Assert.Equal(0, scheduler.StepCallCount);
    }

    [Fact]
    public void OnBatchEnd_MultipleBatches_LearningRateDecreases()
    {
        var scheduler = new MockStepScheduler(decay: 0.9f);
        var callback = new LRSchedulerCallback(scheduler);
        var metrics = new Dictionary<string, float>();

        float initialLR = scheduler.CurrentLearningRate;

        callback.OnBatchEnd(0, metrics);

        Assert.True(scheduler.CurrentLearningRate < initialLR);
    }

    #endregion

    #region Epoch-Based Scheduler Tests

    [Fact]
    public void OnEpochEnd_EpochScheduler_StepsOnEachEpoch()
    {
        var scheduler = new MockEpochScheduler();
        var callback = new LRSchedulerCallback(scheduler);
        var metrics = new Dictionary<string, float>();

        // Simulate 3 epochs
        for (int i = 0; i < 3; i++)
        {
            callback.OnEpochEnd(i, metrics);
        }

        Assert.Equal(3, scheduler.StepEpochCallCount);
    }

    [Fact]
    public void OnBatchEnd_EpochScheduler_DoesNotStep()
    {
        var scheduler = new MockEpochScheduler();
        var callback = new LRSchedulerCallback(scheduler);
        var metrics = new Dictionary<string, float>();

        for (int i = 0; i < 5; i++)
        {
            callback.OnBatchEnd(i, metrics);
        }

        Assert.Equal(0, scheduler.StepEpochCallCount);
    }

    [Fact]
    public void OnEpochEnd_EpochScheduler_WithManualFalse_DoesNotStep()
    {
        var scheduler = new MockEpochScheduler();
        var callback = new LRSchedulerCallback(scheduler, stepOnEpoch: false);
        var metrics = new Dictionary<string, float>();

        for (int i = 0; i < 3; i++)
        {
            callback.OnEpochEnd(i, metrics);
        }

        Assert.Equal(0, scheduler.StepEpochCallCount);
    }

    [Fact]
    public void OnEpochEnd_MultipleEpochs_LearningRateDecreases()
    {
        var scheduler = new MockEpochScheduler(decay: 0.8f);
        var callback = new LRSchedulerCallback(scheduler);
        var metrics = new Dictionary<string, float>();

        float initialLR = scheduler.CurrentLearningRate;

        callback.OnEpochEnd(0, metrics);

        Assert.True(scheduler.CurrentLearningRate < initialLR);
    }

    #endregion

    #region Metric-Based Scheduler Tests

    [Fact]
    public void OnEpochEnd_MetricScheduler_UpdatesMetric()
    {
        var scheduler = new MockMetricScheduler();
        var callback = new LRSchedulerCallback(scheduler, metricName: "loss");
        var metrics = new Dictionary<string, float> { { "loss", 0.5f } };

        callback.OnEpochEnd(0, metrics);

        Assert.True(scheduler.MetricsReceived.ContainsKey("loss"));
        Assert.Equal(0.5f, scheduler.MetricsReceived["loss"]);
    }

    [Fact]
    public void OnValidationEnd_MetricScheduler_UpdatesMetric()
    {
        var scheduler = new MockMetricScheduler();
        var callback = new LRSchedulerCallback(scheduler, metricName: "val_loss");
        var metrics = new Dictionary<string, float> { { "val_loss", 0.3f } };

        callback.OnValidationEnd(metrics);

        Assert.True(scheduler.MetricsReceived.ContainsKey("val_loss"));
        Assert.Equal(0.3f, scheduler.MetricsReceived["val_loss"]);
    }

    [Fact]
    public void OnEpochEnd_MetricScheduler_MetricNotFound_DoesNotThrow()
    {
        var scheduler = new MockMetricScheduler();
        var callback = new LRSchedulerCallback(scheduler, metricName: "loss");
        var metrics = new Dictionary<string, float> { { "accuracy", 0.9f } };

        // Should not throw
        callback.OnEpochEnd(0, metrics);

        Assert.False(scheduler.MetricsReceived.ContainsKey("loss"));
    }

    [Fact]
    public void OnEpochEnd_MetricScheduler_NullMetricName_DoesNotUpdate()
    {
        var scheduler = new MockMetricScheduler();
        var callback = new LRSchedulerCallback(scheduler);
        var metrics = new Dictionary<string, float> { { "loss", 0.5f } };

        callback.OnEpochEnd(0, metrics);

        Assert.Empty(scheduler.MetricsReceived);
    }

    [Fact]
    public void OnEpochEnd_MetricScheduler_MultipleUpdates_RecordsAll()
    {
        var scheduler = new MockMetricScheduler();
        var callback = new LRSchedulerCallback(scheduler, metricName: "loss");

        for (int i = 0; i < 5; i++)
        {
            var metrics = new Dictionary<string, float> { { "loss", 0.5f - (i * 0.1f) } };
            callback.OnEpochEnd(i, metrics);
        }

        Assert.Equal(5, scheduler.MetricsReceived.Count);
        Assert.Equal(0.0f, scheduler.MetricsReceived["loss"]);
    }

    #endregion

    #region Multiple Interface Tests

    [Fact]
    public void MultiInterfaceScheduler_AutoDetection_StepsOnBatch()
    {
        var scheduler = new MockMultiInterfaceScheduler();
        var callback = new LRSchedulerCallback(scheduler);
        var metrics = new Dictionary<string, float>();

        callback.OnBatchEnd(0, metrics);
        callback.OnEpochEnd(0, metrics);

        // Should step on batch (IStepScheduler takes precedence in auto-detection)
        Assert.Equal(1, scheduler.StepCallCount);
        Assert.Equal(0, scheduler.StepEpochCallCount);
    }

    [Fact]
    public void MultiInterfaceScheduler_ManualControl_StepsBoth()
    {
        var scheduler = new MockMultiInterfaceScheduler();
        var callback = new LRSchedulerCallback(scheduler, stepOnBatch: true, stepOnEpoch: true);
        var metrics = new Dictionary<string, float>();

        callback.OnBatchEnd(0, metrics);
        callback.OnEpochEnd(0, metrics);

        Assert.Equal(1, scheduler.StepCallCount);
        Assert.Equal(1, scheduler.StepEpochCallCount);
    }

    #endregion

    #region Training Loop Simulation Tests

    [Fact]
    public void TrainingLoop_Simulation_StepsCorrectly()
    {
        var scheduler = new MockStepScheduler();
        var callback = new LRSchedulerCallback(scheduler);

        // Simulate training loop: 3 epochs, 5 batches each
        for (int epoch = 0; epoch < 3; epoch++)
        {
            for (int batch = 0; batch < 5; batch++)
            {
                var metrics = new Dictionary<string, float> { { "loss", 0.5f } };
                callback.OnBatchEnd(batch, metrics);
            }

            var epochMetrics = new Dictionary<string, float> { { "epoch_loss", 0.3f } };
            callback.OnEpochEnd(epoch, epochMetrics);
        }

        // Should step 15 times (3 epochs * 5 batches)
        Assert.Equal(15, scheduler.StepCallCount);
    }

    [Fact]
    public void TrainingLoop_WithValidation_UpdatesMetrics()
    {
        var scheduler = new MockMetricScheduler();
        var callback = new LRSchedulerCallback(scheduler, metricName: "val_accuracy");

        // Simulate training with validation
        for (int epoch = 0; epoch < 3; epoch++)
        {
            // Training batches
            for (int batch = 0; batch < 5; batch++)
            {
                var metrics = new Dictionary<string, float>();
                callback.OnBatchEnd(batch, metrics);
            }

            // Validation
            var valMetrics = new Dictionary<string, float> { { "val_accuracy", 0.8f + (epoch * 0.05f) } };
            callback.OnValidationEnd(valMetrics);
        }

        // Should have 3 metric updates (one per epoch)
        Assert.Equal(3, scheduler.MetricsReceived.Count);
    }

    #endregion
}
