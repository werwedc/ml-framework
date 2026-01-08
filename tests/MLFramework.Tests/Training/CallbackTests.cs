using System;
using System.Collections.Generic;
using MLFramework.Training;
using MLFramework.Schedulers;
using Xunit;

namespace MLFramework.Tests.Training;

/// <summary>
/// Tests for the base Callback class.
/// </summary>
public class CallbackTests
{
    /// <summary>
    /// Test callback that tracks method calls.
    /// </summary>
    private class TestCallback : Callback
    {
        public int OnBatchBeginCallCount { get; private set; }
        public int OnBatchEndCallCount { get; private set; }
        public int OnEpochBeginCallCount { get; private set; }
        public int OnEpochEndCallCount { get; private set; }
        public int OnValidationEndCallCount { get; private set; }
        public int OnTrainBeginCallCount { get; private set; }
        public int OnTrainEndCallCount { get; private set; }

        public List<int> OnBatchBeginBatchIndices { get; } = new List<int>();
        public List<int> OnBatchEndBatchIndices { get; } = new List<int>();
        public List<int> OnEpochBeginEpochIndices { get; } = new List<int>();
        public List<int> OnEpochEndEpochIndices { get; } = new List<int>();

        public override void OnBatchBegin(int batch)
        {
            base.OnBatchBegin(batch);
            OnBatchBeginCallCount++;
            OnBatchBeginBatchIndices.Add(batch);
        }

        public override void OnBatchEnd(int batch, Dictionary<string, float> metrics)
        {
            base.OnBatchEnd(batch, metrics);
            OnBatchEndCallCount++;
            OnBatchEndBatchIndices.Add(batch);
        }

        public override void OnEpochBegin(int epoch)
        {
            base.OnEpochBegin(epoch);
            OnEpochBeginCallCount++;
            OnEpochBeginEpochIndices.Add(epoch);
        }

        public override void OnEpochEnd(int epoch, Dictionary<string, float> metrics)
        {
            base.OnEpochEnd(epoch, metrics);
            OnEpochEndCallCount++;
            OnEpochEndEpochIndices.Add(epoch);
        }

        public override void OnValidationEnd(Dictionary<string, float> metrics)
        {
            base.OnValidationEnd(metrics);
            OnValidationEndCallCount++;
        }

        public override void OnTrainBegin(object model)
        {
            base.OnTrainBegin(model);
            OnTrainBeginCallCount++;
        }

        public override void OnTrainEnd(Dictionary<string, float> metrics)
        {
            base.OnTrainEnd(metrics);
            OnTrainEndCallCount++;
        }
    }

    [Fact]
    public void Callback_OnBatchBegin_IsCalled()
    {
        var callback = new TestCallback();

        callback.OnBatchBegin(5);

        Assert.Equal(1, callback.OnBatchBeginCallCount);
        Assert.Single(callback.OnBatchBeginBatchIndices);
        Assert.Equal(5, callback.OnBatchBeginBatchIndices[0]);
    }

    [Fact]
    public void Callback_OnBatchEnd_IsCalled()
    {
        var callback = new TestCallback();
        var metrics = new Dictionary<string, float> { { "loss", 0.5f } };

        callback.OnBatchEnd(5, metrics);

        Assert.Equal(1, callback.OnBatchEndCallCount);
        Assert.Single(callback.OnBatchEndBatchIndices);
        Assert.Equal(5, callback.OnBatchEndBatchIndices[0]);
    }

    [Fact]
    public void Callback_OnEpochBegin_IsCalled()
    {
        var callback = new TestCallback();

        callback.OnEpochBegin(3);

        Assert.Equal(1, callback.OnEpochBeginCallCount);
        Assert.Single(callback.OnEpochBeginEpochIndices);
        Assert.Equal(3, callback.OnEpochBeginEpochIndices[0]);
    }

    [Fact]
    public void Callback_OnEpochEnd_IsCalled()
    {
        var callback = new TestCallback();
        var metrics = new Dictionary<string, float> { { "accuracy", 0.9f } };

        callback.OnEpochEnd(3, metrics);

        Assert.Equal(1, callback.OnEpochEndCallCount);
        Assert.Single(callback.OnEpochEndEpochIndices);
        Assert.Equal(3, callback.OnEpochEndEpochIndices[0]);
    }

    [Fact]
    public void Callback_OnValidationEnd_IsCalled()
    {
        var callback = new TestCallback();
        var metrics = new Dictionary<string, float> { { "val_loss", 0.3f } };

        callback.OnValidationEnd(metrics);

        Assert.Equal(1, callback.OnValidationEndCallCount);
    }

    [Fact]
    public void Callback_OnTrainBegin_IsCalled()
    {
        var callback = new TestCallback();
        var model = new object();

        callback.OnTrainBegin(model);

        Assert.Equal(1, callback.OnTrainBeginCallCount);
    }

    [Fact]
    public void Callback_OnTrainEnd_IsCalled()
    {
        var callback = new TestCallback();
        var metrics = new Dictionary<string, float> { { "final_loss", 0.2f } };

        callback.OnTrainEnd(metrics);

        Assert.Equal(1, callback.OnTrainEndCallCount);
    }

    [Fact]
    public void Callback_MultipleCalls_AllTracked()
    {
        var callback = new TestCallback();
        var metrics = new Dictionary<string, float>();

        for (int i = 0; i < 5; i++)
        {
            callback.OnBatchBegin(i);
            callback.OnBatchEnd(i, metrics);
        }

        Assert.Equal(5, callback.OnBatchBeginCallCount);
        Assert.Equal(5, callback.OnBatchEndCallCount);
        Assert.Equal(5, callback.OnBatchBeginBatchIndices.Count);
        Assert.Equal(5, callback.OnBatchEndBatchIndices.Count);
    }

    [Fact]
    public void Callback_DefaultImplementation_DoesNotThrow()
    {
        var callback = new Callback();
        var metrics = new Dictionary<string, float>();

        // Should not throw exceptions
        callback.OnTrainBegin(new object());
        callback.OnBatchBegin(0);
        callback.OnBatchEnd(0, metrics);
        callback.OnEpochBegin(0);
        callback.OnEpochEnd(0, metrics);
        callback.OnValidationEnd(metrics);
        callback.OnTrainEnd(metrics);
    }
}
