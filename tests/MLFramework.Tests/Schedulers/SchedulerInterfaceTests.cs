using MLFramework.Schedulers;
using Xunit;

namespace MLFramework.Tests.Schedulers;

public class SchedulerInterfaceTests
{
    private class TestStepScheduler : BaseScheduler, IStepScheduler
    {
        public override float GetLearningRate(int step, float baseLearningRate)
        {
            return baseLearningRate;
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

    private class TestEpochScheduler : BaseScheduler, IEpochScheduler
    {
        public override float GetLearningRate(int step, float baseLearningRate)
        {
            return baseLearningRate;
        }

        public override StateDict GetState()
        {
            var state = new StateDict();
            state.Set("epochCount", _epochCount);
            return state;
        }

        public override void LoadState(StateDict state)
        {
            _epochCount = state.Get<int>("epochCount");
        }
    }

    private class TestMetricScheduler : BaseScheduler, IMetricBasedScheduler
    {
        private readonly Dictionary<string, float> _metrics = new Dictionary<string, float>();

        public override float GetLearningRate(int step, float baseLearningRate)
        {
            return baseLearningRate;
        }

        public void UpdateMetric(string metricName, float value)
        {
            _metrics[metricName] = value;
        }

        public float GetMetric(string metricName)
        {
            return _metrics.ContainsKey(metricName) ? _metrics[metricName] : float.NaN;
        }

        public override StateDict GetState()
        {
            var state = new StateDict();
            return state;
        }

        public override void LoadState(StateDict state)
        {
        }
    }

    [Fact]
    public void IStepScheduler_IsAssignableFromBaseScheduler()
    {
        var scheduler = new TestStepScheduler();

        Assert.IsAssignableFrom<ILearningRateScheduler>(scheduler);
        Assert.IsAssignableFrom<IStepScheduler>(scheduler);
    }

    [Fact]
    public void IStepScheduler_ImplementsILearningRateScheduler()
    {
        IStepScheduler scheduler = new TestStepScheduler();

        // Should be able to call ILearningRateScheduler methods
        scheduler.Step();
        scheduler.Reset();

        Assert.Equal(1, scheduler.GetState().Get<int>("stepCount"));
    }

    [Fact]
    public void IEpochScheduler_IsAssignableFromBaseScheduler()
    {
        var scheduler = new TestEpochScheduler();

        Assert.IsAssignableFrom<ILearningRateScheduler>(scheduler);
        Assert.IsAssignableFrom<IEpochScheduler>(scheduler);
    }

    [Fact]
    public void IEpochScheduler_ImplementsILearningRateScheduler()
    {
        IEpochScheduler scheduler = new TestEpochScheduler();

        // Should be able to call ILearningRateScheduler methods
        scheduler.Step();
        scheduler.StepEpoch();

        Assert.Equal(1, scheduler.GetState().Get<int>("epochCount"));
    }

    [Fact]
    public void IEpochScheduler_StepEpoch_IncrementsCount()
    {
        IEpochScheduler scheduler = new TestEpochScheduler();

        scheduler.StepEpoch();
        scheduler.StepEpoch();
        scheduler.StepEpoch();

        Assert.Equal(3, scheduler.GetState().Get<int>("epochCount"));
    }

    [Fact]
    public void IMetricBasedScheduler_IsAssignableFromBaseScheduler()
    {
        var scheduler = new TestMetricScheduler();

        Assert.IsAssignableFrom<ILearningRateScheduler>(scheduler);
        Assert.IsAssignableFrom<IMetricBasedScheduler>(scheduler);
    }

    [Fact]
    public void IMetricBasedScheduler_UpdateMetric_StoresValue()
    {
        var scheduler = new TestMetricScheduler();

        scheduler.UpdateMetric("val_loss", 0.5f);

        Assert.Equal(0.5f, scheduler.GetMetric("val_loss"));
    }

    [Fact]
    public void IMetricBasedScheduler_UpdateMultipleMetrics()
    {
        var scheduler = new TestMetricScheduler();

        scheduler.UpdateMetric("val_loss", 0.5f);
        scheduler.UpdateMetric("train_acc", 0.95f);
        scheduler.UpdateMetric("val_acc", 0.93f);

        Assert.Equal(0.5f, scheduler.GetMetric("val_loss"));
        Assert.Equal(0.95f, scheduler.GetMetric("train_acc"));
        Assert.Equal(0.93f, scheduler.GetMetric("val_acc"));
    }

    [Fact]
    public void IMetricBasedScheduler_UpdateMetric_OverwritesValue()
    {
        var scheduler = new TestMetricScheduler();

        scheduler.UpdateMetric("val_loss", 0.5f);
        scheduler.UpdateMetric("val_loss", 0.3f);

        Assert.Equal(0.3f, scheduler.GetMetric("val_loss"));
    }

    [Fact]
    public void TypeCheck_DistinguishesInterfaceTypes()
    {
        var stepScheduler = new TestStepScheduler();
        var epochScheduler = new TestEpochScheduler();
        var metricScheduler = new TestMetricScheduler();

        Assert.True(stepScheduler is IStepScheduler);
        Assert.False(stepScheduler is IEpochScheduler);
        Assert.False(stepScheduler is IMetricBasedScheduler);

        Assert.False(epochScheduler is IStepScheduler);
        Assert.True(epochScheduler is IEpochScheduler);
        Assert.False(epochScheduler is IMetricBasedScheduler);

        Assert.False(metricScheduler is IStepScheduler);
        Assert.False(metricScheduler is IEpochScheduler);
        Assert.True(metricScheduler is IMetricBasedScheduler);
    }
}
