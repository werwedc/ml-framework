using MLFramework.Schedulers;
using Xunit;

namespace MLFramework.Tests.Schedulers;

public class BaseSchedulerTests
{
    private class TestScheduler : BaseScheduler
    {
        public override float GetLearningRate(int step, float baseLearningRate)
        {
            return baseLearningRate;
        }

        public override StateDict GetState()
        {
            var state = new StateDict();
            state.Set("stepCount", _stepCount);
            state.Set("epochCount", _epochCount);
            return state;
        }

        public override void LoadState(StateDict state)
        {
            _stepCount = state.Get<int>("stepCount");
            _epochCount = state.Get<int>("epochCount");
        }
    }

    [Fact]
    public void StepCount_InitializesToZero()
    {
        var scheduler = new TestScheduler();

        Assert.Equal(0, scheduler.StepCount);
    }

    [Fact]
    public void EpochCount_InitializesToZero()
    {
        var scheduler = new TestScheduler();

        Assert.Equal(0, scheduler.EpochCount);
    }

    [Fact]
    public void Step_IncrementsStepCount()
    {
        var scheduler = new TestScheduler();

        scheduler.Step();

        Assert.Equal(1, scheduler.StepCount);
    }

    [Fact]
    public void Step_MultipleIncrements()
    {
        var scheduler = new TestScheduler();

        for (int i = 0; i < 5; i++)
        {
            scheduler.Step();
        }

        Assert.Equal(5, scheduler.StepCount);
    }

    [Fact]
    public void StepEpoch_IncrementsEpochCount()
    {
        var scheduler = new TestScheduler();

        scheduler.StepEpoch();

        Assert.Equal(1, scheduler.EpochCount);
    }

    [Fact]
    public void StepEpoch_MultipleIncrements()
    {
        var scheduler = new TestScheduler();

        for (int i = 0; i < 3; i++)
        {
            scheduler.StepEpoch();
        }

        Assert.Equal(3, scheduler.EpochCount);
    }

    [Fact]
    public void Reset_ResetsStepCountToZero()
    {
        var scheduler = new TestScheduler();
        scheduler.Step();
        scheduler.Step();
        scheduler.Step();

        scheduler.Reset();

        Assert.Equal(0, scheduler.StepCount);
    }

    [Fact]
    public void Reset_ResetsEpochCountToZero()
    {
        var scheduler = new TestScheduler();
        scheduler.StepEpoch();
        scheduler.StepEpoch();

        scheduler.Reset();

        Assert.Equal(0, scheduler.EpochCount);
    }

    [Fact]
    public void Reset_ResetsBothCounters()
    {
        var scheduler = new TestScheduler();
        scheduler.Step();
        scheduler.Step();
        scheduler.StepEpoch();
        scheduler.Step();

        scheduler.Reset();

        Assert.Equal(0, scheduler.StepCount);
        Assert.Equal(0, scheduler.EpochCount);
    }

    [Fact]
    public void GetState_SavesCorrectState()
    {
        var scheduler = new TestScheduler();
        scheduler.Step();
        scheduler.Step();
        scheduler.StepEpoch();

        var state = scheduler.GetState();

        Assert.Equal(2, state.Get<int>("stepCount"));
        Assert.Equal(1, state.Get<int>("epochCount"));
    }

    [Fact]
    public void LoadState_RestoresCorrectState()
    {
        var scheduler1 = new TestScheduler();
        scheduler1.Step();
        scheduler1.Step();
        scheduler1.StepEpoch();

        var state = scheduler1.GetState();

        var scheduler2 = new TestScheduler();
        scheduler2.LoadState(state);

        Assert.Equal(2, scheduler2.StepCount);
        Assert.Equal(1, scheduler2.EpochCount);
    }

    [Fact]
    public void LoadState_AfterOperations_RestoresCorrectly()
    {
        var scheduler = new TestScheduler();
        scheduler.Step();
        scheduler.StepEpoch();

        var state = scheduler.GetState();
        scheduler.Reset();  // Reset to zero
        Assert.Equal(0, scheduler.StepCount);

        scheduler.LoadState(state);

        Assert.Equal(1, scheduler.StepCount);
        Assert.Equal(1, scheduler.EpochCount);
    }

    [Fact]
    public void GetLearningRate_UsesBaseLearningRate()
    {
        var scheduler = new TestScheduler();

        float lr = scheduler.GetLearningRate(0, 0.1f);

        Assert.Equal(0.1f, lr);
    }

    [Fact]
    public void GetLearningRate_IgnoresStepCountInBaseImplementation()
    {
        var scheduler = new TestScheduler();
        scheduler.Step();
        scheduler.Step();
        scheduler.Step();

        float lr1 = scheduler.GetLearningRate(10, 0.1f);
        float lr2 = scheduler.GetLearningRate(100, 0.1f);

        Assert.Equal(lr1, lr2);
    }
}
