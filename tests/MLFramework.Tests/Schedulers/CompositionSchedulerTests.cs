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
