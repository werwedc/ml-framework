using Xunit;
using MLFramework.Schedulers;

namespace MLFramework.Tests.Schedulers;

public class CompositionSchedulerTests
{
    #region ConstantLR Tests

    [Fact]
    public void ConstantLR_AlwaysReturnsSpecifiedLearningRate()
    {
        var scheduler = new ConstantLR(0.001f);

        float lr1 = scheduler.GetLearningRate(0, 0.1f);
        float lr2 = scheduler.GetLearningRate(100, 0.1f);
        float lr3 = scheduler.GetLearningRate(1000, 0.1f);

        Assert.Equal(0.001f, lr1);
        Assert.Equal(0.001f, lr2);
        Assert.Equal(0.001f, lr3);
    }

    [Fact]
    public void ConstantLR_DifferentLearningRates()
    {
        var s1 = new ConstantLR(1e-3f);
        var s2 = new ConstantLR(1e-4f);
        var s3 = new ConstantLR(5e-5f);

        Assert.Equal(1e-3f, s1.GetLearningRate(0, 0.1f));
        Assert.Equal(1e-4f, s2.GetLearningRate(100, 0.1f));
        Assert.Equal(5e-5f, s3.GetLearningRate(1000, 0.1f));
    }

    [Fact]
    public void ConstantLR_StateSerializationAndDeserialization()
    {
        var scheduler = new ConstantLR(0.001f);
        scheduler.Step();
        scheduler.Step();

        var state = scheduler.GetState();
        var newScheduler = new ConstantLR(0.001f);
        newScheduler.LoadState(state);

        Assert.Equal(2, newScheduler.StepCount);
        Assert.Equal(0.001f, newScheduler.GetLearningRate(0, 0.1f));
    }

    #endregion

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
    public void ChainedScheduler_VerifiesFinalLRIsProductOfAllOutputs()
    {
        var s1 = new ConstantLR(0.5f);
        var s2 = new ConstantLR(0.4f);
        var s3 = new ConstantLR(0.3f);
        var scheduler = new ChainedScheduler(s1, s2, s3);

        float lr = scheduler.GetLearningRate(0, 1.0f);

        // 1.0 * 0.5 = 0.5, then 0.5 * 0.4 = 0.2, then 0.2 * 0.3 = 0.06
        Assert.Equal(0.06f, lr);
    }

    [Fact]
    public void ChainedScheduler_MultipleSchedulers()
    {
        var s1 = new ConstantLR(0.9f);
        var s2 = new ConstantLR(0.8f);
        var s3 = new ConstantLR(0.7f);
        var scheduler = new ChainedScheduler(s1, s2, s3);

        float lr = scheduler.GetLearningRate(0, 0.1f);

        // 0.1 * 0.9 * 0.8 * 0.7 = 0.0504
        Assert.Equal(0.0504f, lr);
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

    [Fact]
    public void ChainedScheduler_ResetAllSchedulers()
    {
        var s1 = new CosineAnnealingScheduler(tMax: 100f);
        var s2 = new StepDecayScheduler(stepSize: 30, gamma: 0.1f);
        var scheduler = new ChainedScheduler(s1, s2);

        scheduler.Step();
        scheduler.Step();
        scheduler.Reset();

        Assert.Equal(0, s1.StepCount);
        Assert.Equal(0, s2.StepCount);
        Assert.Equal(0, scheduler.StepCount);
    }

    [Fact]
    public void ChainedScheduler_StateSerializationWithMultipleSchedulers()
    {
        var s1 = new ConstantLR(0.5f);
        var s2 = new ConstantLR(0.4f);
        var scheduler = new ChainedScheduler(s1, s2);

        scheduler.Step();
        scheduler.Step();

        var state = scheduler.GetState();
        var newScheduler = new ChainedScheduler(
            new ConstantLR(0.5f),
            new ConstantLR(0.4f)
        );
        newScheduler.LoadState(state);

        Assert.Equal(2, newScheduler.StepCount);
    }

    [Fact]
    public void ChainedScheduler_SingleScheduler_BehavesIdenticallyToScheduler()
    {
        var innerScheduler = new ConstantLR(0.001f);
        var scheduler = new ChainedScheduler(innerScheduler);

        float lr1 = scheduler.GetLearningRate(100, 0.1f);
        float lr2 = innerScheduler.GetLearningRate(100, 0.1f);

        Assert.Equal(lr2, lr1);
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
    public void SequentialScheduler_AtExactBoundary_SwitchesToNextScheduler()
    {
        var s1 = new ConstantLR(0.001f);
        var s2 = new ConstantLR(0.0001f);
        var scheduler = new SequentialScheduler(
            (s1, 1000),
            (s2, 2000)
        );

        // At step 1000, should use second scheduler with relative step 0
        float lr = scheduler.GetLearningRate(1000, 0.1f);

        Assert.Equal(0.0001f, lr);
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

    [Fact]
    public void SequentialScheduler_MoreThanTwoSchedulers_SwitchesCorrectly()
    {
        var s1 = new ConstantLR(0.001f);
        var s2 = new ConstantLR(0.0005f);
        var s3 = new ConstantLR(0.0001f);
        var scheduler = new SequentialScheduler(
            (s1, 1000),
            (s2, 1000),
            (s3, 1000)
        );

        float lr1 = scheduler.GetLearningRate(500, 0.1f);   // First scheduler
        float lr2 = scheduler.GetLearningRate(1500, 0.1f);  // Second scheduler
        float lr3 = scheduler.GetLearningRate(2500, 0.1f);  // Third scheduler

        Assert.Equal(0.001f, lr1);
        Assert.Equal(0.0005f, lr2);
        Assert.Equal(0.0001f, lr3);
    }

    [Fact]
    public void SequentialScheduler_StepOnlyStepsActiveScheduler()
    {
        var s1 = new ConstantLR(0.001f);
        var s2 = new ConstantLR(0.0001f);
        var scheduler = new SequentialScheduler(
            (s1, 1000),
            (s2, 2000)
        );

        scheduler.Step();
        scheduler.Step();

        // Active scheduler is s1
        Assert.Equal(2, scheduler.StepCount);

        // Advance past first scheduler
        for (int i = 0; i < 1000; i++)
        {
            scheduler.Step();
        }

        // Now s2 is active, step counts should be on s2
        Assert.Equal(1002, scheduler.StepCount);
    }

    [Fact]
    public void SequentialScheduler_StateSerializationWithSchedulerSequence()
    {
        var s1 = new ConstantLR(0.001f);
        var s2 = new ConstantLR(0.0001f);
        var scheduler = new SequentialScheduler(
            (s1, 1000),
            (s2, 2000)
        );

        scheduler.Step();
        scheduler.Step();

        var state = scheduler.GetState();
        var newScheduler = new SequentialScheduler(
            (new ConstantLR(0.001f), 1000),
            (new ConstantLR(0.0001f), 2000)
        );
        newScheduler.LoadState(state);

        Assert.Equal(2, newScheduler.StepCount);
        Assert.Equal(0.001f, newScheduler.GetLearningRate(500, 0.1f));
    }

    [Fact]
    public void SequentialScheduler_EmptySequence_ThrowsException()
    {
        Assert.Throws<ArgumentException>(() => new SequentialScheduler());
    }

    [Fact]
    public void SequentialScheduler_NullScheduler_ThrowsException()
    {
        Assert.Throws<ArgumentException>(() =>
            new SequentialScheduler((null, 1000))
        );
    }

    [Fact]
    public void SequentialScheduler_NegativeDuration_ThrowsException()
    {
        var s1 = new ConstantLR(0.001f);
        Assert.Throws<ArgumentException>(() =>
            new SequentialScheduler((s1, -100))
        );
    }

    #endregion
}
