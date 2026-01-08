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

    [Fact]
    public void StepDecayScheduler_InvalidStepSize_ThrowsException()
    {
        Assert.Throws<ArgumentException>(() =>
            new StepDecayScheduler(stepSize: 0, gamma: 0.1f));

        Assert.Throws<ArgumentException>(() =>
            new StepDecayScheduler(stepSize: -10, gamma: 0.1f));
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

    [Fact]
    public void MultiStepDecayScheduler_NullMilestones_NoDecay()
    {
        var scheduler = new MultiStepDecayScheduler(
            milestones: null,
            gamma: 0.1f
        );
        float baseLR = 0.1f;

        Assert.Equal(0.1f, scheduler.GetLearningRate(0, baseLR));
        Assert.Equal(0.1f, scheduler.GetLearningRate(1000, baseLR));
    }

    [Fact]
    public void MultiStepDecayScheduler_StateSaveLoad_RestoresCorrectState()
    {
        var scheduler = new MultiStepDecayScheduler(
            milestones: new[] { 30, 60, 90 },
            gamma: 0.1f
        );
        scheduler.Step();
        scheduler.StepEpoch();

        var state = scheduler.GetState();
        var newScheduler = new MultiStepDecayScheduler(
            milestones: new[] { 30, 60, 90 },
            gamma: 0.1f
        );
        newScheduler.LoadState(state);

        Assert.Equal(1, newScheduler.StepCount);
        Assert.Equal(1, newScheduler.EpochCount);
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

    [Fact]
    public void ExponentialDecayScheduler_Reset_ClearsStepCount()
    {
        var scheduler = new ExponentialDecayScheduler(gamma: 0.95f);
        scheduler.Step();
        scheduler.Step();

        scheduler.Reset();

        Assert.Equal(0, scheduler.StepCount);
        Assert.Equal(0, scheduler.EpochCount);
    }

    [Fact]
    public void ExponentialDecayScheduler_DifferentGammaValues()
    {
        var scheduler1 = new ExponentialDecayScheduler(gamma: 0.9f);
        var scheduler2 = new ExponentialDecayScheduler(gamma: 0.99f);
        float baseLR = 1.0f;

        // gamma=0.9 decays faster than gamma=0.99
        float lr1 = scheduler1.GetLearningRate(10, baseLR);
        float lr2 = scheduler2.GetLearningRate(10, baseLR);

        Assert.True(lr1 < lr2);  // 0.9 decays faster
    }

    #endregion
}
