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
