using Xunit;
using MLFramework.Schedulers;

namespace MLFramework.Tests.Schedulers;

public class WarmupSchedulerTests
{
    #region LinearWarmupScheduler Tests

    [Fact]
    public void LinearWarmupScheduler_AtZero_ReturnsStartLR()
    {
        var baseScheduler = new CosineAnnealingScheduler(tMax: 9000f);
        var scheduler = new LinearWarmupScheduler(baseScheduler, 1000);

        float lr = scheduler.GetLearningRate(0, 0.1f);

        Assert.Equal(0f, lr);
    }

    [Fact]
    public void LinearWarmupScheduler_DuringWarmup_ReturnsCorrectLR()
    {
        var baseScheduler = new CosineAnnealingScheduler(tMax: 9000f);
        var scheduler = new LinearWarmupScheduler(baseScheduler, 1000);

        float lr = scheduler.GetLearningRate(500, 0.1f);

        Assert.Equal(0.05f, lr);  // Halfway through warmup
    }

    [Fact]
    public void LinearWarmupScheduler_AfterWarmup_DelegatesToBase()
    {
        var baseScheduler = new CosineAnnealingScheduler(tMax: 100f, etaMin: 0f);
        var scheduler = new LinearWarmupScheduler(baseScheduler, 1000);

        float lr = scheduler.GetLearningRate(1500, 0.1f);

        // Should call base with step = 1500 - 1000 = 500
        // Base scheduler at step 500 with tMax=100
        float expectedLR = baseScheduler.GetLearningRate(500, 0.1f);
        Assert.Equal(expectedLR, lr);
    }

    #endregion

    #region ConstantWarmupScheduler Tests

    [Fact]
    public void ConstantWarmupScheduler_DuringWarmup_ReturnsWarmupLR()
    {
        var baseScheduler = new CosineAnnealingScheduler(tMax: 9000f);
        var scheduler = new ConstantWarmupScheduler(baseScheduler, 500, 1e-6f);

        float lr = scheduler.GetLearningRate(0, 0.1f);
        Assert.Equal(1e-6f, lr);

        lr = scheduler.GetLearningRate(499, 0.1f);
        Assert.Equal(1e-6f, lr);
    }

    [Fact]
    public void ConstantWarmupScheduler_AfterWarmup_DelegatesToBase()
    {
        var baseScheduler = new CosineAnnealingScheduler(tMax: 100f, etaMin: 0f);
        var scheduler = new ConstantWarmupScheduler(baseScheduler, 500, 1e-6f);

        float lr = scheduler.GetLearningRate(600, 0.1f);

        // Should call base with step = 600 - 500 = 100
        float expectedLR = baseScheduler.GetLearningRate(100, 0.1f);
        Assert.Equal(expectedLR, lr);
    }

    #endregion
}
