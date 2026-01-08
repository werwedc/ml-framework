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

    [Fact]
    public void LinearWarmupScheduler_HalfwayThroughWarmup_ReturnsAverageLR()
    {
        var baseScheduler = new CosineAnnealingScheduler(tMax: 9000f);
        var scheduler = new LinearWarmupScheduler(baseScheduler, 1000);

        float lr = scheduler.GetLearningRate(500, 0.1f);

        // At halfway point, should be average of startLR and baseLR
        Assert.Equal(0.05f, lr);
    }

    [Fact]
    public void LinearWarmupScheduler_EndOfWarmup_CloseToBaseLR()
    {
        var baseScheduler = new CosineAnnealingScheduler(tMax: 9000f);
        var scheduler = new LinearWarmupScheduler(baseScheduler, 1000);

        float lr = scheduler.GetLearningRate(999, 0.1f);

        // At end of warmup, should be very close to baseLR
        Assert.Equal(0.0999f, lr);
    }

    [Fact]
    public void LinearWarmupScheduler_WithCustomStartLR_ReturnsCorrectLR()
    {
        var baseScheduler = new CosineAnnealingScheduler(tMax: 9000f);
        var scheduler = new LinearWarmupScheduler(baseScheduler, 1000, startLearningRate: 1e-5f);

        float lr = scheduler.GetLearningRate(0, 0.1f);
        Assert.Equal(1e-5f, lr);

        lr = scheduler.GetLearningRate(500, 0.1f);
        Assert.Equal((1e-5f + 0.1f) / 2f, lr);
    }

    [Fact]
    public void LinearWarmupScheduler_WithZeroWarmupSteps_DelegatesImmediately()
    {
        var baseScheduler = new CosineAnnealingScheduler(tMax: 100f, etaMin: 0f);
        var scheduler = new LinearWarmupScheduler(baseScheduler, 0);

        float lr = scheduler.GetLearningRate(0, 0.1f);

        // Should immediately delegate to base scheduler
        float expectedLR = baseScheduler.GetLearningRate(0, 0.1f);
        Assert.Equal(expectedLR, lr);
    }

    [Fact]
    public void LinearWarmupScheduler_StateSerialization_SerializesCorrectly()
    {
        var baseScheduler = new CosineAnnealingScheduler(tMax: 100f, etaMin: 0f);
        var scheduler = new LinearWarmupScheduler(baseScheduler, 1000, startLearningRate: 1e-5f);

        // Advance state
        scheduler.Step();
        scheduler.Step();

        var state = scheduler.GetState();

        Assert.Equal(1000, state.Get<int>("warmup_steps"));
        Assert.Equal(2, state.Get<int>("step_count"));
        Assert.Equal(0, state.Get<int>("epoch_count"));
        Assert.Equal(1e-5f, state.Get<float>("start_lr"));
        Assert.NotNull(state.Get<StateDict>("base_scheduler_state"));
    }

    [Fact]
    public void LinearWarmupScheduler_StateDeserialization_LoadsCorrectly()
    {
        var baseScheduler = new CosineAnnealingScheduler(tMax: 100f, etaMin: 0f);
        var scheduler1 = new LinearWarmupScheduler(baseScheduler, 1000, startLearningRate: 1e-5f);

        // Advance state
        for (int i = 0; i < 10; i++)
        {
            scheduler1.Step();
        }

        var state = scheduler1.GetState();

        // Create new scheduler and load state
        var baseScheduler2 = new CosineAnnealingScheduler(tMax: 100f, etaMin: 0f);
        var scheduler2 = new LinearWarmupScheduler(baseScheduler2, 1000, startLearningRate: 1e-5f);
        scheduler2.LoadState(state);

        // Verify state was loaded
        Assert.Equal(10, scheduler2.StepCount);
        Assert.Equal(0, scheduler2.EpochCount);
    }

    [Fact]
    public void LinearWarmupScheduler_Reset_ResetsBothWarmupAndBaseScheduler()
    {
        var baseScheduler = new CosineAnnealingScheduler(tMax: 100f, etaMin: 0f);
        var scheduler = new LinearWarmupScheduler(baseScheduler, 1000);

        // Advance state
        for (int i = 0; i < 10; i++)
        {
            scheduler.Step();
        }

        Assert.Equal(10, scheduler.StepCount);

        // Reset
        scheduler.Reset();

        Assert.Equal(0, scheduler.StepCount);
        Assert.Equal(0, scheduler.EpochCount);
    }

    [Fact]
    public void LinearWarmupScheduler_NullBaseScheduler_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() =>
        {
            new LinearWarmupScheduler(null!, 1000);
        });
    }

    [Fact]
    public void LinearWarmupScheduler_NegativeWarmupSteps_ThrowsArgumentException()
    {
        var baseScheduler = new CosineAnnealingScheduler(tMax: 9000f);
        Assert.Throws<ArgumentException>(() =>
        {
            new LinearWarmupScheduler(baseScheduler, -1);
        });
    }

    [Fact]
    public void LinearWarmupScheduler_NegativeStartLR_ThrowsArgumentException()
    {
        var baseScheduler = new CosineAnnealingScheduler(tMax: 9000f);
        Assert.Throws<ArgumentException>(() =>
        {
            new LinearWarmupScheduler(baseScheduler, 1000, startLearningRate: -1e-5f);
        });
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

    [Fact]
    public void ConstantWarmupScheduler_DuringWarmup_IndependentOfStep()
    {
        var baseScheduler = new CosineAnnealingScheduler(tMax: 9000f);
        var scheduler = new ConstantWarmupScheduler(baseScheduler, 500, 1e-6f);

        // Should return same LR regardless of step during warmup
        float lr1 = scheduler.GetLearningRate(0, 0.1f);
        float lr2 = scheduler.GetLearningRate(100, 0.1f);
        float lr3 = scheduler.GetLearningRate(499, 0.1f);

        Assert.Equal(lr1, lr2);
        Assert.Equal(lr2, lr3);
        Assert.Equal(1e-6f, lr1);
    }

    [Fact]
    public void ConstantWarmupScheduler_DuringWarmup_IndependentOfBaseLR()
    {
        var baseScheduler = new CosineAnnealingScheduler(tMax: 9000f);
        var scheduler = new ConstantWarmupScheduler(baseScheduler, 500, 1e-6f);

        // Should return same LR regardless of baseLR during warmup
        float lr1 = scheduler.GetLearningRate(100, 0.001f);
        float lr2 = scheduler.GetLearningRate(100, 0.1f);
        float lr3 = scheduler.GetLearningRate(100, 1.0f);

        Assert.Equal(lr1, lr2);
        Assert.Equal(lr2, lr3);
        Assert.Equal(1e-6f, lr1);
    }

    [Fact]
    public void ConstantWarmupScheduler_WithZeroWarmupSteps_DelegatesImmediately()
    {
        var baseScheduler = new CosineAnnealingScheduler(tMax: 100f, etaMin: 0f);
        var scheduler = new ConstantWarmupScheduler(baseScheduler, 0, 1e-6f);

        float lr = scheduler.GetLearningRate(0, 0.1f);

        // Should immediately delegate to base scheduler
        float expectedLR = baseScheduler.GetLearningRate(0, 0.1f);
        Assert.Equal(expectedLR, lr);
    }

    [Fact]
    public void ConstantWarmupScheduler_StateSerialization_SerializesCorrectly()
    {
        var baseScheduler = new CosineAnnealingScheduler(tMax: 100f, etaMin: 0f);
        var scheduler = new ConstantWarmupScheduler(baseScheduler, 500, 1e-6f);

        // Advance state
        scheduler.Step();
        scheduler.Step();

        var state = scheduler.GetState();

        Assert.Equal(500, state.Get<int>("warmup_steps"));
        Assert.Equal(2, state.Get<int>("step_count"));
        Assert.Equal(0, state.Get<int>("epoch_count"));
        Assert.Equal(1e-6f, state.Get<float>("warmup_lr"));
        Assert.NotNull(state.Get<StateDict>("base_scheduler_state"));
    }

    [Fact]
    public void ConstantWarmupScheduler_StateDeserialization_LoadsCorrectly()
    {
        var baseScheduler = new CosineAnnealingScheduler(tMax: 100f, etaMin: 0f);
        var scheduler1 = new ConstantWarmupScheduler(baseScheduler, 500, 1e-6f);

        // Advance state
        for (int i = 0; i < 10; i++)
        {
            scheduler1.Step();
        }

        var state = scheduler1.GetState();

        // Create new scheduler and load state
        var baseScheduler2 = new CosineAnnealingScheduler(tMax: 100f, etaMin: 0f);
        var scheduler2 = new ConstantWarmupScheduler(baseScheduler2, 500, 1e-6f);
        scheduler2.LoadState(state);

        // Verify state was loaded
        Assert.Equal(10, scheduler2.StepCount);
        Assert.Equal(0, scheduler2.EpochCount);
    }

    [Fact]
    public void ConstantWarmupScheduler_Reset_ResetsBothWarmupAndBaseScheduler()
    {
        var baseScheduler = new CosineAnnealingScheduler(tMax: 100f, etaMin: 0f);
        var scheduler = new ConstantWarmupScheduler(baseScheduler, 500, 1e-6f);

        // Advance state
        for (int i = 0; i < 10; i++)
        {
            scheduler.Step();
        }

        Assert.Equal(10, scheduler.StepCount);

        // Reset
        scheduler.Reset();

        Assert.Equal(0, scheduler.StepCount);
        Assert.Equal(0, scheduler.EpochCount);
    }

    [Fact]
    public void ConstantWarmupScheduler_NullBaseScheduler_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() =>
        {
            new ConstantWarmupScheduler(null!, 500, 1e-6f);
        });
    }

    [Fact]
    public void ConstantWarmupScheduler_NegativeWarmupSteps_ThrowsArgumentException()
    {
        var baseScheduler = new CosineAnnealingScheduler(tMax: 9000f);
        Assert.Throws<ArgumentException>(() =>
        {
            new ConstantWarmupScheduler(baseScheduler, -1, 1e-6f);
        });
    }

    [Fact]
    public void ConstantWarmupScheduler_NonPositiveWarmupLR_ThrowsArgumentException()
    {
        var baseScheduler = new CosineAnnealingScheduler(tMax: 9000f);
        Assert.Throws<ArgumentException>(() =>
        {
            new ConstantWarmupScheduler(baseScheduler, 500, -1e-6f);
        });

        Assert.Throws<ArgumentException>(() =>
        {
            new ConstantWarmupScheduler(baseScheduler, 500, 0f);
        });
    }

    #endregion

    #region WarmupSchedulerBase Tests

    [Fact]
    public void WarmupSchedulerBase_WithZeroWarmupSteps_DelegatesImmediately()
    {
        var baseScheduler = new CosineAnnealingScheduler(tMax: 100f, etaMin: 0f);
        var scheduler = new LinearWarmupScheduler(baseScheduler, 0);

        float lr = scheduler.GetLearningRate(0, 0.1f);
        float expectedLR = baseScheduler.GetLearningRate(0, 0.1f);

        Assert.Equal(expectedLR, lr);
    }

    [Fact]
    public void WarmupSchedulerBase_Step_InvokesBaseSchedulerStep()
    {
        var baseScheduler = new CosineAnnealingScheduler(tMax: 100f, etaMin: 0f);
        var scheduler = new LinearWarmupScheduler(baseScheduler, 1000);

        scheduler.Step();

        Assert.Equal(1, scheduler.StepCount);
        Assert.Equal(1, baseScheduler.StepCount);
    }

    [Fact]
    public void WarmupSchedulerBase_Reset_InvokesBaseSchedulerReset()
    {
        var baseScheduler = new CosineAnnealingScheduler(tMax: 100f, etaMin: 0f);
        var scheduler = new LinearWarmupScheduler(baseScheduler, 1000);

        scheduler.Step();
        scheduler.Step();
        scheduler.Step();

        Assert.Equal(3, scheduler.StepCount);
        Assert.Equal(3, baseScheduler.StepCount);

        scheduler.Reset();

        Assert.Equal(0, scheduler.StepCount);
        Assert.Equal(0, baseScheduler.StepCount);
    }

    [Fact]
    public void WarmupSchedulerBase_State_PreservesBaseSchedulerState()
    {
        var baseScheduler = new CosineAnnealingScheduler(tMax: 100f, etaMin: 0f);
        var scheduler = new LinearWarmupScheduler(baseScheduler, 1000);

        // Advance both schedulers
        for (int i = 0; i < 5; i++)
        {
            scheduler.Step();
        }

        var state = scheduler.GetState();
        var baseState = state.Get<StateDict>("base_scheduler_state");

        Assert.NotNull(baseState);
        Assert.Equal(5, baseState.Get<int>("step_count"));
    }

    #endregion

    #region Mathematical Verification Tests

    [Fact]
    public void LinearWarmupScheduler_MathematicalVerification()
    {
        var baseScheduler = new CosineAnnealingScheduler(tMax: 9000f);
        var scheduler = new LinearWarmupScheduler(baseScheduler, 1000);

        // step=0:    LR = 0 + (0.1 - 0) * 0/1000 = 0
        Assert.Equal(0f, scheduler.GetLearningRate(0, 0.1f));

        // step=500:  LR = 0 + (0.1 - 0) * 500/1000 = 0.05
        Assert.Equal(0.05f, scheduler.GetLearningRate(500, 0.1f));

        // step=999:  LR = 0 + (0.1 - 0) * 999/1000 ≈ 0.0999
        Assert.Equal(0.0999f, scheduler.GetLearningRate(999, 0.1f));

        // step=1000: LR = baseScheduler.GetLearningRate(0, 0.1f)
        Assert.Equal(baseScheduler.GetLearningRate(0, 0.1f), scheduler.GetLearningRate(1000, 0.1f));
    }

    [Fact]
    public void ConstantWarmupScheduler_MathematicalVerification()
    {
        var baseScheduler = new CosineAnnealingScheduler(tMax: 9000f);
        var scheduler = new ConstantWarmupScheduler(baseScheduler, 500, 1e-6f);

        // step=0-499: LR = 1e-6
        Assert.Equal(1e-6f, scheduler.GetLearningRate(0, 0.1f));
        Assert.Equal(1e-6f, scheduler.GetLearningRate(100, 0.1f));
        Assert.Equal(1e-6f, scheduler.GetLearningRate(499, 0.1f));

        // step=500: LR = baseScheduler.GetLearningRate(0, 0.1f)
        Assert.Equal(baseScheduler.GetLearningRate(0, 0.1f), scheduler.GetLearningRate(500, 0.1f));
    }

    #endregion
}
