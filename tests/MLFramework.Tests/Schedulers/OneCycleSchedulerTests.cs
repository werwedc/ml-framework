using MLFramework.Schedulers;
using Xunit;

namespace MLFramework.Tests.Schedulers;

public class OneCycleSchedulerTests
{
    [Fact]
    public void Constructor_ValidParameters_InitializesCorrectly()
    {
        var scheduler = new OneCycleScheduler(
            maxLearningRate: 0.1f,
            totalSteps: 100f,
            pctStart: 0.3f
        );

        Assert.NotNull(scheduler);
    }

    [Fact]
    public void Constructor_NegativeMaxLearningRate_ThrowsException()
    {
        Assert.Throws<ArgumentException>(() => new OneCycleScheduler(
            maxLearningRate: -0.1f,
            totalSteps: 100f
        ));
    }

    [Fact]
    public void Constructor_ZeroTotalSteps_ThrowsException()
    {
        Assert.Throws<ArgumentException>(() => new OneCycleScheduler(
            maxLearningRate: 0.1f,
            totalSteps: 0f
        ));
    }

    [Fact]
    public void Constructor_InvalidPctStart_ThrowsException()
    {
        Assert.Throws<ArgumentException>(() => new OneCycleScheduler(
            maxLearningRate: 0.1f,
            totalSteps: 100f,
            pctStart: 0f
        ));

        Assert.Throws<ArgumentException>(() => new OneCycleScheduler(
            maxLearningRate: 0.1f,
            totalSteps: 100f,
            pctStart: 1f
        ));
    }

    [Fact]
    public void Constructor_InvalidAnnealStrategy_ThrowsException()
    {
        Assert.Throws<ArgumentException>(() => new OneCycleScheduler(
            maxLearningRate: 0.1f,
            totalSteps: 100f,
            annealStrategy: "invalid"
        ));
    }

    [Fact]
    public void GetLearningRate_StepZero_ReturnsInitialLearningRate()
    {
        var scheduler = new OneCycleScheduler(
            maxLearningRate: 0.1f,
            totalSteps: 100f
        );

        float lr = scheduler.GetLearningRate(0, 0.1f);

        // initialLR = maxLR / divFactor = 0.1 / 25 = 0.004
        Assert.Equal(0.004f, lr, precision: 6);
    }

    [Fact]
    public void GetLearningRate_IncreasingPhase_LinearIncrease()
    {
        var scheduler = new OneCycleScheduler(
            maxLearningRate: 0.1f,
            totalSteps: 10f,
            pctStart: 0.3f
        );

        // pctStart * totalSteps = 3 steps for increasing phase
        float lr0 = scheduler.GetLearningRate(0, 0.1f);  // initialLR = 0.004
        float lr1 = scheduler.GetLearningRate(1, 0.1f);
        float lr2 = scheduler.GetLearningRate(2, 0.1f);
        float lr3 = scheduler.GetLearningRate(3, 0.1f);  // maxLR = 0.1

        Assert.True(lr0 < lr1);
        Assert.True(lr1 < lr2);
        Assert.True(lr2 < lr3);
        Assert.Equal(0.1f, lr3, precision: 6);
    }

    [Fact]
    public void GetLearningRate_DecreasingPhase_CosineAnnealing()
    {
        var scheduler = new OneCycleScheduler(
            maxLearningRate: 0.1f,
            totalSteps: 10f,
            pctStart: 0.3f,
            annealStrategy: "cos"
        );

        // Decreasing phase from step 3 to 9
        float lr3 = scheduler.GetLearningRate(3, 0.1f);  // maxLR = 0.1
        float lr9 = scheduler.GetLearningRate(9, 0.1f);  // near finalLR

        Assert.True(lr3 > lr9);
        // finalLR = maxLR / finalDivFactor = 0.1 / 1e4 = 1e-5
        Assert.Equal(1e-5f, lr9, precision: 6);
    }

    [Fact]
    public void GetLearningRate_DecreasingPhase_LinearAnnealing()
    {
        var scheduler = new OneCycleScheduler(
            maxLearningRate: 0.1f,
            totalSteps: 10f,
            pctStart: 0.3f,
            annealStrategy: "linear"
        );

        // Decreasing phase from step 3 to 9
        float lr3 = scheduler.GetLearningRate(3, 0.1f);  // maxLR = 0.1
        float lr9 = scheduler.GetLearningRate(9, 0.1f);  // finalLR

        Assert.True(lr3 > lr9);
    }

    [Fact]
    public void GetLearningRate_StepBeyondTotalSteps_ReturnsFinalLR()
    {
        var scheduler = new OneCycleScheduler(
            maxLearningRate: 0.1f,
            totalSteps: 10f
        );

        float lr10 = scheduler.GetLearningRate(10, 0.1f);
        float lr100 = scheduler.GetLearningRate(100, 0.1f);

        // Should return same as last step
        Assert.Equal(lr10, lr100);
    }

    [Fact]
    public void GetLearningRate_DifferentDivFactors_InitialLRCorrect()
    {
        var scheduler1 = new OneCycleScheduler(
            maxLearningRate: 0.1f,
            totalSteps: 100f,
            divFactor: 25f
        );

        var scheduler2 = new OneCycleScheduler(
            maxLearningRate: 0.1f,
            totalSteps: 100f,
            divFactor: 10f
        );

        float lr1 = scheduler1.GetLearningRate(0, 0.1f);
        float lr2 = scheduler2.GetLearningRate(0, 0.1f);

        // 0.1 / 25 = 0.004 vs 0.1 / 10 = 0.01
        Assert.Equal(0.004f, lr1, precision: 6);
        Assert.Equal(0.01f, lr2, precision: 6);
    }

    [Fact]
    public void GetLearningRate_DifferentFinalDivFactors_FinalLRCorrect()
    {
        var scheduler1 = new OneCycleScheduler(
            maxLearningRate: 0.1f,
            totalSteps: 10f,
            finalDivFactor: 1e4f
        );

        var scheduler2 = new OneCycleScheduler(
            maxLearningRate: 0.1f,
            totalSteps: 10f,
            finalDivFactor: 100f
        );

        float lr1 = scheduler1.GetLearningRate(9, 0.1f);
        float lr2 = scheduler2.GetLearningRate(9, 0.1f);

        // 0.1 / 1e4 = 1e-5 vs 0.1 / 100 = 1e-3
        Assert.Equal(1e-5f, lr1, precision: 6);
        Assert.Equal(0.001f, lr2, precision: 6);
    }

    [Fact]
    public void GetState_SavesAllParameters()
    {
        var scheduler = new OneCycleScheduler(
            maxLearningRate: 0.1f,
            totalSteps: 100f,
            pctStart: 0.3f,
            annealStrategy: "cos",
            divFactor: 25f,
            finalDivFactor: 1e4f
        );

        scheduler.Step();
        scheduler.StepEpoch();

        var state = scheduler.GetState();

        Assert.Equal(0.1f, state.Get<float>("max_lr"));
        Assert.Equal(100f, state.Get<float>("total_steps"));
        Assert.Equal(0.3f, state.Get<float>("pct_start"));
        Assert.Equal("cos", state.Get<string>("anneal_strategy"));
        Assert.Equal(25f, state.Get<float>("div_factor"));
        Assert.Equal(1e4f, state.Get<float>("final_div_factor"));
        Assert.Equal(0.004f, state.Get<float>("initial_lr"));
        Assert.Equal(1e-5f, state.Get<float>("final_lr"));
        Assert.Equal(1, state.Get<int>("step_count"));
        Assert.Equal(1, state.Get<int>("epoch_count"));
    }

    [Fact]
    public void LoadState_RestoresStepAndEpochCounts()
    {
        var scheduler1 = new OneCycleScheduler(
            maxLearningRate: 0.1f,
            totalSteps: 100f
        );
        scheduler1.Step();
        scheduler1.Step();
        scheduler1.StepEpoch();

        var state = scheduler1.GetState();

        var scheduler2 = new OneCycleScheduler(
            maxLearningRate: 0.1f,
            totalSteps: 100f
        );
        scheduler2.LoadState(state);

        Assert.Equal(2, scheduler2.StepCount);
        Assert.Equal(1, scheduler2.EpochCount);
    }

    [Fact]
    public void Reset_ResetsToInitialValues()
    {
        var scheduler = new OneCycleScheduler(
            maxLearningRate: 0.1f,
            totalSteps: 100f
        );
        scheduler.Step();
        scheduler.Step();
        scheduler.StepEpoch();

        scheduler.Reset();

        Assert.Equal(0, scheduler.StepCount);
        Assert.Equal(0, scheduler.EpochCount);
    }

    [Fact]
    public void MathematicalVerification_Example1()
    {
        // From spec: maxLR=0.1, totalSteps=10, pctStart=0.3
        // step=0: initialLR = 0.1/25 = 0.004
        var scheduler = new OneCycleScheduler(
            maxLearningRate: 0.1f,
            totalSteps: 10f,
            pctStart: 0.3f
        );

        float lr0 = scheduler.GetLearningRate(0, 0.1f);
        Assert.Equal(0.004f, lr0, precision: 6);
    }

    [Fact]
    public void MathematicalVerification_Example2()
    {
        // From spec: maxLR=0.1, totalSteps=10, pctStart=0.3
        // step=2: (2/3) of increasing phase -> LR ≈ 0.068
        var scheduler = new OneCycleScheduler(
            maxLearningRate: 0.1f,
            totalSteps: 10f,
            pctStart: 0.3f
        );

        float lr2 = scheduler.GetLearningRate(2, 0.1f);
        // LR = 0.004 + (0.1 - 0.004) * (2/3) = 0.004 + 0.096 * 0.666... = 0.068
        Assert.Equal(0.068f, lr2, precision: 3);
    }

    [Fact]
    public void MathematicalVerification_Example3()
    {
        // From spec: maxLR=0.1, totalSteps=10, pctStart=0.3
        // step=3: start of decreasing phase -> LR = maxLR = 0.1
        var scheduler = new OneCycleScheduler(
            maxLearningRate: 0.1f,
            totalSteps: 10f,
            pctStart: 0.3f
        );

        float lr3 = scheduler.GetLearningRate(3, 0.1f);
        Assert.Equal(0.1f, lr3, precision: 6);
    }

    [Fact]
    public void MathematicalVerification_Example4()
    {
        // From spec: maxLR=0.1, totalSteps=10, pctStart=0.3
        // step=9: near end of cycle -> LR ≈ finalLR = 0.1/1e4
        var scheduler = new OneCycleScheduler(
            maxLearningRate: 0.1f,
            totalSteps: 10f,
            pctStart: 0.3f,
            annealStrategy: "cos"
        );

        float lr9 = scheduler.GetLearningRate(9, 0.1f);
        Assert.Equal(1e-5f, lr9, precision: 6);
    }
}
