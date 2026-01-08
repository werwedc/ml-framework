using MLFramework.Schedulers;
using Xunit;

namespace MLFramework.Tests.Schedulers;

public class CyclicLRSchedulerTests
{
    [Fact]
    public void Constructor_ValidParameters_InitializesCorrectly()
    {
        var scheduler = new CyclicLRScheduler(
            baseLearningRate: 0.001f,
            maxLearningRate: 0.1f,
            stepSizeUp: 2000f
        );

        Assert.NotNull(scheduler);
    }

    [Fact]
    public void Constructor_NegativeBaseLearningRate_ThrowsException()
    {
        Assert.Throws<ArgumentException>(() => new CyclicLRScheduler(
            baseLearningRate: -0.001f,
            maxLearningRate: 0.1f,
            stepSizeUp: 2000f
        ));
    }

    [Fact]
    public void Constructor_MaxLearningRateLessThanOrEqualBase_ThrowsException()
    {
        Assert.Throws<ArgumentException>(() => new CyclicLRScheduler(
            baseLearningRate: 0.1f,
            maxLearningRate: 0.001f,
            stepSizeUp: 2000f
        ));

        Assert.Throws<ArgumentException>(() => new CyclicLRScheduler(
            baseLearningRate: 0.1f,
            maxLearningRate: 0.1f,
            stepSizeUp: 2000f
        ));
    }

    [Fact]
    public void Constructor_InvalidMode_ThrowsException()
    {
        Assert.Throws<ArgumentException>(() => new CyclicLRScheduler(
            baseLearningRate: 0.001f,
            maxLearningRate: 0.1f,
            stepSizeUp: 2000f,
            mode: "invalid"
        ));
    }

    [Fact]
    public void Constructor_InvalidGamma_ThrowsException()
    {
        Assert.Throws<ArgumentException>(() => new CyclicLRScheduler(
            baseLearningRate: 0.001f,
            maxLearningRate: 0.1f,
            stepSizeUp: 2000f,
            gamma: 0f
        ));

        Assert.Throws<ArgumentException>(() => new CyclicLRScheduler(
            baseLearningRate: 0.001f,
            maxLearningRate: 0.1f,
            stepSizeUp: 2000f,
            gamma: 1f
        ));
    }

    [Fact]
    public void GetLearningRate_TriangularMode_CompletesCycle()
    {
        var scheduler = new CyclicLRScheduler(
            baseLearningRate: 0.001f,
            maxLearningRate: 0.1f,
            stepSizeUp: 5f,
            mode: "triangular"
        );

        // One complete cycle is 2 * stepSizeUp = 10 steps
        float lr0 = scheduler.GetLearningRate(0, 0.1f);   // baseLR
        float lr5 = scheduler.GetLearningRate(5, 0.1f);   // baseLR (end of first half cycle)
        float lr10 = scheduler.GetLearningRate(10, 0.1f); // baseLR (end of full cycle)

        Assert.Equal(0.001f, lr0, precision: 6);
        Assert.Equal(0.001f, lr5, precision: 6);
        Assert.Equal(0.001f, lr10, precision: 6);
    }

    [Fact]
    public void GetLearningRate_TriangularMode_ReachesMaxLR()
    {
        var scheduler = new CyclicLRScheduler(
            baseLearningRate: 0.001f,
            maxLearningRate: 0.1f,
            stepSizeUp: 5f,
            mode: "triangular"
        );

        float lr0 = scheduler.GetLearningRate(0, 0.1f);   // baseLR
        float lr2_5 = scheduler.GetLearningRate(2, 0.1f);  // 40% of increasing phase
        float lr5 = scheduler.GetLearningRate(5, 0.1f);   // maxLR (peak)

        Assert.Equal(0.001f, lr0, precision: 6);
        // LR = 0.001 + (0.1 - 0.001) * (2/5) = 0.001 + 0.099 * 0.4 = 0.0406
        Assert.Equal(0.0406f, lr2_5, precision: 3);
        Assert.Equal(0.1f, lr5, precision: 6);
    }

    [Fact]
    public void GetLearningRate_Triangular2Mode_AmplitudeDecays()
    {
        var scheduler = new CyclicLRScheduler(
            baseLearningRate: 0.001f,
            maxLearningRate: 0.1f,
            stepSizeUp: 5f,
            mode: "triangular2"
        );

        // First cycle peak (step 5)
        float lrCycle1 = scheduler.GetLearningRate(5, 0.1f);
        // Second cycle peak (step 15) - amplitude should be halved
        float lrCycle2 = scheduler.GetLearningRate(15, 0.1f);

        // Cycle 1: LR = baseLR + (maxLR - baseLR) * 1 = 0.1
        Assert.Equal(0.1f, lrCycle1, precision: 6);
        // Cycle 2: LR = baseLR + (maxLR - baseLR) * 0.5 = 0.001 + 0.099 * 0.5 = 0.0505
        Assert.Equal(0.0505f, lrCycle2, precision: 3);
    }

    [Fact]
    public void GetLearningRate_ExpRangeMode_ExponentialDecay()
    {
        var scheduler = new CyclicLRScheduler(
            baseLearningRate: 0.001f,
            maxLearningRate: 0.1f,
            stepSizeUp: 5f,
            mode: "exp_range",
            gamma: 0.99994f
        );

        float lr0 = scheduler.GetLearningRate(0, 0.1f);   // baseLR
        float lr5 = scheduler.GetLearningRate(5, 0.1f);   // peak
        float lr10 = scheduler.GetLearningRate(10, 0.1f); // second peak with decay

        Assert.Equal(0.001f, lr0, precision: 6);
        Assert.Equal(0.1f, lr5, precision: 6);
        // Due to gamma^step, the second peak should be lower
        Assert.True(lr10 < lr5);
    }

    [Fact]
    public void GetLearningRate_SymmetricCycle_CorrectBehavior()
    {
        var scheduler = new CyclicLRScheduler(
            baseLearningRate: 0.001f,
            maxLearningRate: 0.1f,
            stepSizeUp: 5f
        );

        // Increasing phase: 0 -> 5
        float lr0 = scheduler.GetLearningRate(0, 0.1f);   // baseLR
        float lr2 = scheduler.GetLearningRate(2, 0.1f);   // increasing
        float lr5 = scheduler.GetLearningRate(5, 0.1f);   // maxLR

        // Decreasing phase: 5 -> 10
        float lr7 = scheduler.GetLearningRate(7, 0.1f);   // decreasing
        float lr10 = scheduler.GetLearningRate(10, 0.1f); // baseLR

        Assert.Equal(0.001f, lr0, precision: 6);
        Assert.True(lr2 > lr0 && lr2 < lr5);
        Assert.Equal(0.1f, lr5, precision: 6);
        Assert.True(lr7 < lr5 && lr7 > lr10);
        Assert.Equal(0.001f, lr10, precision: 6);
    }

    [Fact]
    public void GetLearningRate_AllModes_DifferentBehavior()
    {
        var triangular = new CyclicLRScheduler(
            baseLearningRate: 0.001f,
            maxLearningRate: 0.1f,
            stepSizeUp: 5f,
            mode: "triangular"
        );

        var triangular2 = new CyclicLRScheduler(
            baseLearningRate: 0.001f,
            maxLearningRate: 0.1f,
            stepSizeUp: 5f,
            mode: "triangular2"
        );

        var expRange = new CyclicLRScheduler(
            baseLearningRate: 0.001f,
            maxLearningRate: 0.1f,
            stepSizeUp: 5f,
            mode: "exp_range",
            gamma: 0.99994f
        );

        float tri1 = triangular.GetLearningRate(5, 0.1f);
        float tri2_1 = triangular2.GetLearningRate(5, 0.1f);
        float exp1 = expRange.GetLearningRate(5, 0.1f);

        float tri2 = triangular.GetLearningRate(15, 0.1f);
        float tri2_2 = triangular2.GetLearningRate(15, 0.1f);
        float exp2 = expRange.GetLearningRate(15, 0.1f);

        // First cycle: all modes reach maxLR
        Assert.Equal(0.1f, tri1, precision: 6);
        Assert.Equal(0.1f, tri2_1, precision: 6);
        Assert.Equal(0.1f, exp1, precision: 6);

        // Second cycle: triangular2 and exp_range decay
        Assert.Equal(0.1f, tri2, precision: 6);  // triangular doesn't decay
        Assert.True(tri2_2 < tri2_1);             // triangular2 decays
        Assert.True(exp2 < exp1);                 // exp_range decays
    }

    [Fact]
    public void GetState_SavesAllParameters()
    {
        var scheduler = new CyclicLRScheduler(
            baseLearningRate: 0.001f,
            maxLearningRate: 0.1f,
            stepSizeUp: 2000f,
            mode: "triangular",
            gamma: 0.99994f
        );

        scheduler.Step();
        scheduler.StepEpoch();

        var state = scheduler.GetState();

        Assert.Equal(0.001f, state.Get<float>("base_lr"));
        Assert.Equal(0.1f, state.Get<float>("max_lr"));
        Assert.Equal(2000f, state.Get<float>("step_size_up"));
        Assert.Equal(2000f, state.Get<float>("step_size_down"));
        Assert.Equal("triangular", state.Get<string>("mode"));
        Assert.Equal(0.99994f, state.Get<float>("gamma"));
        Assert.Equal(1, state.Get<int>("step_count"));
        Assert.Equal(1, state.Get<int>("epoch_count"));
    }

    [Fact]
    public void LoadState_RestoresStepAndEpochCounts()
    {
        var scheduler1 = new CyclicLRScheduler(
            baseLearningRate: 0.001f,
            maxLearningRate: 0.1f,
            stepSizeUp: 2000f
        );
        scheduler1.Step();
        scheduler1.Step();
        scheduler1.StepEpoch();

        var state = scheduler1.GetState();

        var scheduler2 = new CyclicLRScheduler(
            baseLearningRate: 0.001f,
            maxLearningRate: 0.1f,
            stepSizeUp: 2000f
        );
        scheduler2.LoadState(state);

        Assert.Equal(2, scheduler2.StepCount);
        Assert.Equal(1, scheduler2.EpochCount);
    }

    [Fact]
    public void Reset_ResetsToInitialValues()
    {
        var scheduler = new CyclicLRScheduler(
            baseLearningRate: 0.001f,
            maxLearningRate: 0.1f,
            stepSizeUp: 2000f
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
        // From spec: baseLR=0.001, maxLR=0.1, stepSizeUp=5
        // step=0: x=1, LR = baseLR = 0.001
        var scheduler = new CyclicLRScheduler(
            baseLearningRate: 0.001f,
            maxLearningRate: 0.1f,
            stepSizeUp: 5f
        );

        float lr0 = scheduler.GetLearningRate(0, 0.1f);
        Assert.Equal(0.001f, lr0, precision: 6);
    }

    [Fact]
    public void MathematicalVerification_Example2()
    {
        // From spec: baseLR=0.001, maxLR=0.1, stepSizeUp=5
        // step=2: x=0.6, LR = baseLR + (maxLR - baseLR) * 0.4
        var scheduler = new CyclicLRScheduler(
            baseLearningRate: 0.001f,
            maxLearningRate: 0.1f,
            stepSizeUp: 5f
        );

        float lr2 = scheduler.GetLearningRate(2, 0.1f);
        // LR = 0.001 + (0.1 - 0.001) * (1 - 0.6) = 0.001 + 0.099 * 0.4 = 0.0406
        Assert.Equal(0.0406f, lr2, precision: 3);
    }

    [Fact]
    public void MathematicalVerification_Example3()
    {
        // From spec: baseLR=0.001, maxLR=0.1, stepSizeUp=5
        // step=5: x=1, LR = baseLR = 0.001 (cycle complete)
        var scheduler = new CyclicLRScheduler(
            baseLearningRate: 0.001f,
            maxLearningRate: 0.1f,
            stepSizeUp: 5f
        );

        float lr5 = scheduler.GetLearningRate(5, 0.1f);
        Assert.Equal(0.001f, lr5, precision: 6);
    }

    [Fact]
    public void MultipleCycles_TriangularMode_NoDecay()
    {
        var scheduler = new CyclicLRScheduler(
            baseLearningRate: 0.001f,
            maxLearningRate: 0.1f,
            stepSizeUp: 5f,
            mode: "triangular"
        );

        float lrPeak1 = scheduler.GetLearningRate(5, 0.1f);   // First cycle peak
        float lrPeak2 = scheduler.GetLearningRate(15, 0.1f);  // Second cycle peak
        float lrPeak3 = scheduler.GetLearningRate(25, 0.1f);  // Third cycle peak

        // All peaks should be the same
        Assert.Equal(0.1f, lrPeak1, precision: 6);
        Assert.Equal(0.1f, lrPeak2, precision: 6);
        Assert.Equal(0.1f, lrPeak3, precision: 6);
    }
}
