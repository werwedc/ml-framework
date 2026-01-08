using MLFramework.Schedulers;
using Xunit;

namespace MLFramework.Tests.Schedulers;

public class ReduceLROnPlateauSchedulerTests
{
    [Fact]
    public void Constructor_ValidParameters_InitializesCorrectly()
    {
        var scheduler = new ReduceLROnPlateauScheduler(
            mode: "min",
            factor: 0.1f,
            patience: 10
        );

        Assert.NotNull(scheduler);
    }

    [Fact]
    public void Constructor_InvalidMode_ThrowsException()
    {
        Assert.Throws<ArgumentException>(() => new ReduceLROnPlateauScheduler(
            mode: "invalid"
        ));
    }

    [Fact]
    public void Constructor_InvalidFactor_ThrowsException()
    {
        Assert.Throws<ArgumentException>(() => new ReduceLROnPlateauScheduler(
            factor: 0f
        ));

        Assert.Throws<ArgumentException>(() => new ReduceLROnPlateauScheduler(
            factor: 1f
        ));
    }

    [Fact]
    public void Constructor_InvalidPatience_ThrowsException()
    {
        Assert.Throws<ArgumentException>(() => new ReduceLROnPlateauScheduler(
            patience: 0
        ));

        Assert.Throws<ArgumentException>(() => new ReduceLROnPlateauScheduler(
            patience: -1
        ));
    }

    [Fact]
    public void Constructor_InvalidThreshold_ThrowsException()
    {
        Assert.Throws<ArgumentException>(() => new ReduceLROnPlateauScheduler(
            threshold: -0.001f
        ));
    }

    [Fact]
    public void Constructor_InvalidCooldown_ThrowsException()
    {
        Assert.Throws<ArgumentException>(() => new ReduceLROnPlateauScheduler(
            cooldown: -1
        ));
    }

    [Fact]
    public void Constructor_InvalidMinLearningRate_ThrowsException()
    {
        Assert.Throws<ArgumentException>(() => new ReduceLROnPlateauScheduler(
            minLearningRate: 0f
        ));

        Assert.Throws<ArgumentException>(() => new ReduceLROnPlateauScheduler(
            minLearningRate: -0.001f
        ));
    }

    [Fact]
    public void GetLearningRate_FirstCall_ReturnsBaseLearningRate()
    {
        var scheduler = new ReduceLROnPlateauScheduler();

        float lr = scheduler.GetLearningRate(0, 0.1f);

        Assert.Equal(0.1f, lr);
    }

    [Fact]
    public void GetLearningRate_SubsequentCalls_ReturnsCurrentLR()
    {
        var scheduler = new ReduceLROnPlateauScheduler();

        float lr1 = scheduler.GetLearningRate(0, 0.1f);
        float lr2 = scheduler.GetLearningRate(1, 0.1f);
        float lr3 = scheduler.GetLearningRate(2, 0.1f);

        Assert.Equal(lr1, lr2);
        Assert.Equal(lr2, lr3);
    }

    [Fact]
    public void UpdateMetric_MinMode_ImprovingMetric_ResetsWait()
    {
        var scheduler = new ReduceLROnPlateauScheduler(
            mode: "min",
            factor: 0.1f,
            patience: 3
        );

        scheduler.UpdateMetric("val_loss", 1.0f);
        scheduler.UpdateMetric("val_loss", 0.9f);
        scheduler.UpdateMetric("val_loss", 0.8f);

        float lr = scheduler.GetLearningRate(0, 0.1f);
        Assert.Equal(0.1f, lr); // No reduction yet
    }

    [Fact]
    public void UpdateMetric_MinMode_NoImprovement_ReducesLR()
    {
        var scheduler = new ReduceLROnPlateauScheduler(
            mode: "min",
            factor: 0.1f,
            patience: 2
        );

        // Initial metric
        scheduler.UpdateMetric("val_loss", 1.0f);

        // No improvement for patience+1 steps
        scheduler.UpdateMetric("val_loss", 1.1f);
        scheduler.UpdateMetric("val_loss", 1.2f);
        scheduler.UpdateMetric("val_loss", 1.3f);

        float lr = scheduler.GetLearningRate(0, 0.1f);
        Assert.Equal(0.01f, lr); // 0.1 * 0.1 = 0.01
    }

    [Fact]
    public void UpdateMetric_MaxMode_ImprovingMetric_ResetsWait()
    {
        var scheduler = new ReduceLROnPlateauScheduler(
            mode: "max",
            factor: 0.1f,
            patience: 3
        );

        scheduler.UpdateMetric("val_accuracy", 0.8f);
        scheduler.UpdateMetric("val_accuracy", 0.85f);
        scheduler.UpdateMetric("val_accuracy", 0.9f);

        float lr = scheduler.GetLearningRate(0, 0.1f);
        Assert.Equal(0.1f, lr); // No reduction yet
    }

    [Fact]
    public void UpdateMetric_MaxMode_NoImprovement_ReducesLR()
    {
        var scheduler = new ReduceLROnPlateauScheduler(
            mode: "max",
            factor: 0.1f,
            patience: 2
        );

        // Initial metric
        scheduler.UpdateMetric("val_accuracy", 0.9f);

        // No improvement for patience+1 steps
        scheduler.UpdateMetric("val_accuracy", 0.89f);
        scheduler.UpdateMetric("val_accuracy", 0.88f);
        scheduler.UpdateMetric("val_accuracy", 0.87f);

        float lr = scheduler.GetLearningRate(0, 0.1f);
        Assert.Equal(0.01f, lr); // 0.1 * 0.1 = 0.01
    }

    [Fact]
    public void UpdateMetric_Threshold_SmallChangeDoesNotTrigger()
    {
        var scheduler = new ReduceLROnPlateauScheduler(
            mode: "min",
            factor: 0.1f,
            patience: 2,
            threshold: 0.01f
        );

        // Initial metric
        scheduler.UpdateMetric("val_loss", 1.0f);

        // Small improvement (less than threshold)
        scheduler.UpdateMetric("val_loss", 0.995f);
        scheduler.UpdateMetric("val_loss", 0.992f);
        scheduler.UpdateMetric("val_loss", 0.991f);

        float lr = scheduler.GetLearningRate(0, 0.1f);
        Assert.Equal(0.1f, lr); // No reduction yet
    }

    [Fact]
    public void UpdateMetric_Cooldown_PreventsMultipleReductions()
    {
        var scheduler = new ReduceLROnPlateauScheduler(
            mode: "min",
            factor: 0.1f,
            patience: 2,
            cooldown: 2
        );

        // Initial metric
        scheduler.UpdateMetric("val_loss", 1.0f);

        // Trigger first reduction
        scheduler.UpdateMetric("val_loss", 1.1f);
        scheduler.UpdateMetric("val_loss", 1.2f);
        scheduler.UpdateMetric("val_loss", 1.3f);

        float lr1 = scheduler.GetLearningRate(0, 0.1f);
        Assert.Equal(0.01f, lr1); // Reduced to 0.01

        // In cooldown period - even with no improvement, no reduction
        scheduler.UpdateMetric("val_loss", 1.4f);
        scheduler.UpdateMetric("val_loss", 1.5f);

        float lr2 = scheduler.GetLearningRate(0, 0.1f);
        Assert.Equal(0.01f, lr2); // Still 0.01

        // After cooldown, trigger second reduction
        scheduler.UpdateMetric("val_loss", 1.6f);
        scheduler.UpdateMetric("val_loss", 1.7f);

        float lr3 = scheduler.GetLearningRate(0, 0.1f);
        Assert.Equal(0.001f, lr3); // Reduced to 0.001
    }

    [Fact]
    public void UpdateMetric_MinLearningRate_RespectsFloor()
    {
        var scheduler = new ReduceLROnPlateauScheduler(
            mode: "min",
            factor: 0.1f,
            patience: 1,
            minLearningRate: 0.005f
        );

        scheduler.UpdateMetric("val_loss", 1.0f);

        // First reduction: 0.1 -> 0.01
        scheduler.UpdateMetric("val_loss", 1.1f);
        scheduler.UpdateMetric("val_loss", 1.2f);

        float lr1 = scheduler.GetLearningRate(0, 0.1f);
        Assert.Equal(0.01f, lr1);

        // Second reduction would be 0.01 -> 0.001, but floor is 0.005
        scheduler.UpdateMetric("val_loss", 1.3f);
        scheduler.UpdateMetric("val_loss", 1.4f);

        float lr2 = scheduler.GetLearningRate(0, 0.1f);
        Assert.Equal(0.005f, lr2); // Floor applied

        // Third reduction stays at floor
        scheduler.UpdateMetric("val_loss", 1.5f);
        scheduler.UpdateMetric("val_loss", 1.6f);

        float lr3 = scheduler.GetLearningRate(0, 0.1f);
        Assert.Equal(0.005f, lr3); // Still at floor
    }

    [Fact]
    public void GetState_SavesAllParameters()
    {
        var scheduler = new ReduceLROnPlateauScheduler(
            mode: "min",
            factor: 0.1f,
            patience: 10,
            threshold: 1e-4f,
            cooldown: 2,
            minLearningRate: 1e-6f
        );

        scheduler.UpdateMetric("val_loss", 1.0f);
        scheduler.Step();
        scheduler.StepEpoch();

        var state = scheduler.GetState();

        Assert.Equal("min", state.Get<string>("mode"));
        Assert.Equal(0.1f, state.Get<float>("factor"));
        Assert.Equal(10, state.Get<int>("patience"));
        Assert.Equal(1e-4f, state.Get<float>("threshold"));
        Assert.Equal(2, state.Get<int>("cooldown"));
        Assert.Equal(1e-6f, state.Get<float>("min_lr"));
        Assert.Equal(1, state.Get<int>("step_count")); // One UpdateMetric call
        Assert.Equal(1, state.Get<int>("epoch_count"));
        Assert.Equal(0, state.Get<int>("wait")); // Reset by improving metric
    }

    [Fact]
    public void LoadState_RestoresAllTrackingVariables()
    {
        var scheduler1 = new ReduceLROnPlateauScheduler(
            mode: "min",
            factor: 0.1f,
            patience: 2
        );

        scheduler1.UpdateMetric("val_loss", 1.0f);
        scheduler1.UpdateMetric("val_loss", 1.1f);
        scheduler1.UpdateMetric("val_loss", 1.2f);

        var state = scheduler1.GetState();

        var scheduler2 = new ReduceLROnPlateauScheduler(
            mode: "min",
            factor: 0.1f,
            patience: 2
        );
        scheduler2.LoadState(state);

        Assert.Equal(3, scheduler2.StepCount);
        Assert.Equal(0, scheduler2.EpochCount);
        Assert.Equal(1.0f, state.Get<float>("best_metric"));
        Assert.Equal(0.01f, state.Get<float>("current_lr")); // Reduced once
    }

    [Fact]
    public void Reset_ResetsAllTrackingVariables()
    {
        var scheduler = new ReduceLROnPlateauScheduler(
            mode: "min",
            factor: 0.1f,
            patience: 2
        );

        scheduler.UpdateMetric("val_loss", 1.0f);
        scheduler.UpdateMetric("val_loss", 1.1f);
        scheduler.UpdateMetric("val_loss", 1.2f);

        float lrBeforeReset = scheduler.GetLearningRate(0, 0.1f);
        Assert.Equal(0.01f, lrBeforeReset); // Reduced once

        scheduler.Reset();

        Assert.Equal(0, scheduler.StepCount);
        Assert.Equal(0, scheduler.EpochCount);

        // After reset, LR should reinitialize on next GetLearningRate call
        float lrAfterReset = scheduler.GetLearningRate(0, 0.1f);
        Assert.Equal(0.1f, lrAfterReset); // Back to initial
    }

    [Fact]
    public void MultipleReductions_LRDecreasesCorrectly()
    {
        var scheduler = new ReduceLROnPlateauScheduler(
            mode: "min",
            factor: 0.5f,
            patience: 1
        );

        // First reduction: 0.1 -> 0.05
        scheduler.UpdateMetric("val_loss", 1.0f);
        scheduler.UpdateMetric("val_loss", 1.1f);
        scheduler.UpdateMetric("val_loss", 1.2f);

        float lr1 = scheduler.GetLearningRate(0, 0.1f);
        Assert.Equal(0.05f, lr1);

        // Second reduction: 0.05 -> 0.025
        scheduler.UpdateMetric("val_loss", 1.3f);
        scheduler.UpdateMetric("val_loss", 1.4f);

        float lr2 = scheduler.GetLearningRate(0, 0.1f);
        Assert.Equal(0.025f, lr2);

        // Third reduction: 0.025 -> 0.0125
        scheduler.UpdateMetric("val_loss", 1.5f);
        scheduler.UpdateMetric("val_loss", 1.6f);

        float lr3 = scheduler.GetLearningRate(0, 0.1f);
        Assert.Equal(0.0125f, lr3);
    }

    [Fact]
    public void ImprovementAfterNoImprovement_ResetsWaitCounter()
    {
        var scheduler = new ReduceLROnPlateauScheduler(
            mode: "min",
            factor: 0.1f,
            patience: 3
        );

        scheduler.UpdateMetric("val_loss", 1.0f);

        // Two steps without improvement
        scheduler.UpdateMetric("val_loss", 1.1f);
        scheduler.UpdateMetric("val_loss", 1.2f);

        // Improvement resets wait
        scheduler.UpdateMetric("val_loss", 0.9f);

        // Two more steps without improvement
        scheduler.UpdateMetric("val_loss", 1.0f);
        scheduler.UpdateMetric("val_loss", 1.1f);

        float lr = scheduler.GetLearningRate(0, 0.1f);
        Assert.Equal(0.1f, lr); // No reduction yet (wait was reset)
    }

    [Fact]
    public void StepCount_IncrementsWithUpdateMetric()
    {
        var scheduler = new ReduceLROnPlateauScheduler();

        Assert.Equal(0, scheduler.StepCount);

        scheduler.UpdateMetric("val_loss", 1.0f);
        Assert.Equal(1, scheduler.StepCount);

        scheduler.UpdateMetric("val_loss", 0.9f);
        Assert.Equal(2, scheduler.StepCount);
    }
}
