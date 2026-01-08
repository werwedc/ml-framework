namespace MLFramework.Schedulers;

/// <summary>
/// Uses a constant warmup learning rate during warmupSteps.
/// After warmup, delegates to the base scheduler.
/// </summary>
public sealed class ConstantWarmupScheduler : WarmupSchedulerBase
{
    private readonly float _warmupLearningRate;

    public ConstantWarmupScheduler(
        ILearningRateScheduler baseScheduler,
        int warmupSteps,
        float warmupLearningRate) : base(baseScheduler, warmupSteps)
    {
        if (warmupLearningRate <= 0)
        {
            throw new ArgumentException("warmupLearningRate must be positive", nameof(warmupLearningRate));
        }

        _warmupLearningRate = warmupLearningRate;
    }

    protected override float GetWarmupLearningRate(int step, float baseLearningRate)
    {
        return _warmupLearningRate;
    }

    public override StateDict GetState()
    {
        var state = base.GetState();
        state.Set("warmup_lr", _warmupLearningRate);
        return state;
    }
}
