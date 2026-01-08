namespace MLFramework.Schedulers;

/// <summary>
/// Linearly increases learning rate from 0 to baseLR over warmupSteps.
/// After warmup, delegates to the base scheduler.
/// </summary>
public sealed class LinearWarmupScheduler : WarmupSchedulerBase
{
    private readonly float _startLearningRate;

    public LinearWarmupScheduler(
        ILearningRateScheduler baseScheduler,
        int warmupSteps,
        float startLearningRate = 0f) : base(baseScheduler, warmupSteps)
    {
        _startLearningRate = startLearningRate;

        if (startLearningRate < 0)
        {
            throw new ArgumentException("startLearningRate must be non-negative", nameof(startLearningRate));
        }
    }

    protected override float GetWarmupLearningRate(int step, float baseLearningRate)
    {
        if (_warmupSteps == 0)
        {
            return baseLearningRate;
        }

        float progress = (float)step / _warmupSteps;
        return _startLearningRate + (baseLearningRate - _startLearningRate) * progress;
    }

    public override StateDict GetState()
    {
        var state = base.GetState();
        state.Set("start_lr", _startLearningRate);
        return state;
    }
}
