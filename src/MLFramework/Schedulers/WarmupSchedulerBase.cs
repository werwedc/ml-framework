namespace MLFramework.Schedulers;

/// <summary>
/// Abstract base class for warmup schedulers.
/// Wraps another scheduler and applies warmup during the initial warmupSteps.
/// </summary>
public abstract class WarmupSchedulerBase : BaseScheduler, IStepScheduler
{
    protected readonly ILearningRateScheduler _baseScheduler;
    protected readonly int _warmupSteps;

    public WarmupSchedulerBase(ILearningRateScheduler baseScheduler, int warmupSteps)
    {
        _baseScheduler = baseScheduler ?? throw new ArgumentNullException(nameof(baseScheduler));
        _warmupSteps = warmupSteps;

        if (warmupSteps < 0)
        {
            throw new ArgumentException("warmupSteps must be non-negative", nameof(warmupSteps));
        }
    }

    protected abstract float GetWarmupLearningRate(int step, float baseLearningRate);

    public override float GetLearningRate(int step, float baseLearningRate)
    {
        if (step < _warmupSteps)
        {
            return GetWarmupLearningRate(step, baseLearningRate);
        }
        else
        {
            // After warmup, delegate to base scheduler
            return _baseScheduler.GetLearningRate(step - _warmupSteps, baseLearningRate);
        }
    }

    public override void Step()
    {
        base.Step();
        _baseScheduler?.Step();
    }

    public override void Reset()
    {
        base.Reset();
        _baseScheduler?.Reset();
    }

    public override StateDict GetState()
    {
        var state = new StateDict();
        state.Set("warmup_steps", _warmupSteps);
        state.Set("step_count", _stepCount);
        state.Set("epoch_count", _epochCount);
        state.Set("base_scheduler_state", _baseScheduler?.GetState());
        return state;
    }

    public override void LoadState(StateDict state)
    {
        _stepCount = state.Get<int>("step_count", 0);
        _epochCount = state.Get<int>("epoch_count", 0);

        var baseState = state.Get<StateDict>("base_scheduler_state");
        if (baseState != null && _baseScheduler != null)
        {
            _baseScheduler.LoadState(baseState);
        }
    }
}
