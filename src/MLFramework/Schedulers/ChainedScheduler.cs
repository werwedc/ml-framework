namespace MLFramework.Schedulers;

/// <summary>
/// Chains multiple schedulers together by multiplying their outputs.
/// Useful for composing warmup with decay schedules.
/// </summary>
public sealed class ChainedScheduler : BaseScheduler, IStepScheduler
{
    private readonly ILearningRateScheduler[] _schedulers;

    public ChainedScheduler(params ILearningRateScheduler[] schedulers)
    {
        _schedulers = schedulers ?? throw new ArgumentNullException(nameof(schedulers));
    }

    public override float GetLearningRate(int step, float baseLearningRate)
    {
        float combinedLR = baseLearningRate;

        foreach (var scheduler in _schedulers)
        {
            // Each scheduler operates on the combined LR with normalized input
            combinedLR = scheduler.GetLearningRate(step, combinedLR);
        }

        return combinedLR;
    }

    public override void Step()
    {
        base.Step();
        foreach (var scheduler in _schedulers)
        {
            scheduler.Step();
        }
    }

    public override void Reset()
    {
        base.Reset();
        foreach (var scheduler in _schedulers)
        {
            scheduler.Reset();
        }
    }

    public override StateDict GetState()
    {
        var state = new StateDict();
        state.Set("scheduler_count", _schedulers.Length);
        state.Set("step_count", _stepCount);
        state.Set("epoch_count", _epochCount);

        // Store states for all chained schedulers
        var schedulerStates = new List<StateDict>();
        foreach (var scheduler in _schedulers)
        {
            schedulerStates.Add(scheduler.GetState());
        }
        state.Set("scheduler_states", schedulerStates);

        return state;
    }

    public override void LoadState(StateDict state)
    {
        _stepCount = state.Get<int>("step_count", 0);
        _epochCount = state.Get<int>("epoch_count", 0);

        var schedulerStates = state.Get<List<StateDict>>("scheduler_states");
        if (schedulerStates != null && schedulerStates.Count == _schedulers.Length)
        {
            for (int i = 0; i < _schedulers.Length; i++)
            {
                _schedulers[i].LoadState(schedulerStates[i]);
            }
        }
    }
}
