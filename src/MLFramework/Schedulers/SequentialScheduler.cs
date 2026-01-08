namespace MLFramework.Schedulers;

/// <summary>
/// Runs schedulers sequentially. Each scheduler runs for a specified duration,
/// then the next scheduler in the sequence takes over.
/// </summary>
public sealed class SequentialScheduler : BaseScheduler, IStepScheduler
{
    private readonly List<(ILearningRateScheduler scheduler, int duration)> _schedulerSequence;
    private int _currentSchedulerIndex;

    public SequentialScheduler(params (ILearningRateScheduler scheduler, int duration)[] sequence)
    {
        if (sequence == null || sequence.Length == 0)
        {
            throw new ArgumentException("Sequence must contain at least one scheduler", nameof(sequence));
        }

        _schedulerSequence = new List<(ILearningRateScheduler, int)>();
        int cumulativeSteps = 0;

        foreach (var (scheduler, duration) in sequence)
        {
            if (scheduler == null)
            {
                throw new ArgumentException("Scheduler cannot be null", nameof(sequence));
            }

            if (duration <= 0)
            {
                throw new ArgumentException($"Duration must be positive for scheduler at index {_schedulerSequence.Count}", nameof(sequence));
            }

            _schedulerSequence.Add((scheduler, duration));
            cumulativeSteps += duration;
        }

        _currentSchedulerIndex = 0;
    }

    public override float GetLearningRate(int step, float baseLearningRate)
    {
        // Find which scheduler is active at the current step
        int cumulativeSteps = 0;
        int schedulerIndex = -1;
        int schedulerStartStep = 0;

        for (int i = 0; i < _schedulerSequence.Count; i++)
        {
            var (scheduler, duration) = _schedulerSequence[i];

            if (step < cumulativeSteps + duration)
            {
                schedulerIndex = i;
                schedulerStartStep = cumulativeSteps;
                break;
            }

            cumulativeSteps += duration;
        }

        // If step is beyond all schedulers, use the last one
        if (schedulerIndex == -1)
        {
            schedulerIndex = _schedulerSequence.Count - 1;
            schedulerStartStep = cumulativeSteps - _schedulerSequence[schedulerIndex].duration;
        }

        var (activeScheduler, _) = _schedulerSequence[schedulerIndex];

        // Call the active scheduler with relative step count
        int relativeStep = step - schedulerStartStep;
        return activeScheduler.GetLearningRate(relativeStep, baseLearningRate);
    }

    public override void Step()
    {
        base.Step();
        _currentSchedulerIndex = DetermineActiveScheduler(_stepCount);
        _schedulerSequence[_currentSchedulerIndex].scheduler.Step();
    }

    private int DetermineActiveScheduler(int step)
    {
        int cumulativeSteps = 0;
        for (int i = 0; i < _schedulerSequence.Count; i++)
        {
            var (_, duration) = _schedulerSequence[i];
            if (step < cumulativeSteps + duration)
            {
                return i;
            }
            cumulativeSteps += duration;
        }
        return _schedulerSequence.Count - 1; // Return last scheduler
    }

    public override void Reset()
    {
        base.Reset();
        _currentSchedulerIndex = 0;
        foreach (var (scheduler, _) in _schedulerSequence)
        {
            scheduler.Reset();
        }
    }

    public override StateDict GetState()
    {
        var state = new StateDict();
        state.Set("scheduler_count", _schedulerSequence.Count);
        state.Set("current_scheduler_index", _currentSchedulerIndex);
        state.Set("step_count", _stepCount);
        state.Set("epoch_count", _epochCount);

        // Store durations and states
        var durations = _schedulerSequence.Select(s => s.duration).ToList();
        state.Set("durations", durations);

        var schedulerStates = _schedulerSequence.Select(s => s.scheduler.GetState()).ToList();
        state.Set("scheduler_states", schedulerStates);

        return state;
    }

    public override void LoadState(StateDict state)
    {
        _stepCount = state.Get<int>("step_count", 0);
        _epochCount = state.Get<int>("epoch_count", 0);
        _currentSchedulerIndex = state.Get<int>("current_scheduler_index", 0);

        var schedulerStates = state.Get<List<StateDict>>("scheduler_states");
        if (schedulerStates != null && schedulerStates.Count == _schedulerSequence.Count)
        {
            for (int i = 0; i < _schedulerSequence.Count; i++)
            {
                _schedulerSequence[i].scheduler.LoadState(schedulerStates[i]);
            }
        }
    }
}
