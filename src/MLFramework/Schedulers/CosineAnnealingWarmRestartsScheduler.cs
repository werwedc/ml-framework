namespace MLFramework.Schedulers;

/// <summary>
/// Cosine annealing scheduler with warm restarts (SGDR).
/// Resets the schedule every T_0 * (T_mult^i) steps.
/// Formula: LR = eta_min + 0.5 * (baseLR - eta_min) * (1 + cos(pi * T_cur / T_i))
/// </summary>
public sealed class CosineAnnealingWarmRestartsScheduler : BaseScheduler, IStepScheduler
{
    private readonly float _t0;
    private readonly float _tMult;
    private readonly float _etaMin;

    // Derived state
    private float _currentCycleLength;
    private int _cycleCount;

    public CosineAnnealingWarmRestartsScheduler(
        float t0,
        float tMult = 2f,
        float etaMin = 1e-6f)
    {
        if (t0 <= 0)
        {
            throw new ArgumentException("T_0 must be positive", nameof(t0));
        }

        if (tMult < 1f)
        {
            throw new ArgumentException("T_mult must be >= 1", nameof(tMult));
        }

        _t0 = t0;
        _tMult = tMult;
        _etaMin = etaMin;

        _currentCycleLength = _t0;
        _cycleCount = 0;
    }

    public override float GetLearningRate(int step, float baseLearningRate)
    {
        // Calculate position within current cycle
        float stepsInCycle = step % _currentCycleLength;

        // If we've completed a cycle, update cycle length and count
        if (step >= (int)_currentCycleLength * (_cycleCount + 1))
        {
            _cycleCount++;
            _currentCycleLength = _t0 * (float)Math.Pow(_tMult, _cycleCount);
        }

        // Cosine annealing within current cycle
        float cycleProgress = stepsInCycle / _currentCycleLength;
        float cosineValue = (float)Math.Cos(cycleProgress * Math.PI);
        float decayFactor = 0.5f * (1.0f + cosineValue);

        return _etaMin + (baseLearningRate - _etaMin) * decayFactor;
    }

    public override void Reset()
    {
        base.Reset();
        _currentCycleLength = _t0;
        _cycleCount = 0;
    }

    public override StateDict GetState()
    {
        var state = new StateDict();
        state.Set("t_0", _t0);
        state.Set("t_mult", _tMult);
        state.Set("eta_min", _etaMin);
        state.Set("step_count", _stepCount);
        state.Set("epoch_count", _epochCount);
        state.Set("cycle_count", _cycleCount);
        state.Set("current_cycle_length", _currentCycleLength);
        return state;
    }

    public override void LoadState(StateDict state)
    {
        _stepCount = state.Get<int>("step_count", 0);
        _epochCount = state.Get<int>("epoch_count", 0);
        _cycleCount = state.Get<int>("cycle_count", 0);
        _currentCycleLength = state.Get<float>("current_cycle_length", _t0);
    }
}
