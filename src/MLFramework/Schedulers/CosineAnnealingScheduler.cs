namespace MLFramework.Schedulers;

/// <summary>
/// Cosine annealing learning rate scheduler.
/// Smoothly decays LR from baseLR to eta_min over T_max steps.
/// Formula: LR = eta_min + 0.5 * (baseLR - eta_min) * (1 + cos(pi * step / T_max))
/// </summary>
public sealed class CosineAnnealingScheduler : BaseScheduler, IStepScheduler
{
    private readonly float _tMax;
    private readonly float _etaMin;
    private readonly float _tMaxInversePi;

    public CosineAnnealingScheduler(float tMax, float etaMin = 0f)
    {
        if (tMax <= 0)
        {
            throw new ArgumentException("T_max must be positive", nameof(tMax));
        }

        _tMax = tMax;
        _etaMin = etaMin;

        // Precompute constant for performance
        _tMaxInversePi = (float)Math.PI / tMax;
    }

    public override float GetLearningRate(int step, float baseLearningRate)
    {
        float cosinePhase = _tMaxInversePi * step;
        float cosineValue = (float)Math.Cos(cosinePhase);
        float decayFactor = 0.5f * (1.0f + cosineValue);

        return _etaMin + (baseLearningRate - _etaMin) * decayFactor;
    }

    public override StateDict GetState()
    {
        var state = new StateDict();
        state.Set("t_max", _tMax);
        state.Set("eta_min", _etaMin);
        state.Set("step_count", _stepCount);
        state.Set("epoch_count", _epochCount);
        return state;
    }

    public override void LoadState(StateDict state)
    {
        _stepCount = state.Get<int>("step_count", 0);
        _epochCount = state.Get<int>("epoch_count", 0);
    }
}
