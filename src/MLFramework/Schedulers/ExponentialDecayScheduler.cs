namespace MLFramework.Schedulers;

/// <summary>
/// Decays learning rate exponentially every step.
/// Formula: LR = baseLR * gamma^step
/// </summary>
public sealed class ExponentialDecayScheduler : BaseScheduler, IStepScheduler
{
    private readonly float _gamma;

    public ExponentialDecayScheduler(float gamma = 0.95f)
    {
        _gamma = gamma;
    }

    public override float GetLearningRate(int step, float baseLearningRate)
    {
        float decayFactor = (float)Math.Pow(_gamma, step);
        return baseLearningRate * decayFactor;
    }

    public override StateDict GetState()
    {
        var state = new StateDict();
        state.Set("gamma", _gamma);
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
