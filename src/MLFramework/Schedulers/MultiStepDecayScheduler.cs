namespace MLFramework.Schedulers;

/// <summary>
/// Decays learning rate by gamma at specific milestone steps.
/// Example: LR * 0.1 at epochs 30, 60, 90.
/// </summary>
public sealed class MultiStepDecayScheduler : BaseScheduler, IStepScheduler
{
    private readonly int[] _milestones;
    private readonly float _gamma;

    public MultiStepDecayScheduler(int[] milestones, float gamma = 0.1f)
    {
        _milestones = milestones ?? Array.Empty<int>();
        _gamma = gamma;
    }

    public override float GetLearningRate(int step, float baseLearningRate)
    {
        int decayCount = _milestones.Count(m => step >= m);
        float decayFactor = (float)Math.Pow(_gamma, decayCount);
        return baseLearningRate * decayFactor;
    }

    public override StateDict GetState()
    {
        var state = new StateDict();
        state.Set("milestones", _milestones);
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
