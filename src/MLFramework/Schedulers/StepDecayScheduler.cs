namespace MLFramework.Schedulers;

/// <summary>
/// Decays learning rate by gamma every step_size steps.
/// Example: LR * 0.1 every 30 epochs.
/// </summary>
public sealed class StepDecayScheduler : BaseScheduler, IStepScheduler
{
    private readonly int _stepSize;
    private readonly float _gamma;

    public StepDecayScheduler(int stepSize, float gamma = 0.1f)
    {
        if (stepSize <= 0)
            throw new ArgumentException("stepSize must be positive", nameof(stepSize));

        _stepSize = stepSize;
        _gamma = gamma;
    }

    public override float GetLearningRate(int step, float baseLearningRate)
    {
        int decayCount = step / _stepSize;
        float decayFactor = (float)Math.Pow(_gamma, decayCount);
        return baseLearningRate * decayFactor;
    }

    public override StateDict GetState()
    {
        var state = new StateDict();
        state.Set("step_size", _stepSize);
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
