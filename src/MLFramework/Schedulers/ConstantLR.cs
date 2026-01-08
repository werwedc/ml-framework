namespace MLFramework.Schedulers;

/// <summary>
/// Simple scheduler that always returns a constant learning rate.
/// </summary>
public sealed class ConstantLR : BaseScheduler, IStepScheduler
{
    private readonly float _learningRate;

    public ConstantLR(float learningRate)
    {
        _learningRate = learningRate;
    }

    public override float GetLearningRate(int step, float baseLearningRate)
    {
        return _learningRate;
    }

    public override StateDict GetState()
    {
        var state = new StateDict();
        state.Set("lr", _learningRate);
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
