namespace MLFramework.Schedulers;

/// <summary>
/// Polynomial decay learning rate scheduler.
/// Decays LR polynomially from initialLR to finalLR over totalSteps.
/// </summary>
public sealed class PolynomialDecayScheduler : BaseScheduler, IStepScheduler
{
    private readonly float _initialLearningRate;
    private readonly float _finalLearningRate;
    private readonly float _totalSteps;
    private readonly float _power;

    public PolynomialDecayScheduler(
        float initialLearningRate,
        float finalLearningRate,
        float totalSteps,
        float power = 1.0f)
    {
        if (initialLearningRate <= 0)
            throw new ArgumentException("initialLearningRate must be positive", nameof(initialLearningRate));
        if (finalLearningRate < 0)
            throw new ArgumentException("finalLearningRate must be non-negative", nameof(finalLearningRate));
        if (totalSteps <= 0)
            throw new ArgumentException("totalSteps must be positive", nameof(totalSteps));
        if (power <= 0)
            throw new ArgumentException("power must be positive", nameof(power));

        _initialLearningRate = initialLearningRate;
        _finalLearningRate = finalLearningRate;
        _totalSteps = totalSteps;
        _power = power;
    }

    public override float GetLearningRate(int step, float baseLearningRate)
    {
        // Use the initialLearningRate if provided, otherwise use baseLR
        float initialLR = _initialLearningRate;
        float finalLR = _finalLearningRate;

        if (step >= _totalSteps)
        {
            return finalLR;
        }

        float progress = (float)step / _totalSteps;
        float decayFactor = (float)Math.Pow(1.0 - progress, _power);

        return initialLR + (finalLR - initialLR) * decayFactor;
    }

    public override StateDict GetState()
    {
        var state = new StateDict();
        state.Set("initial_lr", _initialLearningRate);
        state.Set("final_lr", _finalLearningRate);
        state.Set("total_steps", _totalSteps);
        state.Set("power", _power);
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
