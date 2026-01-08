namespace MLFramework.Schedulers;

/// <summary>
/// One cycle learning rate scheduler from fast.ai.
/// Increases LR from initialLR to maxLR, then decreases to finalLR over one cycle.
/// </summary>
public sealed class OneCycleScheduler : BaseScheduler, IStepScheduler
{
    private readonly float _maxLearningRate;
    private readonly float _totalSteps;
    private readonly float _pctStart;
    private readonly string _annealStrategy;
    private readonly float _divFactor;
    private readonly float _finalDivFactor;

    // Derived parameters
    private readonly float _initialLearningRate;
    private readonly float _finalLearningRate;

    public OneCycleScheduler(
        float maxLearningRate,
        float totalSteps,
        float pctStart = 0.3f,
        string annealStrategy = "cos",
        float divFactor = 25f,
        float finalDivFactor = 1e4f)
    {
        if (maxLearningRate <= 0)
            throw new ArgumentException("maxLearningRate must be positive", nameof(maxLearningRate));
        if (totalSteps <= 0)
            throw new ArgumentException("totalSteps must be positive", nameof(totalSteps));
        if (pctStart <= 0 || pctStart >= 1)
            throw new ArgumentException("pctStart must be in (0, 1)", nameof(pctStart));
        if (annealStrategy != "cos" && annealStrategy != "linear")
            throw new ArgumentException("annealStrategy must be 'cos' or 'linear'", nameof(annealStrategy));

        _maxLearningRate = maxLearningRate;
        _totalSteps = totalSteps;
        _pctStart = pctStart;
        _annealStrategy = annealStrategy;
        _divFactor = divFactor;
        _finalDivFactor = finalDivFactor;

        _initialLearningRate = maxLearningRate / divFactor;
        _finalLearningRate = maxLearningRate / finalDivFactor;
    }

    public override float GetLearningRate(int step, float baseLearningRate)
    {
        if (step >= _totalSteps)
        {
            step = (int)_totalSteps - 1;
        }

        float pctStartSteps = _pctStart * _totalSteps;

        if (step < pctStartSteps)
        {
            // Increasing phase: initialLR -> maxLR
            float pct = step / pctStartSteps;
            return _initialLearningRate + (_maxLearningRate - _initialLearningRate) * pct;
        }
        else
        {
            // Decreasing phase: maxLR -> finalLR
            float pct = (step - pctStartSteps) / (_totalSteps - pctStartSteps);

            if (_annealStrategy == "cos")
            {
                float cosineValue = (float)Math.Cos(pct * Math.PI);
                return _finalLearningRate + (_maxLearningRate - _finalLearningRate) * (1 + cosineValue) / 2;
            }
            else // linear
            {
                return _maxLearningRate - (_maxLearningRate - _finalLearningRate) * pct;
            }
        }
    }

    public override StateDict GetState()
    {
        var state = new StateDict();
        state.Set("max_lr", _maxLearningRate);
        state.Set("total_steps", _totalSteps);
        state.Set("pct_start", _pctStart);
        state.Set("anneal_strategy", _annealStrategy);
        state.Set("div_factor", _divFactor);
        state.Set("final_div_factor", _finalDivFactor);
        state.Set("initial_lr", _initialLearningRate);
        state.Set("final_lr", _finalLearningRate);
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
