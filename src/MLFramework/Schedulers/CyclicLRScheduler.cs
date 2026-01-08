namespace MLFramework.Schedulers;

/// <summary>
/// Cyclic learning rate scheduler.
/// Oscillates LR between baseLR and maxLR using triangular or exponential policy.
/// </summary>
public sealed class CyclicLRScheduler : BaseScheduler, IStepScheduler
{
    private readonly float _baseLearningRate;
    private readonly float _maxLearningRate;
    private readonly float _stepSizeUp;
    private readonly string _mode;
    private readonly float _gamma;
    private readonly float _stepSizeDown;

    public CyclicLRScheduler(
        float baseLearningRate,
        float maxLearningRate,
        float stepSizeUp,
        string mode = "triangular",
        float gamma = 0.99994f)
    {
        if (baseLearningRate <= 0)
            throw new ArgumentException("baseLearningRate must be positive", nameof(baseLearningRate));
        if (maxLearningRate <= 0)
            throw new ArgumentException("maxLearningRate must be positive", nameof(maxLearningRate));
        if (maxLearningRate <= baseLearningRate)
            throw new ArgumentException("maxLearningRate must be > baseLearningRate", nameof(maxLearningRate));
        if (stepSizeUp <= 0)
            throw new ArgumentException("stepSizeUp must be positive", nameof(stepSizeUp));
        if (mode != "triangular" && mode != "triangular2" && mode != "exp_range")
            throw new ArgumentException("mode must be 'triangular', 'triangular2', or 'exp_range'", nameof(mode));
        if (gamma <= 0 || gamma >= 1)
            throw new ArgumentException("gamma must be in (0, 1)", nameof(gamma));

        _baseLearningRate = baseLearningRate;
        _maxLearningRate = maxLearningRate;
        _stepSizeUp = stepSizeUp;
        _mode = mode;
        _gamma = gamma;
        _stepSizeDown = stepSizeUp; // Default: symmetric cycle
    }

    public override float GetLearningRate(int step, float baseLearningRate)
    {
        float cycle = (float)Math.Floor(1 + step / (2 * _stepSizeUp));
        float x = Math.Abs(step / _stepSizeUp - 2 * cycle + 1);
        float baseHeight = Math.Max(0, 1 - x);

        float scale;
        if (_mode == "triangular")
        {
            scale = 1;
        }
        else if (_mode == "triangular2")
        {
            scale = (float)Math.Pow(2, -(cycle - 1));
        }
        else // exp_range
        {
            scale = (float)Math.Pow(_gamma, step);
        }

        return _baseLearningRate + (_maxLearningRate - _baseLearningRate) * baseHeight * scale;
    }

    public override StateDict GetState()
    {
        var state = new StateDict();
        state.Set("base_lr", _baseLearningRate);
        state.Set("max_lr", _maxLearningRate);
        state.Set("step_size_up", _stepSizeUp);
        state.Set("step_size_down", _stepSizeDown);
        state.Set("mode", _mode);
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
