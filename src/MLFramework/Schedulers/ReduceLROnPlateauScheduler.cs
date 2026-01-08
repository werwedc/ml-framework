namespace MLFramework.Schedulers;

/// <summary>
/// Reduces learning rate when a metric has stopped improving.
/// </summary>
public sealed class ReduceLROnPlateauScheduler : BaseScheduler, IMetricBasedScheduler
{
    private readonly string _mode;
    private readonly float _factor;
    private readonly int _patience;
    private readonly float _threshold;
    private readonly int _cooldown;
    private readonly float _minLearningRate;

    // State
    private int _wait;
    private int _cooldownCounter;
    private float _bestMetric;
    private float _currentLearningRate;

    public ReduceLROnPlateauScheduler(
        string mode = "min",
        float factor = 0.1f,
        int patience = 10,
        float threshold = 1e-4f,
        int cooldown = 0,
        float minLearningRate = 1e-6f)
    {
        if (mode != "min" && mode != "max")
            throw new ArgumentException("mode must be 'min' or 'max'", nameof(mode));
        if (factor <= 0 || factor >= 1)
            throw new ArgumentException("factor must be in (0, 1)", nameof(factor));
        if (patience <= 0)
            throw new ArgumentException("patience must be positive", nameof(patience));
        if (threshold < 0)
            throw new ArgumentException("threshold must be non-negative", nameof(threshold));
        if (cooldown < 0)
            throw new ArgumentException("cooldown must be non-negative", nameof(cooldown));
        if (minLearningRate <= 0)
            throw new ArgumentException("minLearningRate must be positive", nameof(minLearningRate));

        _mode = mode;
        _factor = factor;
        _patience = patience;
        _threshold = threshold;
        _cooldown = cooldown;
        _minLearningRate = minLearningRate;

        Reset();
    }

    public override float GetLearningRate(int step, float baseLearningRate)
    {
        // Initialize with base learning rate on first call if not set
        if (_currentLearningRate == float.MaxValue)
        {
            _currentLearningRate = baseLearningRate;
        }
        return _currentLearningRate;
    }

    public void UpdateMetric(string metricName, float value)
    {
        _stepCount++; // Update step count on metric update

        // Check if we're in cooldown
        if (_cooldownCounter > 0)
        {
            _cooldownCounter--;
            return;
        }

        bool isBetter;
        if (_mode == "min")
        {
            // For 'min' mode, lower metric is better (e.g., loss)
            isBetter = value < _bestMetric * (1 - _threshold);
        }
        else // 'max' mode
        {
            // For 'max' mode, higher metric is better (e.g., accuracy)
            isBetter = value > _bestMetric * (1 + _threshold);
        }

        if (isBetter)
        {
            _bestMetric = value;
            _wait = 0;
        }
        else
        {
            _wait++;
            if (_wait >= _patience)
            {
                // Reduce learning rate
                float newLR = _currentLearningRate * _factor;
                _currentLearningRate = Math.Max(newLR, _minLearningRate);
                _wait = 0;
                _cooldownCounter = _cooldown;
            }
        }
    }

    public override void Reset()
    {
        base.Reset();
        _wait = 0;
        _cooldownCounter = 0;
        _bestMetric = _mode == "min" ? float.MaxValue : float.MinValue;
        _currentLearningRate = float.MaxValue; // Will be set on first GetLearningRate call
    }

    public override StateDict GetState()
    {
        var state = new StateDict();
        state.Set("mode", _mode);
        state.Set("factor", _factor);
        state.Set("patience", _patience);
        state.Set("threshold", _threshold);
        state.Set("cooldown", _cooldown);
        state.Set("min_lr", _minLearningRate);
        state.Set("wait", _wait);
        state.Set("cooldown_counter", _cooldownCounter);
        state.Set("best_metric", _bestMetric);
        state.Set("current_lr", _currentLearningRate);
        state.Set("step_count", _stepCount);
        state.Set("epoch_count", _epochCount);
        return state;
    }

    public override void LoadState(StateDict state)
    {
        _stepCount = state.Get<int>("step_count", 0);
        _epochCount = state.Get<int>("epoch_count", 0);
        _wait = state.Get<int>("wait", 0);
        _cooldownCounter = state.Get<int>("cooldown_counter", 0);
        _bestMetric = state.Get<float>("best_metric", _mode == "min" ? float.MaxValue : float.MinValue);
        _currentLearningRate = state.Get<float>("current_lr", float.MaxValue);
    }
}
