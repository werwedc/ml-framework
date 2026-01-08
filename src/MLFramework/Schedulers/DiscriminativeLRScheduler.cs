namespace MLFramework.Schedulers;

/// <summary>
/// Applies different learning rates to different parameter groups.
/// Commonly used in transfer learning for fine-tuning.
/// </summary>
public sealed class DiscriminativeLRScheduler : BaseScheduler, IStepScheduler
{
    private readonly float _baseLearningRate;
    private readonly float[] _layerMultipliers;
    private readonly string[] _layerNames;

    public DiscriminativeLRScheduler(
        float baseLearningRate,
        float[] layerMultipliers,
        string[] layerNames = null)
    {
        if (baseLearningRate <= 0)
            throw new ArgumentException("baseLearningRate must be positive", nameof(baseLearningRate));
        if (layerMultipliers == null || layerMultipliers.Length == 0)
            throw new ArgumentException("layerMultipliers must not be empty", nameof(layerMultipliers));

        _baseLearningRate = baseLearningRate;
        _layerMultipliers = layerMultipliers;
        _layerNames = layerNames ?? Array.Empty<string>();
    }

    /// <summary>
    /// Gets the learning rate for a specific parameter group.
    /// </summary>
    /// <param name="groupIndex">Index of the parameter group.</param>
    /// <param name="layerName">Name of the layer (optional, for named lookup).</param>
    /// <returns>Learning rate for this parameter group.</returns>
    public float GetGroupLearningRate(int groupIndex, string layerName = null)
    {
        float multiplier;

        if (layerName != null && _layerNames.Contains(layerName))
        {
            // Find by name
            int nameIndex = Array.IndexOf(_layerNames, layerName);
            if (nameIndex >= 0 && nameIndex < _layerMultipliers.Length)
            {
                multiplier = _layerMultipliers[nameIndex];
            }
            else
            {
                multiplier = 1.0f;  // Default if name not found
            }
        }
        else
        {
            // Find by index
            if (groupIndex >= 0 && groupIndex < _layerMultipliers.Length)
            {
                multiplier = _layerMultipliers[groupIndex];
            }
            else
            {
                multiplier = _layerMultipliers[_layerMultipliers.Length - 1];  // Use last multiplier
            }
        }

        return _baseLearningRate * multiplier;
    }

    public override float GetLearningRate(int step, float baseLearningRate)
    {
        // Returns the base learning rate (default group)
        // Use GetGroupLearningRate for specific groups
        return _baseLearningRate;
    }

    public override StateDict GetState()
    {
        var state = new StateDict();
        state.Set("base_lr", _baseLearningRate);
        state.Set("layer_multipliers", _layerMultipliers);
        state.Set("layer_names", _layerNames);
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
