namespace MLFramework.Schedulers;

/// <summary>
/// Applies layer-wise learning rate decay.
/// Later layers get higher learning rates, earlier layers get lower.
/// </summary>
public sealed class LayerWiseLRDecayScheduler : BaseScheduler, IStepScheduler
{
    private readonly float _decayFactor;
    private readonly string[] _excludedLayers;

    public LayerWiseLRDecayScheduler(
        float decayFactor = 0.8f,
        string[] excludedLayers = null)
    {
        if (decayFactor <= 0 || decayFactor >= 1)
            throw new ArgumentException("decayFactor must be in (0, 1)", nameof(decayFactor));

        _decayFactor = decayFactor;
        _excludedLayers = excludedLayers ?? Array.Empty<string>();
    }

    /// <summary>
    /// Gets the learning rate multiplier for a specific layer.
    /// </summary>
    /// <param name="layerIndex">Index of the layer (0 = input layer).</param>
    /// <param name="totalLayers">Total number of layers.</param>
    /// <param name="layerName">Name of the layer (optional).</param>
    /// <returns>Learning rate multiplier for this layer.</returns>
    public float GetLayerMultiplier(int layerIndex, int totalLayers, string layerName = null)
    {
        // Check if layer is excluded
        if (layerName != null && _excludedLayers.Contains(layerName))
        {
            return 1.0f;  // No decay for excluded layers
        }

        // Calculate decay based on layer position
        // Layer 0 (earliest) has lowest multiplier
        // Layer N-1 (latest) has multiplier = 1.0
        int positionFromEnd = totalLayers - 1 - layerIndex;
        float multiplier = (float)Math.Pow(_decayFactor, positionFromEnd);

        return multiplier;
    }

    public override float GetLearningRate(int step, float baseLearningRate)
    {
        // This is a special scheduler - it doesn't change over time
        // But must implement the interface
        // Actual layer-specific LR is obtained via GetLayerMultiplier
        return baseLearningRate;
    }

    public override StateDict GetState()
    {
        var state = new StateDict();
        state.Set("decay_factor", _decayFactor);
        state.Set("excluded_layers", _excludedLayers);
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
