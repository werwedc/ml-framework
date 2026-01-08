namespace MLFramework.Checkpointing.Strategies;

/// <summary>
/// Uses heuristics to determine optimal checkpointing strategy
/// </summary>
public class SmartCheckpointStrategy : ICheckpointStrategy
{
    private readonly List<LayerInfo> _layerInfo = new List<LayerInfo>();
    private readonly Dictionary<string, long> _layerActivationSizes = new Dictionary<string, long>();
    private readonly Dictionary<string, int> _layerAccessCounts = new Dictionary<string, int>();
    private readonly HashSet<string> _excludeLayers;
    private bool _initialized;

    /// <summary>
    /// Initializes a new instance of SmartCheckpointStrategy
    /// </summary>
    /// <param name="excludeLayers">List of layer IDs to exclude</param>
    public SmartCheckpointStrategy(IEnumerable<string>? excludeLayers = null)
    {
        _excludeLayers = excludeLayers?.ToHashSet() ?? new HashSet<string>();
        _initialized = false;
    }

    /// <summary>
    /// Gets the strategy name
    /// </summary>
    public string Name => "Smart";

    /// <summary>
    /// Determines whether to checkpoint a layer
    /// </summary>
    public bool ShouldCheckpoint(string layerId, Tensor activation, int layerIndex)
    {
        // Update layer statistics
        UpdateLayerStatistics(layerId, activation);

        // If not initialized, collect data first
        if (!_initialized)
        {
            // Collect data for first pass, don't checkpoint yet
            if (layerIndex > 10) // After collecting data from first 10 layers
            {
                _initialized = true;
            }
            return false;
        }

        // If in exclude list, don't checkpoint
        if (_excludeLayers.Contains(layerId))
        {
            return false;
        }

        // Use heuristics to determine if should checkpoint
        return ShouldCheckpointHeuristic(layerId, layerIndex);
    }

    private void UpdateLayerStatistics(string layerId, Tensor activation)
    {
        var activationSize = activation.ElementCount * activation.DataTypeSize;

        if (!_layerActivationSizes.ContainsKey(layerId))
        {
            _layerActivationSizes[layerId] = activationSize;
            _layerAccessCounts[layerId] = 0;
            _layerInfo.Add(new LayerInfo
            {
                LayerId = layerId,
                ActivationSize = activationSize,
                AccessCount = 0
            });
        }

        _layerAccessCounts[layerId]++;
    }

    private bool ShouldCheckpointHeuristic(string layerId, int layerIndex)
    {
        if (!_layerActivationSizes.TryGetValue(layerId, out var activationSize))
        {
            return false;
        }

        var accessCount = _layerAccessCounts[layerId];
        var avgSize = _layerInfo.Average(l => l.ActivationSize);

        // Heuristic 1: Checkpoint large activations
        if (activationSize > avgSize * 1.5)
        {
            return true;
        }

        // Heuristic 2: Checkpoint infrequently accessed layers
        if (accessCount < 2)
        {
            return true;
        }

        // Heuristic 3: Checkpoint layers later in the network
        if (layerIndex > _layerInfo.Count * 0.7)
        {
            return true;
        }

        // Default: don't checkpoint
        return false;
    }

    /// <summary>
    /// Resets the strategy state
    /// </summary>
    public void Reset()
    {
        _layerInfo.Clear();
        _layerActivationSizes.Clear();
        _layerAccessCounts.Clear();
        _initialized = false;
    }

    private class LayerInfo
    {
        public string LayerId { get; set; } = string.Empty;
        public long ActivationSize { get; set; }
        public int AccessCount { get; set; }
    }
}
