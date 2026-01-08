namespace MLFramework.Checkpointing.Strategies;

/// <summary>
/// Checkpoints only specific layers defined by the user
/// </summary>
public class SelectiveCheckpointStrategy : ICheckpointStrategy
{
    private readonly HashSet<string> _checkpointLayers;
    private readonly HashSet<string> _excludeLayers;

    /// <summary>
    /// Initializes a new instance of SelectiveCheckpointStrategy
    /// </summary>
    /// <param name="checkpointLayers">List of layer IDs to checkpoint</param>
    /// <param name="excludeLayers">List of layer IDs to exclude from checkpointing</param>
    public SelectiveCheckpointStrategy(
        IEnumerable<string>? checkpointLayers = null,
        IEnumerable<string>? excludeLayers = null)
    {
        _checkpointLayers = checkpointLayers?.ToHashSet() ?? new HashSet<string>();
        _excludeLayers = excludeLayers?.ToHashSet() ?? new HashSet<string>();

        // Check for overlaps
        var overlap = _checkpointLayers.Intersect(_excludeLayers).ToList();
        if (overlap.Count > 0)
        {
            throw new ArgumentException(
                $"Layers cannot be both checkpointed and excluded: {string.Join(", ", overlap)}");
        }
    }

    /// <summary>
    /// Gets the strategy name
    /// </summary>
    public string Name => "Selective";

    /// <summary>
    /// Determines whether to checkpoint a layer
    /// </summary>
    public bool ShouldCheckpoint(string layerId, Tensor activation, int layerIndex)
    {
        // If in exclude list, don't checkpoint
        if (_excludeLayers.Contains(layerId))
        {
            return false;
        }

        // If in checkpoint list, do checkpoint
        if (_checkpointLayers.Contains(layerId))
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
        // No state to reset for this strategy
    }
}
