namespace MLFramework.Checkpointing.Strategies;

/// <summary>
/// Checkpoints layers based on activation size
/// </summary>
public class SizeBasedCheckpointStrategy : ICheckpointStrategy
{
    private readonly long _minActivationSizeBytes;
    private readonly HashSet<string> _excludeLayers;
    private int _checkpointedCount;

    /// <summary>
    /// Initializes a new instance of SizeBasedCheckpointStrategy
    /// </summary>
    /// <param name="minActivationSizeBytes">Minimum activation size to trigger checkpointing</param>
    /// <param name="excludeLayers">List of layer IDs to exclude</param>
    public SizeBasedCheckpointStrategy(
        long minActivationSizeBytes = 1024 * 1024, // 1MB default
        IEnumerable<string>? excludeLayers = null)
    {
        if (minActivationSizeBytes <= 0)
            throw new ArgumentException("MinActivationSizeBytes must be greater than 0");

        _minActivationSizeBytes = minActivationSizeBytes;
        _excludeLayers = excludeLayers?.ToHashSet() ?? new HashSet<string>();
        _checkpointedCount = 0;
    }

    /// <summary>
    /// Gets the strategy name
    /// </summary>
    public string Name => $"SizeBased({FormatBytes(_minActivationSizeBytes)})";

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

        // Calculate activation size
        var activationSize = CalculateActivationSize(activation);

        // Checkpoint if size exceeds threshold
        var shouldCheckpoint = activationSize >= _minActivationSizeBytes;

        if (shouldCheckpoint)
        {
            _checkpointedCount++;
        }

        return shouldCheckpoint;
    }

    private long CalculateActivationSize(Tensor activation)
    {
        return activation.ElementCount * activation.DataTypeSize;
    }

    private string FormatBytes(long bytes)
    {
        if (bytes < 1024) return $"{bytes}B";
        if (bytes < 1024 * 1024) return $"{bytes / 1024}KB";
        if (bytes < 1024 * 1024 * 1024) return $"{bytes / (1024 * 1024)}MB";
        return $"{bytes / (1024 * 1024 * 1024)}GB";
    }

    /// <summary>
    /// Resets the strategy state
    /// </summary>
    public void Reset()
    {
        _checkpointedCount = 0;
    }
}
