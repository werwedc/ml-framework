namespace MLFramework.Checkpointing.Strategies;

/// <summary>
/// Interface for checkpointing strategies
/// </summary>
public interface ICheckpointStrategy
{
    /// <summary>
    /// Determines whether to checkpoint a layer
    /// </summary>
    /// <param name="layerId">Unique identifier for the layer</param>
    /// <param name="activation">The activation tensor</param>
    /// <param name="layerIndex">Index of the layer in the network</param>
    /// <returns>True if should checkpoint, false otherwise</returns>
    bool ShouldCheckpoint(string layerId, Tensor activation, int layerIndex);

    /// <summary>
    /// Gets the strategy name
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Resets the strategy state
    /// </summary>
    void Reset();
}
