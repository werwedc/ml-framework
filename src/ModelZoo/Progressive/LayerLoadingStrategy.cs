namespace MLFramework.ModelZoo.Progressive;

/// <summary>
/// Defines the strategy for loading model layers progressively.
/// </summary>
public enum LayerLoadingStrategy
{
    /// <summary>
    /// Load a layer when it is first accessed during a forward pass.
    /// </summary>
    OnDemand,

    /// <summary>
    /// Load layers sequentially during forward pass execution.
    /// </summary>
    Sequential,

    /// <summary>
    /// Load multiple layers concurrently for maximum throughput.
    /// </summary>
    Parallel,

    /// <summary>
    /// Load next layers while processing current layer to reduce latency.
    /// </summary>
    Prefetch
}
