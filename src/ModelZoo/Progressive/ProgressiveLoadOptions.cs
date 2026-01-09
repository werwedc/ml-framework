using RitterFramework.Core;

namespace MLFramework.ModelZoo.Progressive;

/// <summary>
/// Configuration options for progressive model loading.
/// </summary>
public class ProgressiveLoadOptions
{
    /// <summary>
    /// Gets or sets the loading strategy to use. Default is OnDemand.
    /// </summary>
    public LayerLoadingStrategy Strategy { get; set; } = LayerLoadingStrategy.OnDemand;

    /// <summary>
    /// Gets or sets the maximum number of layers to load concurrently. Default is 3.
    /// </summary>
    public int MaxConcurrentLoads { get; set; } = 3;

    /// <summary>
    /// Gets or sets the number of layers to prefetch ahead. Default is 1.
    /// </summary>
    public int PrefetchCount { get; set; } = 1;

    /// <summary>
    /// Gets or sets when to unload layers from memory. Default is Never.
    /// </summary>
    public UnloadStrategy UnloadStrategy { get; set; } = UnloadStrategy.Never;

    /// <summary>
    /// Gets or sets the maximum number of layers to keep in memory. -1 means unlimited. Default is -1.
    /// </summary>
    public int MaxLoadedLayers { get; set; } = -1;

    /// <summary>
    /// Gets or sets the memory threshold in bytes for triggering unloads. Default is 100MB.
    /// </summary>
    public long MemoryPressureThreshold { get; set; } = 100 * 1024 * 1024; // 100MB

    /// <summary>
    /// Creates a default instance of ProgressiveLoadOptions.
    /// </summary>
    public ProgressiveLoadOptions()
    {
    }

    /// <summary>
    /// Creates a ProgressiveLoadOptions instance with the specified strategy.
    /// </summary>
    /// <param name="strategy">The loading strategy to use.</param>
    public ProgressiveLoadOptions(LayerLoadingStrategy strategy)
    {
        Strategy = strategy;
    }
}
