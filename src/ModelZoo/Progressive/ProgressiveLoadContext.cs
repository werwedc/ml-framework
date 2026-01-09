using MLFramework.Core;
using MLFramework.ModelVersioning;

namespace MLFramework.ModelZoo.Progressive;

/// <summary>
/// Context for progressive model loading.
/// </summary>
public class ProgressiveLoadContext
{
    /// <summary>
    /// Gets the metadata for the model.
    /// </summary>
    public ModelMetadata Metadata { get; }

    /// <summary>
    /// Gets the target device for loaded tensors.
    /// </summary>
    public Device Device { get; }

    /// <summary>
    /// Gets or sets the path to the cached model file.
    /// </summary>
    public string CachePath { get; set; }

    /// <summary>
    /// Gets the loading strategy being used.
    /// </summary>
    public LayerLoadingStrategy LoadingStrategy { get; }

    /// <summary>
    /// Gets the set of loaded layer names.
    /// </summary>
    public HashSet<string> LoadedLayers { get; }

    /// <summary>
    /// Gets the load order for layers.
    /// </summary>
    public LayerLoadOrder LayerLoadOrder { get; }

    /// <summary>
    /// Gets or sets the current loading progress (0-1).
    /// </summary>
    public double LoadingProgress { get; set; }

    /// <summary>
    /// Gets the total number of layers.
    /// </summary>
    public int TotalLayers { get; }

    /// <summary>
    /// Gets the loading options.
    /// </summary>
    public ProgressiveLoadOptions Options { get; }

    /// <summary>
    /// Gets the memory manager for this context.
    /// </summary>
    public MemoryManager MemoryManager { get; }

    /// <summary>
    /// Creates a new ProgressiveLoadContext.
    /// </summary>
    /// <param name="metadata">The model metadata.</param>
    /// <param name="device">The target device.</param>
    /// <param name="cachePath">Path to the cached model file.</param>
    /// <param name="options">Loading options.</param>
    /// <param name="layerLoadOrder">The load order for layers.</param>
    public ProgressiveLoadContext(
        ModelMetadata metadata,
        Device device,
        string cachePath,
        ProgressiveLoadOptions options,
        LayerLoadOrder layerLoadOrder)
    {
        Metadata = metadata ?? throw new ArgumentNullException(nameof(metadata));
        Device = device ?? throw new ArgumentNullException(nameof(device));
        CachePath = cachePath ?? throw new ArgumentNullException(nameof(cachePath));
        Options = options ?? throw new ArgumentNullException(nameof(options));
        LayerLoadOrder = layerLoadOrder ?? throw new ArgumentNullException(nameof(layerLoadOrder));
        LoadingStrategy = options.Strategy;
        LoadedLayers = new HashSet<string>();
        LoadingProgress = 0.0;
        TotalLayers = layerLoadOrder.OrderedLayers.Count;
        MemoryManager = new MemoryManager(options);
    }

    /// <summary>
    /// Creates a new ProgressiveLoadContext with default options.
    /// </summary>
    /// <param name="metadata">The model metadata.</param>
    /// <param name="device">The target device.</param>
    /// <param name="cachePath">Path to the cached model file.</param>
    public ProgressiveLoadContext(
        ModelMetadata metadata,
        Device device,
        string cachePath)
        : this(metadata, device, cachePath, new ProgressiveLoadOptions(), new LayerLoadOrder(ModelArchitectureType.Custom))
    {
    }

    /// <summary>
    /// Checks if a layer is loaded.
    /// </summary>
    /// <param name="layerName">The layer name.</param>
    /// <returns>True if the layer is loaded, false otherwise.</returns>
    public bool IsLayerLoaded(string layerName)
    {
        return LoadedLayers.Contains(layerName);
    }

    /// <summary>
    /// Marks a layer as loaded and updates progress.
    /// </summary>
    /// <param name="layerName">The layer name.</param>
    public void MarkLayerLoaded(string layerName)
    {
        LoadedLayers.Add(layerName);
        UpdateProgress();
    }

    /// <summary>
    /// Updates the loading progress based on loaded layers.
    /// </summary>
    private void UpdateProgress()
    {
        if (TotalLayers > 0)
        {
            LoadingProgress = (double)LoadedLayers.Count / TotalLayers;
        }
    }

    /// <summary>
    /// Checks if all layers are loaded.
    /// </summary>
    /// <returns>True if all layers are loaded, false otherwise.</returns>
    public bool IsFullyLoaded()
    {
        return LoadedLayers.Count >= TotalLayers;
    }

    /// <summary>
    /// Gets the list of layers that should be loaded next based on the strategy.
    /// </summary>
    /// <param name="currentLayer">The current layer being processed (optional).</param>
    /// <returns>The list of layer names to load next.</returns>
    public List<string> GetNextLayersToLoad(string? currentLayer = null)
    {
        var loadOrder = LayerLoadOrder.GetLoadOrder();
        var result = new List<string>();

        if (LoadingStrategy == LayerLoadingStrategy.OnDemand)
        {
            // Only load requested layers
            return result;
        }

        int currentIndex = currentLayer != null ? loadOrder.IndexOf(currentLayer) : -1;
        int prefetchCount = Options.PrefetchCount;
        int concurrentLoads = Options.MaxConcurrentLoads;

        if (LoadingStrategy == LayerLoadingStrategy.Sequential || LoadingStrategy == LayerLoadingStrategy.Prefetch)
        {
            // Load next layers sequentially
            for (int i = currentIndex + 1; i < loadOrder.Count && result.Count < prefetchCount; i++)
            {
                if (!LoadedLayers.Contains(loadOrder[i]))
                {
                    result.Add(loadOrder[i]);
                }
            }
        }
        else if (LoadingStrategy == LayerLoadingStrategy.Parallel)
        {
            // Load all unloaded layers up to the concurrent limit
            foreach (var layer in loadOrder)
            {
                if (!LoadedLayers.Contains(layer) && result.Count < concurrentLoads)
                {
                    result.Add(layer);
                }
            }
        }

        return result;
    }
}
