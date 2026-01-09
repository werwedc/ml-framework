using MLFramework.Core;
using MLFramework.ModelVersioning;
using System.Collections.Concurrent;

namespace MLFramework.ModelZoo.Progressive;

/// <summary>
/// Manages progressive loading for large models.
/// </summary>
public class ProgressiveModelLoader
{
    private readonly ProgressiveLoadContext _context;
    private readonly ConcurrentDictionary<string, LazyParameter> _lazyParameters;
    private readonly object _prefetchLock = new();

    /// <summary>
    /// Gets the progressive load context.
    /// </summary>
    public ProgressiveLoadContext Context => _context;

    /// <summary>
    /// Gets the collection of lazy parameters.
    /// </summary>
    public IReadOnlyDictionary<string, LazyParameter> LazyParameters => _lazyParameters;

    /// <summary>
    /// Event fired when a layer finishes loading.
    /// </summary>
    public event EventHandler<LayerLoadedEventArgs>? OnLayerLoaded;

    /// <summary>
    /// Event fired when overall loading progress changes.
    /// </summary>
    public event EventHandler<ProgressChangedEventArgs>? OnProgressChanged;

    /// <summary>
    /// Event fired when all layers are fully loaded.
    /// </summary>
    public event EventHandler? OnFullyLoaded;

    /// <summary>
    /// Creates a new ProgressiveModelLoader.
    /// </summary>
    /// <param name="context">The progressive load context.</param>
    public ProgressiveModelLoader(ProgressiveLoadContext context)
    {
        _context = context ?? throw new ArgumentNullException(nameof(context));
        _lazyParameters = new ConcurrentDictionary<string, LazyParameter>();
    }

    /// <summary>
    /// Creates a model with progressive loading.
    /// </summary>
    /// <param name="modelName">The model name.</param>
    /// <param name="version">The model version (optional).</param>
    /// <param name="device">The target device (optional).</param>
    /// <param name="options">Loading options (optional).</param>
    /// <returns>A ProgressiveModelLoader instance.</returns>
    public static ProgressiveModelLoader LoadProgressive(
        string modelName,
        string? version = null,
        Device? device = null,
        ProgressiveLoadOptions? options = null)
    {
        // Create metadata (simplified - in practice, this would fetch from ModelZoo)
        var metadata = new ModelMetadata
        {
            ModelName = modelName,
            Architecture = "Custom"
        };

        // Create cache path
        string cachePath = GetCachePath(modelName, version);

        // Create device if not provided
        device ??= Device.CreateCpu();

        // Create options if not provided
        options ??= new ProgressiveLoadOptions();

        // Create layer load order
        var layerLoadOrder = new LayerLoadOrder(ModelArchitectureType.Custom);

        // Create context
        var context = new ProgressiveLoadContext(metadata, device, cachePath, options, layerLoadOrder);

        return new ProgressiveModelLoader(context);
    }

    /// <summary>
    /// Creates a model from metadata with progressive loading.
    /// </summary>
    /// <param name="metadata">The model metadata.</param>
    /// <param name="device">The target device (optional).</param>
    /// <param name="options">Loading options (optional).</param>
    /// <returns>A ProgressiveModelLoader instance.</returns>
    public static ProgressiveModelLoader LoadProgressive(
        ModelMetadata metadata,
        Device? device = null,
        ProgressiveLoadOptions? options = null)
    {
        if (metadata == null)
        {
            throw new ArgumentNullException(nameof(metadata));
        }

        string cachePath = GetCachePath(metadata.ModelName, null);
        device ??= Device.CreateCpu();
        options ??= new ProgressiveLoadOptions();

        var layerLoadOrder = DetermineLoadOrder(metadata);
        var context = new ProgressiveLoadContext(metadata, device, cachePath, options, layerLoadOrder);

        return new ProgressiveModelLoader(context);
    }

    /// <summary>
    /// Gets the current loading progress (0-1).
    /// </summary>
    /// <returns>The loading progress.</returns>
    public double GetLoadingProgress()
    {
        return _context.LoadingProgress;
    }

    /// <summary>
    /// Checks if all weights are loaded.
    /// </summary>
    /// <returns>True if fully loaded, false otherwise.</returns>
    public bool IsFullyLoaded()
    {
        return _context.IsFullyLoaded();
    }

    /// <summary>
    /// Prefetches a specific layer.
    /// </summary>
    /// <param name="layerName">The layer name to prefetch.</param>
    public void PrefetchLayer(string layerName)
    {
        if (_lazyParameters.TryGetValue(layerName, out var param))
        {
            param.Prefetch();
            TriggerLayerLoadedEvent(layerName);
        }
    }

    /// <summary>
    /// Prefetches multiple layers.
    /// </summary>
    /// <param name="layerNames">The layer names to prefetch.</param>
    public void PrefetchLayers(string[] layerNames)
    {
        if (layerNames == null)
        {
            throw new ArgumentNullException(nameof(layerNames));
        }

        foreach (var layerName in layerNames)
        {
            PrefetchLayer(layerName);
        }
    }

    /// <summary>
    /// Registers a lazy parameter for a layer.
    /// </summary>
    /// <param name="layerName">The layer name.</param>
    /// <param name="weightPath">The path to the weight file.</param>
    /// <param name="shape">The shape of the parameter.</param>
    /// <param name="requiresGrad">Whether gradients are required.</param>
    /// <returns>The created LazyParameter.</returns>
    public LazyParameter RegisterParameter(
        string layerName,
        string weightPath,
        int[] shape,
        bool requiresGrad = true)
    {
        var param = new LazyParameter(layerName, weightPath, shape, _context, requiresGrad);

        // Subscribe to layer loaded event
        param.OnLayerLoaded += (sender, args) =>
        {
            TriggerLayerLoadedEvent(layerName);
            TriggerProgressChangedEvent();
        };

        _lazyParameters[layerName] = param;

        return param;
    }

    /// <summary>
    /// Gets a lazy parameter by layer name.
    /// </summary>
    /// <param name="layerName">The layer name.</param>
    /// <returns>The LazyParameter, or null if not found.</returns>
    public LazyParameter? GetParameter(string layerName)
    {
        return _lazyParameters.TryGetValue(layerName, out var param) ? param : null;
    }

    /// <summary>
    /// Triggers prefetching for the next layers based on the strategy.
    /// </summary>
    /// <param name="currentLayer">The current layer being processed.</param>
    public void PrefetchNextLayers(string? currentLayer = null)
    {
        var nextLayers = _context.GetNextLayersToLoad(currentLayer);

        if (nextLayers.Count > 0)
        {
            PrefetchLayers(nextLayers.ToArray());
        }
    }

    /// <summary>
    /// Manages memory by checking for memory pressure and unloading layers if needed.
    /// </summary>
    public void ManageMemory()
    {
        if (_context.MemoryManager.HasMemoryPressure())
        {
            var layersToUnload = _context.MemoryManager.GetLayersToUnload();

            foreach (var layerName in layersToUnload)
            {
                if (_lazyParameters.TryGetValue(layerName, out var param))
                {
                    param.Unload();
                }
            }
        }
    }

    /// <summary>
    /// Waits for all layers to be loaded (useful for testing).
    /// </summary>
    /// <param name="timeoutMilliseconds">Timeout in milliseconds (default: 60000 = 1 minute).</param>
    /// <returns>True if all layers loaded successfully, false if timeout.</returns>
    public bool WaitForFullyLoaded(int timeoutMilliseconds = 60000)
    {
        var startTime = DateTime.UtcNow;

        while (!IsFullyLoaded() && (DateTime.UtcNow - startTime).TotalMilliseconds < timeoutMilliseconds)
        {
            Thread.Sleep(100);
        }

        return IsFullyLoaded();
    }

    private void TriggerLayerLoadedEvent(string layerName)
    {
        OnLayerLoaded?.Invoke(this, new LayerLoadedEventArgs(layerName));
    }

    private void TriggerProgressChangedEvent()
    {
        double progress = _context.LoadingProgress;
        OnProgressChanged?.Invoke(this, new ProgressChangedEventArgs(progress));

        if (progress >= 1.0)
        {
            OnFullyLoaded?.Invoke(this, EventArgs.Empty);
        }
    }

    private static string GetCachePath(string modelName, string? version)
    {
        string cacheDir = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "MLFramework",
            "ModelCache");

        Directory.CreateDirectory(cacheDir);

        string fileName = version != null ? $"{modelName}_{version}.bin" : $"{modelName}.bin";
        return Path.Combine(cacheDir, fileName);
    }

    private static LayerLoadOrder DetermineLoadOrder(ModelMetadata metadata)
    {
        // Simple heuristic based on architecture type
        if (metadata.Architecture.Contains("CNN", StringComparison.OrdinalIgnoreCase) ||
            metadata.Architecture.Contains("Conv", StringComparison.OrdinalIgnoreCase))
        {
            return LayerLoadOrder.ForCNN();
        }

        if (metadata.Architecture.Contains("Transformer", StringComparison.OrdinalIgnoreCase) ||
            metadata.Architecture.Contains("Attention", StringComparison.OrdinalIgnoreCase))
        {
            return LayerLoadOrder.ForTransformer();
        }

        if (metadata.Architecture.Contains("RNN", StringComparison.OrdinalIgnoreCase) ||
            metadata.Architecture.Contains("LSTM", StringComparison.OrdinalIgnoreCase) ||
            metadata.Architecture.Contains("GRU", StringComparison.OrdinalIgnoreCase))
        {
            return LayerLoadOrder.ForRNN();
        }

        return LayerLoadOrder.ForCNN(); // Default
    }
}

/// <summary>
/// Event arguments for layer loaded events.
/// </summary>
public class LayerLoadedEventArgs : EventArgs
{
    /// <summary>
    /// Gets the layer name that was loaded.
    /// </summary>
    public string LayerName { get; }

    /// <summary>
    /// Creates a new LayerLoadedEventArgs.
    /// </summary>
    /// <param name="layerName">The layer name.</param>
    public LayerLoadedEventArgs(string layerName)
    {
        LayerName = layerName;
    }
}

/// <summary>
/// Event arguments for progress changed events.
/// </summary>
public class ProgressChangedEventArgs : EventArgs
{
    /// <summary>
    /// Gets the current progress (0-1).
    /// </summary>
    public double Progress { get; }

    /// <summary>
    /// Creates a new ProgressChangedEventArgs.
    /// </summary>
    /// <param name="progress">The progress value.</param>
    public ProgressChangedEventArgs(double progress)
    {
        Progress = progress;
    }
}
