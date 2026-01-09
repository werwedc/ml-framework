using System.Collections.Concurrent;

namespace MLFramework.ModelZoo.Progressive;

/// <summary>
/// Manages memory during progressive loading, including tracking usage and unloading layers.
/// </summary>
public class MemoryManager
{
    private readonly ProgressiveLoadOptions _options;
    private readonly ConcurrentDictionary<string, LayerMemoryInfo> _loadedLayers;
    private readonly LinkedList<string> _lruList;
    private readonly object _lock = new();
    private long _totalLoadedBytes;

    /// <summary>
    /// Gets the total number of bytes currently loaded in memory.
    /// </summary>
    public long TotalLoadedBytes => _totalLoadedBytes;

    /// <summary>
    /// Gets the number of layers currently loaded.
    /// </summary>
    public int LoadedLayerCount => _loadedLayers.Count;

    /// <summary>
    /// Creates a new MemoryManager with the specified options.
    /// </summary>
    /// <param name="options">The progressive load options.</param>
    public MemoryManager(ProgressiveLoadOptions options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _loadedLayers = new ConcurrentDictionary<string, LayerMemoryInfo>();
        _lruList = new LinkedList<string>();
        _totalLoadedBytes = 0;
    }

    /// <summary>
    /// Registers a layer as loaded with its memory size.
    /// </summary>
    /// <param name="layerName">The layer name.</param>
    /// <param name="sizeInBytes">The size of the layer in bytes.</param>
    public void RegisterLayer(string layerName, long sizeInBytes)
    {
        var info = new LayerMemoryInfo
        {
            LayerName = layerName,
            SizeInBytes = sizeInBytes,
            LastAccessTime = DateTime.UtcNow,
            LoadTime = DateTime.UtcNow
        };

        _loadedLayers[layerName] = info;
        Interlocked.Add(ref _totalLoadedBytes, sizeInBytes);

        lock (_lock)
        {
            // Add to LRU list (at front for most recently used)
            _lruList.AddFirst(layerName);
        }
    }

    /// <summary>
    /// Records that a layer was accessed.
    /// </summary>
    /// <param name="layerName">The layer name.</param>
    public void RecordAccess(string layerName)
    {
        if (_loadedLayers.TryGetValue(layerName, out var info))
        {
            info.LastAccessTime = DateTime.UtcNow;

            lock (_lock)
            {
                // Move to front of LRU list
                _lruList.Remove(layerName);
                _lruList.AddFirst(layerName);
            }
        }
    }

    /// <summary>
    /// Unregisters a layer and frees its memory.
    /// </summary>
    /// <param name="layerName">The layer name.</param>
    /// <returns>The size in bytes of the unloaded layer, or 0 if not found.</returns>
    public long UnregisterLayer(string layerName)
    {
        if (_loadedLayers.TryRemove(layerName, out var info))
        {
            Interlocked.Add(ref _totalLoadedBytes, -info.SizeInBytes);

            lock (_lock)
            {
                _lruList.Remove(layerName);
            }

            return info.SizeInBytes;
        }

        return 0;
    }

    /// <summary>
    /// Checks if memory pressure exists and needs unloading.
    /// </summary>
    /// <returns>True if memory pressure exists, false otherwise.</returns>
    public bool HasMemoryPressure()
    {
        if (_options.UnloadStrategy == UnloadStrategy.Never)
        {
            return false;
        }

        if (_options.UnloadStrategy == UnloadStrategy.MemoryPressure)
        {
            return _totalLoadedBytes > _options.MemoryPressureThreshold;
        }

        if (_options.UnloadStrategy == UnloadStrategy.LRU)
        {
            return _options.MaxLoadedLayers > 0 && _loadedLayers.Count >= _options.MaxLoadedLayers;
        }

        return false;
    }

    /// <summary>
    /// Gets layers that should be unloaded based on the strategy.
    /// </summary>
    /// <param name="count">Maximum number of layers to unload.</param>
    /// <returns>List of layer names to unload.</returns>
    public List<string> GetLayersToUnload(int count = 1)
    {
        var layersToUnload = new List<string>();

        if (_options.UnloadStrategy == UnloadStrategy.Never)
        {
            return layersToUnload;
        }

        lock (_lock)
        {
            int toUnload = Math.Min(count, _lruList.Count);

            if (_options.UnloadStrategy == UnloadStrategy.LRU)
            {
                // Unload least recently used (from the end of the list)
                var node = _lruList.Last;
                for (int i = 0; i < toUnload && node != null; i++)
                {
                    layersToUnload.Add(node.Value);
                    node = node.Previous;
                }
            }
            else if (_options.UnloadStrategy == UnloadStrategy.MemoryPressure)
            {
                // Unload oldest layers until memory pressure is relieved
                long bytesToFree = _totalLoadedBytes - _options.MemoryPressureThreshold;
                var node = _lruList.Last;
                long freedBytes = 0;

                while (node != null && layersToUnload.Count < toUnload && freedBytes < bytesToFree)
                {
                    if (_loadedLayers.TryGetValue(node.Value, out var info))
                    {
                        layersToUnload.Add(node.Value);
                        freedBytes += info.SizeInBytes;
                    }
                    node = node.Previous;
                }
            }
        }

        return layersToUnload;
    }

    /// <summary>
    /// Gets the memory usage information for all loaded layers.
    /// </summary>
    /// <returns>Dictionary mapping layer names to their memory info.</returns>
    public Dictionary<string, LayerMemoryInfo> GetMemoryInfo()
    {
        return new Dictionary<string, LayerMemoryInfo>(_loadedLayers);
    }

    /// <summary>
    /// Clears all loaded layers.
    /// </summary>
    public void Clear()
    {
        _loadedLayers.Clear();
        lock (_lock)
        {
            _lruList.Clear();
        }
        _totalLoadedBytes = 0;
    }
}

/// <summary>
/// Information about a layer's memory usage.
/// </summary>
public class LayerMemoryInfo
{
    /// <summary>
    /// Gets or sets the layer name.
    /// </summary>
    public string LayerName { get; set; }

    /// <summary>
    /// Gets or sets the size in bytes.
    /// </summary>
    public long SizeInBytes { get; set; }

    /// <summary>
    /// Gets or sets the time when the layer was loaded.
    /// </summary>
    public DateTime LoadTime { get; set; }

    /// <summary>
    /// Gets or sets the time when the layer was last accessed.
    /// </summary>
    public DateTime LastAccessTime { get; set; }
}
