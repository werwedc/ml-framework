using Microsoft.Extensions.Logging;
using System.Collections.Concurrent;

namespace MLFramework.Memory;

/// <summary>
/// Handles memory pressure by evicting least-recently-used model versions.
/// </summary>
public class MemoryPressureHandler : IMemoryPressureHandler
{
    private readonly ConcurrentDictionary<string, ModelMemoryInfo> _loadedModels;
    private readonly object _evictionLock = new();
    private long _memoryThresholdBytes;
    private readonly ILogger<MemoryPressureHandler>? _logger;

    // Callbacks for dependencies
    private readonly Func<string, string, Task<bool>> _canEvictModelCallback;
    private readonly Func<string, string, Task> _unloadModelCallback;

    public MemoryPressureHandler(
        Func<string, string, Task<bool>>? canEvictModelCallback = null,
        Func<string, string, Task>? unloadModelCallback = null,
        ILogger<MemoryPressureHandler>? logger = null,
        long memoryThresholdBytes = 1_073_741_824) // Default 1GB
    {
        _loadedModels = new ConcurrentDictionary<string, ModelMemoryInfo>();
        _memoryThresholdBytes = memoryThresholdBytes;
        _logger = logger;
        _canEvictModelCallback = canEvictModelCallback ?? ((_, _) => Task.FromResult(true));
        _unloadModelCallback = unloadModelCallback ?? ((_, _) => Task.CompletedTask);
    }

    public void TrackModelLoad(string modelName, string version, long memoryBytes)
    {
        if (memoryBytes <= 0)
        {
            throw new ArgumentException("Memory bytes must be positive", nameof(memoryBytes));
        }

        var key = GetKey(modelName, version);
        var info = new ModelMemoryInfo(modelName, version, memoryBytes);

        if (_loadedModels.TryAdd(key, info))
        {
            _logger?.LogDebug("Tracked model load: {ModelName} {Version}, Size: {MemoryBytes:N0} bytes",
                modelName, version, memoryBytes);
        }
        else
        {
            _logger?.LogWarning("Model {ModelName} {Version} is already being tracked", modelName, version);
        }
    }

    public void TrackModelAccess(string modelName, string version)
    {
        var key = GetKey(modelName, version);

        if (_loadedModels.TryGetValue(key, out var info))
        {
            info.RecordAccess();
            _logger?.LogDebug("Tracked model access: {ModelName} {Version}, AccessCount: {AccessCount}",
                modelName, version, info.AccessCount);
        }
    }

    public void PinModel(string modelName, string version)
    {
        var key = GetKey(modelName, version);

        if (_loadedModels.TryGetValue(key, out var info))
        {
            info.IsPinned = true;
            _logger?.LogDebug("Pinned model: {ModelName} {Version}", modelName, version);
        }
    }

    public void UnpinModel(string modelName, string version)
    {
        var key = GetKey(modelName, version);

        if (_loadedModels.TryGetValue(key, out var info))
        {
            info.IsPinned = false;
            _logger?.LogDebug("Unpinned model: {ModelName} {Version}", modelName, version);
        }
    }

    public long GetTotalMemoryUsage()
    {
        return _loadedModels.Values.Sum(m => m.MemoryBytes);
    }

    public void SetMemoryThreshold(long thresholdBytes)
    {
        if (thresholdBytes <= 0)
        {
            throw new ArgumentException("Threshold must be positive", nameof(thresholdBytes));
        }

        _memoryThresholdBytes = thresholdBytes;
        _logger?.LogDebug("Memory threshold set to {Threshold:N0} bytes", thresholdBytes);
    }

    public async Task<EvictionResult> EvictIfNeededAsync(long requiredBytes = 0)
    {
        var totalUsage = GetTotalMemoryUsage();
        var targetThreshold = _memoryThresholdBytes + requiredBytes;

        if (totalUsage <= targetThreshold)
        {
            return new EvictionResult(new List<string>(), 0, true);
        }

        var stopwatch = System.Diagnostics.Stopwatch.StartNew();

        var evictedVersions = new List<string>();
        long bytesFreed = 0;

        lock (_evictionLock)
        {
            // Sort models by weight (LRU+LFU hybrid)
            var candidates = _loadedModels.Values
                .Where(m => !m.IsPinned)
                .OrderBy(m => m.Weight)
                .ToList();

            foreach (var model in candidates)
            {
                if (totalUsage - bytesFreed <= targetThreshold)
                {
                    break; // We've freed enough space
                }

                var key = GetKey(model.ModelName, model.Version);

                // Check if model can be evicted (no active references)
                var canEvict = _canEvictModelCallback(model.ModelName, model.Version).GetAwaiter().GetResult();
                if (!canEvict)
                {
                    _logger?.LogDebug("Skipping eviction of {ModelName} {Version} - has active references",
                        model.ModelName, model.Version);
                    continue;
                }

                // Evict the model
                if (_loadedModels.TryRemove(key, out _))
                {
                    _unloadModelCallback(model.ModelName, model.Version).GetAwaiter().GetResult();

                    evictedVersions.Add(key);
                    bytesFreed += model.MemoryBytes;

                    _logger?.LogInformation("Evicted model: {ModelName} {Version}, Freed: {Bytes:N0} bytes, " +
                        "AccessCount: {AccessCount}, LastAccess: {LastAccess}",
                        model.ModelName, model.Version, model.MemoryBytes, model.AccessCount, model.LastAccessTime);
                }
            }
        }

        stopwatch.Stop();

        var sufficient = (totalUsage - bytesFreed) <= targetThreshold;
        _logger?.LogInformation("Eviction completed: {EvictedCount} models evicted, {BytesFreed:N0} bytes freed, " +
            "Time: {ElapsedMs:F2}ms, Sufficient: {Sufficient}",
            evictedVersions.Count, bytesFreed, stopwatch.Elapsed.TotalMilliseconds, sufficient);

        return new EvictionResult(evictedVersions, bytesFreed, sufficient);
    }

    public IEnumerable<ModelMemoryInfo> GetLoadedModelsInfo()
    {
        return _loadedModels.Values.ToList();
    }

    public void UntrackModel(string modelName, string version)
    {
        var key = GetKey(modelName, version);

        if (_loadedModels.TryRemove(key, out var info))
        {
            _logger?.LogDebug("Untracked model: {ModelName} {Version}, Size: {MemoryBytes:N0} bytes",
                modelName, version, info.MemoryBytes);
        }
    }

    private static string GetKey(string modelName, string version)
    {
        return $"{modelName}:{version}";
    }
}
