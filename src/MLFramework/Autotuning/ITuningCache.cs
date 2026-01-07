using MLFramework.Fusion;

namespace MLFramework.Autotuning;

/// <summary>
/// Interface for tuning cache
/// </summary>
public interface ITuningCache
{
    TuningCacheEntry? Get(MLFramework.Fusion.FusedOperation fusedOp, DeviceInfo device);
    void Put(MLFramework.Fusion.FusedOperation fusedOp, AutotuningResult result, DeviceInfo device);
    void Clear();
    int Count { get; }
}

/// <summary>
/// Entry in the tuning cache
/// </summary>
public record TuningCacheEntry
{
    public required MLFramework.Fusion.FusedOperation Operation { get; init; }
    public required AutotuningResult Result { get; init; }
    public required DeviceInfo Device { get; init; }
    public required DateTime CachedAt { get; init; }
    public required int HitCount { get; init; }
}

/// <summary>
/// In-memory implementation of tuning cache
/// </summary>
public class InMemoryTuningCache : ITuningCache
{
    private readonly Dictionary<string, TuningCacheEntry> _cache = new();
    private readonly object _lock = new();

    public int Count => _cache.Count;

    public TuningCacheEntry? Get(MLFramework.Fusion.FusedOperation fusedOp, DeviceInfo device)
    {
        var key = ComputeCacheKey(fusedOp, device);

        lock (_lock)
        {
            if (_cache.TryGetValue(key, out var entry))
            {
                // Update hit count
                _cache[key] = entry with { HitCount = entry.HitCount + 1 };
                return entry;
            }
        }

        return null;
    }

    public void Put(MLFramework.Fusion.FusedOperation fusedOp, AutotuningResult result, DeviceInfo device)
    {
        var key = ComputeCacheKey(fusedOp, device);

        var entry = new TuningCacheEntry
        {
            Operation = fusedOp,
            Result = result,
            Device = device,
            CachedAt = DateTime.UtcNow,
            HitCount = 0
        };

        lock (_lock)
        {
            _cache[key] = entry;
        }
    }

    public void Clear()
    {
        lock (_lock)
        {
            _cache.Clear();
        }
    }

    private string ComputeCacheKey(MLFramework.Fusion.FusedOperation fusedOp, DeviceInfo device)
    {
        // Key based on operation signature and device
        var opSignature = fusedOp.Pattern.Name;
        var deviceSignature = $"{device.Architecture}_{device.ComputeCapability}";

        return $"{opSignature}_{deviceSignature}";
    }
}
