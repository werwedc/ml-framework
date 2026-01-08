using System.Collections.Concurrent;

namespace MLFramework.Compilation;

/// <summary>
/// LRU (Least Recently Used) kernel cache implementation
/// </summary>
public class LRUKernelCache<TKernel> : IKernelCache<TKernel>
{
    private readonly ConcurrentDictionary<ShapeSignature, KernelCacheEntry> _cache;
    private readonly int _maxSize;
    private readonly object _lockObject = new object();
    private long _totalHits;
    private long _totalMisses;

    /// <summary>
    /// Gets the maximum number of kernels to cache
    /// </summary>
    public int MaxSize => _maxSize;

    /// <summary>
    /// Gets the current size of the cache
    /// </summary>
    public int CurrentSize => _cache.Count;

    /// <summary>
    /// Creates a new LRU kernel cache
    /// </summary>
    /// <param name="maxSize">Maximum number of kernels to cache</param>
    public LRUKernelCache(int maxSize = 100)
    {
        _maxSize = maxSize;
        _cache = new ConcurrentDictionary<ShapeSignature, KernelCacheEntry>();
        _totalHits = 0;
        _totalMisses = 0;
    }

    /// <summary>
    /// Gets a kernel from the cache
    /// </summary>
    public TKernel? Get(ShapeSignature sig)
    {
        if (_cache.TryGetValue(sig, out var entry))
        {
            Interlocked.Increment(ref _totalHits);
            entry.UpdateAccessTime();
            entry.IncrementUseCount();
            return (TKernel)entry.CompiledKernel;
        }

        Interlocked.Increment(ref _totalMisses);
        return default;
    }

    /// <summary>
    /// Adds a kernel to the cache
    /// </summary>
    public void Set(ShapeSignature sig, TKernel kernel)
    {
        var entry = KernelCacheEntry.Create(sig, kernel!, 0);

        // If we're at capacity, evict least recently used
        if (_cache.Count >= _maxSize && !_cache.ContainsKey(sig))
        {
            EvictLeastRecentlyUsed();
        }

        _cache.AddOrUpdate(sig, entry, (_, existing) =>
        {
            existing.UpdateAccessTime();
            return existing;
        });
    }

    /// <summary>
    /// Checks if the cache contains a kernel for the given signature
    /// </summary>
    public bool Contains(ShapeSignature sig)
    {
        return _cache.ContainsKey(sig);
    }

    /// <summary>
    /// Removes a kernel from the cache
    /// </summary>
    public void Remove(ShapeSignature sig)
    {
        _cache.TryRemove(sig, out _);
    }

    /// <summary>
    /// Clears the cache
    /// </summary>
    public void Clear()
    {
        _cache.Clear();
        Interlocked.Exchange(ref _totalHits, 0);
        Interlocked.Exchange(ref _totalMisses, 0);
    }

    /// <summary>
    /// Gets cache statistics
    /// </summary>
    public CacheStats GetStats()
    {
        return new CacheStats
        {
            TotalKernels = _cache.Count,
            TotalHits = Interlocked.Read(ref _totalHits),
            TotalMisses = Interlocked.Read(ref _totalMisses)
        };
    }

    /// <summary>
    /// Evicts the least recently used kernel from the cache
    /// </summary>
    private void EvictLeastRecentlyUsed()
    {
        lock (_lockObject)
        {
            if (_cache.Count >= _maxSize)
            {
                var oldestEntry = _cache
                    .OrderBy(kvp => kvp.Value.LastUsed)
                    .FirstOrDefault();

                if (!oldestEntry.Equals(default(KeyValuePair<ShapeSignature, KernelCacheEntry>)))
                {
                    _cache.TryRemove(oldestEntry.Key, out _);
                }
            }
        }
    }

    /// <summary>
    /// Gets eviction candidates for cleanup
    /// </summary>
    public List<ShapeSignature> GetEvictionCandidates(int count)
    {
        lock (_lockObject)
        {
            return _cache
                .OrderBy(kvp => kvp.Value.LastUsed)
                .Take(count)
                .Select(kvp => kvp.Key)
                .ToList();
        }
    }
}
