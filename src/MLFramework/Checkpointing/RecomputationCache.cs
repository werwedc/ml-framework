namespace MLFramework.Checkpointing;

/// <summary>
/// Caches recomputed activations to avoid redundant computation
/// </summary>
public class RecomputationCache : IDisposable
{
    private class CacheEntry
    {
        public Tensor Tensor { get; set; } = null!;
        public long SizeBytes { get; set; }
        public DateTime LastAccessedAt { get; set; }
        public int AccessCount { get; set; }
    }

    private readonly Dictionary<string, CacheEntry> _cache;
    private readonly long _maxSizeBytes;
    private readonly object _lock = new object();
    private bool _disposed;
    private int _cacheHits;
    private int _cacheMisses;
    private long _currentSizeBytes;

    /// <summary>
    /// Initializes a new instance of RecomputationCache
    /// </summary>
    /// <param name="maxSizeBytes">Maximum cache size in bytes</param>
    public RecomputationCache(long maxSizeBytes)
    {
        if (maxSizeBytes <= 0)
            throw new ArgumentException("Max size must be greater than 0", nameof(maxSizeBytes));

        _cache = new Dictionary<string, CacheEntry>();
        _maxSizeBytes = maxSizeBytes;
        _disposed = false;
        _cacheHits = 0;
        _cacheMisses = 0;
        _currentSizeBytes = 0;
    }

    /// <summary>
    /// Gets a cached activation or null if not cached
    /// </summary>
    public Tensor? Get(string layerId)
    {
        if (string.IsNullOrWhiteSpace(layerId))
            throw new ArgumentException("Layer ID cannot be null or whitespace");

        lock (_lock)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(RecomputationCache));

            if (_cache.TryGetValue(layerId, out var entry))
            {
                entry.LastAccessedAt = DateTime.UtcNow;
                entry.AccessCount++;
                _cacheHits++;
                return entry.Tensor;
            }

            _cacheMisses++;
            return null;
        }
    }

    /// <summary>
    /// Adds an activation to the cache
    /// </summary>
    /// <param name="layerId">Unique identifier for the layer</param>
    /// <param name="activation">Activation tensor to cache</param>
    public void Add(string layerId, Tensor activation)
    {
        if (string.IsNullOrWhiteSpace(layerId))
            throw new ArgumentException("Layer ID cannot be null or whitespace");
        if (activation == null)
            throw new ArgumentNullException(nameof(activation));

        lock (_lock)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(RecomputationCache));

            // Calculate tensor size
            var sizeBytes = activation.ElementCount * activation.DataTypeSize;

            // Remove existing entry if present
            if (_cache.TryGetValue(layerId, out var existingEntry))
            {
                _currentSizeBytes -= existingEntry.SizeBytes;
                _cache.Remove(layerId);
            }

            // Evict items if necessary (LRU policy)
            while (_currentSizeBytes + sizeBytes > _maxSizeBytes && _cache.Count > 0)
            {
                EvictLRU();
            }

            // If still not enough space, don't cache
            if (sizeBytes > _maxSizeBytes)
                return;

            // Add new entry
            _cache[layerId] = new CacheEntry
            {
                Tensor = activation,
                SizeBytes = sizeBytes,
                LastAccessedAt = DateTime.UtcNow,
                AccessCount = 0
            };

            _currentSizeBytes += sizeBytes;
        }
    }

    /// <summary>
    /// Checks if an activation is cached
    /// </summary>
    public bool Contains(string layerId)
    {
        if (string.IsNullOrWhiteSpace(layerId))
            throw new ArgumentException("Layer ID cannot be null or whitespace");

        lock (_lock)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(RecomputationCache));

            return _cache.ContainsKey(layerId);
        }
    }

    /// <summary>
    /// Clears the cache
    /// </summary>
    public void Clear()
    {
        lock (_lock)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(RecomputationCache));

            _cache.Clear();
            _currentSizeBytes = 0;
            _cacheHits = 0;
            _cacheMisses = 0;
        }
    }

    /// <summary>
    /// Gets cache statistics
    /// </summary>
    public CacheStats GetStats()
    {
        lock (_lock)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(RecomputationCache));

            return new CacheStats
            {
                CachedItemsCount = _cache.Count,
                CurrentSizeBytes = _currentSizeBytes,
                MaxSizeBytes = _maxSizeBytes,
                CacheHits = _cacheHits,
                CacheMisses = _cacheMisses
            };
        }
    }

    private void EvictLRU()
    {
        if (_cache.Count == 0)
            return;

        // Find least recently used entry
        var lruKey = _cache.OrderBy(kvp => kvp.Value.LastAccessedAt).First().Key;
        var entry = _cache[lruKey];

        _currentSizeBytes -= entry.SizeBytes;
        _cache.Remove(lruKey);
    }

    /// <summary>
    /// Disposes the cache and releases resources
    /// </summary>
    public void Dispose()
    {
        if (!_disposed)
        {
            lock (_lock)
            {
                _cache.Clear();
                _currentSizeBytes = 0;
            }
            _disposed = true;
        }
    }
}
