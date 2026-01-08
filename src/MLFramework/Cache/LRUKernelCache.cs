using System;
using System.Collections.Generic;
using System.Linq;

namespace MLFramework.Cache
{
    /// <summary>
    /// A thread-safe kernel cache with LRU (Least Recently Used) eviction policy.
    /// </summary>
    /// <typeparam name="TKernel">The type of the compiled kernel.</typeparam>
    public class LRUKernelCache<TKernel> : IKernelCache<TKernel>
    {
        private readonly Dictionary<ShapeSignature, KernelCacheEntry<TKernel>> _cache;
        private readonly object _lock;
        private readonly int _maxSize;
        private readonly CacheStats _stats;

        /// <summary>
        /// Gets the maximum number of kernels that can be cached.
        /// </summary>
        public int MaxSize => _maxSize;

        /// <summary>
        /// Gets the current number of kernels in the cache.
        /// </summary>
        public int CurrentSize
        {
            get
            {
                lock (_lock)
                {
                    return _cache.Count;
                }
            }
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="LRUKernelCache{TKernel}"/> class.
        /// </summary>
        /// <param name="maxSize">Maximum number of kernels to cache. Must be positive.</param>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when maxSize is less than or equal to zero.</exception>
        public LRUKernelCache(int maxSize = 100)
        {
            if (maxSize <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(maxSize), "Max size must be positive.");
            }

            _maxSize = maxSize;
            _cache = new Dictionary<ShapeSignature, KernelCacheEntry<TKernel>>();
            _lock = new object();
            _stats = new CacheStats();
        }

        /// <summary>
        /// Gets a compiled kernel from the cache, or null if not found.
        /// Updates the access time and use count if found.
        /// </summary>
        /// <param name="sig">The shape signature of the kernel.</param>
        /// <returns>The compiled kernel if found; otherwise, null.</returns>
        public TKernel? Get(ShapeSignature sig)
        {
            lock (_lock)
            {
                if (_cache.TryGetValue(sig, out var entry))
                {
                    // Update access time and use count
                    entry.UpdateAccessTime();
                    entry.IncrementUseCount();

                    // Update statistics
                    _stats.TotalHits++;

                    return entry.CompiledKernel;
                }
                else
                {
                    // Update statistics
                    _stats.TotalMisses++;

                    return default;
                }
            }
        }

        /// <summary>
        /// Adds or updates a compiled kernel in the cache.
        /// If the cache is full, evicts the least recently used kernel.
        /// </summary>
        /// <param name="sig">The shape signature of the kernel.</param>
        /// <param name="kernel">The compiled kernel to cache.</param>
        /// <param name="compilationTimeMs">Time taken to compile the kernel (in milliseconds).</param>
        public void Set(ShapeSignature sig, TKernel kernel, long compilationTimeMs)
        {
            if (compilationTimeMs < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(compilationTimeMs), "Compilation time cannot be negative.");
            }

            lock (_lock)
            {
                // If the signature already exists, update the entry
                if (_cache.ContainsKey(sig))
                {
                    var existingEntry = _cache[sig];
                    _stats.TotalCompilationTimeMs -= existingEntry.CompilationTimeMs;
                    var newEntry = new KernelCacheEntry<TKernel>(sig, kernel, compilationTimeMs);
                    _cache[sig] = newEntry;
                }
                else
                {
                    // Check if we need to evict
                    while (_cache.Count >= _maxSize)
                    {
                        EvictLeastRecentlyUsed();
                    }

                    // Add new entry
                    var entry = new KernelCacheEntry<TKernel>(sig, kernel, compilationTimeMs);
                    _cache[sig] = entry;
                    _stats.TotalCompilationTimeMs += compilationTimeMs;
                }

                _stats.TotalKernels = _cache.Count;
            }
        }

        /// <summary>
        /// Determines whether a compiled kernel for the given signature exists in the cache.
        /// </summary>
        /// <param name="sig">The shape signature to check.</param>
        /// <returns>True if the cache contains a kernel for the signature; otherwise, false.</returns>
        public bool Contains(ShapeSignature sig)
        {
            lock (_lock)
            {
                return _cache.ContainsKey(sig);
            }
        }

        /// <summary>
        /// Removes a compiled kernel from the cache.
        /// </summary>
        /// <param name="sig">The shape signature of the kernel to remove.</param>
        /// <returns>True if the kernel was removed; otherwise, false.</returns>
        public bool Remove(ShapeSignature sig)
        {
            lock (_lock)
            {
                if (_cache.TryGetValue(sig, out var entry))
                {
                    _stats.TotalCompilationTimeMs -= entry.CompilationTimeMs;
                    bool removed = _cache.Remove(sig);
                    _stats.TotalKernels = _cache.Count;
                    return removed;
                }
                return false;
            }
        }

        /// <summary>
        /// Clears all entries from the cache.
        /// </summary>
        public void Clear()
        {
            lock (_lock)
            {
                _cache.Clear();
                _stats.TotalKernels = 0;
                _stats.TotalCompilationTimeMs = 0;
            }
        }

        /// <summary>
        /// Gets statistics about the cache's performance and usage.
        /// </summary>
        /// <returns>A <see cref="CacheStats"/> object containing cache statistics.</returns>
        public CacheStats GetStats()
        {
            lock (_lock)
            {
                return _stats.Clone();
            }
        }

        /// <summary>
        /// Evicts the least recently used kernel from the cache.
        /// </summary>
        private void EvictLeastRecentlyUsed()
        {
            if (_cache.Count == 0)
            {
                return;
            }

            // Find the entry with the oldest LastUsed time
            var lruSignature = _cache
                .OrderBy(kvp => kvp.Value.LastUsed)
                .ThenBy(kvp => kvp.Value.UseCount)
                .First()
                .Key;

            Remove(lruSignature);
        }

        /// <summary>
        /// Gets a list of signatures for kernels that would be evicted if space were needed.
        /// Returns up to <paramref name="count"/> candidates, ordered from least to most recently used.
        /// </summary>
        /// <param name="count">The number of eviction candidates to return.</param>
        /// <returns>A list of shape signatures, ordered by least recently used.</returns>
        public List<ShapeSignature> GetEvictionCandidates(int count)
        {
            lock (_lock)
            {
                return _cache
                    .OrderBy(kvp => kvp.Value.LastUsed)
                    .ThenBy(kvp => kvp.Value.UseCount)
                    .Take(count)
                    .Select(kvp => kvp.Key)
                    .ToList();
            }
        }
    }
}
