using System;
using System.Collections.Concurrent;
using System.Collections.Generic;

namespace MLFramework.Data
{
    /// <summary>
    /// Centralized manager for multiple pools.
    /// </summary>
    public class PoolManager
    {
        private readonly ConcurrentDictionary<string, IPool> _pools;
        private static readonly Lazy<PoolManager> _instance = new Lazy<PoolManager>(() => new PoolManager());

        /// <summary>
        /// Gets the singleton instance of the PoolManager.
        /// </summary>
        public static PoolManager Instance => _instance.Value;

        private PoolManager()
        {
            _pools = new ConcurrentDictionary<string, IPool>();
        }

        /// <summary>
        /// Gets an existing pool with the specified key.
        /// </summary>
        /// <typeparam name="T">The type of objects managed by the pool.</typeparam>
        /// <param name="key">The key identifying the pool.</param>
        /// <returns>The pool if found; otherwise, throws <see cref="KeyNotFoundException"/>.</returns>
        public IPool<T> GetPool<T>(string key)
        {
            if (string.IsNullOrWhiteSpace(key))
                throw new ArgumentException("Key cannot be null or whitespace.", nameof(key));

            if (_pools.TryGetValue(key, out var pool))
            {
                if (pool is IPool<T> typedPool)
                {
                    return typedPool;
                }

                throw new InvalidCastException($"Pool with key '{key}' exists but is not of type {typeof(T).Name}.");
            }

            throw new KeyNotFoundException($"Pool with key '{key}' not found.");
        }

        /// <summary>
        /// Gets an existing pool or creates a new one if it doesn't exist.
        /// </summary>
        /// <typeparam name="T">The type of objects managed by the pool.</typeparam>
        /// <param name="key">The key identifying the pool.</param>
        /// <param name="factory">Function to create new instances of T.</param>
        /// <param name="reset">Optional action to reset items when returned to pool.</param>
        /// <param name="initialSize">Number of items to pre-allocate.</param>
        /// <param name="maxSize">Maximum number of items to keep in pool.</param>
        /// <returns>The existing or newly created pool.</returns>
        public IPool<T> GetOrCreatePool<T>(
            string key,
            Func<T> factory,
            Action<T>? reset = null,
            int initialSize = 0,
            int maxSize = 100)
        {
            if (string.IsNullOrWhiteSpace(key))
                throw new ArgumentException("Key cannot be null or whitespace.", nameof(key));

            return (IPool<T>)_pools.GetOrAdd(
                key,
                _ => new ObjectPool<T>(factory, reset, initialSize, maxSize));
        }

        /// <summary>
        /// Creates or replaces a pool with the specified key.
        /// </summary>
        /// <typeparam name="T">The type of objects managed by the pool.</typeparam>
        /// <param name="key">The key identifying the pool.</param>
        /// <param name="factory">Function to create new instances of T.</param>
        /// <param name="reset">Optional action to reset items when returned to pool.</param>
        /// <param name="initialSize">Number of items to pre-allocate.</param>
        /// <param name="maxSize">Maximum number of items to keep in pool.</param>
        /// <returns>The newly created pool.</returns>
        public IPool<T> CreatePool<T>(
            string key,
            Func<T> factory,
            Action<T>? reset = null,
            int initialSize = 0,
            int maxSize = 100)
        {
            if (string.IsNullOrWhiteSpace(key))
                throw new ArgumentException("Key cannot be null or whitespace.", nameof(key));

            var pool = new ObjectPool<T>(factory, reset, initialSize, maxSize);
            _pools.AddOrUpdate(key, pool, (_, _) => pool);
            return pool;
        }

        /// <summary>
        /// Registers an existing pool with the specified key.
        /// </summary>
        /// <typeparam name="T">The type of objects managed by the pool.</typeparam>
        /// <param name="key">The key identifying the pool.</param>
        /// <param name="pool">The pool to register.</param>
        public void RegisterPool<T>(string key, IPool<T> pool)
        {
            if (string.IsNullOrWhiteSpace(key))
                throw new ArgumentException("Key cannot be null or whitespace.", nameof(key));

            if (pool == null)
                throw new ArgumentNullException(nameof(pool));

            _pools.AddOrUpdate(key, pool, (_, _) => pool);
        }

        /// <summary>
        /// Checks if a pool with the specified key exists.
        /// </summary>
        /// <param name="key">The key identifying the pool.</param>
        /// <returns>True if the pool exists; otherwise, false.</returns>
        public bool HasPool(string key)
        {
            if (string.IsNullOrWhiteSpace(key))
                return false;

            return _pools.ContainsKey(key);
        }

        /// <summary>
        /// Removes and disposes a pool with the specified key.
        /// </summary>
        /// <param name="key">The key identifying the pool to remove.</param>
        /// <returns>True if the pool was found and removed; otherwise, false.</returns>
        public bool RemovePool(string key)
        {
            if (string.IsNullOrWhiteSpace(key))
                return false;

            if (_pools.TryRemove(key, out var pool))
            {
                pool.Dispose();
                return true;
            }

            return false;
        }

        /// <summary>
        /// Clears all managed pools.
        /// </summary>
        /// <remarks>
        /// Useful for cleanup or memory pressure response.
        /// </remarks>
        public void ClearAll()
        {
            foreach (var pool in _pools.Values)
            {
                pool.Clear();
            }
        }

        /// <summary>
        /// Removes and disposes all managed pools.
        /// </summary>
        public void DisposeAll()
        {
            foreach (var pool in _pools.Values)
            {
                pool.Dispose();
            }

            _pools.Clear();
        }

        /// <summary>
        /// Gets statistics for all managed pools.
        /// </summary>
        /// <returns>A <see cref="PoolManagerStatistics"/> object containing aggregate statistics.</returns>
        public PoolManagerStatistics GetStatistics()
        {
            var stats = new PoolManagerStatistics();

            foreach (var kvp in _pools)
            {
                var pool = kvp.Value;
                stats.PoolSizes[kvp.Key] = pool.AvailableCount;
                stats.TotalAvailableItems += pool.AvailableCount;
                stats.TotalCreatedItems += pool.TotalCount;
                stats.PoolCount++;
            }

            return stats;
        }
    }

    /// <summary>
    /// Statistics for the PoolManager.
    /// </summary>
    public class PoolManagerStatistics
    {
        /// <summary>
        /// Gets the number of pools managed by the PoolManager.
        /// </summary>
        public int PoolCount { get; internal set; }

        /// <summary>
        /// Gets the total number of available items across all pools.
        /// </summary>
        public int TotalAvailableItems { get; internal set; }

        /// <summary>
        /// Gets the total number of items created across all pools.
        /// </summary>
        public int TotalCreatedItems { get; internal set; }

        /// <summary>
        /// Gets a dictionary mapping pool keys to their current sizes.
        /// </summary>
        public Dictionary<string, int> PoolSizes { get; internal set; } = new Dictionary<string, int>();
    }
}
