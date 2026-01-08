using System;
using System.Collections.Concurrent;
using System.Threading;

namespace MLFramework.Data
{
    /// <summary>
    /// Generic thread-safe object pool implementation.
    /// </summary>
    /// <typeparam name="T">The type of objects managed by the pool.</typeparam>
    public class ObjectPool<T> : IPool<T>
    {
        private readonly Func<T> _factory;
        private readonly Action<T>? _reset;
        private readonly ConcurrentBag<T> _availableItems;
        private int _totalCount;
        private readonly int _maxSize;
        private readonly PoolStatistics _statistics;
        private bool _disposed;

        /// <summary>
        /// Initializes a new instance of the <see cref="ObjectPool{T}"/> class.
        /// </summary>
        /// <param name="factory">Function to create new instances.</param>
        /// <param name="reset">Optional action to reset item when returned to pool.</param>
        /// <param name="initialSize">Number of items to pre-allocate.</param>
        /// <param name="maxSize">Maximum number of items to keep in pool.</param>
        public ObjectPool(
            Func<T> factory,
            Action<T>? reset = null,
            int initialSize = 0,
            int maxSize = 100)
        {
            _factory = factory ?? throw new ArgumentNullException(nameof(factory));
            _reset = reset;
            _maxSize = maxSize > 0 ? maxSize : throw new ArgumentOutOfRangeException(nameof(maxSize), "MaxSize must be greater than 0.");
            _availableItems = new ConcurrentBag<T>();
            _statistics = new PoolStatistics();

            // Pre-allocate items if initialSize > 0
            for (int i = 0; i < initialSize; i++)
            {
                var item = _factory();
                _availableItems.Add(item);
                Interlocked.Increment(ref _totalCount);
            }
        }

        /// <inheritdoc/>
        public int AvailableCount => _availableItems.Count;

        /// <inheritdoc/>
        public int TotalCount => _totalCount;

        /// <summary>
        /// Gets the maximum number of items that can be kept in the pool.
        /// </summary>
        public int MaxSize => _maxSize;

        /// <summary>
        /// Gets the statistics for this pool.
        /// </summary>
        public PoolStatistics Statistics => _statistics;

        /// <inheritdoc/>
        public T Rent()
        {
            ThrowIfDisposed();

            _statistics.IncrementRent();

            if (_availableItems.TryTake(out var item))
            {
                return item;
            }

            // Pool is empty, create new item
            _statistics.IncrementMiss();
            Interlocked.Increment(ref _totalCount);
            return _factory();
        }

        /// <inheritdoc/>
        public void Return(T item)
        {
            ThrowIfDisposed();

            _statistics.IncrementReturn();

            // Apply reset action if provided
            _reset?.Invoke(item);

            // Return to pool if not at max size
            if (_availableItems.Count < _maxSize)
            {
                _availableItems.Add(item);
            }
            else
            {
                // Discard item if pool is full
                _statistics.IncrementDiscard();
            }
        }

        /// <inheritdoc/>
        public void Clear()
        {
            ThrowIfDisposed();

            // Clear all items from the pool
            while (_availableItems.TryTake(out _))
            {
                // Items are not disposed; caller is responsible
            }
        }

        /// <summary>
        /// Gets the current statistics for the pool.
        /// </summary>
        public PoolStatistics GetStatistics()
        {
            return _statistics;
        }

        /// <summary>
        /// Resets all statistics counters to zero.
        /// </summary>
        public void ResetStatistics()
        {
            _statistics.Reset();
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(ObjectPool<T>));
        }

        /// <inheritdoc/>
        public void Dispose()
        {
            if (!_disposed)
            {
                Clear();
                _disposed = true;
            }
        }
    }
}
