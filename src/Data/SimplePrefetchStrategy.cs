using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Threading.Tasks;

namespace MLFramework.Data
{
    /// <summary>
    /// Simple prefetching strategy using a fixed-size buffer.
    /// Maintains a buffer of pre-fetched items to reduce wait times.
    /// </summary>
    /// <typeparam name="T">The type of items to prefetch.</typeparam>
    public class SimplePrefetchStrategy<T> : IPrefetchStrategy<T>
    {
        private readonly SharedQueue<T> _sourceQueue;
        private readonly PrefetchBuffer<T> _buffer;
        private readonly CancellationToken? _cancellationToken;
        private readonly PrefetchStatistics _statistics;
        private readonly object _statisticsLock = new object();
        private Task? _prefetchTask;
        private bool _isDisposed;

        /// <summary>
        /// Initializes a new instance of the SimplePrefetchStrategy class.
        /// </summary>
        /// <param name="sourceQueue">Queue to prefetch from.</param>
        /// <param name="prefetchCount">Number of items to keep preloaded.</param>
        /// <param name="cancellationToken">Optional cancellation token.</param>
        /// <exception cref="ArgumentNullException">Thrown when sourceQueue is null.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when prefetchCount is less than or equal to zero.</exception>
        public SimplePrefetchStrategy(
            SharedQueue<T> sourceQueue,
            int prefetchCount,
            CancellationToken? cancellationToken = null)
        {
            _sourceQueue = sourceQueue ?? throw new ArgumentNullException(nameof(sourceQueue));
            if (prefetchCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(prefetchCount), "Prefetch count must be greater than zero.");
            }

            _buffer = new PrefetchBuffer<T>(prefetchCount);
            _cancellationToken = cancellationToken;
            _statistics = new PrefetchStatistics();
        }

        /// <summary>
        /// Gets whether prefetched items are available in the buffer.
        /// </summary>
        public bool IsAvailable => !_buffer.IsEmpty;

        /// <summary>
        /// Gets the current number of prefetched items in the buffer.
        /// </summary>
        public int BufferCount => _buffer.Count;

        /// <summary>
        /// Gets the prefetch buffer capacity.
        /// </summary>
        public int BufferCapacity => _buffer.Capacity;

        /// <summary>
        /// Gets the next prefetched item asynchronously.
        /// Returns immediately if an item is available in the prefetch buffer,
        /// otherwise waits for the next item from the source queue.
        /// </summary>
        /// <param name="cancellationToken">Optional cancellation token.</param>
        /// <returns>The next prefetched item.</returns>
        /// <exception cref="OperationCanceledException">Thrown when cancelled.</exception>
        public async Task<T> GetNextAsync(CancellationToken cancellationToken)
        {
            ThrowIfDisposed();

            var stopwatch = Stopwatch.StartNew();
            T item;

            // First, try to get from buffer (cache hit)
            if (_buffer.TryGet(out item))
            {
                UpdateStatisticsOnCacheHit(stopwatch);
                // Trigger background refill after taking item
                _ = Task.Run(() => RefillBufferAsync(cancellationToken), cancellationToken);
                return item;
            }

            // Cache miss - item not available in buffer
            UpdateStatisticsOnCacheMiss(stopwatch);

            // Wait for next item from source queue
            item = await Task.Run(() => _sourceQueue.Dequeue(), cancellationToken).ConfigureAwait(false);

            UpdateStatisticsOnFetch(stopwatch);

            // Trigger background refill
            _ = Task.Run(() => RefillBufferAsync(cancellationToken), cancellationToken);

            return item;
        }

        /// <summary>
        /// Starts a background task to prefetch the specified number of items.
        /// </summary>
        /// <param name="count">Number of items to prefetch.</param>
        /// <param name="cancellationToken">Optional cancellation token.</param>
        /// <returns>A task representing the prefetch operation.</returns>
        public async Task PrefetchAsync(int count, CancellationToken cancellationToken)
        {
            ThrowIfDisposed();

            await Task.Run(() => RefillBufferWithCountAsync(count, cancellationToken), cancellationToken).ConfigureAwait(false);
        }

        /// <summary>
        /// Resets the prefetch strategy, clearing the internal buffer.
        /// </summary>
        public void Reset()
        {
            ThrowIfDisposed();
            _buffer.Clear();
            _statistics.Reset();
        }

        /// <summary>
        /// Gets the current prefetch statistics.
        /// </summary>
        /// <returns>Prefetch statistics object.</returns>
        public PrefetchStatistics GetStatistics()
        {
            lock (_statisticsLock)
            {
                return new PrefetchStatistics
                {
                    CacheHits = _statistics.CacheHits,
                    CacheMisses = _statistics.CacheMisses,
                    AverageLatencyMs = _statistics.AverageLatencyMs,
                    RefillCount = _statistics.RefillCount,
                    StarvationCount = _statistics.StarvationCount
                };
            }
        }

        /// <summary>
        /// Releases all resources used by the prefetch strategy.
        /// </summary>
        public void Dispose()
        {
            if (_isDisposed)
            {
                return;
            }

            _isDisposed = true;
            _buffer.Clear();
            GC.SuppressFinalize(this);
        }

        #region Private Methods

        private async Task RefillBufferAsync(CancellationToken cancellationToken)
        {
            // Fill buffer until full or source queue is exhausted
            while (!_buffer.IsFull && !_isDisposed)
            {
                if (_sourceQueue.TryDequeue(out T item, 100))
                {
                    _buffer.Add(item);
                }
                else
                {
                    // Queue is empty or timeout, stop refilling
                    break;
                }

                // Check cancellation
                if (cancellationToken.IsCancellationRequested ||
                    (_cancellationToken.HasValue && _cancellationToken.Value.IsCancellationRequested))
                {
                    break;
                }
            }

            if (_buffer.Count > 0)
            {
                lock (_statisticsLock)
                {
                    _statistics.RefillCount++;
                }
            }
        }

        private async Task RefillBufferWithCountAsync(int count, CancellationToken cancellationToken)
        {
            int itemsPrefetched = 0;

            while (itemsPrefetched < count && !_buffer.IsFull && !_isDisposed)
            {
                if (_sourceQueue.TryDequeue(out T item, 100))
                {
                    _buffer.Add(item);
                    itemsPrefetched++;
                }
                else
                {
                    // Queue is empty or timeout, stop refilling
                    break;
                }

                // Check cancellation
                if (cancellationToken.IsCancellationRequested ||
                    (_cancellationToken.HasValue && _cancellationToken.Value.IsCancellationRequested))
                {
                    break;
                }
            }

            if (itemsPrefetched > 0)
            {
                lock (_statisticsLock)
                {
                    _statistics.RefillCount++;
                }
            }
        }

        private void UpdateStatisticsOnCacheHit(Stopwatch stopwatch)
        {
            lock (_statisticsLock)
            {
                _statistics.CacheHits++;
                _statistics.AverageLatencyMs = UpdateAverageLatency(
                    _statistics.AverageLatencyMs,
                    _statistics.TotalRequests,
                    stopwatch.ElapsedMilliseconds);
            }
        }

        private void UpdateStatisticsOnCacheMiss(Stopwatch stopwatch)
        {
            lock (_statisticsLock)
            {
                _statistics.CacheMisses++;
                _statistics.StarvationCount++;
            }
        }

        private void UpdateStatisticsOnFetch(Stopwatch stopwatch)
        {
            lock (_statisticsLock)
            {
                _statistics.AverageLatencyMs = UpdateAverageLatency(
                    _statistics.AverageLatencyMs,
                    _statistics.TotalRequests,
                    stopwatch.ElapsedMilliseconds);
            }
        }

        private double UpdateAverageLatency(double currentAverage, int totalCount, double newLatency)
        {
            if (totalCount == 0)
            {
                return newLatency;
            }

            return (currentAverage * totalCount + newLatency) / (totalCount + 1);
        }

        private void ThrowIfDisposed()
        {
            if (_isDisposed)
            {
                throw new ObjectDisposedException(nameof(SimplePrefetchStrategy<T>));
            }
        }

        #endregion
    }
}
