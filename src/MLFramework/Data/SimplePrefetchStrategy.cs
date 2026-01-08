using System.Collections.Concurrent;
using System.Diagnostics;

namespace MLFramework.Data;

/// <summary>
/// Simple prefetching strategy using a fixed-size buffer for fast item retrieval.
/// </summary>
/// <typeparam name="T">The type of items being prefetched.</typeparam>
public sealed class SimplePrefetchStrategy<T> : IPrefetchStrategy<T>, IDisposable
{
    private readonly SharedQueue<T> _sourceQueue;
    private readonly int _prefetchCount;
    private readonly CancellationTokenSource _cancellationTokenSource;
        private readonly PrefetchBuffer<T> _buffer;
    private readonly Stopwatch _latencyStopwatch;
    private Task? _prefetchTask;
    private bool _disposed;
    private int _cacheHits;
    private int _cacheMisses;
    private int _refillCount;
    private int _starvationCount;

    /// <summary>
    /// Initializes a new instance of the SimplePrefetchStrategy class.
    /// </summary>
    /// <param name="sourceQueue">The queue to prefetch from.</param>
    /// <param name="prefetchCount">Number of items to keep preloaded in the buffer.</param>
    /// <param name="cancellationToken">Optional cancellation token for graceful shutdown.</param>
    /// <exception cref="ArgumentNullException">Thrown when sourceQueue is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when prefetchCount is not positive.</exception>
    public SimplePrefetchStrategy(SharedQueue<T> sourceQueue, int prefetchCount, CancellationToken? cancellationToken = null)
    {
        _sourceQueue = sourceQueue ?? throw new ArgumentNullException(nameof(sourceQueue));

        if (prefetchCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(prefetchCount), prefetchCount, "PrefetchCount must be > 0.");

        _prefetchCount = prefetchCount;
        _cancellationTokenSource = CancellationTokenSource.CreateLinkedTokenSource(
            cancellationToken ?? CancellationToken.None);
        _buffer = new PrefetchBuffer<T>(prefetchCount);
        _latencyStopwatch = new Stopwatch();
    }

    /// <summary>
    /// Gets whether prefetched items are available in the buffer.
    /// </summary>
    public bool IsAvailable => !_buffer.IsEmpty;

    /// <summary>
    /// Gets the next item asynchronously.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token for graceful shutdown.</param>
    /// <returns>The next item, either from the prefetch buffer or by waiting for the source.</returns>
    /// <exception cref="ObjectDisposedException">Thrown when the strategy has been disposed.</exception>
    public async Task<T> GetNextAsync(CancellationToken cancellationToken)
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(SimplePrefetchStrategy<T>));

        var combinedToken = CancellationTokenSource.CreateLinkedTokenSource(
            _cancellationTokenSource.Token, cancellationToken).Token;

        _latencyStopwatch.Restart();

        // Try to get from buffer first (cache hit)
        if (_buffer.TryGet(out var item))
        {
            _cacheHits++;
            _latencyStopwatch.Stop();
            TriggerRefillIfNeeded(combinedToken);
            return item;
        }

        // Cache miss - fetch from source queue
        _cacheMisses++;
        _starvationCount++;

        try
        {
            item = await Task.Run(() => _sourceQueue.Dequeue(), combinedToken);
        }
        catch (OperationCanceledException)
        {
            _latencyStopwatch.Stop();
            throw;
        }

        _latencyStopwatch.Stop();
        TriggerRefillIfNeeded(combinedToken);

        return item;
    }

    /// <summary>
    /// Prefetches the specified number of items in the background.
    /// </summary>
    /// <param name="count">The number of items to prefetch.</param>
    /// <param name="cancellationToken">Cancellation token for graceful shutdown.</param>
    /// <returns>A task representing the prefetch operation.</returns>
    public Task PrefetchAsync(int count, CancellationToken cancellationToken)
    {
        if (_disposed)
            return Task.CompletedTask;

        var combinedToken = CancellationTokenSource.CreateLinkedTokenSource(
            _cancellationTokenSource.Token, cancellationToken).Token;

        return Task.Run(() =>
        {
            var itemsToFetch = Math.Min(count, _buffer.Capacity - _buffer.Count);

            for (int i = 0; i < itemsToFetch; i++)
            {
                try
                {
                    if (_cancellationTokenSource.IsCancellationRequested)
                        break;

                    var item = _sourceQueue.Dequeue();
                    _buffer.Add(item);
                }
                catch (OperationCanceledException)
                {
                    break;
                }
            }

            _refillCount++;
        }, combinedToken);
    }

    /// <summary>
    /// Resets the prefetch strategy, clearing the internal buffer and counters.
    /// </summary>
    public void Reset()
    {
        _buffer.Clear();
        _cacheHits = 0;
        _cacheMisses = 0;
        _refillCount = 0;
        _starvationCount = 0;
    }

    /// <summary>
    /// Gets statistics about prefetch operations.
    /// </summary>
    /// <returns>PrefetchStatistics instance with current metrics.</returns>
    public PrefetchStatistics GetStatistics()
    {
        var totalRequests = _cacheHits + _cacheMisses;
        return new PrefetchStatistics(
            cacheHits: _cacheHits,
            cacheMisses: _cacheMisses,
            averageLatencyMs: totalRequests > 0 ? _latencyStopwatch.ElapsedMilliseconds / (double)totalRequests : 0,
            refillCount: _refillCount,
            starvationCount: _starvationCount);
    }

    private void TriggerRefillIfNeeded(CancellationToken cancellationToken)
    {
        // Trigger refill if buffer is less than half full
        if (_buffer.Count < _buffer.Capacity / 2)
        {
            _prefetchTask = PrefetchAsync(_prefetchCount - _buffer.Count, cancellationToken);
        }
    }

    /// <summary>
    /// Disposes of all resources used by the prefetch strategy.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
            return;

        _cancellationTokenSource.Cancel();
        _cancellationTokenSource.Dispose();
        _disposed = true;
    }
}

/// <summary>
/// Internal thread-safe buffer for holding prefetched items.
/// </summary>
internal sealed class PrefetchBuffer<T>
{
    private readonly ConcurrentQueue<T> _queue;
    private readonly int _capacity;

    /// <summary>
    /// Gets the number of items in the buffer.
    /// </summary>
    public int Count => _queue.Count;

    /// <summary>
    /// Gets the capacity of the buffer.
    /// </summary>
    public int Capacity => _capacity;

    /// <summary>
    /// Gets whether the buffer is empty.
    /// </summary>
    public bool IsEmpty => _queue.IsEmpty;

    /// <summary>
    /// Gets whether the buffer is full.
    /// </summary>
    public bool IsFull => Count >= Capacity;

    /// <summary>
    /// Initializes a new instance of the PrefetchBuffer class.
    /// </summary>
    /// <param name="capacity">The capacity of the buffer.</param>
    public PrefetchBuffer(int capacity)
    {
        _capacity = capacity;
        _queue = new ConcurrentQueue<T>();
    }

    /// <summary>
    /// Adds an item to the buffer.
    /// </summary>
    /// <param name="item">The item to add.</param>
    /// <exception cref="InvalidOperationException">Thrown when the buffer is full.</exception>
    public void Add(T item)
    {
        if (IsFull)
            throw new InvalidOperationException("Cannot add to full buffer.");

        _queue.Enqueue(item);
    }

    /// <summary>
    /// Gets the next item from the buffer in FIFO order.
    /// </summary>
    /// <returns>The next item.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the buffer is empty.</exception>
    public T GetNext()
    {
        if (!TryGet(out var item))
            throw new InvalidOperationException("Buffer is empty.");

        return item;
    }

    /// <summary>
    /// Tries to get the next item from the buffer.
    /// </summary>
    /// <param name="item">The item if successful.</param>
    /// <returns>True if an item was retrieved, false if the buffer is empty.</returns>
    public bool TryGet(out T item)
    {
        return _queue.TryDequeue(out item!);
    }

    /// <summary>
    /// Peeks at the next item without removing it.
    /// </summary>
    /// <returns>The next item.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the buffer is empty.</exception>
    public T Peek()
    {
        if (!TryPeek(out var item))
            throw new InvalidOperationException("Buffer is empty.");

        return item;
    }

    /// <summary>
    /// Tries to peek at the next item without removing it.
    /// </summary>
    /// <param name="item">The item if successful.</param>
    /// <returns>True if an item was peeked, false if the buffer is empty.</returns>
    public bool TryPeek(out T item)
    {
        return _queue.TryPeek(out item!);
    }

    /// <summary>
    /// Clears all items from the buffer.
    /// </summary>
    public void Clear()
    {
        while (!_queue.IsEmpty)
        {
            _queue.TryDequeue(out _);
        }
    }
}

/// <summary>
/// Statistics for prefetch operations.
/// </summary>
public sealed class PrefetchStatistics
{
    /// <summary>
    /// Gets the number of cache hits (items retrieved from prefetch buffer).
    /// </summary>
    public int CacheHits { get; }

    /// <summary>
    /// Gets the number of cache misses (items retrieved directly from source).
    /// </summary>
    public int CacheMisses { get; }

    /// <summary>
    /// Gets the cache hit rate as a percentage.
    /// </summary>
    public double CacheHitRate
    {
        get
        {
            var total = CacheHits + CacheMisses;
            return total > 0 ? (double)CacheHits / total * 100 : 0;
        }
    }

    /// <summary>
    /// Gets the average latency in milliseconds for retrieving items.
    /// </summary>
    public double AverageLatencyMs { get; }

    /// <summary>
    /// Gets the number of times the prefetch buffer was refilled.
    /// </summary>
    public int RefillCount { get; }

    /// <summary>
    /// Gets the number of times the prefetch buffer was empty when an item was requested.
    /// </summary>
    public int StarvationCount { get; }

    internal PrefetchStatistics(
        int cacheHits,
        int cacheMisses,
        double averageLatencyMs,
        int refillCount,
        int starvationCount)
    {
        CacheHits = cacheHits;
        CacheMisses = cacheMisses;
        AverageLatencyMs = averageLatencyMs;
        RefillCount = refillCount;
        StarvationCount = starvationCount;
    }

    public override string ToString()
    {
        return $"PrefetchStatistics {{ CacheHits: {CacheHits}, CacheMisses: {CacheMisses}, " +
               $"CacheHitRate: {CacheHitRate:F2}%, AverageLatencyMs: {AverageLatencyMs:F2}, " +
               $"RefillCount: {RefillCount}, StarvationCount: {StarvationCount} }}";
    }
}
