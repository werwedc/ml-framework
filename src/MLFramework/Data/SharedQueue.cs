using System.Collections.Concurrent;
using System.Diagnostics;

namespace MLFramework.Data;

/// <summary>
/// Thread-safe, blocking queue for communication between data loading workers and the main training process.
/// Implements the producer-consumer pattern with support for graceful shutdown.
/// </summary>
/// <typeparam name="T">The type of items in the queue.</typeparam>
public sealed class SharedQueue<T> : IDisposable
{
    private readonly BlockingCollection<T> _collection;
    private readonly CancellationTokenSource _cancellationTokenSource;
    private readonly Stopwatch _enqueueStopwatch;
    private readonly Stopwatch _dequeueStopwatch;
    private readonly object _statsLock;
    private readonly bool _enableStatistics;
    private int _totalEnqueued;
    private int _totalDequeued;
    private long _totalWaitTimeMs;
    private int _producerWaitCount;
    private int _consumerWaitCount;
    private int _maxQueueSize;

    /// <summary>
    /// Initializes a new instance of the SharedQueue class.
    /// </summary>
    /// <param name="capacity">Maximum number of items in the queue (must be > 0).</param>
    /// <param name="cancellationToken">Optional cancellation token for graceful shutdown.</param>
    /// <param name="enableStatistics">Whether to track statistics (default: true).</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when capacity is not positive.</exception>
    public SharedQueue(int capacity, CancellationToken? cancellationToken = null, bool enableStatistics = true)
    {
        if (capacity <= 0)
            throw new ArgumentOutOfRangeException(nameof(capacity), capacity, "Capacity must be > 0.");

        _collection = new BlockingCollection<T>(capacity);
        _cancellationTokenSource = CancellationTokenSource.CreateLinkedTokenSource(
            cancellationToken ?? CancellationToken.None);
        _enableStatistics = enableStatistics;
        _statsLock = new object();
        _enqueueStopwatch = new Stopwatch();
        _dequeueStopwatch = new Stopwatch();
        _totalEnqueued = 0;
        _totalDequeued = 0;
        _totalWaitTimeMs = 0;
        _producerWaitCount = 0;
        _consumerWaitCount = 0;
        _maxQueueSize = 0;
    }

    /// <summary>
    /// Gets the current number of items in the queue.
    /// </summary>
    public int Count => _collection.Count;

    /// <summary>
    /// Gets whether the queue is marked as complete and empty.
    /// </summary>
    public bool IsCompleted => _collection.IsCompleted;

    /// <summary>
    /// Gets the maximum capacity of the queue.
    /// </summary>
    public int Capacity => _collection.BoundedCapacity;

    /// <summary>
    /// Enqueues an item, blocking if the queue is full until space becomes available.
    /// </summary>
    /// <param name="item">The item to enqueue.</param>
    /// <exception cref="OperationCanceledException">Thrown when cancellation token is triggered.</exception>
    /// <exception cref="InvalidOperationException">Thrown when queue is marked complete.</exception>
    public void Enqueue(T item)
    {
        var stopwatch = _enableStatistics ? Stopwatch.StartNew() : null;

        try
        {
            _collection.Add(item, _cancellationTokenSource.Token);

            if (_enableStatistics)
            {
                stopwatch!.Stop();
                lock (_statsLock)
                {
                    _totalEnqueued++;
                    _totalWaitTimeMs += stopwatch.ElapsedMilliseconds;
                    if (stopwatch.ElapsedMilliseconds > 0)
                        _producerWaitCount++;

                    if (_collection.Count > _maxQueueSize)
                        _maxQueueSize = _collection.Count;
                }
            }
        }
        catch (OperationCanceledException)
        {
            if (_enableStatistics && stopwatch != null)
                stopwatch.Stop();
            throw;
        }
    }

    /// <summary>
    /// Tries to enqueue an item with a timeout.
    /// </summary>
    /// <param name="item">The item to enqueue.</param>
    /// <param name="timeoutMilliseconds">Timeout in milliseconds.</param>
    /// <returns>True if the item was enqueued successfully, false if timeout elapsed.</returns>
    /// <exception cref="OperationCanceledException">Thrown when cancellation token is triggered.</exception>
    /// <exception cref="InvalidOperationException">Thrown when queue is marked complete.</exception>
    public bool TryEnqueue(T item, int timeoutMilliseconds)
    {
        var stopwatch = _enableStatistics ? Stopwatch.StartNew() : null;

        try
        {
            bool success = _collection.TryAdd(item, timeoutMilliseconds, _cancellationTokenSource.Token);

            if (_enableStatistics)
            {
                stopwatch!.Stop();
                lock (_statsLock)
                {
                    if (success)
                        _totalEnqueued++;

                    if (stopwatch.ElapsedMilliseconds > 0)
                        _producerWaitCount++;
                }
            }

            return success;
        }
        catch (OperationCanceledException)
        {
            if (_enableStatistics && stopwatch != null)
                stopwatch.Stop();
            throw;
        }
    }

    /// <summary>
    /// Dequeues an item, blocking if the queue is empty until an item is available.
    /// </summary>
    /// <returns>The next item in FIFO order.</returns>
    /// <exception cref="OperationCanceledException">Thrown when cancellation token is triggered.</exception>
    public T Dequeue()
    {
        var stopwatch = _enableStatistics ? Stopwatch.StartNew() : null;

        try
        {
            T item = _collection.Take(_cancellationTokenSource.Token);

            if (_enableStatistics)
            {
                stopwatch!.Stop();
                lock (_statsLock)
                {
                    _totalDequeued++;
                    if (stopwatch.ElapsedMilliseconds > 0)
                        _consumerWaitCount++;
                }
            }

            return item;
        }
        catch (OperationCanceledException)
        {
            if (_enableStatistics && stopwatch != null)
                stopwatch.Stop();
            throw;
        }
    }

    /// <summary>
    /// Tries to dequeue an item with a timeout.
    /// </summary>
    /// <param name="item">The dequeued item if successful.</param>
    /// <param name="timeoutMilliseconds">Timeout in milliseconds.</param>
    /// <returns>True if an item was dequeued successfully, false if timeout elapsed.</returns>
    /// <exception cref="OperationCanceledException">Thrown when cancellation token is triggered.</exception>
    public bool TryDequeue(out T item, int timeoutMilliseconds)
    {
        var stopwatch = _enableStatistics ? Stopwatch.StartNew() : null;

        try
        {
            bool success = _collection.TryTake(out item!, timeoutMilliseconds, _cancellationTokenSource.Token);

            if (_enableStatistics)
            {
                stopwatch!.Stop();
                lock (_statsLock)
                {
                    if (success)
                        _totalDequeued++;

                    if (stopwatch.ElapsedMilliseconds > 0)
                        _consumerWaitCount++;
                }
            }

            return success;
        }
        catch (OperationCanceledException)
        {
            if (_enableStatistics && stopwatch != null)
                stopwatch.Stop();
            item = default!;
            return false;
        }
    }

    /// <summary>
    /// Tries to peek at the next item without removing it from the queue.
    /// </summary>
    /// <param name="item">The peeked item if successful.</param>
    /// <returns>True if an item is available, false if queue is empty.</returns>
    public bool TryPeek(out T item)
    {
        return _collection.TryTake(out item!, 0) && _collection.TryAdd(item);
    }

    /// <summary>
    /// Marks the queue as complete, indicating no more items will be added.
    /// Existing items can still be dequeued.
    /// </summary>
    public void CompleteAdding()
    {
        _collection.CompleteAdding();
    }

    /// <summary>
    /// Blocks until all items are dequeued from the queue.
    /// Returns immediately if already complete.
    /// </summary>
    public void WaitForCompletion()
    {
        _collection.CompleteAdding();
        while (!_collection.IsCompleted || _collection.Count > 0)
        {
            if (_cancellationTokenSource.IsCancellationRequested)
                return;
            Thread.Sleep(10);
        }
    }

    /// <summary>
    /// Immediately shuts down the queue, cancelling all blocking operations.
    /// </summary>
    public void Shutdown()
    {
        _cancellationTokenSource.Cancel();
        _collection.CompleteAdding();
    }

    /// <summary>
    /// Enqueues a batch of items atomically.
    /// </summary>
    /// <param name="items">The items to enqueue.</param>
    /// <exception cref="OperationCanceledException">Thrown when cancellation token is triggered.</exception>
    /// <exception cref="InvalidOperationException">Thrown when queue is marked complete or insufficient space.</exception>
    public void EnqueueBatch(IEnumerable<T> items)
    {
        var itemList = items.ToList();
        if (itemList.Count == 0)
            return;

        // Check if we have enough space
        if (Count + itemList.Count > Capacity)
            throw new InvalidOperationException($"Not enough space in queue. Capacity: {Capacity}, Available: {Capacity - Count}, Required: {itemList.Count}");

        foreach (var item in itemList)
        {
            Enqueue(item);
        }
    }

    /// <summary>
    /// Attempts to dequeue a batch of items.
    /// </summary>
    /// <param name="batchSize">The number of items to dequeue.</param>
    /// <param name="timeoutMilliseconds">Timeout in milliseconds.</param>
    /// <returns>An array of dequeued items (may be fewer than batchSize if timeout).</returns>
    /// <exception cref="OperationCanceledException">Thrown when cancellation token is triggered.</exception>
    public T[] DequeueBatch(int batchSize, int timeoutMilliseconds)
    {
        var items = new List<T>();
        var startTime = DateTime.UtcNow;

        while (items.Count < batchSize)
        {
            var elapsed = (int)(DateTime.UtcNow - startTime).TotalMilliseconds;
            var remainingTimeout = Math.Max(0, timeoutMilliseconds - elapsed);

            if (remainingTimeout <= 0)
                break;

            if (TryDequeue(out var item, remainingTimeout))
            {
                items.Add(item);
            }
            else
            {
                break;
            }
        }

        return items.ToArray();
    }

    /// <summary>
    /// Gets statistics about queue operations.
    /// </summary>
    /// <returns>QueueStatistics instance with current metrics.</returns>
    public QueueStatistics GetStatistics()
    {
        lock (_statsLock)
        {
            return new QueueStatistics(
                totalEnqueued: _totalEnqueued,
                totalDequeued: _totalDequeued,
                averageWaitTimeMs: _totalEnqueued > 0 ? _totalWaitTimeMs / (double)_totalEnqueued : 0,
                maxQueueSize: _maxQueueSize,
                producerWaitCount: _producerWaitCount,
                consumerWaitCount: _consumerWaitCount);
        }
    }

    /// <summary>
    /// Disposes of all resources used by the queue.
    /// </summary>
    public void Dispose()
    {
        _cancellationTokenSource.Cancel();
        _collection.Dispose();
        _cancellationTokenSource.Dispose();
    }
}

/// <summary>
/// Statistics for SharedQueue operations.
/// </summary>
public sealed class QueueStatistics
{
    /// <summary>
    /// Gets the total number of items enqueued.
    /// </summary>
    public int TotalEnqueued { get; }

    /// <summary>
    /// Gets the total number of items dequeued.
    /// </summary>
    public int TotalDequeued { get; }

    /// <summary>
    /// Gets the average wait time in milliseconds.
    /// </summary>
    public double AverageWaitTimeMs { get; }

    /// <summary>
    /// Gets the maximum size the queue reached.
    /// </summary>
    public int MaxQueueSize { get; }

    /// <summary>
    /// Gets the number of times producers had to wait.
    /// </summary>
    public int ProducerWaitCount { get; }

    /// <summary>
    /// Gets the number of times consumers had to wait.
    /// </summary>
    public int ConsumerWaitCount { get; }

    internal QueueStatistics(
        int totalEnqueued,
        int totalDequeued,
        double averageWaitTimeMs,
        int maxQueueSize,
        int producerWaitCount,
        int consumerWaitCount)
    {
        TotalEnqueued = totalEnqueued;
        TotalDequeued = totalDequeued;
        AverageWaitTimeMs = averageWaitTimeMs;
        MaxQueueSize = maxQueueSize;
        ProducerWaitCount = producerWaitCount;
        ConsumerWaitCount = consumerWaitCount;
    }

    public override string ToString()
    {
        return $"QueueStatistics {{ TotalEnqueued: {TotalEnqueued}, TotalDequeued: {TotalDequeued}, " +
               $"AverageWaitTimeMs: {AverageWaitTimeMs:F2}, MaxQueueSize: {MaxQueueSize}, " +
               $"ProducerWaitCount: {ProducerWaitCount}, ConsumerWaitCount: {ConsumerWaitCount} }}";
    }
}
