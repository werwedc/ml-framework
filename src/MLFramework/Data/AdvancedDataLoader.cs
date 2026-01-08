using System.Collections.Concurrent;
using System.Diagnostics;
using MLFramework.Data.Worker;

namespace MLFramework.Data;

/// <summary>
/// High-performance data loader that integrates worker pool, shared queue, prefetching, and memory management.
/// Supports both synchronous and asynchronous iteration for flexible training loops.
/// </summary>
/// <typeparam name="T">The type of data items being loaded.</typeparam>
public sealed class AdvancedDataLoader<T> : IDataLoader<T>, IAsyncEnumerable<T>
{
    private readonly IDataset<T> _dataset;
    private readonly DataLoaderConfig _config;
    private readonly CancellationTokenSource _cancellationTokenSource;
    private readonly Random _random;
    private readonly object _lock;
    private SharedQueue<T>? _queue;
    private WorkerPool? _workerPool;
    private SimplePrefetchStrategy<T>? _prefetchStrategy;
    private int[] _indices;
    private IEnumerator<int[]>? _batchIndexEnumerator;
    private bool _isRunning;
    private bool _disposed;
    private int _batchesLoaded;

    /// <summary>
    /// Gets whether the data loader is currently running.
    /// </summary>
    public bool IsRunning => _isRunning;

    /// <summary>
    /// Gets the configuration used by this data loader.
    /// </summary>
    public DataLoaderConfig Config => _config;

    /// <summary>
    /// Gets the number of batches that will be produced.
    /// </summary>
    public int BatchCount => (int)Math.Ceiling((double)_dataset.Length / _config.BatchSize);

    /// <summary>
    /// Initializes a new instance of the DataLoader class.
    /// </summary>
    /// <param name="dataset">The dataset to load data from.</param>
    /// <param name="config">Configuration for loading behavior.</param>
    /// <exception cref="ArgumentNullException">Thrown when dataset is null.</exception>
    /// <exception cref="ArgumentNullException">Thrown when config is null.</exception>
    public AdvancedDataLoader(IDataset<T> dataset, DataLoaderConfig config)
    {
        _dataset = dataset ?? throw new ArgumentNullException(nameof(dataset));
        _config = config ?? throw new ArgumentNullException(nameof(config));
        _cancellationTokenSource = new CancellationTokenSource();
        _random = new Random(config.Seed);
        _lock = new object();
        _indices = new int[dataset.Length];
        _batchesLoaded = 0;

        // Initialize indices array
        for (int i = 0; i < _indices.Length; i++)
        {
            _indices[i] = i;
        }

        // Shuffle if configured
        if (config.Shuffle)
        {
            ShuffleIndices();
        }
    }

    /// <summary>
    /// Starts the data loading process, initializing workers and prefetching.
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown when the loader is already running.</exception>
    public void Start()
    {
        if (_isRunning)
            throw new InvalidOperationException("DataLoader is already running.");

        lock (_lock)
        {
            if (_isRunning)
                throw new InvalidOperationException("DataLoader is already running.");

            // Create shared queue with configured size
            _queue = new SharedQueue<T>(_config.QueueSize, _cancellationTokenSource.Token);

            // Initialize worker pool
            _workerPool = new WorkerPool(_config.NumWorkers);

            // Create prefetch strategy
            _prefetchStrategy = new SimplePrefetchStrategy<T>(_queue, _config.PrefetchCount, _cancellationTokenSource.Token);

            // Start worker pool
            _workerPool.Start();

            // Start batch preparation in background
            _ = Task.Run(() => PrepareBatchesAsync(_cancellationTokenSource.Token));

            // Start prefetching
            _ = _prefetchStrategy.PrefetchAsync(_config.PrefetchCount, _cancellationTokenSource.Token);

            _isRunning = true;
        }
    }

    /// <summary>
    /// Stops the data loading process gracefully.
    /// </summary>
    public void Stop()
    {
        lock (_lock)
        {
            if (!_isRunning)
                return;

            // Signal workers to stop
            _cancellationTokenSource.Cancel();

            // Stop prefetching
            _prefetchStrategy?.Dispose();

            // Mark queue as complete
            _queue?.CompleteAdding();

            // Wait for workers to complete (with timeout)
            _workerPool?.Stop();

            // Cleanup
            _workerPool?.Dispose();
            _queue?.Dispose();

            _isRunning = false;
        }
    }

    /// <summary>
    /// Resets the data loader, clearing internal state and preparing for a new iteration.
    /// </summary>
    public void Reset()
    {
        lock (_lock)
        {
            // Stop if running
            if (_isRunning)
            {
                Stop();
            }

            // Clear internal state
            _batchesLoaded = 0;
            _prefetchStrategy?.Reset();

            // Re-initialize indices
            _indices = new int[_dataset.Length];
            for (int i = 0; i < _indices.Length; i++)
            {
                _indices[i] = i;
            }

            // Shuffle if configured
            if (_config.Shuffle)
            {
                ShuffleIndices();
            }

            // Reset batch iterator
            _batchIndexEnumerator = null;
        }
    }

    /// <summary>
    /// Disposes of all resources used by the data loader.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
            return;

        Stop();
        _cancellationTokenSource.Dispose();
        _disposed = true;
    }

    /// <summary>
    /// Returns a synchronous enumerator for batches.
    /// </summary>
    /// <returns>An enumerator for iterating through batches.</returns>
    public IEnumerator<T> GetEnumerator()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(AdvancedDataLoader<T>));

        if (!_isRunning)
            throw new InvalidOperationException("DataLoader is not started. Call Start() before iterating.");

        return new DataLoaderEnumerator(this, _cancellationTokenSource.Token);
    }

    /// <summary>
    /// Returns a non-generic enumerator for batches (required by IEnumerable interface).
    /// </summary>
    /// <returns>A non-generic enumerator for iterating through batches.</returns>
    System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
    {
        return GetEnumerator();
    }

    /// <summary>
    /// Returns an asynchronous enumerator for batches.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token for async iteration.</param>
    /// <returns>An async enumerator for iterating through batches.</returns>
    public async IAsyncEnumerator<T> GetAsyncEnumerator(CancellationToken cancellationToken = default)
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(AdvancedDataLoader<T>));

        if (!_isRunning)
            throw new InvalidOperationException("DataLoader is not started. Call Start() before iterating.");

        var combinedToken = CancellationTokenSource.CreateLinkedTokenSource(
            _cancellationTokenSource.Token, cancellationToken).Token;

        await using var enumerator = new AsyncDataLoaderEnumerator(this, combinedToken);
        while (await enumerator.MoveNextAsync())
        {
            yield return enumerator.Current;
        }
    }

    /// <summary>
    /// Gets statistics about data loading performance.
    /// </summary>
    /// <returns>DataLoaderStatistics instance with current metrics.</returns>
    public DataLoaderStatistics GetStatistics()
    {
        var queueStats = _queue?.GetStatistics();
        var prefetchStats = _prefetchStrategy?.GetStatistics();

        return new DataLoaderStatistics(
            batchesLoaded: _batchesLoaded,
            totalSamples: _dataset.Length,
            averageBatchTimeMs: 0, // Can be enhanced with timing
            throughputSamplesPerSecond: 0, // Can be enhanced with timing
            queueStatistics: queueStats,
            prefetchStatistics: prefetchStats);
    }

    private void ShuffleIndices()
    {
        // Fisher-Yates shuffle algorithm
        for (int i = _indices.Length - 1; i > 0; i--)
        {
            int j = _random.Next(i + 1);
            (_indices[i], _indices[j]) = (_indices[j], _indices[i]);
        }
    }

    private IEnumerable<int[]> GenerateBatchIndices()
    {
        for (int i = 0; i < _dataset.Length; i += _config.BatchSize)
        {
            int batchSize = Math.Min(_config.BatchSize, _dataset.Length - i);
            int[] batchIndices = new int[batchSize];
            Array.Copy(_indices, i, batchIndices, 0, batchSize);
            yield return batchIndices;
        }
    }

    private async Task PrepareBatchesAsync(CancellationToken cancellationToken)
    {
        _batchIndexEnumerator = GenerateBatchIndices().GetEnumerator();

        try
        {
            while (_batchIndexEnumerator.MoveNext() && !cancellationToken.IsCancellationRequested)
            {
                int[] batchIndices = _batchIndexEnumerator.Current;

                // Submit batch preparation to worker pool
                _workerPool!.SubmitTask(() => CreateBatch(batchIndices));

                // Wait a bit to avoid overwhelming the queue
                await Task.Delay(10, cancellationToken);
            }

            // Signal that all batches have been prepared
            _queue?.CompleteAdding();
        }
        catch (OperationCanceledException)
        {
            // Graceful shutdown
        }
    }

    private T CreateBatch(int[] indices)
    {
        // In a real implementation, this would create a proper batch structure
        // For now, we'll just fetch the items and return the first one as a placeholder
        // This will be enhanced when tensor structures are available

        if (indices.Length == 0)
            throw new ArgumentException("Batch indices array is empty.", nameof(indices));

        // Fetch all items
        var items = new T[indices.Length];
        for (int i = 0; i < indices.Length; i++)
        {
            items[i] = _dataset.GetItem(indices[i]);
        }

        // For now, return the first item as a simple batch
        // TODO: Implement proper batch assembly when tensor structures are available
        return items[0];
    }

    private T GetNextBatch(CancellationToken cancellationToken)
    {
        if (_prefetchStrategy == null)
            throw new InvalidOperationException("Prefetch strategy not initialized.");

        var batch = _prefetchStrategy.GetNextAsync(cancellationToken).GetAwaiter().GetResult();
        Interlocked.Increment(ref _batchesLoaded);
        return batch;
    }

    private class DataLoaderEnumerator : IEnumerator<T>
    {
        private readonly AdvancedDataLoader<T> _dataLoader;
        private readonly CancellationToken _cancellationToken;
        private T? _current;
        private bool _disposed;

        public DataLoaderEnumerator(AdvancedDataLoader<T> dataLoader, CancellationToken cancellationToken)
        {
            _dataLoader = dataLoader;
            _cancellationToken = cancellationToken;
        }

        public T Current => _current!;

        object System.Collections.IEnumerator.Current => Current!;

        public bool MoveNext()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(DataLoaderEnumerator));

            if (_cancellationToken.IsCancellationRequested)
                return false;

            try
            {
                _current = _dataLoader.GetNextBatch(_cancellationToken);
                return true;
            }
            catch (OperationCanceledException)
            {
                return false;
            }
            catch (Exception)
            {
                return false;
            }
        }

        public void Reset()
        {
            throw new NotSupportedException("Reset is not supported. Use DataLoader.Reset() instead.");
        }

        public void Dispose()
        {
            _disposed = true;
        }
    }

    private class AsyncDataLoaderEnumerator : IAsyncEnumerator<T>
    {
        private readonly AdvancedDataLoader<T> _dataLoader;
        private readonly CancellationToken _cancellationToken;
        private T? _current;
        private bool _disposed;

        public AsyncDataLoaderEnumerator(AdvancedDataLoader<T> dataLoader, CancellationToken cancellationToken)
        {
            _dataLoader = dataLoader;
            _cancellationToken = cancellationToken;
        }

        public T Current => _current!;

        public async ValueTask<bool> MoveNextAsync()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(AsyncDataLoaderEnumerator));

            if (_cancellationToken.IsCancellationRequested)
                return false;

            try
            {
                _current = await _dataLoader._prefetchStrategy!.GetNextAsync(_cancellationToken);
                Interlocked.Increment(ref _dataLoader._batchesLoaded);
                return true;
            }
            catch (OperationCanceledException)
            {
                return false;
            }
            catch (Exception)
            {
                return false;
            }
        }

        public ValueTask DisposeAsync()
        {
            _disposed = true;
            return default;
        }
    }
}
