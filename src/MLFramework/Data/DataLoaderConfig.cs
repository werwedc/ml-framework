namespace MLFramework.Data;

/// <summary>
/// Immutable configuration class for DataLoader parameters with validation and sensible defaults.
/// </summary>
public sealed class DataLoaderConfig
{
    /// <summary>
    /// Gets the number of parallel workers for data loading.
    /// </summary>
    public int NumWorkers { get; }

    /// <summary>
    /// Gets the number of samples per batch.
    /// </summary>
    public int BatchSize { get; }

    /// <summary>
    /// Gets the number of batches to prefetch in the background.
    /// </summary>
    public int PrefetchCount { get; }

    /// <summary>
    /// Gets the maximum number of batches in the internal queue.
    /// </summary>
    public int QueueSize { get; }

    /// <summary>
    /// Gets whether to randomize the data order.
    /// </summary>
    public bool Shuffle { get; }

    /// <summary>
    /// Gets the random seed for reproducibility.
    /// </summary>
    public int Seed { get; }

    /// <summary>
    /// Gets whether to use pinned memory for faster GPU transfers.
    /// </summary>
    public bool PinMemory { get; }

    /// <summary>
    /// Gets the error policy that determines how errors are handled.
    /// </summary>
    public ErrorPolicy ErrorPolicy { get; }

    /// <summary>
    /// Gets the maximum number of retry attempts for failed workers.
    /// </summary>
    public int MaxWorkerRetries { get; }

    /// <summary>
    /// Gets the timeout for worker operations before considering them stalled.
    /// </summary>
    public TimeSpan WorkerTimeout { get; }

    /// <summary>
    /// Gets whether to log errors to the console.
    /// </summary>
    public bool LogErrors { get; }

    /// <summary>
    /// Initializes a new instance of the DataLoaderConfig class with validation.
    /// </summary>
    /// <param name="numWorkers">Number of parallel workers (must be >= 0, defaults to processor count if 0).</param>
    /// <param name="batchSize">Samples per batch (must be > 0).</param>
    /// <param name="prefetchCount">Number of batches to prefetch (must be >= 0).</param>
    /// <param name="queueSize">Maximum batches in queue (must be >= prefetchCount + 1).</param>
    /// <param name="shuffle">Whether to randomize data order.</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    /// <param name="pinMemory">Whether to use pinned memory.</param>
    /// <param name="errorPolicy">Error handling policy for worker failures.</param>
    /// <param name="maxWorkerRetries">Maximum number of retry attempts per worker.</param>
    /// <param name="workerTimeout">Timeout for worker operations before considering them stalled.</param>
    /// <param name="logErrors">Whether to log errors to the console.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when parameters are invalid.</exception>
    public DataLoaderConfig(
        int numWorkers = 4,
        int batchSize = 32,
        int prefetchCount = 2,
        int queueSize = 10,
        bool shuffle = true,
        int seed = 42,
        bool pinMemory = true,
        ErrorPolicy errorPolicy = ErrorPolicy.Continue,
        int maxWorkerRetries = 3,
        TimeSpan? workerTimeout = null,
        bool logErrors = true)
    {
        if (numWorkers < 0)
            throw new ArgumentOutOfRangeException(nameof(numWorkers), numWorkers, "NumWorkers must be >= 0.");

        if (batchSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(batchSize), batchSize, "BatchSize must be > 0.");

        if (prefetchCount < 0)
            throw new ArgumentOutOfRangeException(nameof(prefetchCount), prefetchCount, "PrefetchCount must be >= 0.");

        if (queueSize < prefetchCount + 1)
            throw new ArgumentOutOfRangeException(nameof(queueSize), queueSize,
                $"QueueSize must be >= PrefetchCount + 1 (>= {prefetchCount + 1}).");

        if (maxWorkerRetries < 0)
            throw new ArgumentOutOfRangeException(nameof(maxWorkerRetries), maxWorkerRetries, "MaxWorkerRetries must be >= 0.");

        var timeout = workerTimeout ?? TimeSpan.FromSeconds(30);
        if (timeout <= TimeSpan.Zero)
            throw new ArgumentOutOfRangeException(nameof(workerTimeout), timeout, "WorkerTimeout must be > TimeSpan.Zero.");

        // Default to processor count if numWorkers is 0
        NumWorkers = numWorkers == 0 ? Environment.ProcessorCount : numWorkers;
        BatchSize = batchSize;
        PrefetchCount = prefetchCount;
        QueueSize = queueSize;
        Shuffle = shuffle;
        Seed = seed;
        PinMemory = pinMemory;
        ErrorPolicy = errorPolicy;
        MaxWorkerRetries = maxWorkerRetries;
        WorkerTimeout = timeout;
        LogErrors = logErrors;
    }

    /// <summary>
    /// Creates a copy of this configuration with the specified numWorkers.
    /// </summary>
    public DataLoaderConfig WithNumWorkers(int numWorkers) =>
        new DataLoaderConfig(numWorkers, BatchSize, PrefetchCount, QueueSize, Shuffle, Seed, PinMemory, ErrorPolicy, MaxWorkerRetries, WorkerTimeout, LogErrors);

    /// <summary>
    /// Creates a copy of this configuration with the specified batchSize.
    /// </summary>
    public DataLoaderConfig WithBatchSize(int batchSize) =>
        new DataLoaderConfig(NumWorkers, batchSize, PrefetchCount, QueueSize, Shuffle, Seed, PinMemory, ErrorPolicy, MaxWorkerRetries, WorkerTimeout, LogErrors);

    /// <summary>
    /// Creates a copy of this configuration with the specified prefetchCount.
    /// </summary>
    public DataLoaderConfig WithPrefetchCount(int prefetchCount) =>
        new DataLoaderConfig(NumWorkers, BatchSize, prefetchCount, QueueSize, Shuffle, Seed, PinMemory, ErrorPolicy, MaxWorkerRetries, WorkerTimeout, LogErrors);

    /// <summary>
    /// Creates a copy of this configuration with the specified queueSize.
    /// </summary>
    public DataLoaderConfig WithQueueSize(int queueSize) =>
        new DataLoaderConfig(NumWorkers, BatchSize, PrefetchCount, queueSize, Shuffle, Seed, PinMemory, ErrorPolicy, MaxWorkerRetries, WorkerTimeout, LogErrors);

    /// <summary>
    /// Creates a copy of this configuration with the specified shuffle flag.
    /// </summary>
    public DataLoaderConfig WithShuffle(bool shuffle) =>
        new DataLoaderConfig(NumWorkers, BatchSize, PrefetchCount, QueueSize, shuffle, Seed, PinMemory, ErrorPolicy, MaxWorkerRetries, WorkerTimeout, LogErrors);

    /// <summary>
    /// Creates a copy of this configuration with the specified seed.
    /// </summary>
    public DataLoaderConfig WithSeed(int seed) =>
        new DataLoaderConfig(NumWorkers, BatchSize, PrefetchCount, QueueSize, Shuffle, seed, PinMemory, ErrorPolicy, MaxWorkerRetries, WorkerTimeout, LogErrors);

    /// <summary>
    /// Creates a copy of this configuration with the specified pinMemory flag.
    /// </summary>
    public DataLoaderConfig WithPinMemory(bool pinMemory) =>
        new DataLoaderConfig(NumWorkers, BatchSize, PrefetchCount, QueueSize, Shuffle, Seed, pinMemory, ErrorPolicy, MaxWorkerRetries, WorkerTimeout, LogErrors);

    /// <summary>
    /// Creates a copy of this configuration with the specified error policy.
    /// </summary>
    public DataLoaderConfig WithErrorPolicy(ErrorPolicy errorPolicy) =>
        new DataLoaderConfig(NumWorkers, BatchSize, PrefetchCount, QueueSize, Shuffle, Seed, PinMemory, errorPolicy, MaxWorkerRetries, WorkerTimeout, LogErrors);

    /// <summary>
    /// Creates a copy of this configuration with the specified max worker retries.
    /// </summary>
    public DataLoaderConfig WithMaxWorkerRetries(int maxWorkerRetries) =>
        new DataLoaderConfig(NumWorkers, BatchSize, PrefetchCount, QueueSize, Shuffle, Seed, PinMemory, ErrorPolicy, maxWorkerRetries, WorkerTimeout, LogErrors);

    /// <summary>
    /// Creates a copy of this configuration with the specified worker timeout.
    /// </summary>
    public DataLoaderConfig WithWorkerTimeout(TimeSpan workerTimeout) =>
        new DataLoaderConfig(NumWorkers, BatchSize, PrefetchCount, QueueSize, Shuffle, Seed, PinMemory, ErrorPolicy, MaxWorkerRetries, workerTimeout, LogErrors);

    /// <summary>
    /// Creates a copy of this configuration with the specified log errors flag.
    /// </summary>
    public DataLoaderConfig WithLogErrors(bool logErrors) =>
        new DataLoaderConfig(NumWorkers, BatchSize, PrefetchCount, QueueSize, Shuffle, Seed, PinMemory, ErrorPolicy, MaxWorkerRetries, WorkerTimeout, logErrors);

    /// <summary>
    /// Creates a clone of this configuration.
    /// </summary>
    public DataLoaderConfig Clone() =>
        new DataLoaderConfig(NumWorkers, BatchSize, PrefetchCount, QueueSize, Shuffle, Seed, PinMemory, ErrorPolicy, MaxWorkerRetries, WorkerTimeout, LogErrors);

    /// <summary>
    /// Returns a human-readable string representation of this configuration.
    /// </summary>
    public override string ToString()
    {
        return $"DataLoaderConfig {{ NumWorkers: {NumWorkers}, BatchSize: {BatchSize}, " +
               $"PrefetchCount: {PrefetchCount}, QueueSize: {QueueSize}, Shuffle: {Shuffle}, " +
               $"Seed: {Seed}, PinMemory: {PinMemory}, ErrorPolicy: {ErrorPolicy}, " +
               $"MaxWorkerRetries: {MaxWorkerRetries}, WorkerTimeout: {WorkerTimeout.TotalSeconds}s, " +
               $"LogErrors: {LogErrors} }}";
    }
}

/// <summary>
/// Static factory class providing convenient presets for common DataLoader scenarios.
/// </summary>
public static class DataLoaderConfigPresets
{
    /// <summary>
    /// Creates a configuration optimized for CPU-bound workloads.
    /// Uses more workers and larger prefetch buffers.
    /// </summary>
    public static DataLoaderConfig ForCPUBound() =>
        new DataLoaderConfig(
            numWorkers: Environment.ProcessorCount,
            batchSize: 64,
            prefetchCount: 3,
            queueSize: 15,
            shuffle: true,
            seed: 42,
            pinMemory: false,
            errorPolicy: ErrorPolicy.Continue,
            maxWorkerRetries: 3,
            workerTimeout: TimeSpan.FromSeconds(30),
            logErrors: true);

    /// <summary>
    /// Creates a configuration optimized for GPU-bound workloads.
    /// Uses fewer workers but enables pinned memory for faster transfers.
    /// </summary>
    public static DataLoaderConfig ForGPUBound() =>
        new DataLoaderConfig(
            numWorkers: 2,
            batchSize: 32,
            prefetchCount: 2,
            queueSize: 10,
            shuffle: true,
            seed: 42,
            pinMemory: true,
            errorPolicy: ErrorPolicy.Restart,
            maxWorkerRetries: 3,
            workerTimeout: TimeSpan.FromSeconds(60),
            logErrors: true);

    /// <summary>
    /// Creates a configuration for small datasets.
    /// Uses single-threaded loading with minimal prefetching.
    /// </summary>
    public static DataLoaderConfig ForSmallDataset() =>
        new DataLoaderConfig(
            numWorkers: 1,
            batchSize: 16,
            prefetchCount: 1,
            queueSize: 5,
            shuffle: true,
            seed: 42,
            pinMemory: false,
            errorPolicy: ErrorPolicy.FailFast,
            maxWorkerRetries: 0,
            workerTimeout: TimeSpan.FromSeconds(30),
            logErrors: true);

    /// <summary>
    /// Creates a configuration for large datasets.
    /// Uses multiple workers with large batch sizes and prefetch buffers.
    /// </summary>
    public static DataLoaderConfig ForLargeDataset() =>
        new DataLoaderConfig(
            numWorkers: 4,
            batchSize: 128,
            prefetchCount: 4,
            queueSize: 20,
            shuffle: true,
            seed: 42,
            pinMemory: true,
            errorPolicy: ErrorPolicy.Restart,
            maxWorkerRetries: 5,
            workerTimeout: TimeSpan.FromSeconds(45),
            logErrors: true);
}
