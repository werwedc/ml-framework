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
    /// Initializes a new instance of the DataLoaderConfig class with validation.
    /// </summary>
    /// <param name="numWorkers">Number of parallel workers (must be >= 0, defaults to processor count if 0).</param>
    /// <param name="batchSize">Samples per batch (must be > 0).</param>
    /// <param name="prefetchCount">Number of batches to prefetch (must be >= 0).</param>
    /// <param name="queueSize">Maximum batches in queue (must be >= prefetchCount + 1).</param>
    /// <param name="shuffle">Whether to randomize data order.</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    /// <param name="pinMemory">Whether to use pinned memory.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when parameters are invalid.</exception>
    public DataLoaderConfig(
        int numWorkers = 4,
        int batchSize = 32,
        int prefetchCount = 2,
        int queueSize = 10,
        bool shuffle = true,
        int seed = 42,
        bool pinMemory = true)
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

        // Default to processor count if numWorkers is 0
        NumWorkers = numWorkers == 0 ? Environment.ProcessorCount : numWorkers;
        BatchSize = batchSize;
        PrefetchCount = prefetchCount;
        QueueSize = queueSize;
        Shuffle = shuffle;
        Seed = seed;
        PinMemory = pinMemory;
    }

    /// <summary>
    /// Creates a copy of this configuration with the specified numWorkers.
    /// </summary>
    public DataLoaderConfig WithNumWorkers(int numWorkers) =>
        new DataLoaderConfig(numWorkers, BatchSize, PrefetchCount, QueueSize, Shuffle, Seed, PinMemory);

    /// <summary>
    /// Creates a copy of this configuration with the specified batchSize.
    /// </summary>
    public DataLoaderConfig WithBatchSize(int batchSize) =>
        new DataLoaderConfig(NumWorkers, batchSize, PrefetchCount, QueueSize, Shuffle, Seed, PinMemory);

    /// <summary>
    /// Creates a copy of this configuration with the specified prefetchCount.
    /// </summary>
    public DataLoaderConfig WithPrefetchCount(int prefetchCount) =>
        new DataLoaderConfig(NumWorkers, BatchSize, prefetchCount, QueueSize, Shuffle, Seed, PinMemory);

    /// <summary>
    /// Creates a copy of this configuration with the specified queueSize.
    /// </summary>
    public DataLoaderConfig WithQueueSize(int queueSize) =>
        new DataLoaderConfig(NumWorkers, BatchSize, PrefetchCount, queueSize, Shuffle, Seed, PinMemory);

    /// <summary>
    /// Creates a copy of this configuration with the specified shuffle flag.
    /// </summary>
    public DataLoaderConfig WithShuffle(bool shuffle) =>
        new DataLoaderConfig(NumWorkers, BatchSize, PrefetchCount, QueueSize, shuffle, Seed, PinMemory);

    /// <summary>
    /// Creates a copy of this configuration with the specified seed.
    /// </summary>
    public DataLoaderConfig WithSeed(int seed) =>
        new DataLoaderConfig(NumWorkers, BatchSize, PrefetchCount, QueueSize, Shuffle, seed, PinMemory);

    /// <summary>
    /// Creates a copy of this configuration with the specified pinMemory flag.
    /// </summary>
    public DataLoaderConfig WithPinMemory(bool pinMemory) =>
        new DataLoaderConfig(NumWorkers, BatchSize, PrefetchCount, QueueSize, Shuffle, Seed, pinMemory);

    /// <summary>
    /// Creates a clone of this configuration.
    /// </summary>
    public DataLoaderConfig Clone() =>
        new DataLoaderConfig(NumWorkers, BatchSize, PrefetchCount, QueueSize, Shuffle, Seed, PinMemory);

    /// <summary>
    /// Returns a human-readable string representation of this configuration.
    /// </summary>
    public override string ToString()
    {
        return $"DataLoaderConfig {{ NumWorkers: {NumWorkers}, BatchSize: {BatchSize}, " +
               $"PrefetchCount: {PrefetchCount}, QueueSize: {QueueSize}, Shuffle: {Shuffle}, " +
               $"Seed: {Seed}, PinMemory: {PinMemory} }}";
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
            pinMemory: false);

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
            pinMemory: true);

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
            pinMemory: false);

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
            pinMemory: true);
}
