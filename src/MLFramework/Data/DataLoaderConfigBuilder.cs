namespace MLFramework.Data;

/// <summary>
/// Fluent builder for creating DataLoaderConfig instances.
/// Provides a convenient way to configure data loader parameters with method chaining.
/// </summary>
public sealed class DataLoaderConfigBuilder
{
    private int _numWorkers = 4;
    private int _batchSize = 32;
    private int _prefetchCount = 2;
    private int _queueSize = 10;
    private bool _shuffle = true;
    private int _seed = 42;
    private bool _pinMemory = true;

    /// <summary>
    /// Initializes a new instance of the DataLoaderConfigBuilder with default values.
    /// </summary>
    public DataLoaderConfigBuilder() { }

    /// <summary>
    /// Sets the number of parallel workers for data loading.
    /// </summary>
    /// <param name="numWorkers">Number of workers (must be >= 0, defaults to processor count if 0).</param>
    /// <returns>The builder instance for method chaining.</returns>
    public DataLoaderConfigBuilder WithNumWorkers(int numWorkers)
    {
        _numWorkers = numWorkers;
        return this;
    }

    /// <summary>
    /// Sets the number of samples per batch.
    /// </summary>
    /// <param name="batchSize">Samples per batch (must be > 0).</param>
    /// <returns>The builder instance for method chaining.</returns>
    public DataLoaderConfigBuilder WithBatchSize(int batchSize)
    {
        _batchSize = batchSize;
        return this;
    }

    /// <summary>
    /// Sets the number of batches to prefetch in the background.
    /// </summary>
    /// <param name="prefetchCount">Number of batches to prefetch (must be >= 0).</param>
    /// <returns>The builder instance for method chaining.</returns>
    public DataLoaderConfigBuilder WithPrefetchCount(int prefetchCount)
    {
        _prefetchCount = prefetchCount;
        return this;
    }

    /// <summary>
    /// Sets the maximum number of batches in the internal queue.
    /// </summary>
    /// <param name="queueSize">Maximum batches in queue (must be >= prefetchCount + 1).</param>
    /// <returns>The builder instance for method chaining.</returns>
    public DataLoaderConfigBuilder WithQueueSize(int queueSize)
    {
        _queueSize = queueSize;
        return this;
    }

    /// <summary>
    /// Sets whether to randomize the data order.
    /// </summary>
    /// <param name="shuffle">True to shuffle data, false to keep original order.</param>
    /// <returns>The builder instance for method chaining.</returns>
    public DataLoaderConfigBuilder WithShuffle(bool shuffle)
    {
        _shuffle = shuffle;
        return this;
    }

    /// <summary>
    /// Sets the random seed for reproducibility.
    /// </summary>
    /// <param name="seed">Random seed value.</param>
    /// <returns>The builder instance for method chaining.</returns>
    public DataLoaderConfigBuilder WithSeed(int seed)
    {
        _seed = seed;
        return this;
    }

    /// <summary>
    /// Sets whether to use pinned memory for faster GPU transfers.
    /// </summary>
    /// <param name="pinMemory">True to use pinned memory, false otherwise.</param>
    /// <returns>The builder instance for method chaining.</returns>
    public DataLoaderConfigBuilder WithPinMemory(bool pinMemory)
    {
        _pinMemory = pinMemory;
        return this;
    }

    /// <summary>
    /// Builds and returns a DataLoaderConfig instance with the configured values.
    /// </summary>
    /// <returns>A new DataLoaderConfig instance.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when configured parameters are invalid.</exception>
    public DataLoaderConfig Build()
    {
        return new DataLoaderConfig(
            numWorkers: _numWorkers,
            batchSize: _batchSize,
            prefetchCount: _prefetchCount,
            queueSize: _queueSize,
            shuffle: _shuffle,
            seed: _seed,
            pinMemory: _pinMemory);
    }
}
