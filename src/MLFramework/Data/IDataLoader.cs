namespace MLFramework.Data;

/// <summary>
/// High-level interface for data loading that supports both synchronous and asynchronous iteration.
/// </summary>
/// <typeparam name="T">The type of data items being loaded.</typeparam>
public interface IDataLoader<T> : IDisposable, IEnumerable<T>
{
    /// <summary>
    /// Starts the data loading process, initializing workers and prefetching.
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown when the loader is already running.</exception>
    void Start();

    /// <summary>
    /// Stops the data loading process gracefully, signaling workers to complete.
    /// </summary>
    void Stop();

    /// <summary>
    /// Resets the data loader, clearing internal state and preparing for a new iteration.
    /// </summary>
    void Reset();

    /// <summary>
    /// Gets whether the data loader is currently running.
    /// </summary>
    bool IsRunning { get; }

    /// <summary>
    /// Gets the configuration used by this data loader.
    /// </summary>
    DataLoaderConfig Config { get; }
}
