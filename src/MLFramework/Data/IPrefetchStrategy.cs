namespace MLFramework.Data;

/// <summary>
/// Defines the contract for prefetching strategies that prepare data batches in the background
/// to ensure zero GPU idle time during training.
/// </summary>
/// <typeparam name="T">The type of items being prefetched.</typeparam>
public interface IPrefetchStrategy<T>
{
    /// <summary>
    /// Gets the next item asynchronously.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token for graceful shutdown.</param>
    /// <returns>The next item, either from the prefetch buffer or by waiting for the source.</returns>
    Task<T> GetNextAsync(CancellationToken cancellationToken);

    /// <summary>
    /// Prefetches the specified number of items in the background.
    /// </summary>
    /// <param name="count">The number of items to prefetch.</param>
    /// <param name="cancellationToken">Cancellation token for graceful shutdown.</param>
    /// <returns>A task representing the prefetch operation.</returns>
    Task PrefetchAsync(int count, CancellationToken cancellationToken);

    /// <summary>
    /// Resets the prefetch strategy, clearing internal state.
    /// </summary>
    void Reset();

    /// <summary>
    /// Gets whether prefetched items are available.
    /// </summary>
    bool IsAvailable { get; }
}
