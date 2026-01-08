using System.Threading.Tasks;

namespace MLFramework.Data
{
    /// <summary>
    /// Defines the contract for prefetching implementations.
    /// Prefetching prepares future batches while current data is being processed,
    /// reducing wait times and improving throughput.
    /// </summary>
    /// <typeparam name="T">The type of items to prefetch.</typeparam>
    public interface IPrefetchStrategy<T>
    {
        /// <summary>
        /// Gets the next prefetched item asynchronously.
        /// Returns immediately if an item is available in the prefetch buffer,
        /// otherwise waits for the next item from the source.
        /// </summary>
        /// <param name="cancellationToken">Optional cancellation token.</param>
        /// <returns>The next prefetched item.</returns>
        /// <exception cref="OperationCanceledException">Thrown when cancelled.</exception>
        Task<T> GetNextAsync(CancellationToken cancellationToken);

        /// <summary>
        /// Starts a background task to prefetch the specified number of items.
        /// </summary>
        /// <param name="count">Number of items to prefetch.</param>
        /// <param name="cancellationToken">Optional cancellation token.</param>
        /// <returns>A task representing the prefetch operation.</returns>
        Task PrefetchAsync(int count, CancellationToken cancellationToken);

        /// <summary>
        /// Resets the prefetch strategy, clearing the internal buffer.
        /// </summary>
        void Reset();

        /// <summary>
        /// Gets whether prefetched items are available in the buffer.
        /// </summary>
        bool IsAvailable { get; }
    }
}
