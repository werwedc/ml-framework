using System;

namespace MLFramework.Data.Worker
{
    /// <summary>
    /// Interface for a prefetch queue that proactively prepares data batches before they're needed.
    /// </summary>
    /// <typeparam name="T">The type of items being prefetched.</typeparam>
    public interface IPrefetchQueue<T> : IDisposable
    {
        /// <summary>
        /// Starts the background prefetching task.
        /// </summary>
        void Start();

        /// <summary>
        /// Stops the background prefetching task.
        /// </summary>
        void Stop();

        /// <summary>
        /// Gets a value indicating whether the prefetch queue is running.
        /// </summary>
        bool IsRunning { get; }

        /// <summary>
        /// Gets the configured prefetch count (number of batches to prepare ahead).
        /// </summary>
        int PrefetchCount { get; }

        /// <summary>
        /// Gets the number of currently available batches in the prefetch buffer.
        /// </summary>
        int AvailableBatches { get; }

        /// <summary>
        /// Gets the next batch from the prefetch queue. Throws if queue is not running or empty.
        /// </summary>
        /// <returns>The next batch.</returns>
        T GetNext();

        /// <summary>
        /// Attempts to get the next batch from the prefetch queue without throwing.
        /// </summary>
        /// <param name="batch">When this method returns, contains the next batch if successful, or default if not.</param>
        /// <returns>True if a batch was successfully retrieved; otherwise, false.</returns>
        bool TryGetNext(out T batch);
    }
}
