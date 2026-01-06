using System;

namespace MLFramework.Data.Worker
{
    /// <summary>
    /// Interface for a shared memory queue for efficient communication between workers and main process.
    /// </summary>
    /// <typeparam name="T">The type of items in the queue.</typeparam>
    public interface ISharedMemoryQueue<T> : IDisposable
    {
        /// <summary>
        /// Adds an item to the queue. Blocks if queue is bounded and full.
        /// </summary>
        /// <param name="item">The item to add.</param>
        void Enqueue(T item);

        /// <summary>
        /// Removes and returns the next item from the queue. Blocks if queue is empty.
        /// </summary>
        /// <returns>The next item in the queue.</returns>
        T Dequeue();

        /// <summary>
        /// Attempts to remove and return the next item from the queue without blocking.
        /// </summary>
        /// <param name="item">When this method returns, contains the next item if the operation succeeded, or default(T) if not.</param>
        /// <returns>True if an item was successfully removed; otherwise, false.</returns>
        bool TryDequeue(out T item);

        /// <summary>
        /// Gets the number of items in the queue.
        /// </summary>
        int Count { get; }

        /// <summary>
        /// Gets a value indicating whether the queue is empty.
        /// </summary>
        bool IsEmpty { get; }

        /// <summary>
        /// Removes all items from the queue.
        /// </summary>
        void Clear();
    }
}
