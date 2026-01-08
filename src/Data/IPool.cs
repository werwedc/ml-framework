using System;

namespace MLFramework.Data
{
    /// <summary>
    /// Generic interface for object pools.
    /// </summary>
    /// <typeparam name="T">The type of objects managed by the pool.</typeparam>
    public interface IPool<T> : IDisposable
    {
        /// <summary>
        /// Rents an object from the pool.
        /// </summary>
        /// <returns>An object from the pool or a newly created object if the pool is empty.</returns>
        T Rent();

        /// <summary>
        /// Returns an object to the pool for reuse.
        /// </summary>
        /// <param name="item">The object to return to the pool.</param>
        void Return(T item);

        /// <summary>
        /// Gets the number of objects currently available for rent in the pool.
        /// </summary>
        int AvailableCount { get; }

        /// <summary>
        /// Gets the total number of objects created by the pool.
        /// </summary>
        int TotalCount { get; }

        /// <summary>
        /// Clears the pool, removing all available items.
        /// </summary>
        /// <remarks>
        /// Does not dispose of items; caller is responsible for disposal.
        /// Useful for cleanup or memory pressure response.
        /// </remarks>
        void Clear();
    }
}
