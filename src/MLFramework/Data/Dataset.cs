using System;

namespace MLFramework.Data;
    /// <summary>
    /// Abstract base class providing common functionality for dataset implementations.
    /// </summary>
    /// <typeparam name="T">The type of items in the dataset</typeparam>
    public abstract class Dataset<T> : IDataset<T>
    {
        /// <summary>
        /// Gets the total number of items in the dataset.
        /// </summary>
        public abstract int Count { get; }

        /// <summary>
        /// Gets the total number of items in the dataset.
        /// Kept for backward compatibility - returns Count.
        /// </summary>
        public virtual int Length => Count;

        /// <summary>
        /// Retrieves a single item by index.
        /// </summary>
        /// <param name="index">The index of the item to retrieve.</param>
        /// <returns>The item at the specified index.</returns>
        public abstract T GetItem(int index);

        /// <summary>
        /// Retrieves multiple items at once for efficiency.
        /// Default implementation calls GetItem for each index.
        /// </summary>
        /// <param name="indices">The indices of items to retrieve.</param>
        /// <returns>An array of items at the specified indices.</returns>
        public virtual T[] GetBatch(int[] indices)
        {
            if (indices == null)
                throw new ArgumentNullException(nameof(indices));

            if (indices.Length == 0)
                return Array.Empty<T>();

            var result = new T[indices.Length];
            for (int i = 0; i < indices.Length; i++)
            {
                result[i] = GetItem(indices[i]);
            }
            return result;
        }

        /// <summary>
        /// Validates and normalizes an index, handling negative indexing.
        /// </summary>
        /// <param name="index">The index to validate and normalize.</param>
        /// <returns>The normalized non-negative index.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when index is out of bounds.</exception>
        protected virtual int NormalizeIndex(int index)
        {
            // Handle negative indexing (e.g., -1 returns last item)
            if (index < 0)
            {
                index = Count + index;
                if (index < 0)
                    throw new ArgumentOutOfRangeException(nameof(index), $"Index {index - Count} is out of range for dataset of size {Count}");
            }

            if (index >= Count)
                throw new ArgumentOutOfRangeException(nameof(index), $"Index {index} is out of range for dataset of size {Count}");

            return index;
        }
    }
