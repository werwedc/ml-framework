using System;
using System.Collections.Generic;

namespace MLFramework.Data;
    /// <summary>
    /// Simple in-memory dataset wrapper around IList{T}.
    /// </summary>
    /// <typeparam name="T">The type of items in the dataset</typeparam>
    public class ListDataset<T> : Dataset<T>
    {
        private readonly IList<T> _items;

        /// <summary>
        /// Gets the total number of items in the dataset.
        /// </summary>
        public override int Count => _items.Count;

        /// <summary>
        /// Initializes a new instance of the ListDataset class.
        /// </summary>
        /// <param name="items">The collection of items to wrap.</param>
        /// <exception cref="ArgumentNullException">Thrown when items is null.</exception>
        public ListDataset(IList<T> items)
        {
            _items = items ?? throw new ArgumentNullException(nameof(items));
        }

        /// <summary>
        /// Retrieves a single item by index.
        /// Thread-safe for concurrent reads.
        /// </summary>
        /// <param name="index">The index of the item to retrieve. Supports negative indexing.</param>
        /// <returns>The item at the specified index.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when index is out of bounds.</exception>
        public override T GetItem(int index)
        {
            int normalizedIndex = NormalizeIndex(index);
            return _items[normalizedIndex];
        }

        /// <summary>
        /// Retrieves multiple items at once for efficiency.
        /// Thread-safe for concurrent reads.
        /// </summary>
        /// <param name="indices">The indices of items to retrieve. Supports negative indexing.</param>
        /// <returns>An array of items at the specified indices.</returns>
        public override T[] GetBatch(int[] indices)
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
    }
