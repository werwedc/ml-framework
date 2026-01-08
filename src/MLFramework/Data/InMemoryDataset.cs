using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace MLFramework.Data;
    /// <summary>
    /// Generic convenience class providing factory methods for easy dataset creation.
    /// </summary>
    public static class InMemoryDataset
    {
        /// <summary>
        /// Creates a dataset from an enumerable, materializing it into the most appropriate implementation.
        /// </summary>
        /// <typeparam name="T">The type of items in the dataset</typeparam>
        /// <param name="items">The enumerable of items to create a dataset from.</param>
        /// <returns>
        /// An ArrayDataset{T} if the input is an array,
        /// or a ListDataset{T} if the input is a list or other collection.
        /// </returns>
        /// <exception cref="ArgumentNullException">Thrown when items is null.</exception>
        public static IDataset<T> FromEnumerable<T>(IEnumerable<T> items)
        {
            if (items == null)
                throw new ArgumentNullException(nameof(items));

            // Check if it's already an array to avoid unnecessary copying
            if (items is T[] array)
            {
                return new ArrayDataset<T>(array);
            }

            // Check if it's a list to avoid copying
            if (items is IList<T> list)
            {
                return new ListDataset<T>(list);
            }

            // Materialize the enumerable into an array for other types
            return new ArrayDataset<T>(items.ToArray());
        }

        /// <summary>
        /// Creates a dataset from a list.
        /// </summary>
        /// <typeparam name="T">The type of items in the dataset</typeparam>
        /// <param name="items">The list of items to create a dataset from.</param>
        /// <returns>A ListDataset{T} wrapping the provided list.</returns>
        /// <exception cref="ArgumentNullException">Thrown when items is null.</exception>
        public static ListDataset<T> FromList<T>(IList<T> items)
        {
            return new ListDataset<T>(items);
        }

        /// <summary>
        /// Creates a dataset from an array.
        /// </summary>
        /// <typeparam name="T">The type of items in the dataset</typeparam>
        /// <param name="items">The array of items to create a dataset from.</param>
        /// <returns>An ArrayDataset{T} wrapping the provided array.</returns>
        /// <exception cref="ArgumentNullException">Thrown when items is null.</exception>
        public static ArrayDataset<T> FromArray<T>(T[] items)
        {
            return new ArrayDataset<T>(items);
        }
    }
