using System;

namespace MLFramework.Data;

/// <summary>
/// Abstract base class for datasets that provide random access to samples via index.
/// Suitable for datasets where all samples can be efficiently accessed by their index.
/// Thread-safe for concurrent reads.
/// </summary>
/// <typeparam name="T">The type of data items in the dataset.</typeparam>
public abstract class MapStyleDataset<T> : IDataset<T>
{
    /// <summary>
    /// Gets item at the specified index.
    /// Supports negative indexing.
    /// </summary>
    /// <param name="index">The index of the item to retrieve.</param>
    /// <returns>The item at the specified index.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when index is out of bounds.</exception>
    public abstract T GetItem(int index);

    /// <summary>
    /// Gets the total number of items in the dataset.
    /// </summary>
    public abstract int Count { get; }

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
    /// Called when dataset is created. Override to perform initialization.
    /// </summary>
    protected virtual void OnDatasetCreated()
    {
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

    /// <summary>
    /// Validates that the given index is within the valid range.
    /// Kept for backward compatibility.
    /// </summary>
    /// <param name="index">The index to validate.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when index is out of range.</exception>
    protected virtual void ValidateIndex(int index)
    {
        if (index < 0 || index >= Count)
        {
            throw new ArgumentOutOfRangeException(nameof(index), $"Index {index} is out of range for dataset of size {Count}.");
        }
    }

    /// <summary>
    /// Gets the item at the specified index with validation.
    /// Kept for backward compatibility.
    /// </summary>
    /// <param name="index">The index of the item to retrieve.</param>
    /// <returns>The item at the specified index.</returns>
    public T GetValidatedItem(int index)
    {
        ValidateIndex(index);
        return GetItem(index);
    }

    /// <summary>
    /// Gets the total number of items in the dataset.
    /// Kept for backward compatibility.
    /// </summary>
    public int Length => Count;
}
