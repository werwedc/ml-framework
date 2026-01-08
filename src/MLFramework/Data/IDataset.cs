namespace MLFramework.Data;

/// <summary>
/// Interface for datasets that provide random access to samples.
/// </summary>
/// <typeparam name="T">The type of data items in the dataset.</typeparam>
public interface IDataset<T>
{
    /// <summary>
    /// Gets the item at the specified index.
    /// Supports negative indexing (e.g., -1 returns last item).
    /// </summary>
    /// <param name="index">The index of the item to retrieve.</param>
    /// <returns>The item at the specified index.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when index is out of bounds.</exception>
    T GetItem(int index);

    /// <summary>
    /// Gets the total number of items in the dataset.
    /// </summary>
    int Count { get; }

    /// <summary>
    /// Gets the total number of items in the dataset.
    /// Kept for backward compatibility - returns Count.
    /// </summary>
    int Length { get; }

    /// <summary>
    /// Retrieves multiple items at once for efficiency.
    /// Supports negative indexing.
    /// </summary>
    /// <param name="indices">The indices of items to retrieve.</param>
    /// <returns>An array of items at the specified indices.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any index is out of bounds.</exception>
    T[] GetBatch(int[] indices);
}
