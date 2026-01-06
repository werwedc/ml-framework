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
    /// Gets the item at the specified index.
    /// </summary>
    /// <param name="index">The index of the item to retrieve.</param>
    /// <returns>The item at the specified index.</returns>
    /// <exception cref="IndexOutOfRangeException">Thrown when index is out of range.</exception>
    public abstract T GetItem(int index);

    /// <summary>
    /// Gets the total number of items in the dataset.
    /// </summary>
    public abstract int Length { get; }

    /// <summary>
    /// Called when the dataset is created. Override to perform initialization.
    /// </summary>
    protected virtual void OnDatasetCreated()
    {
    }

    /// <summary>
    /// Validates that the given index is within the valid range.
    /// </summary>
    /// <param name="index">The index to validate.</param>
    /// <exception cref="IndexOutOfRangeException">Thrown when index is out of range.</exception>
    protected virtual void ValidateIndex(int index)
    {
        if (index < 0 || index >= Length)
        {
            throw new IndexOutOfRangeException($"Index {index} is out of range for dataset of length {Length}.");
        }
    }

    /// <summary>
    /// Gets the item at the specified index with validation.
    /// </summary>
    /// <param name="index">The index of the item to retrieve.</param>
    /// <returns>The item at the specified index.</returns>
    public T GetValidatedItem(int index)
    {
        ValidateIndex(index);
        return GetItem(index);
    }
}
