namespace MLFramework.Data;

/// <summary>
/// Interface for datasets that provide random access to samples.
/// </summary>
/// <typeparam name="T">The type of data items in the dataset.</typeparam>
public interface IDataset<T>
{
    /// <summary>
    /// Gets the item at the specified index.
    /// </summary>
    /// <param name="index">The index of the item to retrieve.</param>
    /// <returns>The item at the specified index.</returns>
    T GetItem(int index);

    /// <summary>
    /// Gets the total number of items in the dataset.
    /// </summary>
    int Length { get; }
}
