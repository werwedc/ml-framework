namespace MLFramework.Data;

/// <summary>
/// Interface for datasets that provide sequential iteration over samples.
/// Suitable for streaming or infinite datasets.
/// </summary>
/// <typeparam name="T">The type of data items in the dataset.</typeparam>
public interface IIterableDataset<T>
{
    /// <summary>
    /// Returns an enumerator that iterates through the dataset.
    /// </summary>
    /// <returns>An enumerator that can be used to iterate through the dataset.</returns>
    IEnumerator<T> GetEnumerator();
}
