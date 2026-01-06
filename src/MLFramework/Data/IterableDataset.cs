namespace MLFramework.Data;

/// <summary>
/// Abstract base class for datasets that provide sequential iteration over samples.
/// Suitable for streaming or infinite datasets where random access is not available or desired.
/// Not thread-safe - single consumer expected.
/// </summary>
/// <typeparam name="T">The type of data items in the dataset.</typeparam>
public abstract class IterableDataset<T> : IIterableDataset<T>
{
    /// <summary>
    /// Returns an enumerator that iterates through the dataset.
    /// </summary>
    /// <returns>An enumerator that can be used to iterate through the dataset.</returns>
    public abstract IEnumerator<T> GetEnumerator();

    /// <summary>
    /// Called when the dataset is created. Override to perform initialization.
    /// </summary>
    protected virtual void OnDatasetCreated()
    {
    }
}
