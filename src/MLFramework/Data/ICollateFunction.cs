namespace MLFramework.Data;

/// <summary>
/// Interface for collate functions that combine individual samples into batches.
/// </summary>
/// <typeparam name="T">The type of individual samples in the batch.</typeparam>
public interface ICollateFunction<T>
{
    /// <summary>
    /// Collates a batch of individual samples into a batched format.
    /// </summary>
    /// <param name="batch">An array of individual samples to collate.</param>
    /// <returns>A batched representation of the samples.</returns>
    object Collate(T[] batch);
}
