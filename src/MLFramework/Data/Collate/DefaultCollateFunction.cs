namespace MLFramework.Data.Collate;

/// <summary>
/// Default collate function that returns the batch unchanged.
/// This is a simple pass-through for primitive types and a placeholder for complex types.
/// </summary>
/// <typeparam name="T">The type of individual samples in the batch.</typeparam>
public class DefaultCollateFunction<T> : ICollateFunction<T>
{
    /// <summary>
    /// Collates a batch of individual samples by returning the array unchanged.
    /// </summary>
    /// <param name="batch">An array of individual samples to collate.</param>
    /// <returns>The input array unchanged.</returns>
    /// <exception cref="ArgumentNullException">Thrown when batch is null.</exception>
    public object Collate(T[] batch)
    {
        if (batch == null)
            throw new ArgumentNullException(nameof(batch));

        return batch;
    }
}
