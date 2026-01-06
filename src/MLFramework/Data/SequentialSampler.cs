namespace MLFramework.Data;

/// <summary>
/// Sampler that returns indices in sequential order: 0, 1, 2, ..., N-1.
/// Deterministic and repeatable. Useful for validation and testing.
/// </summary>
public class SequentialSampler : ISampler
{
    private readonly int _size;

    /// <summary>
    /// Initializes a new instance of the SequentialSampler class.
    /// </summary>
    /// <param name="size">The size of the dataset to sample from.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when size is negative.</exception>
    public SequentialSampler(int size)
    {
        if (size < 0)
            throw new ArgumentOutOfRangeException(nameof(size), "Size must be non-negative.");

        _size = size;
        Length = size;
    }

    /// <summary>
    /// Gets the total number of samples that will be returned.
    /// </summary>
    public int Length { get; }

    /// <summary>
    /// Iterates over indices in sequential order.
    /// </summary>
    /// <returns>An enumerable of indices from 0 to size-1.</returns>
    public IEnumerable<int> Iterate()
    {
        for (int i = 0; i < _size; i++)
            yield return i;
    }
}
