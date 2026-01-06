namespace MLFramework.Data;

/// <summary>
/// Sampler that returns indices in random order without replacement.
/// Guarantees each index appears exactly once per epoch.
/// Configurable seed for deterministic behavior.
/// </summary>
public class RandomSampler : ISampler
{
    private readonly int _size;
    private readonly Random _random;

    /// <summary>
    /// Initializes a new instance of the RandomSampler class.
    /// </summary>
    /// <param name="size">The size of the dataset to sample from.</param>
    /// <param name="seed">Optional seed for reproducible randomization.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when size is negative.</exception>
    public RandomSampler(int size, int? seed = null)
    {
        if (size < 0)
            throw new ArgumentOutOfRangeException(nameof(size), "Size must be non-negative.");

        _size = size;
        _random = seed.HasValue ? new Random(seed.Value) : new Random();
        Length = size;
    }

    /// <summary>
    /// Gets the total number of samples that will be returned.
    /// </summary>
    public int Length { get; }

    /// <summary>
    /// Iterates over indices in random order using Fisher-Yates shuffle algorithm.
    /// </summary>
    /// <returns>An enumerable of shuffled indices.</returns>
    public IEnumerable<int> Iterate()
    {
        var indices = Enumerable.Range(0, _size).ToList();

        // Fisher-Yates shuffle (O(n) time complexity)
        for (int i = indices.Count - 1; i > 0; i--)
        {
            int j = _random.Next(i + 1);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }

        return indices;
    }
}
