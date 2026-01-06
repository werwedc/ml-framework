namespace MLFramework.Data;

/// <summary>
/// Helper utilities for sampler operations.
/// </summary>
internal static class SamplerHelper
{
    /// <summary>
    /// Generate a permutation of [0, n) using Fisher-Yates shuffle with given seed.
    /// </summary>
    public static int[] Shuffle(int n, int seed)
    {
        if (n <= 0)
            return Array.Empty<int>();

        var indices = new int[n];
        var random = new Random(seed);

        // Initialize array with [0, 1, 2, ..., n-1]
        for (int i = 0; i < n; i++)
        {
            indices[i] = i;
        }

        // Fisher-Yates shuffle
        for (int i = n - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);  // inclusive
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }

        return indices;
    }

    /// <summary>
    /// Generate a range [0, n) in order.
    /// </summary>
    public static int[] Range(int n)
    {
        if (n <= 0)
            return Array.Empty<int>();

        var indices = new int[n];
        for (int i = 0; i < n; i++)
        {
            indices[i] = i;
        }

        return indices;
    }
}
