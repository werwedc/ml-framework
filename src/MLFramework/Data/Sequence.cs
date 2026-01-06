namespace MLFramework.Data;

/// <summary>
/// Represents a variable-length sequence for dynamic batching.
/// Commonly used for text, time series, and other sequential data.
/// </summary>
public class Sequence
{
    /// <summary>
    /// Gets the tokens in the sequence.
    /// </summary>
    public int[] Tokens { get; set; }

    /// <summary>
    /// Gets the length of the sequence.
    /// </summary>
    public int Length => Tokens?.Length ?? 0;

    /// <summary>
    /// Initializes a new instance of the Sequence class.
    /// </summary>
    /// <param name="tokens">The tokens in the sequence.</param>
    /// <exception cref="ArgumentNullException">Thrown when tokens is null.</exception>
    public Sequence(int[] tokens)
    {
        Tokens = tokens ?? throw new ArgumentNullException(nameof(tokens));
    }

    /// <summary>
    /// Initializes a new instance of the Sequence class with an empty sequence.
    /// </summary>
    public Sequence()
    {
        Tokens = Array.Empty<int>();
    }

    /// <summary>
    /// Gets a slice of the sequence.
    /// </summary>
    /// <param name="start">The start index (inclusive).</param>
    /// <param name="end">The end index (exclusive).</param>
    /// <returns>A new Sequence containing the sliced tokens.</returns>
    public Sequence Slice(int start, int end)
    {
        if (start < 0 || start > Length)
            throw new ArgumentOutOfRangeException(nameof(start));

        if (end < start || end > Length)
            throw new ArgumentOutOfRangeException(nameof(end));

        int sliceLength = end - start;
        int[] slicedTokens = new int[sliceLength];
        Array.Copy(Tokens, start, slicedTokens, 0, sliceLength);

        return new Sequence(slicedTokens);
    }

    /// <summary>
    /// Returns a string representation of the sequence.
    /// </summary>
    /// <returns>A string containing the tokens.</returns>
    public override string ToString()
    {
        return $"[{string.Join(", ", Tokens)}]";
    }
}
