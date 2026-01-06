namespace MLFramework.Data;

/// <summary>
/// Interface for samplers that group indices into batches.
/// </summary>
public interface IBatchSampler
{
    /// <summary>
    /// Iterates over batches of indices.
    /// </summary>
    /// <returns>An enumerable of index arrays, where each array represents a batch.</returns>
    IEnumerable<int[]> Iterate();

    /// <summary>
    /// Gets the batch size.
    /// </summary>
    int BatchSize { get; }
}
