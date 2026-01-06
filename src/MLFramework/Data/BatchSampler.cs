namespace MLFramework.Data;

/// <summary>
/// Sampler that wraps an ISampler and groups its indices into batches.
/// Supports variable batch sizes for handling remainder samples.
/// </summary>
public class BatchSampler : IBatchSampler
{
    private readonly ISampler _sampler;
    private readonly int _batchSize;
    private readonly bool _dropLast;

    /// <summary>
    /// Initializes a new instance of the BatchSampler class.
    /// </summary>
    /// <param name="sampler">The underlying sampler to get indices from.</param>
    /// <param name="batchSize">The size of each batch.</param>
    /// <param name="dropLast">If true, drops the last incomplete batch.</param>
    /// <exception cref="ArgumentNullException">Thrown when sampler is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when batchSize is less than or equal to zero.</exception>
    public BatchSampler(ISampler sampler, int batchSize, bool dropLast = false)
    {
        _sampler = sampler ?? throw new ArgumentNullException(nameof(sampler));

        if (batchSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(batchSize), "Batch size must be positive.");

        _batchSize = batchSize;
        _dropLast = dropLast;
        BatchSize = batchSize;
    }

    /// <summary>
    /// Gets the batch size.
    /// </summary>
    public int BatchSize { get; }

    /// <summary>
    /// Iterates over batches of indices from the underlying sampler.
    /// </summary>
    /// <returns>An enumerable of index arrays, where each array represents a batch.</returns>
    public IEnumerable<int[]> Iterate()
    {
        var batch = new List<int>(_batchSize);

        foreach (var index in _sampler.Iterate())
        {
            batch.Add(index);

            if (batch.Count == _batchSize)
            {
                yield return batch.ToArray();
                batch.Clear();
            }
        }

        // Handle remaining samples
        if (batch.Count > 0 && !_dropLast)
        {
            yield return batch.ToArray();
        }
    }
}
