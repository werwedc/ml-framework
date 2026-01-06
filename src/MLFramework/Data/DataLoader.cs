namespace MLFramework.Data;

/// <summary>
/// Provides single-threaded data loading with batching and transformation support.
/// Integrates Dataset, Sampler, BatchSampler, and Collate function.
/// </summary>
/// <typeparam name="T">The type of data items in the dataset.</typeparam>
public class DataLoader<T> : IEnumerable<object>
{
    private readonly IDataset<T> _dataset;
    private readonly int _batchSize;
    private readonly ISampler _sampler;
    private readonly IBatchSampler _batchSampler;
    private readonly Func<T[], object> _collateFn;
    private readonly bool _dropLast;

    /// <summary>
    /// Gets the dataset being loaded from.
    /// </summary>
    public IDataset<T> Dataset => _dataset;

    /// <summary>
    /// Gets the batch size.
    /// </summary>
    public int BatchSize => _batchSize;

    /// <summary>
    /// Gets the length of the dataset.
    /// </summary>
    public int DatasetLength => _dataset.Length;

    /// <summary>
    /// Gets the total number of batches that will be produced.
    /// </summary>
    public long NumBatches { get; }

    /// <summary>
    /// Initializes a new instance of the DataLoader class.
    /// </summary>
    /// <param name="dataset">The dataset to load data from.</param>
    /// <param name="batchSize">The batch size.</param>
    /// <param name="sampler">Optional custom sampler. If null, creates one based on shuffle parameter.</param>
    /// <param name="batchSampler">Optional custom batch sampler. If null, creates one using the sampler.</param>
    /// <param name="collateFn">Optional custom collate function. If null, uses default.</param>
    /// <param name="dropLast">If true, drops the last incomplete batch.</param>
    /// <param name="shuffle">If true, shuffles the data. Only used when sampler is null.</param>
    /// <exception cref="ArgumentNullException">Thrown when dataset is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when batchSize is less than or equal to zero.</exception>
    /// <exception cref="InvalidOperationException">Thrown when dataset is empty and dropLast is true.</exception>
    public DataLoader(
        IDataset<T> dataset,
        int batchSize,
        ISampler? sampler = null,
        IBatchSampler? batchSampler = null,
        Func<T[], object>? collateFn = null,
        bool dropLast = false,
        bool shuffle = false)
    {
        _dataset = dataset ?? throw new ArgumentNullException(nameof(dataset));

        if (batchSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(batchSize), "Batch size must be positive.");

        if (_dataset.Length == 0 && dropLast)
            throw new InvalidOperationException("Cannot drop last batch from an empty dataset.");

        _batchSize = batchSize;
        _dropLast = dropLast;

        // Create sampler if not provided
        _sampler = sampler ?? (shuffle
            ? (ISampler)new RandomSampler(dataset.Length)
            : new SequentialSampler(dataset.Length));

        // Create batchSampler if not provided
        _batchSampler = batchSampler ?? new BatchSampler(_sampler, batchSize, dropLast);

        _collateFn = collateFn ?? DefaultCollate;

        // Calculate number of batches
        int totalSamples = dataset.Length;
        int fullBatches = totalSamples / batchSize;
        int remainder = totalSamples % batchSize;
        NumBatches = dropLast ? fullBatches : fullBatches + (remainder > 0 ? 1 : 0);
    }

    /// <summary>
    /// Default collate function that simply returns the batch array.
    /// </summary>
    /// <param name="batch">The batch of samples.</param>
    /// <returns>The batch array.</returns>
    private static object DefaultCollate(T[] batch)
    {
        // Simple stacking logic
        // More sophisticated implementations will be added later
        return batch;
    }

    /// <summary>
    /// Returns an enumerator that iterates through batches of data.
    /// </summary>
    /// <returns>An enumerator for the batches.</returns>
    public IEnumerator<object> GetEnumerator()
    {
        foreach (var batchIndices in _batchSampler.Iterate())
        {
            var samples = new T[batchIndices.Length];

            for (int i = 0; i < batchIndices.Length; i++)
            {
                samples[i] = _dataset.GetItem(batchIndices[i]);
            }

            yield return _collateFn(samples);
        }
    }

    /// <summary>
    /// Returns an enumerator that iterates through batches of data.
    /// </summary>
    /// <returns>An enumerator for the batches.</returns>
    System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
    {
        return GetEnumerator();
    }
}
