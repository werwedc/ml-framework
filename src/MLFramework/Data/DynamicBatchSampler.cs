namespace MLFramework.Data;

/// <summary>
/// Batch sampler that implements dynamic batching strategies for variable-length sequences.
/// Supports three strategies: PadToMax, Bucket, and Dynamic.
/// </summary>
public class DynamicBatchSampler : IBatchSampler
{
    private readonly IDataset<Sequence> _dataset;
    private readonly DynamicBatchStrategy _strategy;
    private readonly int _maxBatchSize;
    private readonly int _maxSequenceLength;
    private readonly int _paddingValue;

    /// <summary>
    /// Gets the batch size. For dynamic strategy, this varies per batch.
    /// </summary>
    public int BatchSize { get; private set; }

    /// <summary>
    /// Initializes a new instance of the DynamicBatchSampler class.
    /// </summary>
    /// <param name="dataset">The dataset containing sequences to batch.</param>
    /// <param name="strategy">The batching strategy to use.</param>
    /// <param name="maxBatchSize">The maximum batch size.</param>
    /// <param name="maxSequenceLength">The maximum sequence length allowed.</param>
    /// <param name="paddingValue">The value to use for padding (not used in batching, but useful for collation).</param>
    /// <exception cref="ArgumentNullException">Thrown when dataset is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when maxBatchSize or maxSequenceLength is less than or equal to zero.</exception>
    public DynamicBatchSampler(
        IDataset<Sequence> dataset,
        DynamicBatchStrategy strategy,
        int maxBatchSize,
        int maxSequenceLength = 512,
        int paddingValue = 0)
    {
        _dataset = dataset ?? throw new ArgumentNullException(nameof(dataset));

        if (maxBatchSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxBatchSize), "Max batch size must be positive.");

        if (maxSequenceLength <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxSequenceLength), "Max sequence length must be positive.");

        _strategy = strategy;
        _maxBatchSize = maxBatchSize;
        _maxSequenceLength = maxSequenceLength;
        _paddingValue = paddingValue;
        BatchSize = maxBatchSize;
    }

    /// <summary>
    /// Iterates over batches of indices based on the selected strategy.
    /// </summary>
    /// <returns>An enumerable of index arrays, where each array represents a batch.</returns>
    public IEnumerable<int[]> Iterate()
    {
        int[] lengths = GetSequenceLengths();

        return _strategy switch
        {
            DynamicBatchStrategy.PadToMax => PadToMaxStrategy(lengths),
            DynamicBatchStrategy.Bucket => BucketStrategy(lengths),
            DynamicBatchStrategy.Dynamic => DynamicStrategy(lengths),
            _ => throw new ArgumentException($"Unknown strategy: {_strategy}")
        };
    }

    /// <summary>
    /// Gets the length of each sequence in the dataset.
    /// </summary>
    /// <returns>An array of sequence lengths.</returns>
    private int[] GetSequenceLengths()
    {
        int[] lengths = new int[_dataset.Length];

        for (int i = 0; i < _dataset.Length; i++)
        {
            var sequence = _dataset.GetItem(i);
            lengths[i] = sequence.Length;
        }

        return lengths;
    }

    /// <summary>
    /// Implements the PadToMax strategy.
    /// Pads all sequences to the maximum length in the batch.
    /// </summary>
    /// <param name="lengths">Array of sequence lengths.</param>
    /// <returns>Batches of indices.</returns>
    private IEnumerable<int[]> PadToMaxStrategy(int[] lengths)
    {
        List<int> batch = new List<int>(_maxBatchSize);

        for (int i = 0; i < lengths.Length; i++)
        {
            batch.Add(i);

            if (batch.Count == _maxBatchSize)
            {
                BatchSize = batch.Count;
                yield return batch.ToArray();
                batch.Clear();
            }
        }

        // Handle remaining samples
        if (batch.Count > 0)
        {
            BatchSize = batch.Count;
            yield return batch.ToArray();
        }
    }

    /// <summary>
    /// Implements the Bucket strategy.
    /// Groups sequences of similar lengths together into buckets.
    /// </summary>
    /// <param name="lengths">Array of sequence lengths.</param>
    /// <returns>Batches of indices.</returns>
    private IEnumerable<int[]> BucketStrategy(int[] lengths)
    {
        // Group indices by length buckets
        var buckets = new Dictionary<int, List<int>>();
        int bucketSize = 64; // Bucket width

        for (int i = 0; i < lengths.Length; i++)
        {
            int bucket = (lengths[i] / bucketSize) * bucketSize;

            if (!buckets.ContainsKey(bucket))
                buckets[bucket] = new List<int>();

            buckets[bucket].Add(i);
        }

        // Create batches from each bucket
        foreach (var bucket in buckets.Values.OrderBy(b => b.Count))
        {
            for (int i = 0; i < bucket.Count; i += _maxBatchSize)
            {
                int take = Math.Min(_maxBatchSize, bucket.Count - i);
                int[] batch = bucket.Skip(i).Take(take).ToArray();
                BatchSize = batch.Length;
                yield return batch;
            }
        }
    }

    /// <summary>
    /// Implements the Dynamic strategy.
    /// Adjusts batch size to fit within a token limit.
    /// </summary>
    /// <param name="lengths">Array of sequence lengths.</param>
    /// <returns>Batches of indices.</returns>
    private IEnumerable<int[]> DynamicStrategy(int[] lengths)
    {
        List<int> batch = new List<int>(_maxBatchSize);
        int totalTokens = 0;
        int maxTokensPerBatch = _maxBatchSize * _maxSequenceLength;

        for (int i = 0; i < lengths.Length; i++)
        {
            int seqLength = Math.Min(lengths[i], _maxSequenceLength);

            // Check if adding this sequence would exceed token limit
            if (batch.Count > 0 && totalTokens + seqLength > maxTokensPerBatch)
            {
                BatchSize = batch.Count;
                yield return batch.ToArray();
                batch.Clear();
                totalTokens = 0;
            }

            batch.Add(i);
            totalTokens += seqLength;
        }

        // Handle remaining samples
        if (batch.Count > 0)
        {
            BatchSize = batch.Count;
            yield return batch.ToArray();
        }
    }
}
