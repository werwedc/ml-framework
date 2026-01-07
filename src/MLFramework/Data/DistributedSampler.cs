using MLFramework.Distributed;

namespace MLFramework.Data;

/// <summary>
/// Distributed sampler that partitions a dataset across multiple devices, ensuring each device
/// processes different data while maintaining reproducible shuffling.
/// </summary>
public class DistributedSampler : IDistributedSampler, IDisposable
{
    private readonly int _datasetSize;
    private readonly int _numReplicas;
    private readonly int _rank;
    private readonly int _seed;
    private readonly bool _dropLast;
    private readonly int _shuffle;

    private int _epoch;
    private int[] _indices;
    private int _numSamples;
    private int _totalSize;
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of the DistributedSampler class.
    /// </summary>
    /// <param name="datasetSize">The total size of the dataset.</param>
    /// <param name="numReplicas">Number of processes participating in distributed training.
    /// If null, uses world size from ProcessGroup.Default.</param>
    /// <param name="rank">Rank of the current process. If null, uses rank from ProcessGroup.Default.</param>
    /// <param name="shuffle">If true, shuffles the indices.</param>
    /// <param name="seed">Random seed for shuffling.</param>
    /// <param name="dropLast">If true, drop the last incomplete batch.</param>
    public DistributedSampler(
        int datasetSize,
        int? numReplicas = null,
        int? rank = null,
        bool shuffle = true,
        int seed = 0,
        bool dropLast = false)
    {
        if (datasetSize < 0)
            throw new ArgumentException("datasetSize must be non-negative");

        _datasetSize = datasetSize;
        _numReplicas = numReplicas ?? ProcessGroup.Default?.WorldSize ?? 1;
        _rank = rank ?? ProcessGroup.Default?.Rank ?? 0;
        _shuffle = shuffle ? 1 : 0;
        _seed = seed;
        _dropLast = dropLast;
        _epoch = 0;

        if (_numReplicas <= 0)
            throw new ArgumentException("numReplicas must be positive");

        if (_rank >= _numReplicas || _rank < 0)
            throw new ArgumentException("rank must be in [0, numReplicas - 1]");

        Initialize();
    }

    /// <summary>
    /// Gets the total number of replicas participating in distributed training.
    /// </summary>
    public int NumReplicas => _numReplicas;

    /// <summary>
    /// Gets the rank of the current process/node.
    /// </summary>
    public int Rank => _rank;

    /// <summary>
    /// Gets the current epoch number.
    /// </summary>
    public int Epoch => _epoch;

    /// <summary>
    /// Gets the total number of samples for this rank.
    /// </summary>
    public int NumSamples => _numSamples;

    /// <summary>
    /// Gets the total number of samples that will be returned for this replica.
    /// </summary>
    public int Length => _numSamples;

    /// <summary>
    /// Sets the current epoch for shuffling.
    /// Different epochs produce different shuffle orders.
    /// </summary>
    /// <param name="epoch">The epoch number.</param>
    public void SetEpoch(int epoch)
    {
        _epoch = epoch;
        Initialize();
    }

    /// <summary>
    /// Get the batch of indices for the given batch index.
    /// </summary>
    /// <param name="batchIndex">The batch index.</param>
    /// <param name="batchSize">The batch size.</param>
    /// <returns>An array of indices for the batch.</returns>
    public int[] GetBatch(int batchIndex, int batchSize)
    {
        if (batchIndex < 0 || batchIndex >= GetNumBatches(batchSize))
            throw new ArgumentOutOfRangeException(nameof(batchIndex), "batchIndex out of range");

        var startIdx = batchIndex * batchSize;
        var endIdx = Math.Min(startIdx + batchSize, _numSamples);
        var batch = new int[endIdx - startIdx];

        for (int i = 0; i < batch.Length; i++)
        {
            batch[i] = _indices[startIdx + i];
        }

        return batch;
    }

    /// <summary>
    /// Get all indices for this rank.
    /// </summary>
    /// <returns>A clone of the indices array.</returns>
    public int[] GetIndices()
    {
        return (int[])_indices.Clone();
    }

    /// <summary>
    /// Get the number of batches for this rank.
    /// </summary>
    /// <param name="batchSize">The batch size.</param>
    /// <returns>The number of batches.</returns>
    public int GetNumBatches(int batchSize)
    {
        if (batchSize <= 0)
            throw new ArgumentException("batchSize must be positive", nameof(batchSize));

        return (int)Math.Ceiling((double)_numSamples / batchSize);
    }

    /// <summary>
    /// Iterates over the sampled indices for this replica.
    /// </summary>
    /// <returns>An enumerable of indices for this replica.</returns>
    public IEnumerable<int> Iterate()
    {
        return _indices;
    }

    /// <summary>
    /// Initialize the sampler indices based on current configuration.
    /// </summary>
    private void Initialize()
    {
        if (_datasetSize == 0)
        {
            _indices = Array.Empty<int>();
            _numSamples = 0;
            _totalSize = 0;
            return;
        }

        // Calculate total size and samples per replica
        if (_dropLast)
        {
            // Drop samples that don't divide evenly
            var numSamplesPerReplica = _datasetSize / _numReplicas;
            _totalSize = numSamplesPerReplica * _numReplicas;
            _numSamples = numSamplesPerReplica;
        }
        else
        {
            // Distribute uneven remainder
            _totalSize = _datasetSize;
            _numSamples = (int)Math.Ceiling((double)_datasetSize / _numReplicas);
        }

        // Generate base indices (shuffled or in order)
        int[] baseIndices;
        if (_shuffle == 1)
        {
            int effectiveSeed = _seed + _epoch;
            baseIndices = SamplerHelper.Shuffle(_totalSize, effectiveSeed);
        }
        else
        {
            baseIndices = SamplerHelper.Range(_totalSize);
        }

        // Assign indices to this rank (interleaved assignment)
        // Each rank gets indices: rank, rank + numReplicas, rank + 2*numReplicas, ...
        _indices = new int[_numSamples];
        for (int i = 0; i < _numSamples; i++)
        {
            int globalIndex = _rank + i * _numReplicas;
            if (globalIndex < _totalSize)
            {
                _indices[i] = baseIndices[globalIndex];
            }
            else
            {
                // This can happen when dropLast is false and rank is near the end
                _indices[i] = -1; // Invalid index
            }
        }
    }

    /// <summary>
    /// Disposes the sampler.
    /// </summary>
    public void Dispose()
    {
        if (!_disposed)
        {
            _indices = Array.Empty<int>();
            _disposed = true;
        }
    }
}
