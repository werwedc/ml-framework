namespace MLFramework.Data;

/// <summary>
/// Distributed sampler that divides the dataset into equal chunks across multiple replicas.
/// Each replica (process/node) gets a distinct subset of the data, ensuring no duplicates
/// across workers in distributed training scenarios.
/// </summary>
public class DistributedSampler : IDistributedSampler
{
    private readonly int _datasetSize;
    private readonly int _numReplicas;
    private readonly int _rank;
    private readonly bool _shuffle;
    private readonly bool _dropLast;
    private readonly int _seed;
    private int _epoch;

    /// <summary>
    /// Initializes a new instance of the DistributedSampler class.
    /// </summary>
    /// <param name="datasetSize">The total size of the dataset.</param>
    /// <param name="numReplicas">The total number of replicas (processes/nodes).</param>
    /// <param name="rank">The rank of the current process/node.</param>
    /// <param name="shuffle">Whether to shuffle the indices within each replica.</param>
    /// <param name="dropLast">Whether to drop the last uneven batch of data.</param>
    /// <param name="seed">The random seed for reproducibility.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown if parameters are out of valid range.</exception>
    public DistributedSampler(
        int datasetSize,
        int numReplicas,
        int rank,
        bool shuffle = true,
        bool dropLast = false,
        int seed = 0)
    {
        if (datasetSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(datasetSize), "Dataset size must be positive.");

        if (numReplicas <= 0)
            throw new ArgumentOutOfRangeException(nameof(numReplicas), "Number of replicas must be positive.");

        if (rank < 0 || rank >= numReplicas)
            throw new ArgumentOutOfRangeException(nameof(rank), $"Rank must be in [0, {numReplicas - 1}].");

        _datasetSize = datasetSize;
        _numReplicas = numReplicas;
        _rank = rank;
        _shuffle = shuffle;
        _dropLast = dropLast;
        _seed = seed;
        _epoch = 0;

        ValidateConfiguration();
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
    /// Gets the total number of samples that will be returned for this replica.
    /// </summary>
    public int Length => CalculatePerReplicaSize();

    /// <summary>
    /// Iterates over the sampled indices for this replica.
    /// </summary>
    /// <returns>An enumerable of indices for this replica.</returns>
    public IEnumerable<int> Iterate()
    {
        int perReplica = CalculatePerReplicaSize();

        // Generate indices for this replica
        var indices = new List<int>(perReplica);

        int startIndex = _rank * perReplica;

        for (int i = 0; i < perReplica; i++)
        {
            int globalIndex = startIndex + i;

            if (globalIndex >= _datasetSize)
                break;

            indices.Add(globalIndex);
        }

        // Shuffle if enabled (different seed per epoch)
        if (_shuffle)
        {
            int epochSeed = _seed + _epoch;
            var random = new Random(epochSeed);

            // Fisher-Yates shuffle
            for (int i = indices.Count - 1; i > 0; i--)
            {
                int j = random.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }
        }

        return indices;
    }

    /// <summary>
    /// Sets the epoch number to ensure different shuffling across epochs.
    /// </summary>
    /// <param name="epoch">The epoch number (must be non-negative).</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown if epoch is negative.</exception>
    public void SetEpoch(int epoch)
    {
        if (epoch < 0)
            throw new ArgumentOutOfRangeException(nameof(epoch), "Epoch must be non-negative.");

        _epoch = epoch;
    }

    /// <summary>
    /// Calculates the number of samples per replica.
    /// </summary>
    /// <returns>The number of samples assigned to this replica.</returns>
    private int CalculatePerReplicaSize()
    {
        int numSamples = _datasetSize;

        if (_dropLast)
        {
            // Drop samples to make evenly divisible
            numSamples = (_datasetSize / _numReplicas) * _numReplicas;
        }

        int perReplica = numSamples / _numReplicas;

        // Last replica may get more samples if not drop_last
        if (!_dropLast && _rank == _numReplicas - 1)
        {
            perReplica += numSamples % _numReplicas;
        }

        return perReplica;
    }

    /// <summary>
    /// Validates the sampler configuration.
    /// </summary>
    /// <exception cref="ArgumentException">Thrown if configuration is invalid.</exception>
    private void ValidateConfiguration()
    {
        if (_numReplicas <= 1)
            throw new ArgumentException("numReplicas must be > 1 for distributed sampling.");

        if (_rank < 0 || _rank >= _numReplicas)
            throw new ArgumentException($"rank must be in [0, {_numReplicas - 1}].");
    }
}
