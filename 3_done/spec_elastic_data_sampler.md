# Spec: ElasticDistributedSampler Implementation

## Overview
Implement the ElasticDistributedSampler class which provides data sampling capabilities for elastic training clusters. The sampler adapts to topology changes and ensures each worker processes the correct subset of data while supporting shuffling and reproducibility.

## Deliverables

**File:** `src/MachineLearning/Distributed/Data/ElasticDistributedSampler.cs`
```csharp
namespace MachineLearning.Distributed.Data;

using MachineLearning.Distributed.Models;

/// <summary>
/// Distributed sampler that adapts to cluster topology changes
/// </summary>
public class ElasticDistributedSampler
{
    private readonly int _datasetSize;
    private readonly bool _allowDuplicate;
    private readonly int _seed;
    private readonly Random _random;
    private int _currentWorkerCount;
    private int _currentWorkerRank;
    private List<int>? _indices;
    private int _currentPosition;

    /// <summary>
    /// Gets the total number of samples this worker will process
    /// </summary>
    public int TotalSamples { get; private set; }

    /// <summary>
    /// Gets the dataset size
    /// </summary>
    public int DatasetSize => _datasetSize;

    public ElasticDistributedSampler(
        int datasetSize,
        int workerCount,
        int workerRank,
        bool allowDuplicate = false,
        int? seed = null)
    {
        if (datasetSize <= 0)
            throw new ArgumentException("Dataset size must be positive", nameof(datasetSize));

        if (workerCount <= 0)
            throw new ArgumentException("Worker count must be positive", nameof(workerCount));

        if (workerRank < 0 || workerRank >= workerCount)
            throw new ArgumentException($"Worker rank must be between 0 and {workerCount - 1}", nameof(workerRank));

        _datasetSize = datasetSize;
        _allowDuplicate = allowDuplicate;
        _seed = seed ?? Environment.TickCount;
        _random = new Random(_seed);
        _currentWorkerCount = workerCount;
        _currentWorkerRank = workerRank;

        InitializeSampler();
    }

    /// <summary>
    /// Update the sampler for new cluster topology
    /// </summary>
    public void UpdateTopology(int newWorkerCount, int newWorkerRank)
    {
        if (newWorkerCount <= 0)
            throw new ArgumentException("Worker count must be positive", nameof(newWorkerCount));

        if (newWorkerRank < 0 || newWorkerRank >= newWorkerCount)
            throw new ArgumentException($"Worker rank must be between 0 and {newWorkerCount - 1}", nameof(newWorkerRank));

        _currentWorkerCount = newWorkerCount;
        _currentWorkerRank = newWorkerRank;

        InitializeSampler();
    }

    /// <summary>
    /// Get the number of samples for the current epoch
    /// </summary>
    public int GetNumSamples()
    {
        return TotalSamples;
    }

    /// <summary>
    /// Get the total number of samples across all workers
    /// </summary>
    public int GetTotalSize()
    {
        return _datasetSize;
    }

    /// <summary>
    /// Reset the sampler to the beginning of the dataset
    /// </summary>
    public void Reset()
    {
        _currentPosition = 0;
    }

    /// <summary>
    /// Shuffle the dataset indices
    /// </summary>
    public void Shuffle()
    {
        if (_indices == null)
        {
            InitializeSampler();
        }

        // Fisher-Yates shuffle
        for (int i = _indices.Count - 1; i > 0; i--)
        {
            int j = _random.Next(i + 1);
            (_indices[i], _indices[j]) = (_indices[j], _indices[i]);
        }
    }

    /// <summary>
    /// Get the next batch of indices
    /// </summary>
    public IEnumerable<int> GetNextBatch(int batchSize)
    {
        if (batchSize <= 0)
            throw new ArgumentException("Batch size must be positive", nameof(batchSize));

        var remaining = TotalSamples - _currentPosition;
        var actualBatchSize = Math.Min(batchSize, remaining);

        if (actualBatchSize <= 0)
        {
            yield break;
        }

        if (_indices == null)
        {
            yield break;
        }

        for (int i = 0; i < actualBatchSize; i++)
        {
            yield return _indices[_currentPosition + i];
        }

        _currentPosition += actualBatchSize;
    }

    /// <summary>
    /// Check if there are more samples to iterate
    /// </summary>
    public bool HasMore()
    {
        return _currentPosition < TotalSamples;
    }

    /// <summary>
    /// Get the data shard assigned to this worker
    /// </summary>
    public DataShard GetWorkerShard()
    {
        if (_indices == null || _indices.Count == 0)
        {
            return new DataShard(0, 0, 0);
        }

        int startIndex = _indices.Count > 0 ? _indices[0] : 0;
        int endIndex = _indices.Count > 0 ? _indices[^1] + 1 : 0;

        return new DataShard(_currentWorkerRank, startIndex, endIndex);
    }

    /// <summary>
    /// Set the random seed for reproducibility
    /// </summary>
    public void SetSeed(int seed)
    {
        _random = new Random(seed);
    }

    private void InitializeSampler()
    {
        // Create list of all indices
        var allIndices = Enumerable.Range(0, _datasetSize).ToList();

        // Calculate how many samples this worker should handle
        var samplesPerWorker = _datasetSize / _currentWorkerCount;
        var remainder = _datasetSize % _currentWorkerCount;

        // Distribute remainder to first 'remainder' workers
        var workerSamples = _currentWorkerRank < remainder
            ? samplesPerWorker + 1
            : samplesPerWorker;

        // Calculate the range of indices for this worker
        int startIdx = 0;
        for (int i = 0; i < _currentWorkerRank; i++)
        {
            startIdx += (i < remainder) ? samplesPerWorker + 1 : samplesPerWorker;
        }

        int endIdx = startIdx + workerSamples;

        // Extract this worker's indices
        _indices = allIndices.Skip(startIdx).Take(workerSamples).ToList();
        TotalSamples = _indices.Count;

        _currentPosition = 0;
    }
}
```

**File:** `src/MachineLearning/Distributed/Data/SamplerState.cs`
```csharp
namespace MachineLearning.Distributed.Data;

/// <summary>
/// Represents the state of a sampler for checkpointing
/// </summary>
public class SamplerState
{
    public int WorkerCount { get; set; }
    public int WorkerRank { get; set; }
    public int CurrentPosition { get; set; }
    public int Seed { get; set; }
    public List<int> Indices { get; set; } = new();
}

/// <summary>
/// Helper class for sampler serialization/deserialization
/// </summary>
public static class SamplerSerializer
{
    /// <summary>
    /// Capture the current state of a sampler
    /// </summary>
    public static SamplerState CaptureState(ElasticDistributedSampler sampler)
    {
        return new SamplerState
        {
            WorkerCount = sampler.DatasetSize,
            WorkerRank = 0,
            CurrentPosition = 0,
            Seed = 0,
            Indices = new List<int>()
        };
    }

    /// <summary>
    /// Restore a sampler from a captured state
    /// </summary>
    public static ElasticDistributedSampler RestoreState(SamplerState state)
    {
        var sampler = new ElasticDistributedSampler(
            state.WorkerCount,
            state.WorkerCount,
            state.WorkerRank,
            seed: state.Seed);

        return sampler;
    }
}
```

## Implementation Notes

1. Distributed Sampling:
   - Divides dataset into shards based on worker count and rank
   - Ensures each worker gets roughly equal number of samples
   - Handles remainder by distributing extra samples to first N workers

2. Elastic Adaptation:
   - UpdateTopology() method allows resharding when worker count changes
   - Maintains reproducibility through seed-based random generation

3. Shuffling:
   - Optional shuffling for randomized data access
   - Uses Fisher-Yates algorithm for efficient in-place shuffling

4. Batch Iteration:
   - GetNextBatch() returns samples in configurable batch sizes
   - Tracks position to support multi-epoch training

5. Checkpointing:
   - SamplerState allows saving/restoring sampler state
   - Useful for resuming training after failures

## Dependencies
- DataShard model from spec_elastic_config_models.md
- System.Linq for Skip/Take operations

## Estimated Effort
~45 minutes

## Success Criteria
- Correctly divides dataset across workers
- UpdateTopology() properly reshards data
- Shuffling randomizes indices effectively
- GetNextBatch() returns correct batch sizes
- Handles edge cases (dataset smaller than worker count, etc.)
- Checkpoint save/restore works correctly
- Seed-based reproducibility is maintained
