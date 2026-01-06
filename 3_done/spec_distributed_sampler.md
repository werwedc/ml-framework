# Spec: DistributedSampler

## Overview
Implement a distributed sampler that partitions data across multiple training processes/nodes.

## Requirements

### Interface

#### IDistributedSampler
```csharp
public interface IDistributedSampler : ISampler
{
    int NumReplicas { get; }
    int Rank { get; }
    int Epoch { get; }
    void SetEpoch(int epoch);
}
```

### Implementation

#### DistributedSampler
- Divides dataset into equal chunks across replicas
- Supports shuffling (each replica gets different subset per epoch)
- Handles drop_last for uneven datasets
- Configurable via numReplicas and rank

**Key Fields:**
```csharp
public class DistributedSampler : IDistributedSampler
{
    private readonly int _datasetSize;
    private readonly int _numReplicas;
    private readonly int _rank;
    private readonly bool _shuffle;
    private readonly bool _dropLast;
    private readonly int _seed;
    private int _epoch;
    private readonly Random _random;
}
```

**Constructor:**
```csharp
public DistributedSampler(
    int datasetSize,
    int numReplicas,
    int rank,
    bool shuffle = true,
    bool dropLast = false,
    int seed = 0)
{
    if (datasetSize <= 0)
        throw new ArgumentOutOfRangeException(nameof(datasetSize));

    if (numReplicas <= 0)
        throw new ArgumentOutOfRangeException(nameof(numReplicas));

    if (rank < 0 || rank >= numReplicas)
        throw new ArgumentOutOfRangeException(nameof(rank));

    _datasetSize = datasetSize;
    _numReplicas = numReplicas;
    _rank = rank;
    _shuffle = shuffle;
    _dropLast = dropLast;
    _seed = seed;
    _epoch = 0;
    _random = new Random(seed);
}
```

**Calculate Per-Replica Size:**
```csharp
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
```

**Iterate:**
```csharp
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
        var epochRandom = new Random(epochSeed);

        for (int i = indices.Count - 1; i > 0; i--)
        {
            int j = epochRandom.Next(i + 1);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }
    }

    return indices;
}
```

**SetEpoch:**
```csharp
public void SetEpoch(int epoch)
{
    if (epoch < 0)
        throw new ArgumentOutOfRangeException(nameof(epoch));

    _epoch = epoch;
}
```

**Properties:**
```csharp
public int Length => CalculatePerReplicaSize();
public int NumReplicas => _numReplicas;
public int Rank => _rank;
public int Epoch => _epoch;
}
```

### Helper Methods

#### Validate Configuration
```csharp
private void ValidateConfiguration()
{
    if (_numReplicas <= 1)
        throw new ArgumentException("numReplicas must be > 1 for distributed sampling");

    if (_rank < 0 || _rank >= _numReplicas)
        throw new ArgumentException($"rank must be in [0, {_numReplicas - 1}]");
}
```

## Acceptance Criteria
1. Dataset divided equally across replicas when drop_last=true
2. Last replica gets extra samples when drop_last=false
3. Each replica processes distinct, non-overlapping samples
4. SetEpoch changes shuffling pattern across epochs
5. Different ranks get different samples per epoch
6. shuffle=false returns sequential indices within replica
7. Seed provides reproducible shuffling
8. Length property correctly reports samples per replica
9. Unit tests verify correct partitioning logic
10. Integration tests simulate multi-process training

## Files to Create
- `src/Data/IDistributedSampler.cs`
- `src/Data/DistributedSampler.cs`

## Tests
- `tests/Data/DistributedSamplerTests.cs`

## Usage Example
```csharp
// Each process creates its own sampler with its rank
int numReplicas = 4; // Total number of GPUs/processes
int rank = 2; // This process's rank

var sampler = new DistributedSampler(
    datasetSize: 1000,
    numReplicas: numReplicas,
    rank: rank,
    shuffle: true,
    dropLast: true
);

// Each epoch, update to ensure different shuffling
sampler.SetEpoch(currentEpoch);

foreach (var index in sampler.Iterate())
{
    // Process sample - no overlap with other ranks
}
```

## Notes
- Critical for distributed training (DDP, FSDP)
- Ensures no duplicate samples across workers
- drop_last recommended for even workload distribution
- SetEpoch must be called each training epoch
- Combines well with DataLoader for end-to-end solution
- Consider adding stratified sampling for imbalanced datasets
- Future: Support for uneven replica configurations
- Common pattern: Use with PyTorch DDP-style training
