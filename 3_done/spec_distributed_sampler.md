# Spec: Distributed Sampler

## Overview
Implement a distributed sampler that partitions a dataset across multiple devices, ensuring each device processes different data while maintaining reproducible shuffling.

## Requirements
- Partition dataset indices evenly across all ranks
- Support deterministic shuffling with epoch-based seeds
- Handle uneven divisions gracefully
- Provide random access to batch indices
- Support dropping last incomplete batch (optional)

## Classes

### 1. DistributedSampler Class
```csharp
public class DistributedSampler : IDisposable
{
    private readonly Dataset _dataset;
    private readonly int _numReplicas;
    private readonly int _rank;
    private readonly int _seed;
    private readonly bool _dropLast;
    private readonly int _shuffle;

    private int _epoch;
    private int[] _indices;
    private int _numSamples;
    private int _totalSize;

    public DistributedSampler(
        Dataset dataset,
        int? numReplicas = null,  // null = use world size from process group
        int? rank = null,         // null = use rank from process group
        bool shuffle = true,
        int seed = 0,
        int dropLast = false)
    {
        _dataset = dataset;
        _numReplicas = numReplicas ?? ProcessGroup.Default?.WorldSize ?? 1;
        _rank = rank ?? ProcessGroup.Default?.Rank ?? 0;
        _shuffle = shuffle ? 1 : 0;
        _seed = seed;
        _dropLast = dropLast;

        if (_numReplicas <= 0)
            throw new ArgumentException("numReplicas must be positive");

        if (_rank >= _numReplicas || _rank < 0)
            throw new ArgumentException("rank must be in [0, numReplicas - 1]");

        Initialize();
    }

    /// <summary>
    /// Get the number of samples for this rank.
    /// </summary>
    public int NumSamples => _numSamples;

    /// <summary>
    /// Set the current epoch for shuffling.
    /// Different epochs produce different shuffle orders.
    /// </summary>
    public void SetEpoch(int epoch)
    {
        _epoch = epoch;
        Initialize();
    }

    /// <summary>
    /// Get the batch of indices for the given batch index.
    /// </summary>
    public int[] GetBatch(int batchIndex, int batchSize)
    {
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
    public int[] GetIndices()
    {
        return (int[])_indices.Clone();
    }

    /// <summary>
    /// Get the number of batches for this rank.
    /// </summary>
    public int GetNumBatches(int batchSize)
    {
        return (int)Math.Ceiling((double)_numSamples / batchSize);
    }

    private void Initialize();

    public void Dispose();
}
```

### 2. SamplerHelper Class (Internal)
```csharp
/// <summary>
/// Helper utilities for sampler operations.
/// </summary>
internal static class SamplerHelper
{
    /// <summary>
    /// Generate a permutation of [0, n) using Fisher-Yates shuffle with given seed.
    /// </summary>
    public static int[] Shuffle(int n, int seed);

    /// <summary>
    /// Generate a range [0, n) in order.
    /// </summary>
    public static int[] Range(int n);
}
```

## Implementation Details

### Partitioning Logic

**Total Dataset Size**: `dataset.Count`

**Partition Size Calculation**:
```csharp
if (dropLast)
{
    // Drop samples that don't divide evenly
    var numSamplesPerReplica = dataset.Count / numReplicas;
    _totalSize = numSamplesPerReplica * numReplicas;
    _numSamples = numSamplesPerReplica;
}
else
{
    // Distribute uneven remainder
    _totalSize = dataset.Count;
    _numSamples = (int)Math.Ceiling((double)dataset.Count / numReplicas);
}
```

**Index Assignment**:
- Each rank gets indices: `rank, rank + numReplicas, rank + 2*numReplicas, ...`
- This interleaves data across ranks (better for gradient diversity)

### Shuffling Strategy

**Seed Generation**: `effectiveSeed = baseSeed + epoch`

**Shuffle Process**:
1. Generate permutation of `[0, totalSize)` using Fisher-Yates shuffle
2. Assign indices to each rank from the shuffled permutation
3. Each rank gets every `numReplicas`-th element from its starting position

**Fisher-Yates Shuffle**:
```csharp
for (int i = n - 1; i > 0; i--)
{
    var j = Random.Next(0, i + 1);  // inclusive
    (arr[i], arr[j]) = (arr[j], arr[i]);
}
```

**Deterministic RNG**: Use `System.Random` with fixed seed for reproducibility

### Edge Cases

**Single Device**: When `numReplicas = 1`, behaves like regular sampler

**Empty Dataset**: Return empty indices array

**Small Dataset**: Handles cases where some ranks get fewer samples

**Batch Size > NumSamples**: Last batch is smaller or dropped (based on `dropLast`)

## Usage Example

```csharp
// Initialize sampler with default process group
var trainDataset = new ImageNetDataset(path: "/data/imagenet");
var sampler = new DistributedSampler(trainDataset, shuffle: true);

// Create data loader with distributed sampler
var loader = new DataLoader(trainDataset, batchSize: 32, sampler: sampler);

// Training loop
for (int epoch = 0; epoch < numEpochs; epoch++)
{
    // Important: set epoch for different shuffling each epoch
    sampler.SetEpoch(epoch);

    foreach (var batch in loader)
    {
        // Each rank gets different batches
        var output = model.Forward(batch);
        var loss = lossFn(output, target);
        loss.Backward();
        optimizer.Step();
    }
}
```

## Success Criteria
- [ ] Each rank gets a disjoint subset of dataset indices
- [ ] All indices are covered when all ranks are combined
- [ ] Shuffling is deterministic for same epoch/seed
- [ ] Different epochs produce different shuffle orders
- [ ] Handles uneven dataset sizes correctly
- [ ] Works with dropLast = true and false
- [ ] Compatible with DataLoader (batch generation)

## Dependencies
- Existing Dataset class (from src/)
- spec_process_group.md (optional, for auto-detecting rank/world_size)

## Testing
- Unit tests in spec_ddp_tests.md will verify:
  - Correct partitioning across ranks
  - Deterministic shuffling
  - Different epochs produce different shuffles
  - All indices covered when combined
  - Edge cases (empty dataset, single device, etc.)
  - Integration with DataLoader
