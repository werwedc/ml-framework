# Spec: BatchSampler

## Overview
Implement a sampler that groups indices into batches for efficient data loading.

## Requirements

### Interface

#### IBatchSampler
```csharp
public interface IBatchSampler
{
    IEnumerable<int[]> Iterate();
    int BatchSize { get; }
}
```

### Implementation

#### BatchSampler
- Wraps an `ISampler` and groups its indices into batches
- Supports variable batch sizes for handling remainder samples
- Configurable `dropLast` to discard final incomplete batch
- Lazy evaluation for memory efficiency

**Key Methods:**
```csharp
public class BatchSampler : IBatchSampler
{
    private readonly ISampler _sampler;
    private readonly int _batchSize;
    private readonly bool _dropLast;

    public BatchSampler(ISampler sampler, int batchSize, bool dropLast = false)
    {
        _sampler = sampler ?? throw new ArgumentNullException(nameof(sampler));

        if (batchSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(batchSize));

        _batchSize = batchSize;
        _dropLast = dropLast;
        BatchSize = batchSize;
    }

    public int BatchSize { get; }

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
```

### Error Handling
- `ArgumentNullException` if sampler is null
- `ArgumentOutOfRangeException` if batchSize <= 0

## Acceptance Criteria
1. BatchSampler groups indices into batches of specified size
2. Last batch is either returned or dropped based on `dropLast` flag
3. Works correctly with SequentialSampler (predictable batches)
4. Works correctly with RandomSampler (shuffled batches)
5. Empty dataset returns no batches
6. Batch size larger than dataset returns single batch (if !dropLast)
7. Unit tests cover all combinations of dropLast and dataset sizes

## Files to Create
- `src/Data/IBatchSampler.cs`
- `src/Data/BatchSampler.cs`

## Tests
- `tests/Data/BatchSamplerTests.cs`

## Usage Example
```csharp
var sampler = new RandomSampler(datasetSize: 100, seed: 42);
var batchSampler = new BatchSampler(sampler, batchSize: 32, dropLast: true);

foreach (var batch in batchSampler.Iterate())
{
    // batch.Length == 32 for 3 batches, no partial final batch
}
```

## Notes
- Return `int[]` not `List<int>` for clarity and immutability
- Memory efficient: only holds one batch in memory at a time
- Compatible with any ISampler implementation
- Consider adding `DropLast` property for introspection
