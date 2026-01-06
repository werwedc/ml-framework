# Spec: Basic DataLoader (Single-Threaded)

## Overview
Implement a DataLoader that provides single-threaded data loading with batching and transformation support.

## Requirements

### Class Definition

#### DataLoader<T>
- Main data loading class (single-threaded, no workers)
- Integrates Dataset, Sampler, BatchSampler, and Collate function
- Supports iteration over batches
- Configurable batch size, sampler, and collate function

**Key Properties:**
```csharp
public class DataLoader<T>
{
    private readonly IDataset<T> _dataset;
    private readonly int _batchSize;
    private readonly ISampler _sampler;
    private readonly IBatchSampler _batchSampler;
    private readonly Func<T[], object> _collateFn;
    private readonly bool _dropLast;

    public IDataset<T> Dataset => _dataset;
    public int BatchSize => _batchSize;
    public int DatasetLength => _dataset.Length;
    public long NumBatches { get; }
}
```

**Constructor:**
```csharp
public DataLoader(
    IDataset<T> dataset,
    int batchSize,
    ISampler sampler = null,
    IBatchSampler batchSampler = null,
    Func<T[], object> collateFn = null,
    bool dropLast = false,
    bool shuffle = false)
{
    _dataset = dataset ?? throw new ArgumentNullException(nameof(dataset));

    if (batchSize <= 0)
        throw new ArgumentOutOfRangeException(nameof(batchSize));

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
```

**Iterator:**
```csharp
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
```

### Default Collate Function
```csharp
private static object DefaultCollate(T[] batch)
{
    // Simple stacking logic
    // More sophisticated implementations will be added later
    return batch;
}
```

### Error Handling
- `ArgumentNullException` if dataset is null
- `ArgumentOutOfRangeException` if batchSize <= 0
- `InvalidOperationException` if dataset is empty and dropLast is true

## Acceptance Criteria
1. DataLoader can iterate over dataset in batches
2. Shuffle parameter controls random vs sequential sampling
3. DropLast parameter controls handling of partial final batch
4. Custom sampler can be provided for advanced sampling strategies
5. Custom batchSampler can be provided
6. NumBatches property correctly calculates expected iterations
7. GetEnumerator returns one batch per iteration
8. Default collate function returns the batch array
9. Unit tests verify all combinations of shuffle and dropLast

## Files to Create
- `src/Data/DataLoader.cs`

## Tests
- `tests/Data/DataLoaderTests.cs`

## Usage Example
```csharp
public class SimpleDataset : MapStyleDataset<int>
{
    public override int GetItem(int index) => index * 2;
    public override int Length => 100;
}

var dataset = new SimpleDataset();
var dataloader = new DataLoader(dataset, batchSize: 32, shuffle: true);

foreach (var batch in dataloader)
{
    // Process batch
}
```

## Notes
- This is the foundation DataLoader - multiprocessing workers added later
- Use `Func<T[], object>` for collate to support various batch types
- Current version is single-threaded - no concurrent data loading
- IEnumerator allows use with foreach loops and LINQ
- Consider implementing IEnumerable<object> explicitly
- Future versions will add transform pipeline support
