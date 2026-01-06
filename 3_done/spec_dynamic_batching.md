# Spec: Dynamic Batching for Variable-Length Sequences

## Overview
Implement dynamic batching strategies for variable-length sequences (e.g., text, time series).

## Requirements

### Batching Strategies

#### 1. Pad to Maximum Length
- Pad all sequences to the maximum length in the batch
- Simple and straightforward
- May waste computation on short sequences

#### 2. Bucket Batching
- Group sequences of similar lengths together
- Reduces padding overhead
- Requires sorting or binning

#### 3. Dynamic Batch Size
- Adjust batch size to fit within token limit
- Ensures uniform computation across batches
- More complex but efficient

### Implementation

#### DynamicBatchSampler
- Extends BatchSampler with dynamic batching logic
- Configurable strategy (pad, bucket, dynamic)

**Key Fields:**
```csharp
public class DynamicBatchSampler : IBatchSampler
{
    private readonly IDataset<Sequence> _dataset;
    private readonly DynamicBatchStrategy _strategy;
    private readonly int _maxBatchSize;
    private readonly int _maxSequenceLength;
    private readonly int _paddingValue;

    public int BatchSize { get; private set; } // Variable for dynamic strategy
}
```

**Enums:**
```csharp
public enum DynamicBatchStrategy
{
    PadToMax,
    Bucket,
    Dynamic
}
```

**Constructor:**
```csharp
public DynamicBatchSampler(
    IDataset<Sequence> dataset,
    DynamicBatchStrategy strategy,
    int maxBatchSize,
    int maxSequenceLength = 512,
    int paddingValue = 0)
{
    _dataset = dataset ?? throw new ArgumentNullException(nameof(dataset));

    if (maxBatchSize <= 0)
        throw new ArgumentOutOfRangeException(nameof(maxBatchSize));

    if (maxSequenceLength <= 0)
        throw new ArgumentOutOfRangeException(nameof(maxSequenceLength));

    _strategy = strategy;
    _maxBatchSize = maxBatchSize;
    _maxSequenceLength = maxSequenceLength;
    _paddingValue = paddingValue;
    BatchSize = maxBatchSize;
}
```

**Get Sequence Lengths:**
```csharp
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
```

**Iterate (Strategy Implementation):**
```csharp
public IEnumerable<int[]> Iterate()
{
    int[] lengths = GetSequenceLengths();

    switch (_strategy)
    {
        case DynamicBatchStrategy.PadToMax:
            return PadToMaxStrategy(lengths);
        case DynamicBatchStrategy.Bucket:
            return BucketStrategy(lengths);
        case DynamicBatchStrategy.Dynamic:
            return DynamicStrategy(lengths);
        default:
            throw new ArgumentException("Unknown strategy");
    }
}
```

**Pad to Max Strategy:**
```csharp
private IEnumerable<int[]> PadToMaxStrategy(int[] lengths)
{
    List<int> batch = new List<int>(_maxBatchSize);
    int currentMax = 0;

    for (int i = 0; i < lengths.Length; i++)
    {
        currentMax = Math.Max(currentMax, lengths[i]);
        batch.Add(i);

        if (batch.Count == _maxBatchSize)
        {
            BatchSize = batch.Count;
            yield return batch.ToArray();
            batch.Clear();
            currentMax = 0;
        }
    }

    if (batch.Count > 0)
    {
        BatchSize = batch.Count;
        yield return batch.ToArray();
    }
}
```

**Bucket Strategy:**
```csharp
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
```

**Dynamic Strategy:**
```csharp
private IEnumerable<int[]> DynamicStrategy(int[] lengths)
{
    List<int> batch = new List<int>(_maxBatchSize);
    int totalTokens = 0;

    for (int i = 0; i < lengths.Length; i++)
    {
        int seqLength = Math.Min(lengths[i], _maxSequenceLength);

        // Check if adding this sequence would exceed token limit
        if (batch.Count > 0 && totalTokens + seqLength > _maxBatchSize * _maxSequenceLength)
        {
            BatchSize = batch.Count;
            yield return batch.ToArray();
            batch.Clear();
            totalTokens = 0;
        }

        batch.Add(i);
        totalTokens += seqLength;
    }

    if (batch.Count > 0)
    {
        BatchSize = batch.Count;
        yield return batch.ToArray();
    }
}
```

### Helper Classes

#### Sequence
- Represents a variable-length sequence

```csharp
public class Sequence
{
    public int[] Tokens { get; set; }
    public int Length => Tokens?.Length ?? 0;

    public Sequence(int[] tokens)
    {
        Tokens = tokens ?? throw new ArgumentNullException(nameof(tokens));
    }
}
```

## Acceptance Criteria
1. PadToMaxStrategy pads to max length in batch
2. BucketStrategy groups similar-length sequences
3. DynamicStrategy adjusts batch size for token limit
4. All strategies produce valid batches
5. No sequence exceeds maxSequenceLength
6. BatchSize property reflects actual batch size
7. Unit tests verify correct batching for each strategy
8. Performance tests measure padding efficiency

## Files to Create
- `src/Data/DynamicBatchSampler.cs`
- `src/Data/Sequence.cs`
- `src/Data/DynamicBatchStrategy.cs`

## Tests
- `tests/Data/DynamicBatchSamplerTests.cs`

## Usage Example
```csharp
var dataset = new TextDataset(textFilePath); // Variable-length text
var sampler = new DynamicBatchSampler(
    dataset,
    strategy: DynamicBatchStrategy.Bucket,
    maxBatchSize: 32,
    maxSequenceLength: 512
);

var dataloader = new DataLoader(dataset, sampler: sampler);

foreach (var batch in dataloader)
{
    // batch contains sequences of similar lengths
    // Less padding waste than naive batching
}
```

## Notes
- Critical for NLP and time series training
- Bucket strategy typically 20-40% efficiency gain
- Dynamic strategy best for strict token budgets
- Consider adding bucket size as parameter
- Future: Add support for padding masks
- Monitor average padding percentage across batches
- Works well with DataLoader for end-to-end solution
