# Spec: IterableDataset Optimizations

## Overview
Add optimizations specifically for streaming/infinite datasets that implement IterableDataset.

## Requirements

### Optimizations

#### 1. Multi-Process Worker Support
- IterableDatasets can be used with multiprocessing workers
- Each worker independently iterates over its own stream
- Avoids cross-worker communication overhead

#### 2. Iterator Caching
- Cache iterators for repeated epochs
- Only re-iterator when explicitly reset

#### 3. Stream Replication
- Replicate stream across workers for parallel processing
- Each worker processes disjoint chunks of the stream

### Implementation

#### OptimizedIterableDataset<T>
- Extends IterableDataset with optimization features
- Configurable worker strategy (shared vs replicated)

**Key Fields:**
```csharp
public class OptimizedIterableDataset<T> : IterableDataset<T>
{
    private readonly Func<IEnumerator<T>> _iteratorFactory;
    private IEnumerator<T> _cachedIterator;
    private readonly bool _enableWorkerSupport;
    private readonly int _workerId;
    private readonly int _totalWorkers;
    private volatile bool _iteratorCreated;
}
```

**Constructor:**
```csharp
public OptimizedIterableDataset(
    Func<IEnumerator<T>> iteratorFactory,
    bool enableWorkerSupport = false,
    int workerId = 0,
    int totalWorkers = 1)
{
    _iteratorFactory = iteratorFactory ?? throw new ArgumentNullException(nameof(iteratorFactory));
    _enableWorkerSupport = enableWorkerSupport;
    _workerId = workerId;
    _totalWorkers = totalWorkers;
    _cachedIterator = null;
    _iteratorCreated = false;
}
```

**GetEnumerator:**
```csharp
public override IEnumerator<T> GetEnumerator()
{
    lock (this)
    {
        if (!_iteratorCreated)
        {
            _cachedIterator = CreateIterator();
            _iteratorCreated = true;
        }
    }

    return CreateStreamIterator();
}
```

**Create Iterator:**
```csharp
private IEnumerator<T> CreateIterator()
{
    if (_enableWorkerSupport && _totalWorkers > 1)
    {
        // Create worker-aware iterator
        return CreateWorkerIterator();
    }

    return _iteratorFactory();
}
```

**Create Worker Iterator:**
```csharp
private IEnumerator<T> CreateWorkerIterator()
{
    // Each worker skips to its position in the stream
    var baseIterator = _iteratorFactory();
    int skipCount = _workerId;

    // Skip samples for other workers
    for (int i = 0; i < skipCount; i++)
    {
        if (!baseIterator.MoveNext())
            break;
    }

    // Yield samples for this worker (stride by totalWorkers)
    int sampleCount = 0;
    while (baseIterator.MoveNext())
    {
        if (sampleCount % _totalWorkers == _workerId)
        {
            yield return baseIterator.Current;
        }
        sampleCount++;
    }
}
```

**Create Stream Iterator:**
```csharp
private IEnumerator<T> CreateStreamIterator()
{
    // Return a new enumerator that wraps the cached iterator
    // This allows multiple concurrent iterations if needed

    if (_cachedIterator == null)
    {
        throw new InvalidOperationException("Iterator not created");
    }

    // For now, just return the cached iterator
    // In a real implementation, we might clone or reset it
    return _cachedIterator;
}
```

**Reset:**
```csharp
public void Reset()
{
    lock (this)
    {
        _cachedIterator?.Dispose();
        _cachedIterator = null;
        _iteratorCreated = false;
    }
}
```

### Helper Classes

#### StreamReplicator
- Replicates a single stream across multiple workers
- Ensures each worker gets disjoint samples

```csharp
public class StreamReplicator<T>
{
    private readonly IEnumerator<T> _sourceStream;
    private readonly int _numReplicas;

    public StreamReplicator(IEnumerator<T> sourceStream, int numReplicas)
    {
        _sourceStream = sourceStream ?? throw new ArgumentNullException(nameof(sourceStream));
        _numReplicas = numReplicas;
    }

    public IEnumerator<T> GetReplicaStream(int replicaId)
    {
        if (replicaId < 0 || replicaId >= _numReplicas)
            throw new ArgumentOutOfRangeException(nameof(replicaId));

        return new ReplicaEnumerator(_sourceStream, replicaId, _numReplicas);
    }

    private class ReplicaEnumerator : IEnumerator<T>
    {
        private readonly IEnumerator<T> _source;
        private readonly int _replicaId;
        private readonly int _stride;
        private int _position;

        public ReplicaEnumerator(IEnumerator<T> source, int replicaId, int stride)
        {
            _source = source;
            _replicaId = replicaId;
            _stride = stride;
            _position = -1;
        }

        public T Current => _source.Current;
        object IEnumerator.Current => Current;

        public bool MoveNext()
        {
            _position++;

            // Move source to this replica's position
            while (_source.MoveNext())
            {
                if (_position % _stride == _replicaId)
                    return true;
            }

            return false;
        }

        public void Reset()
        {
            throw new NotSupportedException("Reset not supported for streaming data");
        }

        public void Dispose()
        {
            // Don't dispose the shared source stream
        }
    }
}
```

### Error Handling
- Validation of worker ID and total workers
- Thread-safe iterator creation
- Graceful handling of exhausted streams

## Acceptance Criteria
1. OptimizedIterableDataset caches iterators
2. Reset clears cached iterator
3. Worker support partitions stream correctly
4. Each worker receives disjoint samples
5. Multiple workers don't read same samples
6. StreamReplicator handles exhausted streams
7. Unit tests verify worker partitioning
8. Integration tests verify multi-worker scenarios

## Files to Create
- `src/Data/OptimizedIterableDataset.cs`
- `src/Data/StreamReplicator.cs`

## Tests
- `tests/Data/OptimizedIterableDatasetTests.cs`
- `tests/Data/StreamReplicatorTests.cs`

## Usage Example
```csharp
// Create streaming dataset (e.g., from API or file)
var dataset = new OptimizedIterableDataset<Sample>(
    iteratorFactory: () => StreamSamplesFromApi(),
    enableWorkerSupport: true,
    workerId: workerRank,
    totalWorkers: worldSize
);

// Use with DataLoader
var dataloader = new DataLoader(dataset, batchSize: 32, shuffle: false);

// Each epoch, reset to re-stream data
dataset.Reset();
```

## Notes
- IterableDataset is critical for streaming/infinite data
- Worker support enables parallel processing of streams
- Stride-based partitioning ensures disjoint samples
- Reset needed between epochs for IterableDataset
- Cannot shuffle IterableDataset (stream is ordered)
- Common use cases: real-time data, web scraping, online learning
- Future: Add support for dynamic batch sizes based on stream rate
- Consider adding buffer management for high-throughput streams
