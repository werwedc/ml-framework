# Spec: Recomputation Engine

## Overview
Implement a recomputation engine that executes computational subgraphs to reconstruct discarded activations. The engine must preserve the exact behavior of the original forward pass and handle various computational patterns.

## Classes

### Location
`src/MLFramework/Checkpointing/RecomputationEngine.cs`

### Class: RecomputationEngine

```csharp
namespace MLFramework.Checkpointing;

/// <summary>
/// Executes recomputation of discarded activations from checkpointed states
/// </summary>
public class RecomputationEngine : IDisposable
{
    /// <summary>
    /// Initializes a new instance of RecomputationEngine
    /// </summary>
    public RecomputationEngine();

    /// <summary>
    /// Registers a computational node for recomputation
    /// </summary>
    /// <param name="layerId">Unique identifier for the layer</param>
    /// <param name="recomputeFunction">Function that recomputes the activation</param>
    /// <exception cref="ArgumentException">Thrown if layerId already registered</exception>
    public void RegisterRecomputeFunction(string layerId, Func<Tensor> recomputeFunction);

    /// <summary>
    /// Recomputes the activation for the specified layer
    /// </summary>
    /// <param name="layerId">Unique identifier for the layer</param>
    /// <returns>The recomputed activation tensor</returns>
    /// <exception cref="KeyNotFoundException">Thrown if layer not registered</exception>
    /// <exception cref="RecomputationException">Thrown if recomputation fails</exception>
    public Tensor Recompute(string layerId);

    /// <summary>
    /// Recomputes multiple activations in dependency order
    /// </summary>
    /// <param name="layerIds">List of layer IDs to recompute</param>
    /// <returns>Dictionary mapping layer IDs to recomputed activations</returns>
    public Dictionary<string, Tensor> RecomputeMultiple(IEnumerable<string> layerIds);

    /// <summary>
    /// Checks if a layer has a registered recompute function
    /// </summary>
    /// <param name="layerId">Unique identifier for the layer</param>
    /// <returns>True if registered, false otherwise</returns>
    public bool HasRecomputeFunction(string layerId);

    /// <summary>
    /// Clears all registered recompute functions
    /// </summary>
    public void Clear();

    /// <summary>
    /// Gets statistics about recomputation operations
    /// </summary>
    /// <returns>RecomputationStats with detailed information</returns>
    public RecomputationStats GetStats();

    /// <summary>
    /// Disposes the engine and releases resources
    /// </summary>
    public void Dispose();
}
```

### Internal Data Structures

```csharp
private class RecomputeEntry
{
    public string LayerId { get; set; }
    public Func<Tensor> RecomputeFunction { get; set; }
    public int CallCount { get; set; }
    public long TotalComputationTimeMs { get; set; }
    public DateTime LastCalledAt { get; set; }
    public List<string> Dependencies { get; set; } = new List<string>();
}

private readonly Dictionary<string, RecomputeEntry> _recomputeFunctions;
private readonly object _lock = new object();
private bool _disposed;
```

## Implementation Details

### RegisterRecomputeFunction

```csharp
public void RegisterRecomputeFunction(string layerId, Func<Tensor> recomputeFunction)
{
    if (string.IsNullOrWhiteSpace(layerId))
        throw new ArgumentException("Layer ID cannot be null or whitespace");
    if (recomputeFunction == null)
        throw new ArgumentNullException(nameof(recomputeFunction));

    lock (_lock)
    {
        if (_recomputeFunctions.ContainsKey(layerId))
        {
            throw new ArgumentException($"Layer '{layerId}' is already registered");
        }

        _recomputeFunctions[layerId] = new RecomputeEntry
        {
            LayerId = layerId,
            RecomputeFunction = recomputeFunction,
            CallCount = 0,
            TotalComputationTimeMs = 0,
            LastCalledAt = DateTime.MinValue
        };
    }
}
```

### Recompute

```csharp
public Tensor Recompute(string layerId)
{
    if (string.IsNullOrWhiteSpace(layerId))
        throw new ArgumentException("Layer ID cannot be null or whitespace");

    RecomputeEntry? entry;

    lock (_lock)
    {
        if (!_recomputeFunctions.TryGetValue(layerId, out entry))
        {
            throw new KeyNotFoundException($"No recompute function registered for layer '{layerId}'");
        }
    }

    try
    {
        var stopwatch = Stopwatch.StartNew();
        var result = entry.RecomputeFunction();
        stopwatch.Stop();

        lock (_lock)
        {
            entry.CallCount++;
            entry.TotalComputationTimeMs += stopwatch.ElapsedMilliseconds;
            entry.LastCalledAt = DateTime.UtcNow;
        }

        return result;
    }
    catch (Exception ex)
    {
        throw new RecomputationException(
            $"Failed to recompute activation for layer '{layerId}'", ex);
    }
}
```

### RecomputeMultiple

```csharp
public Dictionary<string, Tensor> RecomputeMultiple(IEnumerable<string> layerIds)
{
    if (layerIds == null)
        throw new ArgumentNullException(nameof(layerIds));

    var layerList = layerIds.ToList();
    if (layerList.Count == 0)
        return new Dictionary<string, Tensor>();

    // Sort layers by dependency (if dependencies are tracked)
    // For now, recompute in the order provided
    var results = new Dictionary<string, Tensor>();

    foreach (var layerId in layerList)
    {
        results[layerId] = Recompute(layerId);
    }

    return results;
}
```

## Exception Classes

```csharp
namespace MLFramework.Checkpointing;

/// <summary>
/// Exception thrown when recomputation fails
/// </summary>
public class RecomputationException : Exception
{
    public RecomputationException(string message) : base(message) { }

    public RecomputationException(string message, Exception innerException)
        : base(message, innerException) { }

    public RecomputationException(string layerId, Exception innerException)
        : base($"Recomputation failed for layer '{layerId}'", innerException)
    {
        LayerId = layerId;
    }

    public string? LayerId { get; set; }
}
```

## Recomputation Statistics

### Class: RecomputationStats

```csharp
namespace MLFramework.Checkpointing;

/// <summary>
/// Statistics about recomputation operations
/// </summary>
public class RecomputationStats
{
    /// <summary>
    /// Total number of recomputation calls
    /// </summary>
    public int TotalRecomputations { get; set; }

    /// <summary>
    /// Total time spent on recomputation (in milliseconds)
    /// </summary>
    public long TotalRecomputationTimeMs { get; set; }

    /// <summary>
    /// Average time per recomputation (in milliseconds)
    /// </summary>
    public double AverageRecomputationTimeMs =>
        TotalRecomputations > 0 ? (double)TotalRecomputationTimeMs / TotalRecomputations : 0.0;

    /// <summary>
    /// Number of layers with registered recompute functions
    /// </summary>
    public int RegisteredLayerCount { get; set; }

    /// <summary>
    /// Per-layer recomputation statistics
    /// </summary>
    public Dictionary<string, LayerRecomputationStats> LayerStats { get; set; } = new();

    /// <summary>
    /// Timestamp when stats were collected
    /// </summary>
    public DateTime Timestamp { get; set; }
}
```

### Class: LayerRecomputationStats

```csharp
namespace MLFramework.Checkpointing;

/// <summary>
/// Recomputation statistics for a specific layer
/// </summary>
public class LayerRecomputationStats
{
    /// <summary>
    /// Unique identifier for the layer
    /// </summary>
    public string LayerId { get; set; } = string.Empty;

    /// <summary>
    /// Number of times this layer was recomputed
    /// </summary>
    public int CallCount { get; set; }

    /// <summary>
    /// Total time spent recomputing this layer (in milliseconds)
    /// </summary>
    public long TotalComputationTimeMs { get; set; }

    /// <summary>
    /// Average time per recomputation for this layer (in milliseconds)
    /// </summary>
    public double AverageComputationTimeMs =>
        CallCount > 0 ? (double)TotalComputationTimeMs / CallCount : 0.0;

    /// <summary>
    /// Timestamp of last recomputation
    /// </summary>
    public DateTime LastCalledAt { get; set; }
}
```

## Dependency Tracking (Advanced)

### Class: DependencyGraph

```csharp
namespace MLFramework.Checkpointing;

/// <summary>
/// Tracks dependencies between layers for optimal recomputation ordering
/// </summary>
public class DependencyGraph
{
    /// <summary>
    /// Adds a dependency relationship
    /// </summary>
    /// <param name="layerId">Layer that depends on other layers</param>
    /// <param name="dependsOn">Layers that this layer depends on</param>
    public void AddDependency(string layerId, IEnumerable<string> dependsOn);

    /// <summary>
    /// Gets layers in topological order (dependencies first)
    /// </summary>
    /// <param name="layerIds">Layers to order</param>
    /// <returns>Layers in topological order</returns>
    public List<string> GetTopologicalOrder(IEnumerable<string> layerIds);

    /// <summary>
    /// Detects cycles in the dependency graph
    /// </summary>
    /// <returns>True if cycle detected, false otherwise</returns>
    public bool HasCycle();

    /// <summary>
    /// Clears all dependencies
    /// </summary>
    public void Clear();
}
```

## Asynchronous Recomputation

### Class: AsyncRecomputationEngine

```csharp
namespace MLFramework.Checkpointing;

/// <summary>
/// Asynchronous recomputation engine for parallel execution
/// </summary>
public class AsyncRecomputationEngine : IDisposable
{
    /// <summary>
    /// Initializes a new instance of AsyncRecomputationEngine
    /// </summary>
    /// <param name="maxDegreeOfParallelism">Maximum parallel operations</param>
    public AsyncRecomputationEngine(int maxDegreeOfParallelism = -1);

    /// <summary>
    /// Registers a recompute function
    /// </summary>
    public void RegisterRecomputeFunction(string layerId, Func<Tensor> recomputeFunction);

    /// <summary>
    /// Asynchronously recomputes an activation
    /// </summary>
    public Task<Tensor> RecomputeAsync(string layerId, CancellationToken cancellationToken = default);

    /// <summary>
    /// Asynchronously recomputes multiple activations in parallel
    /// </summary>
    public Task<Dictionary<string, Tensor>> RecomputeMultipleAsync(
        IEnumerable<string> layerIds,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Disposes the engine and releases resources
    /// </summary>
    public void Dispose();
}
```

## Recomputation Cache

### Class: RecomputationCache

```csharp
namespace MLFramework.Checkpointing;

/// <summary>
/// Caches recomputed activations to avoid redundant computation
/// </summary>
public class RecomputationCache : IDisposable
{
    /// <summary>
    /// Initializes a new instance of RecomputationCache
    /// </summary>
    /// <param name="maxSizeBytes">Maximum cache size in bytes</param>
    public RecomputationCache(long maxSizeBytes);

    /// <summary>
    /// Gets a cached activation or null if not cached
    /// </summary>
    public Tensor? Get(string layerId);

    /// <summary>
    /// Adds an activation to the cache
    /// </summary>
    /// <param name="layerId">Unique identifier for the layer</param>
    /// <param name="activation">Activation tensor to cache</param>
    public void Add(string layerId, Tensor activation);

    /// <summary>
    /// Checks if an activation is cached
    /// </summary>
    public bool Contains(string layerId);

    /// <summary>
    /// Clears the cache
    /// </summary>
    public void Clear();

    /// <summary>
    /// Gets cache statistics
    /// </summary>
    public CacheStats GetStats();

    /// <summary>
    /// Disposes the cache and releases resources
    /// </summary>
    public void Dispose();
}

/// <summary>
/// Statistics for recomputation cache
/// </summary>
public class CacheStats
{
    public int CachedItemsCount { get; set; }
    public long CurrentSizeBytes { get; set; }
    public long MaxSizeBytes { get; set; }
    public int CacheHits { get; set; }
    public int CacheMisses { get; set; }
    public double HitRate => CacheHits + CacheMisses > 0
        ? (double)CacheHits / (CacheHits + CacheMisses) : 0.0;
}
```

## Testing Requirements

### Unit Tests

1. **RecomputationEngine Basic Tests**
   - [ ] Successfully register recompute function
   - [ ] Throw exception when registering duplicate layer
   - [ ] Successfully recompute activation
   - [ ] Throw exception for unregistered layer
   - [ ] Handle recomputation function that throws exception

2. **RecomputationStatistics Tests**
   - [ ] Correctly track call count
   - [ ] Correctly track computation time
   - [ ] Calculate correct average computation time
   - [ ] Get stats returns correct information

3. **RecomputeMultiple Tests**
   - [ ] Successfully recompute multiple layers
   - [ ] Handle empty list of layers
   - [ ] Handle null layer list
   - [ ] Preserve order in results dictionary

4. **DependencyGraph Tests**
   - [ ] Correctly add dependencies
   - [ ] Correctly get topological order
   - [ ] Detect cycles correctly
   - [ ] Handle circular dependencies

5. **AsyncRecomputationEngine Tests**
   - [ ] Successfully recompute asynchronously
   - [ ] Cancel async recomputation
   - [ ] Handle parallel recomputations
   - [ ] Respect max degree of parallelism

6. **RecomputationCache Tests**
   - [ ] Successfully add and retrieve cached items
   - [ ] Handle cache eviction when size limit exceeded
   - [ ] Correctly track cache hits and misses
   - [ ] Calculate correct hit rate
   - [ ] Clear cache removes all items

7. **Exception Handling Tests**
   - [ ] RecomputationException thrown on failure
   - [ ] Exception includes layer ID
   - [ ] Exception preserves inner exception

8. **Thread Safety Tests**
   - [ ] Multiple threads can register functions
   - [ ] Multiple threads can recompute
   - [ ] Stats remain consistent under concurrent access

9. **Edge Cases**
   - [ ] Handle recompute function that returns null
   - [ ] Handle very large tensors
   - [ ] Handle very slow recompute functions
   - [ ] Handle rapid successive recomputations

## Implementation Notes

1. **Thread Safety**:
   - All public methods should be thread-safe
   - Use lock for mutable shared state
   - Consider lock-free approaches for high-performance scenarios

2. **Performance**:
   - Minimize overhead of recomputation calls
   - Use efficient caching strategies (LRU, LFU)
   - Consider memory vs. computation trade-offs

3. **Error Handling**:
   - Provide clear error messages
   - Preserve original exceptions
   - Allow recovery from failures

4. **Extensibility**:
   - Design to support different recomputation strategies
   - Allow custom caching policies
   - Support monitoring and profiling

## Dependencies on Other Specs

This spec depends on:
- **Checkpoint Manager Core** (spec_1) for integration
- **Checkpoint Configuration** (spec_2) for async recomputation settings

This spec is independent and can be implemented in parallel with other specs.

## Estimated Implementation Time
45-60 minutes
