using System.Diagnostics;

namespace MLFramework.Checkpointing;

/// <summary>
/// Executes recomputation of discarded activations from checkpointed states
/// </summary>
public class RecomputationEngine : IDisposable
{
    private class RecomputeEntry
    {
        public string LayerId { get; set; } = string.Empty;
        public Func<Tensor> RecomputeFunction { get; set; } = null!;
        public int CallCount { get; set; }
        public long TotalComputationTimeMs { get; set; }
        public DateTime LastCalledAt { get; set; }
        public List<string> Dependencies { get; set; } = new List<string>();
    }

    private readonly Dictionary<string, RecomputeEntry> _recomputeFunctions;
    private readonly object _lock = new object();
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of RecomputationEngine
    /// </summary>
    public RecomputationEngine()
    {
        _recomputeFunctions = new Dictionary<string, RecomputeEntry>();
        _disposed = false;
    }

    /// <summary>
    /// Registers a computational node for recomputation
    /// </summary>
    /// <param name="layerId">Unique identifier for the layer</param>
    /// <param name="recomputeFunction">Function that recomputes the activation</param>
    /// <exception cref="ArgumentException">Thrown if layerId already registered</exception>
    public void RegisterRecomputeFunction(string layerId, Func<Tensor> recomputeFunction)
    {
        if (string.IsNullOrWhiteSpace(layerId))
            throw new ArgumentException("Layer ID cannot be null or whitespace");
        if (recomputeFunction == null)
            throw new ArgumentNullException(nameof(recomputeFunction));

        lock (_lock)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(RecomputationEngine));

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

    /// <summary>
    /// Registers a computational node with dependencies for recomputation
    /// </summary>
    /// <param name="layerId">Unique identifier for the layer</param>
    /// <param name="recomputeFunction">Function that recomputes the activation</param>
    /// <param name="dependencies">List of layer IDs this layer depends on</param>
    public void RegisterRecomputeFunction(
        string layerId,
        Func<Tensor> recomputeFunction,
        IEnumerable<string> dependencies)
    {
        if (dependencies == null)
            throw new ArgumentNullException(nameof(dependencies));

        lock (_lock)
        {
            RegisterRecomputeFunction(layerId, recomputeFunction);

            _recomputeFunctions[layerId].Dependencies = dependencies.ToList();
        }
    }

    /// <summary>
    /// Recomputes the activation for the specified layer
    /// </summary>
    /// <param name="layerId">Unique identifier for the layer</param>
    /// <returns>The recomputed activation tensor</returns>
    /// <exception cref="KeyNotFoundException">Thrown if layer not registered</exception>
    /// <exception cref="RecomputationException">Thrown if recomputation fails</exception>
    public Tensor Recompute(string layerId)
    {
        if (string.IsNullOrWhiteSpace(layerId))
            throw new ArgumentException("Layer ID cannot be null or whitespace");

        RecomputeEntry? entry;

        lock (_lock)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(RecomputationEngine));

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
            throw RecomputationException.ForLayer(layerId, ex);
        }
    }

    /// <summary>
    /// Recomputes multiple activations in dependency order
    /// </summary>
    /// <param name="layerIds">List of layer IDs to recompute</param>
    /// <returns>Dictionary mapping layer IDs to recomputed activations</returns>
    public Dictionary<string, Tensor> RecomputeMultiple(IEnumerable<string> layerIds)
    {
        if (layerIds == null)
            throw new ArgumentNullException(nameof(layerIds));

        var layerList = layerIds.ToList();
        if (layerList.Count == 0)
            return new Dictionary<string, Tensor>();

        // Use dependency graph to determine optimal order
        var graph = new DependencyGraph();
        var layerSet = new HashSet<string>(layerList);

        lock (_lock)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(RecomputationEngine));

            // Build dependency graph for requested layers
            foreach (var layerId in layerList)
            {
                if (_recomputeFunctions.TryGetValue(layerId, out var entry))
                {
                    var relevantDeps = entry.Dependencies
                        .Where(dep => layerSet.Contains(dep))
                        .ToList();

                    if (relevantDeps.Count > 0)
                    {
                        graph.AddDependency(layerId, relevantDeps);
                    }
                }
            }
        }

        // Get layers in topological order
        var orderedLayers = graph.HasCycle()
            ? layerList // Fall back to provided order if there's a cycle
            : graph.GetTopologicalOrder(layerList);

        // Recompute in order
        var results = new Dictionary<string, Tensor>();
        foreach (var layerId in orderedLayers)
        {
            results[layerId] = Recompute(layerId);
        }

        return results;
    }

    /// <summary>
    /// Checks if a layer has a registered recompute function
    /// </summary>
    /// <param name="layerId">Unique identifier for the layer</param>
    /// <returns>True if registered, false otherwise</returns>
    public bool HasRecomputeFunction(string layerId)
    {
        if (string.IsNullOrWhiteSpace(layerId))
            return false;

        lock (_lock)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(RecomputationEngine));

            return _recomputeFunctions.ContainsKey(layerId);
        }
    }

    /// <summary>
    /// Clears all registered recompute functions
    /// </summary>
    public void Clear()
    {
        lock (_lock)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(RecomputationEngine));

            _recomputeFunctions.Clear();
        }
    }

    /// <summary>
    /// Gets statistics about recomputation operations
    /// </summary>
    /// <returns>RecomputationStats with detailed information</returns>
    public RecomputationStats GetStats()
    {
        lock (_lock)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(RecomputationEngine));

            var layerStats = new Dictionary<string, LayerRecomputationStats>();

            foreach (var kvp in _recomputeFunctions)
            {
                var entry = kvp.Value;
                layerStats[kvp.Key] = new LayerRecomputationStats
                {
                    LayerId = entry.LayerId,
                    CallCount = entry.CallCount,
                    TotalComputationTimeMs = entry.TotalComputationTimeMs,
                    LastCalledAt = entry.LastCalledAt
                };
            }

            return new RecomputationStats
            {
                TotalRecomputations = layerStats.Values.Sum(s => s.CallCount),
                TotalRecomputationTimeMs = layerStats.Values.Sum(s => s.TotalComputationTimeMs),
                RegisteredLayerCount = _recomputeFunctions.Count,
                LayerStats = layerStats,
                Timestamp = DateTime.UtcNow
            };
        }
    }

    /// <summary>
    /// Gets statistics for a specific layer
    /// </summary>
    /// <param name="layerId">Unique identifier for the layer</param>
    /// <returns>LayerRecomputationStats or null if not found</returns>
    public LayerRecomputationStats? GetLayerStats(string layerId)
    {
        if (string.IsNullOrWhiteSpace(layerId))
            return null;

        lock (_lock)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(RecomputationEngine));

            if (!_recomputeFunctions.TryGetValue(layerId, out var entry))
                return null;

            return new LayerRecomputationStats
            {
                LayerId = entry.LayerId,
                CallCount = entry.CallCount,
                TotalComputationTimeMs = entry.TotalComputationTimeMs,
                LastCalledAt = entry.LastCalledAt
            };
        }
    }

    /// <summary>
    /// Disposes the engine and releases resources
    /// </summary>
    public void Dispose()
    {
        if (!_disposed)
        {
            lock (_lock)
            {
                _recomputeFunctions.Clear();
            }
            _disposed = true;
        }
    }
}
