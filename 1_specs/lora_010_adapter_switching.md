# Spec: Adapter Switching Mechanism

## Overview
Implement runtime adapter switching for seamless task switching without model reloading. This is critical for multi-tenant serving and applications that need to switch between specialized behaviors quickly.

## Implementation Details

### 1. AdapterSwitcher Class
**File**: `src/LoRA/AdapterSwitcher.cs`

```csharp
/// <summary>
/// Provides runtime adapter switching for models
/// </summary>
public class AdapterSwitcher
{
    private readonly IModule _model;
    private readonly LoRAAdapterRegistry _registry;
    private string _currentAdapterId = string.Empty;
    private AdapterState? _currentState;
    private Dictionary<string, AdapterState> _cachedAdapters;

    public string CurrentAdapterId => _currentAdapterId;

    public AdapterSwitcher(IModule model, LoRAAdapterRegistry registry, int cacheSize = 5)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _registry = registry ?? throw new ArgumentNullException(nameof(registry));
        _cachedAdapters = new Dictionary<string, AdapterState>();
        _registry.SetModel(model);
    }

    /// <summary>
    /// Switches to a different adapter
    /// </summary>
    /// <param name="adapterId">ID of the adapter to switch to</param>
    /// <param name="useCache">Whether to use cached adapter state</param>
    /// <returns>Time taken to switch in milliseconds</returns>
    public long SwitchAdapter(string adapterId, bool useCache = true)
    {
        var stopwatch = Stopwatch.StartNew();

        // Save current adapter state if needed
        if (!string.IsNullOrEmpty(_currentAdapterId) && _currentState != null)
        {
            if (_cachedAdapters.Count < 5 || _cachedAdapters.ContainsKey(_currentAdapterId))
            {
                // Update cached state
                _currentState = ExtractCurrentState();
                _cachedAdapters[_currentAdapterId] = _currentState;
            }
        }

        // Load new adapter state
        AdapterState newState;
        if (useCache && _cachedAdapters.TryGetValue(adapterId, out var cachedState))
        {
            newState = cachedState;
        }
        else
        {
            newState = LoadAdapterState(adapterId);
            if (useCache)
            {
                _cachedAdapters[adapterId] = newState;
            }
        }

        // Apply new adapter state
        ApplyAdapterState(newState);

        _currentAdapterId = adapterId;
        _currentState = newState;

        stopwatch.Stop();
        return stopwatch.ElapsedMilliseconds;
    }

    /// <summary>
    /// Switches to a different adapter asynchronously
    /// </summary>
    public async Task<long> SwitchAdapterAsync(string adapterId, bool useCache = true)
    {
        return await Task.Run(() => SwitchAdapter(adapterId, useCache));
    }

    /// <summary>
    /// Preloads an adapter into cache
    /// </summary>
    public void PreloadAdapter(string adapterId)
    {
        if (!_cachedAdapters.ContainsKey(adapterId))
        {
            var state = LoadAdapterState(adapterId);
            _cachedAdapters[adapterId] = state;
        }
    }

    /// <summary>
    /// Preloads multiple adapters asynchronously
    /// </summary>
    public async Task PreloadAdaptersAsync(string[] adapterIds)
    {
        var tasks = adapterIds
            .Where(id => !_cachedAdapters.ContainsKey(id))
            .Select(id => Task.Run(() =>
            {
                var state = LoadAdapterState(id);
                _cachedAdapters[id] = state;
            }));

        await Task.WhenAll(tasks);
    }

    /// <summary>
    /// Removes an adapter from cache
    /// </summary>
    public void UnloadAdapter(string adapterId)
    {
        _cachedAdapters.Remove(adapterId);
    }

    /// <summary>
    /// Clears the adapter cache
    /// </summary>
    public void ClearCache()
    {
        _cachedAdapters.Clear();
    }

    /// <summary>
    /// Gets cache statistics
    /// </summary>
    public AdapterCacheStats GetCacheStats()
    {
        return new AdapterCacheStats
        {
            CachedAdapters = _cachedAdapters.Count,
            CurrentAdapter = _currentAdapterId,
            MemoryUsageMB = EstimateCacheMemoryUsage()
        };
    }

    /// <summary>
    /// Compares performance of different adapters
    /// </summary>
    public Dictionary<string, AdapterPerformance> BenchmarkAdapters(
        ITensor testInput,
        int warmupRuns = 3,
        int benchmarkRuns = 10)
    {
        var results = new Dictionary<string, AdapterPerformance>();
        var availableAdapters = _registry.ListAdapters()
            .Select(m => m.Id)
            .ToList();

        foreach (var adapterId in availableAdapters)
        {
            // Warmup
            for (int i = 0; i < warmupRuns; i++)
            {
                SwitchAdapter(adapterId);
                _ = _model.Forward(testInput);
            }

            // Benchmark
            var timings = new List<long>();
            for (int i = 0; i < benchmarkRuns; i++)
            {
                var switchTime = SwitchAdapter(adapterId);
                var inferenceStart = Stopwatch.StartNew();
                _ = _model.Forward(testInput);
                var inferenceTime = inferenceStart.ElapsedMilliseconds;

                timings.Add(switchTime + inferenceTime);
            }

            results[adapterId] = new AdapterPerformance
            {
                AdapterId = adapterId,
                AvgSwitchTime = timings.Average(),
                MinTime = timings.Min(),
                MaxTime = timings.Max(),
                StdDev = CalculateStdDev(timings)
            };
        }

        return results;
    }

    private AdapterState ExtractCurrentState()
    {
        var state = new AdapterState
        {
            Metadata = new AdapterMetadata
            {
                Id = _currentAdapterId,
                ModifiedAt = DateTime.UtcNow
            }
        };

        void ExtractFromModule(IModule module, string name)
        {
            if (module is ILoRAAdapter adapter)
            {
                var (matrixA, matrixB) = adapter.GetAdapterWeights();

                var weights = new AdapterWeights
                {
                    MatrixA = matrixA!.Clone(),
                    MatrixB = matrixB!.Clone(),
                    LayerType = module.GetType().Name
                };

                if (adapter is LoRALinear linearAdapter)
                {
                    weights.Bias = linearAdapter.GetBias()?.Clone();
                }

                state.Weights[name] = weights;
            }

            if (module is IHasSubmodules hasSubmodules)
            {
                foreach (var (subName, subModule) in hasSubmodules.NamedChildren())
                {
                    var fullName = string.IsNullOrEmpty(name) ? subName : $"{name}.{subName}";
                    ExtractFromModule(subModule, fullName);
                }
            }
        }

        ExtractFromModule(_model, "");
        return state;
    }

    private AdapterState LoadAdapterState(string adapterId)
    {
        // Load from registry
        var state = new AdapterState();

        void LoadIntoModule(IModule module, string name)
        {
            if (module is ILoRAAdapter adapter)
            {
                var (matrixA, matrixB) = adapter.GetAdapterWeights();

                var weights = new AdapterWeights
                {
                    MatrixA = matrixA,
                    MatrixB = matrixB,
                    LayerType = module.GetType().Name
                };

                if (adapter is LoRALinear linearAdapter)
                {
                    weights.Bias = linearAdapter.GetBias();
                }

                state.Weights[name] = weights;
            }

            if (module is IHasSubmodules hasSubmodules)
            {
                foreach (var (subName, subModule) in hasSubmodules.NamedChildren())
                {
                    var fullName = string.IsNullOrEmpty(name) ? subName : $"{name}.{subName}";
                    LoadIntoModule(subModule, fullName);
                }
            }
        }

        LoadIntoModule(_model, "");
        return state;
    }

    private void ApplyAdapterState(AdapterState state)
    {
        void ApplyToModule(IModule module, string name)
        {
            if (module is ILoRAAdapter adapter && state.Weights.TryGetValue(name, out var weights))
            {
                adapter.SetAdapterWeights(weights.MatrixA, weights.MatrixB);

                if (weights.Bias != null && adapter is LoRALinear linearAdapter)
                {
                    linearAdapter.SetBias(weights.Bias);
                }
            }

            if (module is IHasSubmodules hasSubmodules)
            {
                foreach (var (subName, subModule) in hasSubmodules.NamedChildren())
                {
                    var fullName = string.IsNullOrEmpty(name) ? subName : $"{name}.{subName}";
                    ApplyToModule(subModule, fullName);
                }
            }
        }

        ApplyToModule(_model, "");
    }

    private double EstimateCacheMemoryUsage()
    {
        // Estimate memory usage of cached adapters
        double totalElements = 0;

        foreach (var (_, state) in _cachedAdapters)
        {
            foreach (var (_, weights) in state.Weights)
            {
                totalElements += weights.MatrixA.NumElements;
                totalElements += weights.MatrixB.NumElements;
                if (weights.Bias != null)
                {
                    totalElements += weights.Bias.NumElements;
                }
            }
        }

        // Assume float32 (4 bytes per element)
        return (totalElements * 4.0) / (1024.0 * 1024.0);
    }

    private double CalculateStdDev(List<long> values)
    {
        var avg = values.Average();
        var sumOfSquares = values.Sum(v => Math.Pow(v - avg, 2));
        return Math.Sqrt(sumOfSquares / values.Count);
    }
}

/// <summary>
/// Statistics about adapter cache
/// </summary>
public class AdapterCacheStats
{
    public int CachedAdapters { get; set; }
    public string CurrentAdapter { get; set; } = string.Empty;
    public double MemoryUsageMB { get; set; }
}

/// <summary>
/// Performance metrics for an adapter
/// </summary>
public class AdapterPerformance
{
    public string AdapterId { get; set; } = string.Empty;
    public double AvgSwitchTime { get; set; }
    public long MinTime { get; set; }
    public long MaxTime { get; set; }
    public double StdDev { get; set; }
}
```

### 2. Extension Methods
**File**: `src/LoRA/SwitcherExtensions.cs`

```csharp
public static class SwitcherExtensions
{
    /// <summary>
    /// Creates an adapter switcher for a model
    /// </summary>
    public static AdapterSwitcher CreateAdapterSwitcher(
        this IModule model,
        LoRAAdapterRegistry registry,
        int cacheSize = 5)
    {
        return new AdapterSwitcher(model, registry, cacheSize);
    }

    /// <summary>
    /// Creates an adapter switcher with default registry
    /// </summary>
    public static AdapterSwitcher CreateAdapterSwitcher(
        this IModule model,
        string registryPath = "./adapters",
        int cacheSize = 5)
    {
        var registry = new LoRAAdapterRegistry(registryPath);
        return new AdapterSwitcher(model, registry, cacheSize);
    }
}
```

## Testing Requirements

**File**: `tests/LoRA/AdapterSwitcherTests.cs`

1. **Switching Tests**
   - Test SwitchAdapter changes model behavior
   - Test adapter state is preserved
   - Test switching speed is acceptable (<100ms)

2. **Caching Tests**
   - Test PreloadAdapter loads adapter into cache
   - Test cache improves switching speed
   - Test ClearCache removes all cached adapters

3. **Async Tests**
   - Test SwitchAdapterAsync works correctly
   - Test PreloadAdaptersAsync loads multiple adapters
   - Test concurrent switching doesn't cause issues

4. **Performance Tests**
   - Test BenchmarkAdapters returns consistent results
   - Test warmup runs affect benchmark
   - Test cache memory estimation

## Dependencies
- IModule interface (existing)
- ILoRAAdapter interface (from spec 001)
- LoRAAdapterRegistry (from spec 008)

## Success Criteria
- Adapter switching works correctly without model reload
- Caching improves switching performance
- Async operations work without blocking
- Performance benchmarks provide useful metrics
- All unit tests pass

## Estimated Time
45 minutes

## Notes
- Switching should be fast enough for real-time serving (<100ms target)
- Cache size should be configurable based on available memory
- Consider adding adapter version checking for compatibility
- Thread safety may be needed for concurrent switching
