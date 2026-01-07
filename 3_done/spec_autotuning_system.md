# Spec: Autotuning System

## Overview
Implement an autotuning system that benchmarks different fusion strategies, selects optimal configurations based on hardware, and caches tuning results for reuse.

## Requirements

### 1. Autotuning Interface
Core interface for kernel autotuning.

```csharp
public interface IAutotuner
{
    /// <summary>
    /// Tunes a fused operation and selects best configuration
    /// </summary>
    AutotuningResult Tune(
        FusedOperation fusedOp,
        AutotuningOptions options,
        Tensor? benchmarkInput = null);

    /// <summary>
    /// Gets cached tuning result if available
    /// </summary>
    TuningCacheEntry? GetCachedResult(FusedOperation fusedOp, DeviceInfo device);

    /// <summary>
    /// Stores tuning result in cache
    /// </summary>
    void CacheResult(FusedOperation fusedOp, AutotuningResult result, DeviceInfo device);
}

public record AutotuningResult
{
    public required KernelLaunchConfiguration BestConfiguration { get; init; }
    public required double BestExecutionTimeMs { get; init; }
    public required IReadOnlyList<TuningBenchmarkResult> Benchmarks { get; init; }
    public required bool CacheHit { get; init; }
    public required DateTime Timestamp { get; init; }
}

public record TuningBenchmarkResult
{
    public required KernelLaunchConfiguration Configuration { get; init; }
    public required double ExecutionTimeMs { get; init; }
    public required double MemoryBandwidthGBps { get; init; }
    public required int SMUtilizationPercent { get; init; }
}

public record AutotuningOptions
{
    public int MaxIterations { get; init; } = 10;
    public int WarmupRuns { get; init; } = 3;
    public int MeasurementRuns { get; init; } = 5;
    public bool EnableMemoryMeasurement { get; init; } = true;
    public bool EnableSMUtilization { get; init; } = true;
    public SearchStrategy SearchStrategy { get; init; } = SearchStrategy.GridSearch;
}

public enum SearchStrategy
{
    GridSearch,
    RandomSearch,
    BayesianOptimization,
    GeneticAlgorithm
}
```

### 2. Tuning Cache
Cache system for storing and retrieving tuning results.

```csharp
public interface ITuningCache
{
    TuningCacheEntry? Get(FusedOperation fusedOp, DeviceInfo device);
    void Put(FusedOperation fusedOp, AutotuningResult result, DeviceInfo device);
    void Clear();
    int Count { get; }
}

public record TuningCacheEntry
{
    public required FusedOperation Operation { get; init; }
    public required AutotuningResult Result { get; init; }
    public required DeviceInfo Device { get; init; }
    public required DateTime CachedAt { get; init; }
    public required int HitCount { get; init; }
}

public record DeviceInfo
{
    public required string DeviceName { get; init; }
    public required int ComputeCapability { get; init; }
    public required int SMCount { get; init; }
    public required long TotalMemoryBytes { get; init; }
    public required int MaxThreadsPerBlock { get; init; }
    public required int MaxSharedMemoryPerBlock { get; init; }
    public required string Architecture { get; init; }
}

public class InMemoryTuningCache : ITuningCache
{
    private readonly Dictionary<string, TuningCacheEntry> _cache = new();
    private readonly object _lock = new();

    public int Count => _cache.Count;

    public TuningCacheEntry? Get(FusedOperation fusedOp, DeviceInfo device)
    {
        var key = ComputeCacheKey(fusedOp, device);

        lock (_lock)
        {
            if (_cache.TryGetValue(key, out var entry))
            {
                // Update hit count
                _cache[key] = entry with { HitCount = entry.HitCount + 1 };
                return entry;
            }
        }

        return null;
    }

    public void Put(FusedOperation fusedOp, AutotuningResult result, DeviceInfo device)
    {
        var key = ComputeCacheKey(fusedOp, device);

        var entry = new TuningCacheEntry
        {
            Operation = fusedOp,
            Result = result,
            Device = device,
            CachedAt = DateTime.UtcNow,
            HitCount = 0
        };

        lock (_lock)
        {
            _cache[key] = entry;
        }
    }

    public void Clear()
    {
        lock (_lock)
        {
            _cache.Clear();
        }
    }

    private string ComputeCacheKey(FusedOperation fusedOp, DeviceInfo device)
    {
        // Key based on operation signature and device
        var opSignature = fusedOp.Pattern.Name;
        var deviceSignature = $"{device.Architecture}_{device.ComputeCapability}";

        return $"{opSignature}_{deviceSignature}";
    }
}
```

### 3. Search Space Generator
Generate candidate configurations to benchmark.

```csharp
public interface ISearchSpaceGenerator
{
    IReadOnlyList<KernelLaunchConfiguration> GenerateSearchSpace(
        FusedOperation fusedOp,
        DeviceInfo device,
        SearchStrategy strategy,
        int maxIterations);
}

public class SearchSpaceGenerator : ISearchSpaceGenerator
{
    public IReadOnlyList<KernelLaunchConfiguration> GenerateSearchSpace(
        FusedOperation fusedOp,
        DeviceInfo device,
        SearchStrategy strategy,
        int maxIterations)
    {
        return strategy switch
        {
            SearchStrategy.GridSearch => GenerateGridSearch(fusedOp, device),
            SearchStrategy.RandomSearch => GenerateRandomSearch(fusedOp, device, maxIterations),
            SearchStrategy.BayesianOptimization => GenerateBayesianOptimization(fusedOp, device, maxIterations),
            SearchStrategy.GeneticAlgorithm => GenerateGeneticAlgorithm(fusedOp, device, maxIterations),
            _ => GenerateGridSearch(fusedOp, device)
        };
    }

    private IReadOnlyList<KernelLaunchConfiguration> GenerateGridSearch(
        FusedOperation fusedOp,
        DeviceInfo device)
    {
        var configurations = new List<KernelLaunchConfiguration>();

        // Grid search over thread block sizes
        int[] threadBlockX = { 32, 64, 128, 256, 512 };
        int[] threadBlockY = { 1, 2, 4, 8 };

        foreach (var tx in threadBlockX)
        {
            foreach (var ty in threadBlockY)
            {
                if (tx * ty > device.MaxThreadsPerBlock)
                    continue;

                var config = CreateConfiguration(fusedOp, tx, ty, 1, device);
                configurations.Add(config);
            }
        }

        return configurations;
    }

    private IReadOnlyList<KernelLaunchConfiguration> GenerateRandomSearch(
        FusedOperation fusedOp,
        DeviceInfo device,
        int maxIterations)
    {
        var configurations = new List<KernelLaunchConfiguration>();
        var random = new Random();

        for (int i = 0; i < maxIterations; i++)
        {
            // Random thread block sizes
            var tx = GetRandomPowerOfTwo(random, device.MaxThreadsPerBlock);
            var ty = GetRandomPowerOfTwo(random, device.MaxThreadsPerBlock / tx);
            var tz = 1;

            var config = CreateConfiguration(fusedOp, tx, ty, tz, device);
            configurations.Add(config);
        }

        return configurations;
    }

    private IReadOnlyList<KernelLaunchConfiguration> GenerateBayesianOptimization(
        FusedOperation fusedOp,
        DeviceInfo device,
        int maxIterations)
    {
        // Simplified version: use a few grid search points as initial samples
        // Full implementation would use Gaussian Process optimization
        var initialConfigs = GenerateGridSearch(fusedOp, device);
        return initialConfigs.Take(Math.Min(maxIterations, initialConfigs.Count)).ToList();
    }

    private IReadOnlyList<KernelLaunchConfiguration> GenerateGeneticAlgorithm(
        FusedOperation fusedOp,
        DeviceInfo device,
        int maxIterations)
    {
        // Simplified version: start with grid search, then evolve
        var population = GenerateGridSearch(fusedOp, device);

        // Evolution would go here in full implementation
        return population.Take(Math.Min(maxIterations, population.Count)).ToList();
    }

    private KernelLaunchConfiguration CreateConfiguration(
        FusedOperation fusedOp,
        int threadIdx,
        int threadIdxY,
        int threadIdxZ,
        DeviceInfo device)
    {
        var totalThreads = threadIdx * threadIdxY * threadIdxZ;
        var outputElements = fusedOp.OutputShape.TotalElements;
        var gridDimX = (outputElements + totalThreads - 1) / totalThreads;

        return new KernelLaunchConfiguration
        {
            BlockDim = new ThreadBlockConfiguration
            {
                X = threadIdx,
                Y = threadIdxY,
                Z = threadIdxZ
            },
            GridDim = new ThreadBlockConfiguration
            {
                X = Math.Min(gridDimX, device.SMCount * 32), // Limit grid size
                Y = 1,
                Z = 1
            },
            SharedMemoryBytes = Math.Min(
                fusedOp.IntermediateRepresentation.MemoryLayout.SharedMemoryBytes,
                device.MaxSharedMemoryPerBlock),
            Parameters = fusedOp.KernelSpec.Parameters.Select(p =>
                new KernelLaunchParameter
                {
                    Name = p.Name,
                    Value = null, // Filled during execution
                    Type = p.Type
                }).ToList()
        };
    }

    private int GetRandomPowerOfTwo(Random random, int maxValue)
    {
        var power = random.Next(1, 6); // 2^1 to 2^5
        var value = 1 << power;

        while (value > maxValue)
        {
            power--;
            value >>= 1;
        }

        return value;
    }
}
```

### 4. Benchmark Runner
Execute benchmarks for different configurations.

```csharp
public interface IBenchmarkRunner
{
    TuningBenchmarkResult Benchmark(
        FusedOperation fusedOp,
        KernelLaunchConfiguration config,
        Tensor input,
        AutotuningOptions options);

    IReadOnlyList<TuningBenchmarkResult> BenchmarkAll(
        FusedOperation fusedOp,
        IReadOnlyList<KernelLaunchConfiguration> configs,
        Tensor input,
        AutotuningOptions options);
}

public class BenchmarkRunner : IBenchmarkRunner
{
    private readonly IKernelExecutor _executor;
    private readonly IPerformanceProfiler _profiler;

    public BenchmarkRunner(IKernelExecutor executor, IPerformanceProfiler profiler)
    {
        _executor = executor;
        _profiler = profiler;
    }

    public TuningBenchmarkResult Benchmark(
        FusedOperation fusedOp,
        KernelLaunchConfiguration config,
        Tensor input,
        AutotuningOptions options)
    {
        // Warmup runs
        for (int i = 0; i < options.WarmupRuns; i++)
        {
            _executor.ExecuteFusedKernel(fusedOp, config, input);
        }

        // Synchronize to ensure accurate timing
        _executor.Synchronize();

        var measurements = new List<double>();

        // Measurement runs
        for (int i = 0; i < options.MeasurementRuns; i++)
        {
            var measurement = _profiler.MeasureExecution(() =>
            {
                _executor.ExecuteFusedKernel(fusedOp, config, input);
            });

            measurements.Add(measurement);
        }

        // Compute statistics
        var executionTimeMs = measurements.Average();
        var memoryBandwidthGBps = options.EnableMemoryMeasurement
            ? _profiler.MeasureMemoryBandwidth(fusedOp, config)
            : 0.0;
        var smUtilization = options.EnableSMUtilization
            ? _profiler.MeasureSMUtilization(fusedOp, config)
            : 0;

        return new TuningBenchmarkResult
        {
            Configuration = config,
            ExecutionTimeMs = executionTimeMs,
            MemoryBandwidthGBps = memoryBandwidthGBps,
            SMUtilizationPercent = smUtilization
        };
    }

    public IReadOnlyList<TuningBenchmarkResult> BenchmarkAll(
        FusedOperation fusedOp,
        IReadOnlyList<KernelLaunchConfiguration> configs,
        Tensor input,
        AutotuningOptions options)
    {
        var results = new List<TuningBenchmarkResult>();

        foreach (var config in configs)
        {
            var result = Benchmark(fusedOp, config, input, options);
            results.Add(result);
        }

        return results;
    }
}
```

### 5. Autotuner Implementation
Main autotuner implementation.

```csharp
public class Autotuner : IAutotuner
{
    private readonly ITuningCache _cache;
    private readonly ISearchSpaceGenerator _searchSpaceGenerator;
    private readonly IBenchmarkRunner _benchmarkRunner;
    private readonly ITensorGenerator _tensorGenerator;
    private readonly IDeviceQuery _deviceQuery;
    private readonly ILogger _logger;

    public Autotuner(
        ITuningCache cache,
        ISearchSpaceGenerator searchSpaceGenerator,
        IBenchmarkRunner benchmarkRunner,
        ITensorGenerator tensorGenerator,
        IDeviceQuery deviceQuery,
        ILogger logger)
    {
        _cache = cache;
        _searchSpaceGenerator = searchSpaceGenerator;
        _benchmarkRunner = benchmarkRunner;
        _tensorGenerator = tensorGenerator;
        _deviceQuery = deviceQuery;
        _logger = logger;
    }

    public AutotuningResult Tune(
        FusedOperation fusedOp,
        AutotuningOptions options,
        Tensor? benchmarkInput = null)
    {
        var device = _deviceQuery.GetCurrentDeviceInfo();

        // Check cache first
        var cachedEntry = GetCachedResult(fusedOp, device);
        if (cachedEntry != null)
        {
            _logger.LogInformation("Cache hit for fused operation {OpId}", fusedOp.Id);
            return cachedEntry.Result with
            {
                CacheHit = true,
                Timestamp = DateTime.UtcNow
            };
        }

        // Generate search space
        var searchSpace = _searchSpaceGenerator.GenerateSearchSpace(
            fusedOp,
            device,
            options.SearchStrategy,
            options.MaxIterations);

        // Generate benchmark input if not provided
        var input = benchmarkInput ?? _tensorGenerator.GenerateRandomTensor(
            fusedOp.InputShape,
            fusedOp.DataType);

        // Benchmark all configurations
        var benchmarks = _benchmarkRunner.BenchmarkAll(
            fusedOp,
            searchSpace,
            input,
            options);

        // Find best configuration
        var bestBenchmark = benchmarks.OrderBy(b => b.ExecutionTimeMs).First();

        // Create result
        var result = new AutotuningResult
        {
            BestConfiguration = bestBenchmark.Configuration,
            BestExecutionTimeMs = bestBenchmark.ExecutionTimeMs,
            Benchmarks = benchmarks,
            CacheHit = false,
            Timestamp = DateTime.UtcNow
        };

        // Cache result
        CacheResult(fusedOp, result, device);

        return result;
    }

    public TuningCacheEntry? GetCachedResult(FusedOperation fusedOp, DeviceInfo device)
    {
        return _cache.Get(fusedOp, device);
    }

    public void CacheResult(FusedOperation fusedOp, AutotuningResult result, DeviceInfo device)
    {
        _cache.Put(fusedOp, result, device);
    }
}
```

### 6. Performance Profiling
Measure performance metrics for benchmarks.

```csharp
public interface IPerformanceProfiler
{
    double MeasureExecution(Action action);
    double MeasureMemoryBandwidth(FusedOperation fusedOp, KernelLaunchConfiguration config);
    int MeasureSMUtilization(FusedOperation fusedOp, KernelLaunchConfiguration config);
}

public class CudaPerformanceProfiler : IPerformanceProfiler
{
    private readonly IKernelExecutor _executor;

    public CudaPerformanceProfiler(IKernelExecutor executor)
    {
        _executor = executor;
    }

    public double MeasureExecution(Action action)
    {
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();

        action();

        _executor.Synchronize();
        stopwatch.Stop();

        return stopwatch.Elapsed.TotalMilliseconds;
    }

    public double MeasureMemoryBandwidth(FusedOperation fusedOp, KernelLaunchConfiguration config)
    {
        // Estimate based on tensor sizes and execution time
        var totalBytes = fusedOp.InputShape.TotalElements * GetDataTypeSize(fusedOp.DataType) * 2; // read + write

        // This would need actual profiling via CUDA events
        return totalBytes / 1e9; // Simplified
    }

    public int MeasureSMUtilization(FusedOperation fusedOp, KernelLaunchConfiguration config)
    {
        // Estimate SM utilization based on thread count and device SM count
        var totalThreads = config.BlockDim.Total * (config.GridDim.X * config.GridDim.Y * config.GridDim.Z);
        var maxThreadsPerSM = 2048; // Typical value

        var smEstimate = (int)((totalThreads / (double)maxThreadsPerSM) * 100);
        return Math.Min(smEstimate, 100);
    }

    private int GetDataTypeSize(TensorDataType dtype)
    {
        return dtype switch
        {
            TensorDataType.Float32 or TensorDataType.Int32 => 4,
            TensorDataType.Float16 or TensorDataType.Int16 => 2,
            TensorDataType.Int8 => 1,
            TensorDataType.Int64 => 8,
            _ => 4
        };
    }
}
```

## Implementation Tasks

1. **Create autotuning interfaces and records** (20 min)
   - IAutotuner interface
   - AutotuningResult and related records
   - AutotuningOptions and enums
   - DeviceInfo record

2. **Implement ITuningCache** (25 min)
   - InMemoryTuningCache
   - Get/Put methods
   - Cache key computation

3. **Implement ISearchSpaceGenerator** (35 min)
   - SearchSpaceGenerator class
   - Grid search generation
   - Random search generation
   - Configuration creation helper

4. **Implement IBenchmarkRunner** (30 min)
   - BenchmarkRunner class
   - Benchmark with warmup
   - BenchmarkAll for multiple configs
   - Statistics computation

5. **Implement Autotuner** (25 min)
   - Main autotuning logic
   - Cache integration
   - Search space generation
   - Benchmark execution

6. **Implement IPerformanceProfiler** (20 min)
   - CudaPerformanceProfiler
   - Execution time measurement
   - Memory bandwidth estimation
   - SM utilization estimation

## Test Cases

```csharp
[Test]
public void Tune_SelectsBestConfiguration()
{
    var autotuner = CreateAutotuner();
    var fusedOp = CreateSimpleFusedOperation();
    var options = new AutotuningOptions { MaxIterations = 5 };

    var result = autotuner.Tune(fusedOp, options);

    Assert.IsNotNull(result.BestConfiguration);
    Assert.Less(result.BestExecutionTimeMs, result.Benchmarks.First().ExecutionTimeMs + 0.01);
    Assert.AreEqual(result.Benchmarks.Min(b => b.ExecutionTimeMs), result.BestExecutionTimeMs);
}

[Test]
public void Cache_StoresAndRetrievesResult()
{
    var cache = new InMemoryTuningCache();
    var fusedOp = CreateSimpleFusedOperation();
    var device = CreateMockDeviceInfo();
    var result = CreateAutotuningResult();

    cache.Put(fusedOp, result, device);
    var retrieved = cache.Get(fusedOp, device);

    Assert.IsNotNull(retrieved);
    Assert.AreEqual(result.BestConfiguration, retrieved.Result.BestConfiguration);
}

[Test]
public void SearchSpaceGenerator_GeneratesValidConfigs()
{
    var generator = new SearchSpaceGenerator();
    var fusedOp = CreateSimpleFusedOperation();
    var device = CreateMockDeviceInfo();

    var configs = generator.GenerateSearchSpace(
        fusedOp,
        device,
        SearchStrategy.GridSearch,
        10);

    Assert.IsNotEmpty(configs);
    Assert.True(configs.All(c => c.BlockDim.Total <= device.MaxThreadsPerBlock));
}

[Test]
public void BenchmarkRunner_MeasuresExecutionTime()
{
    var runner = CreateBenchmarkRunner();
    var fusedOp = CreateSimpleFusedOperation();
    var config = CreateLaunchConfiguration();
    var input = CreateTestTensor();
    var options = new AutotuningOptions { WarmupRuns = 2, MeasurementRuns = 3 };

    var result = runner.Benchmark(fusedOp, config, input, options);

    Assert.Greater(result.ExecutionTimeMs, 0);
}
```

## Success Criteria
- Autotuner finds optimal configuration from search space
- Cache correctly stores and retrieves tuning results
- Benchmark runner accurately measures execution time
- Search space generator produces valid configurations
- Performance profiler estimates bandwidth and utilization
- Cache hit logic works correctly

## Dependencies
- FusedOperation from fusion engine
- IKernelExecutor (to be defined)
- ITensorGenerator (to be defined)
- IDeviceQuery (to be defined)
- ILogger (to be defined)
