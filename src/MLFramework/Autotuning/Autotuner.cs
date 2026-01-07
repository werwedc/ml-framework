using MLFramework.Fusion;
using RitterFramework.Core;
using Tensor = RitterFramework.Core.Tensor.Tensor;

namespace MLFramework.Autotuning;

/// <summary>
/// Main autotuner implementation
/// </summary>
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
        MLFramework.Fusion.FusedOperation fusedOp,
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

    public TuningCacheEntry? GetCachedResult(MLFramework.Fusion.FusedOperation fusedOp, DeviceInfo device)
    {
        return _cache.Get(fusedOp, device);
    }

    public void CacheResult(MLFramework.Fusion.FusedOperation fusedOp, AutotuningResult result, DeviceInfo device)
    {
        _cache.Put(fusedOp, result, device);
    }
}
