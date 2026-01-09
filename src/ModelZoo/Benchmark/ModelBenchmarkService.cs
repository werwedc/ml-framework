using System.Diagnostics;

namespace ModelZoo.Benchmark;

/// <summary>
/// Interface for models that can be benchmarked.
/// </summary>
public interface IBenchmarkableModel
{
    /// <summary>
    /// Gets the model name.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Runs inference on the given input.
    /// </summary>
    /// <param name="input">The input data.</param>
    /// <returns>The inference output.</returns>
    object Infer(object input);
}

/// <summary>
/// Service for benchmarking model performance on datasets.
/// </summary>
public class ModelBenchmarkService
{
    private readonly BenchmarkHistory _history;

    /// <summary>
    /// Initializes a new instance of the ModelBenchmarkService class.
    /// </summary>
    public ModelBenchmarkService()
    {
        _history = new BenchmarkHistory();
    }

    /// <summary>
    /// Initializes a new instance of the ModelBenchmarkService class with a history instance.
    /// </summary>
    /// <param name="history">The benchmark history to use.</param>
    public ModelBenchmarkService(BenchmarkHistory history)
    {
        _history = history ?? throw new ArgumentNullException(nameof(history));
    }

    /// <summary>
    /// Benchmarks a model by name using the specified options.
    /// </summary>
    /// <param name="modelName">The name of the model to benchmark.</param>
    /// <param name="options">The benchmark options.</param>
    /// <returns>The benchmark result.</returns>
    public BenchmarkResult Benchmark(string modelName, BenchmarkOptions options)
    {
        options.Validate();

        // In a real implementation, we would load the model from a registry
        // For now, we'll create a mock model
        var model = CreateMockModel(modelName);

        return BenchmarkModel(model, options);
    }

    /// <summary>
    /// Benchmarks a model object directly.
    /// </summary>
    /// <param name="model">The model to benchmark.</param>
    /// <param name="options">The benchmark options.</param>
    /// <returns>The benchmark result.</returns>
    public BenchmarkResult BenchmarkModel(IBenchmarkableModel model, BenchmarkOptions options)
    {
        if (model == null)
        {
            throw new ArgumentNullException(nameof(model));
        }

        options.Validate();

        var result = new BenchmarkResult
        {
            ModelName = model.Name,
            Dataset = options.Dataset,
            Timestamp = DateTime.UtcNow
        };

        var stopwatch = Stopwatch.StartNew();
        var latencyMeasurements = new List<float>();
        var memorySnapshots = new List<long>();

        // Create mock data loader
        var dataLoader = CreateMockDataLoader(options);

        // Determine number of batches
        var numBatches = options.NumBatches ?? 100;
        var totalSamples = 0;

        try
        {
            // Warmup iterations
            for (int i = 0; i < options.WarmupIterations; i++)
            {
                var batch = dataLoader.GetBatch(options.BatchSize);
                model.Infer(batch);
                GC.Collect();
                GC.WaitForPendingFinalizers();
            }

            // Benchmark iterations
            for (int batchIdx = 0; batchIdx < numBatches; batchIdx++)
            {
                var batch = dataLoader.GetBatch(options.BatchSize);
                totalSamples += options.BatchSize;

                // Memory snapshot before (if profiling)
                if (options.IncludeMemoryProfile)
                {
                    memorySnapshots.Add(GC.GetTotalMemory(false));
                }

                // Preprocess time
                var preprocessStart = DateTime.UtcNow;

                // Inference iterations
                for (int iter = 0; iter < options.NumIterations; iter++)
                {
                    var iterStart = DateTime.UtcNow;
                    model.Infer(batch);
                    var iterEnd = DateTime.UtcNow;

                    if (!options.Preprocess && !options.Postprocess)
                    {
                        latencyMeasurements.Add((float)(iterEnd - iterStart).TotalMilliseconds);
                    }
                }

                // Postprocess time
                var postprocessEnd = DateTime.UtcNow;

                if (options.Preprocess || options.Postprocess)
                {
                    var totalLatency = (float)(postprocessEnd - preprocessStart).TotalMilliseconds;
                    latencyMeasurements.Add(totalLatency);
                }

                // Memory snapshot after (if profiling)
                if (options.IncludeMemoryProfile)
                {
                    memorySnapshots.Add(GC.GetTotalMemory(false));
                }
            }

            stopwatch.Stop();

            // Calculate statistics
            result.TotalSamples = totalSamples;
            result.BenchmarkDuration = stopwatch.Elapsed;
            result.LatencyMeasurements = latencyMeasurements;

            if (latencyMeasurements.Count > 0)
            {
                result.AvgLatency = latencyMeasurements.Average();
                result.MinLatency = latencyMeasurements.Min();
                result.MaxLatency = latencyMeasurements.Max();
                result.P50Latency = result.CalculatePercentile(50);
                result.P95Latency = result.CalculatePercentile(95);
                result.P99Latency = result.CalculatePercentile(99);
            }

            if (memorySnapshots.Count > 0)
            {
                result.MemoryPeak = memorySnapshots.Max();
                result.MemoryAvg = (long)memorySnapshots.Average();
            }

            // Calculate throughput
            if (result.BenchmarkDuration.TotalSeconds > 0)
            {
                result.Throughput = result.TotalSamples / (float)result.BenchmarkDuration.TotalSeconds;
            }

            // Mock accuracy calculation (would be real in production)
            result.Accuracy = CalculateMockAccuracy(totalSamples);

            // Save to history
            _history.SaveResult(result);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Benchmark failed for model {model.Name}", ex);
        }

        return result;
    }

    /// <summary>
    /// Compares multiple models using the specified options.
    /// </summary>
    /// <param name="modelNames">The names of the models to compare.</param>
    /// <param name="options">The comparison options.</param>
    /// <returns>The comparison result.</returns>
    public ComparisonResult Compare(string[] modelNames, ComparisonOptions options)
    {
        if (modelNames == null || modelNames.Length == 0)
        {
            throw new ArgumentException("At least one model must be specified.", nameof(modelNames));
        }

        options.Validate();

        var result = new ComparisonResult
        {
            Options = options,
            Timestamp = DateTime.UtcNow
        };

        // Benchmark each model
        foreach (var modelName in modelNames)
        {
            var benchmarkOptions = new BenchmarkOptions
            {
                Dataset = options.Dataset,
                Subset = options.Subset,
                BatchSize = options.BatchSize,
                Device = options.Device,
                NumBatches = options.NumBatches,
                NumIterations = options.NumIterations,
                WarmupIterations = options.WarmupIterations,
                Preprocess = true,
                Postprocess = true,
                IncludeMemoryProfile = false
            };

            var modelResult = Benchmark(modelName, benchmarkOptions);
            result.ModelResults[modelName] = modelResult;
        }

        // Determine winner based on primary metric
        result.Winner = DetermineWinner(result, options.PrimaryMetric);

        // Calculate rankings by metric
        result.RankByMetric = CalculateRankings(result);

        // Calculate statistical significance
        result.StatisticalSignificance = CalculateStatisticalSignificance(result);

        return result;
    }

    /// <summary>
    /// Runs multiple benchmarks in parallel.
    /// </summary>
    /// <param name="benchmarks">Dictionary mapping model names to benchmark options.</param>
    /// <returns>Dictionary mapping model names to benchmark results.</returns>
    public Dictionary<string, BenchmarkResult> BenchmarkBatch(
        Dictionary<string, BenchmarkOptions> benchmarks)
    {
        if (benchmarks == null || benchmarks.Count == 0)
        {
            throw new ArgumentException("At least one benchmark must be specified.", nameof(benchmarks));
        }

        var results = new ConcurrentDictionary<string, BenchmarkResult>();
        var tasks = benchmarks.Select(kvp =>
            Task.Run(() =>
            {
                var result = Benchmark(kvp.Key, kvp.Value);
                results.TryAdd(kvp.Key, result);
            })
        );

        Task.WaitAll(tasks.ToArray());

        return results.ToDictionary(kvp => kvp.Key, kvp => kvp.Value);
    }

    /// <summary>
    /// Gets the benchmark history instance.
    /// </summary>
    public BenchmarkHistory History => _history;

    /// <summary>
    /// Determines the winning model based on the specified metric.
    /// </summary>
    private static string? DetermineWinner(ComparisonResult result, string metric)
    {
        if (result.ModelResults.Count == 0)
        {
            return null;
        }

        return metric.ToLower() switch
        {
            "throughput" or "accuracy" => result.ModelResults
                .OrderByDescending(kvp => GetMetricValue(kvp.Value, metric))
                .First().Key,

            "latency" or "memory" => result.ModelResults
                .OrderBy(kvp => GetMetricValue(kvp.Value, metric))
                .First().Key,

            _ => throw new ArgumentException($"Unknown metric: {metric}", nameof(metric))
        };
    }

    /// <summary>
    /// Gets the value of a specific metric from a benchmark result.
    /// </summary>
    private static float GetMetricValue(BenchmarkResult result, string metric)
    {
        return metric.ToLower() switch
        {
            "throughput" => result.Throughput,
            "latency" => result.AvgLatency,
            "accuracy" => result.Accuracy,
            "memory" => result.MemoryAvg,
            _ => 0f
        };
    }

    /// <summary>
    /// Calculates rankings for each metric.
    /// </summary>
    private static Dictionary<string, string[]> CalculateRankings(ComparisonResult result)
    {
        var rankings = new Dictionary<string, string[]>();

        foreach (var metric in result.Options!.Metrics)
        {
            var sortedModels = metric.ToLower() switch
            {
                "throughput" or "accuracy" => result.ModelResults
                    .OrderByDescending(kvp => GetMetricValue(kvp.Value, metric))
                    .Select(kvp => kvp.Key)
                    .ToArray(),

                "latency" or "memory" => result.ModelResults
                    .OrderBy(kvp => GetMetricValue(kvp.Value, metric))
                    .Select(kvp => kvp.Key)
                    .ToArray(),

                _ => result.ModelResults.Keys.ToArray()
            };

            rankings[metric] = sortedModels;
        }

        return rankings;
    }

    /// <summary>
    /// Calculates statistical significance between model pairs.
    /// </summary>
    private static Dictionary<string, bool> CalculateStatisticalSignificance(ComparisonResult result)
    {
        var significance = new Dictionary<string, bool>();
        var models = result.ModelResults.Keys.ToList();

        for (int i = 0; i < models.Count; i++)
        {
            for (int j = i + 1; j < models.Count; j++)
            {
                var model1 = models[i];
                var model2 = models[j];
                var result1 = result.ModelResults[model1];
                var result2 = result.ModelResults[model2];

                // Simple t-test approximation for statistical significance
                var isSignificant = PerformTTest(result1.LatencyMeasurements, result2.LatencyMeasurements);
                significance[$"{model1}_vs_{model2}"] = isSignificant;
            }
        }

        return significance;
    }

    /// <summary>
    /// Performs a two-sample t-test to determine if differences are statistically significant.
    /// </summary>
    private static bool PerformTTest(List<float> sample1, List<float> sample2)
    {
        if (sample1.Count < 2 || sample2.Count < 2)
        {
            return false;
        }

        var mean1 = sample1.Average();
        var mean2 = sample2.Average();
        var var1 = sample1.Sum(x => Math.Pow(x - mean1, 2)) / (sample1.Count - 1);
        var var2 = sample2.Sum(x => Math.Pow(x - mean2, 2)) / (sample2.Count - 1);

        var pooledStdErr = Math.Sqrt(var1 / sample1.Count + var2 / sample2.Count);
        var tStat = (mean1 - mean2) / pooledStdErr;
        var degreesOfFreedom = sample1.Count + sample2.Count - 2;

        // Critical value for 95% confidence (approximate)
        var criticalValue = 1.96;

        return Math.Abs(tStat) > criticalValue;
    }

    /// <summary>
    /// Creates a mock model for testing purposes.
    /// </summary>
    private static IBenchmarkableModel CreateMockModel(string name)
    {
        return new MockModel(name);
    }

    /// <summary>
    /// Creates a mock data loader for testing purposes.
    /// </summary>
    private static MockDataLoader CreateMockDataLoader(BenchmarkOptions options)
    {
        return new MockDataLoader(options.Dataset);
    }

    /// <summary>
    /// Calculates a mock accuracy for testing purposes.
    /// </summary>
    private static float CalculateMockAccuracy(int numSamples)
    {
        // Random accuracy between 0.8 and 0.99
        var random = new Random();
        return 0.8f + (float)random.NextDouble() * 0.19f;
    }

    /// <summary>
    /// Mock model implementation for testing.
    /// </summary>
    private class MockModel : IBenchmarkableModel
    {
        private readonly Random _random = new Random();

        public MockModel(string name)
        {
            Name = name;
        }

        public string Name { get; }

        public object Infer(object input)
        {
            // Simulate inference delay
            Thread.Sleep(1);
            return new object();
        }
    }

    /// <summary>
    /// Mock data loader for testing.
    /// </summary>
    private class MockDataLoader
    {
        private readonly string _datasetName;
        private int _batchIndex;

        public MockDataLoader(string datasetName)
        {
            _datasetName = datasetName;
        }

        public object[] GetBatch(int batchSize)
        {
            return Enumerable.Range(0, batchSize)
                .Select(i => new object())
                .ToArray();
        }
    }
}
