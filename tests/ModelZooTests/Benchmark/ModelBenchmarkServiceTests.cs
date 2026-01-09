using Xunit;
using ModelZoo.Benchmark;

namespace ModelZooTests.Benchmark;

/// <summary>
/// Unit tests for the ModelBenchmarkService class.
/// </summary>
public class ModelBenchmarkServiceTests
{
    private readonly ModelBenchmarkService _service;

    public ModelBenchmarkServiceTests()
    {
        _service = new ModelBenchmarkService();
    }

    [Fact]
    public void Benchmark_WithValidOptions_ReturnsBenchmarkResult()
    {
        // Arrange
        var options = new BenchmarkOptions
        {
            Dataset = "test_dataset",
            BatchSize = 32,
            NumBatches = 5,
            NumIterations = 1,
            WarmupIterations = 2
        };

        // Act
        var result = _service.Benchmark("test_model", options);

        // Assert
        Assert.NotNull(result);
        Assert.Equal("test_model", result.ModelName);
        Assert.Equal("test_dataset", result.Dataset);
        Assert.True(result.TotalSamples > 0);
        Assert.True(result.BenchmarkDuration > TimeSpan.Zero);
        Assert.True(result.Throughput > 0);
        Assert.True(result.AvgLatency > 0);
        Assert.Equal(5, options.NumBatches);
    }

    [Fact]
    public void Benchmark_WithInvalidOptions_ThrowsArgumentException()
    {
        // Arrange
        var options = new BenchmarkOptions
        {
            Dataset = "", // Invalid
            BatchSize = 32
        };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => _service.Benchmark("test_model", options));
    }

    [Fact]
    public void Benchmark_WithWarmupIterations_WarmupDoesNotAffectResults()
    {
        // Arrange
        var optionsWithWarmup = new BenchmarkOptions
        {
            Dataset = "test_dataset",
            BatchSize = 32,
            NumBatches = 5,
            WarmupIterations = 10
        };

        var optionsWithoutWarmup = new BenchmarkOptions
        {
            Dataset = "test_dataset",
            BatchSize = 32,
            NumBatches = 5,
            WarmupIterations = 0
        };

        // Act
        var resultWithWarmup = _service.Benchmark("model1", optionsWithWarmup);
        var resultWithoutWarmup = _service.Benchmark("model2", optionsWithoutWarmup);

        // Assert
        // Results should be similar (within reasonable margin due to randomness)
        var latencyDiff = Math.Abs(resultWithWarmup.AvgLatency - resultWithoutWarmup.AvgLatency);
        Assert.True(latencyDiff < 5.0, "Latency difference should be small");
    }

    [Fact]
    public void Compare_WithMultipleModels_ReturnsComparisonResult()
    {
        // Arrange
        var options = new ComparisonOptions
        {
            Models = new[] { "model1", "model2", "model3" },
            Dataset = "test_dataset",
            BatchSize = 32,
            NumBatches = 5,
            PrimaryMetric = "throughput",
            Metrics = new[] { "throughput", "latency", "accuracy" }
        };

        // Act
        var result = _service.Compare(options.Models, options);

        // Assert
        Assert.NotNull(result);
        Assert.Equal(3, result.ModelCount);
        Assert.NotNull(result.Winner);
        Assert.True(result.ModelResults.ContainsKey("model1"));
        Assert.True(result.ModelResults.ContainsKey("model2"));
        Assert.True(result.ModelResults.ContainsKey("model3"));
        Assert.True(result.RankByMetric.ContainsKey("throughput"));
        Assert.True(result.RankByMetric.ContainsKey("latency"));
        Assert.True(result.RankByMetric.ContainsKey("accuracy"));
    }

    [Fact]
    public void Compare_WithThroughputPrimaryMetric_SelectsHighestThroughputModel()
    {
        // Arrange
        var options = new ComparisonOptions
        {
            Models = new[] { "model1", "model2" },
            Dataset = "test_dataset",
            BatchSize = 32,
            NumBatches = 10,
            PrimaryMetric = "throughput",
            Metrics = new[] { "throughput" }
        };

        // Act
        var result = _service.Compare(options.Models, options);

        // Assert
        Assert.NotNull(result.Winner);
        var winnerResult = result.GetModelResult(result.Winner);
        var loserModel = result.ModelResults.Keys.First(m => m != result.Winner);
        var loserResult = result.GetModelResult(loserModel);

        if (winnerResult != null && loserResult != null)
        {
            Assert.True(winnerResult.Throughput >= loserResult.Throughput);
        }
    }

    [Fact]
    public void Compare_WithLatencyPrimaryMetric_SelectsLowestLatencyModel()
    {
        // Arrange
        var options = new ComparisonOptions
        {
            Models = new[] { "model1", "model2" },
            Dataset = "test_dataset",
            BatchSize = 32,
            NumBatches = 10,
            PrimaryMetric = "latency",
            Metrics = new[] { "latency" }
        };

        // Act
        var result = _service.Compare(options.Models, options);

        // Assert
        Assert.NotNull(result.Winner);
        var winnerResult = result.GetModelResult(result.Winner);
        var loserModel = result.ModelResults.Keys.First(m => m != result.Winner);
        var loserResult = result.GetModelResult(loserModel);

        if (winnerResult != null && loserResult != null)
        {
            Assert.True(winnerResult.AvgLatency <= loserResult.AvgLatency);
        }
    }

    [Fact]
    public void BenchmarkBatch_WithMultipleBenchmarks_ReturnsAllResults()
    {
        // Arrange
        var benchmarks = new Dictionary<string, BenchmarkOptions>
        {
            ["model1"] = new BenchmarkOptions { Dataset = "test1", BatchSize = 32, NumBatches = 5 },
            ["model2"] = new BenchmarkOptions { Dataset = "test2", BatchSize = 32, NumBatches = 5 },
            ["model3"] = new BenchmarkOptions { Dataset = "test3", BatchSize = 32, NumBatches = 5 }
        };

        // Act
        var results = _service.BenchmarkBatch(benchmarks);

        // Assert
        Assert.Equal(3, results.Count);
        Assert.True(results.ContainsKey("model1"));
        Assert.True(results.ContainsKey("model2"));
        Assert.True(results.ContainsKey("model3"));
    }

    [Fact]
    public void Benchmark_WithMemoryProfileEnabled_IncludesMemoryMetrics()
    {
        // Arrange
        var options = new BenchmarkOptions
        {
            Dataset = "test_dataset",
            BatchSize = 32,
            NumBatches = 5,
            IncludeMemoryProfile = true
        };

        // Act
        var result = _service.Benchmark("test_model", options);

        // Assert
        Assert.True(result.MemoryPeak > 0);
        Assert.True(result.MemoryAvg > 0);
    }

    [Fact]
    public void Benchmark_WithMemoryProfileDisabled_MemoryMetricsAreZero()
    {
        // Arrange
        var options = new BenchmarkOptions
        {
            Dataset = "test_dataset",
            BatchSize = 32,
            NumBatches = 5,
            IncludeMemoryProfile = false
        };

        // Act
        var result = _service.Benchmark("test_model", options);

        // Assert
        Assert.Equal(0L, result.MemoryPeak);
        Assert.Equal(0L, result.MemoryAvg);
    }

    [Fact]
    public void Benchmark_WithAccuracyCalculation_IncludesAccuracyMetric()
    {
        // Arrange
        var options = new BenchmarkOptions
        {
            Dataset = "test_dataset",
            BatchSize = 32,
            NumBatches = 10
        };

        // Act
        var result = _service.Benchmark("test_model", options);

        // Assert
        Assert.True(result.Accuracy >= 0);
        Assert.True(result.Accuracy <= 1.0);
    }
}
