using Xunit;
using ModelZoo.Benchmark;

namespace ModelZooTests.Benchmark;

/// <summary>
/// Unit tests for the BenchmarkResult class.
/// </summary>
public class BenchmarkResultTests
{
    [Fact]
    public void CalculatePercentile_WithValidMeasurements_ReturnsCorrectValue()
    {
        // Arrange
        var result = new BenchmarkResult
        {
            LatencyMeasurements = new List<float> { 10f, 20f, 30f, 40f, 50f }
        };

        // Act
        var p50 = result.CalculatePercentile(50);
        var p95 = result.CalculatePercentile(95);
        var p99 = result.CalculatePercentile(99);

        // Assert
        Assert.Equal(30f, p50);
        Assert.Equal(50f, p95);
        Assert.Equal(50f, p99);
    }

    [Fact]
    public void CalculatePercentile_WithEmptyMeasurements_ReturnsZero()
    {
        // Arrange
        var result = new BenchmarkResult
        {
            LatencyMeasurements = new List<float>()
        };

        // Act
        var percentile = result.CalculatePercentile(50);

        // Assert
        Assert.Equal(0f, percentile);
    }

    [Fact]
    public void CalculatePercentile_WithSingleMeasurement_ReturnsThatValue()
    {
        // Arrange
        var result = new BenchmarkResult
        {
            LatencyMeasurements = new List<float> { 25f }
        };

        // Act
        var percentile = result.CalculatePercentile(50);

        // Assert
        Assert.Equal(25f, percentile);
    }

    [Fact]
    public void GetSummary_ReturnsFormattedString()
    {
        // Arrange
        var result = new BenchmarkResult
        {
            ModelName = "test_model",
            Dataset = "test_dataset",
            Throughput = 100.5f,
            AvgLatency = 10.2f,
            P95Latency = 15.5f,
            Accuracy = 0.95f
        };

        // Act
        var summary = result.GetSummary();

        // Assert
        Assert.Contains("test_model", summary);
        Assert.Contains("test_dataset", summary);
        Assert.Contains("100.50", summary);
        Assert.Contains("10.20", summary);
        Assert.Contains("15.50", summary);
        Assert.Contains("0.9500", summary);
    }

    [Fact]
    public void BenchmarkResult_DefaultValues_AreSetCorrectly()
    {
        // Arrange & Act
        var result = new BenchmarkResult();

        // Assert
        Assert.Equal(string.Empty, result.ModelName);
        Assert.Equal(string.Empty, result.Dataset);
        Assert.Equal(0, result.TotalSamples);
        Assert.Equal(0f, result.Throughput);
        Assert.Equal(0f, result.AvgLatency);
        Assert.Equal(0f, result.MinLatency);
        Assert.Equal(0f, result.MaxLatency);
        Assert.Equal(0f, result.P50Latency);
        Assert.Equal(0f, result.P95Latency);
        Assert.Equal(0f, result.P99Latency);
        Assert.Equal(0L, result.MemoryPeak);
        Assert.Equal(0L, result.MemoryAvg);
        Assert.Equal(0f, result.Accuracy);
        Assert.Equal(TimeSpan.Zero, result.BenchmarkDuration);
        Assert.NotEqual(default, result.Timestamp);
        Assert.NotNull(result.LatencyMeasurements);
        Assert.Empty(result.LatencyMeasurements);
    }
}
