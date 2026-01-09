using Xunit;
using ModelZoo.Benchmark;

namespace ModelZooTests.Benchmark;

/// <summary>
/// Unit tests for the ComparisonOptions class.
/// </summary>
public class ComparisonOptionsTests
{
    [Fact]
    public void Validate_WithValidOptions_DoesNotThrow()
    {
        // Arrange
        var options = new ComparisonOptions
        {
            Models = new[] { "model1", "model2" },
            Dataset = "test_dataset",
            Subset = "test",
            BatchSize = 32,
            Metrics = new[] { "throughput", "latency" },
            PrimaryMetric = "throughput",
            Parallel = true
        };

        // Act & Assert
        options.Validate();
    }

    [Fact]
    public void Validate_WithEmptyModels_ThrowsArgumentException()
    {
        // Arrange
        var options = new ComparisonOptions
        {
            Models = Array.Empty<string>(),
            Dataset = "test_dataset"
        };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => options.Validate());
    }

    [Fact]
    public void Validate_WithNullModels_ThrowsArgumentException()
    {
        // Arrange
        var options = new ComparisonOptions
        {
            Dataset = "test_dataset"
        };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => options.Validate());
    }

    [Fact]
    public void Validate_WithEmptyDataset_ThrowsArgumentException()
    {
        // Arrange
        var options = new ComparisonOptions
        {
            Models = new[] { "model1" },
            Dataset = ""
        };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => options.Validate());
    }

    [Fact]
    public void Validate_WithInvalidBatchSize_ThrowsArgumentException()
    {
        // Arrange
        var options = new ComparisonOptions
        {
            Models = new[] { "model1" },
            Dataset = "test_dataset",
            BatchSize = 0
        };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => options.Validate());
    }

    [Fact]
    public void Validate_WithInvalidMetrics_ThrowsArgumentException()
    {
        // Arrange
        var options = new ComparisonOptions
        {
            Models = new[] { "model1" },
            Dataset = "test_dataset",
            Metrics = new[] { "invalid_metric" }
        };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => options.Validate());
    }

    [Fact]
    public void Validate_WithInvalidPrimaryMetric_ThrowsArgumentException()
    {
        // Arrange
        var options = new ComparisonOptions
        {
            Models = new[] { "model1" },
            Dataset = "test_dataset",
            PrimaryMetric = "invalid_metric"
        };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => options.Validate());
    }

    [Fact]
    public void Validate_WithMixedValidInvalidMetrics_ThrowsArgumentException()
    {
        // Arrange
        var options = new ComparisonOptions
        {
            Models = new[] { "model1" },
            Dataset = "test_dataset",
            Metrics = new[] { "throughput", "invalid_metric" }
        };

        // Act & Assert
        var ex = Assert.Throws<ArgumentException>(() => options.Validate());
        Assert.Contains("invalid_metric", ex.Message);
    }
}
