using Xunit;
using ModelZoo.Benchmark;

namespace ModelZooTests.Benchmark;

/// <summary>
/// Unit tests for the BenchmarkOptions class.
/// </summary>
public class BenchmarkOptionsTests
{
    [Fact]
    public void Validate_WithValidOptions_DoesNotThrow()
    {
        // Arrange
        var options = new BenchmarkOptions
        {
            Dataset = "test_dataset",
            Subset = "test",
            BatchSize = 32,
            NumBatches = 10,
            NumIterations = 1,
            WarmupIterations = 5,
            Preprocess = true,
            Postprocess = true,
            IncludeMemoryProfile = false
        };

        // Act & Assert
        options.Validate();
    }

    [Fact]
    public void Validate_WithEmptyDataset_ThrowsArgumentException()
    {
        // Arrange
        var options = new BenchmarkOptions
        {
            Dataset = "",
            BatchSize = 32
        };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => options.Validate());
    }

    [Fact]
    public void Validate_WithInvalidBatchSize_ThrowsArgumentException()
    {
        // Arrange
        var options = new BenchmarkOptions
        {
            Dataset = "test_dataset",
            BatchSize = 0
        };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => options.Validate());
    }

    [Fact]
    public void Validate_WithInvalidNumBatches_ThrowsArgumentException()
    {
        // Arrange
        var options = new BenchmarkOptions
        {
            Dataset = "test_dataset",
            BatchSize = 32,
            NumBatches = -1
        };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => options.Validate());
    }

    [Fact]
    public void Validate_WithInvalidNumIterations_ThrowsArgumentException()
    {
        // Arrange
        var options = new BenchmarkOptions
        {
            Dataset = "test_dataset",
            BatchSize = 32,
            NumIterations = 0
        };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => options.Validate());
    }

    [Fact]
    public void Validate_WithInvalidWarmupIterations_ThrowsArgumentException()
    {
        // Arrange
        var options = new BenchmarkOptions
        {
            Dataset = "test_dataset",
            BatchSize = 32,
            WarmupIterations = -1
        };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => options.Validate());
    }
}
