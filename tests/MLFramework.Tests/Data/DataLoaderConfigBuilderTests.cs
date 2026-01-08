using Xunit;
using MLFramework.Data;
using System;

namespace MLFramework.Tests.Data;

/// <summary>
/// Unit tests for DataLoaderConfigBuilder class.
/// </summary>
public class DataLoaderConfigBuilderTests
{
    [Fact]
    public void Build_WithDefaultValues_ShouldCreateValidConfig()
    {
        // Arrange
        var builder = new DataLoaderConfigBuilder();

        // Act
        var config = builder.Build();

        // Assert
        Assert.Equal(4, config.NumWorkers);
        Assert.Equal(32, config.BatchSize);
        Assert.Equal(2, config.PrefetchCount);
        Assert.Equal(10, config.QueueSize);
        Assert.True(config.Shuffle);
        Assert.Equal(42, config.Seed);
        Assert.True(config.PinMemory);
    }

    [Fact]
    public void Build_WithNumWorkersZero_ShouldUseProcessorCount()
    {
        // Arrange
        var builder = new DataLoaderConfigBuilder().WithNumWorkers(0);

        // Act
        var config = builder.Build();

        // Assert
        Assert.Equal(Environment.ProcessorCount, config.NumWorkers);
    }

    [Fact]
    public void Build_WithNegativeNumWorkers_ShouldThrowException()
    {
        // Arrange
        var builder = new DataLoaderConfigBuilder().WithNumWorkers(-1);

        // Act & Assert
        var exception = Assert.Throws<ArgumentOutOfRangeException>(() => builder.Build());
        Assert.Equal("numWorkers", exception.ParamName);
        Assert.Contains("NumWorkers must be >= 0", exception.Message);
    }

    [Fact]
    public void Build_WithInvalidBatchSize_ShouldThrowException()
    {
        // Arrange
        var builder = new DataLoaderConfigBuilder().WithBatchSize(0);

        // Act & Assert
        var exception = Assert.Throws<ArgumentOutOfRangeException>(() => builder.Build());
        Assert.Equal("batchSize", exception.ParamName);
        Assert.Contains("BatchSize must be > 0", exception.Message);
    }

    [Fact]
    public void Build_WithNegativePrefetchCount_ShouldThrowException()
    {
        // Arrange
        var builder = new DataLoaderConfigBuilder().WithPrefetchCount(-1);

        // Act & Assert
        var exception = Assert.Throws<ArgumentOutOfRangeException>(() => builder.Build());
        Assert.Equal("prefetchCount", exception.ParamName);
        Assert.Contains("PrefetchCount must be >= 0", exception.Message);
    }

    [Fact]
    public void Build_WithQueueSizeSmallerThanPrefetchCountPlusOne_ShouldThrowException()
    {
        // Arrange
        var builder = new DataLoaderConfigBuilder()
            .WithPrefetchCount(2)
            .WithQueueSize(2);

        // Act & Assert
        var exception = Assert.Throws<ArgumentOutOfRangeException>(() => builder.Build());
        Assert.Equal("queueSize", exception.ParamName);
        Assert.Contains("QueueSize must be >= PrefetchCount + 1 (>= 3)", exception.Message);
    }

    [Fact]
    public void WithNumWorkers_ShouldReturnBuilderForChaining()
    {
        // Arrange
        var builder = new DataLoaderConfigBuilder();

        // Act
        var result = builder.WithNumWorkers(8);

        // Assert
        Assert.Same(builder, result);
    }

    [Fact]
    public void WithBatchSize_ShouldReturnBuilderForChaining()
    {
        // Arrange
        var builder = new DataLoaderConfigBuilder();

        // Act
        var result = builder.WithBatchSize(64);

        // Assert
        Assert.Same(builder, result);
    }

    [Fact]
    public void WithPrefetchCount_ShouldReturnBuilderForChaining()
    {
        // Arrange
        var builder = new DataLoaderConfigBuilder();

        // Act
        var result = builder.WithPrefetchCount(4);

        // Assert
        Assert.Same(builder, result);
    }

    [Fact]
    public void WithQueueSize_ShouldReturnBuilderForChaining()
    {
        // Arrange
        var builder = new DataLoaderConfigBuilder();

        // Act
        var result = builder.WithQueueSize(20);

        // Assert
        Assert.Same(builder, result);
    }

    [Fact]
    public void WithShuffle_ShouldReturnBuilderForChaining()
    {
        // Arrange
        var builder = new DataLoaderConfigBuilder();

        // Act
        var result = builder.WithShuffle(false);

        // Assert
        Assert.Same(builder, result);
    }

    [Fact]
    public void WithSeed_ShouldReturnBuilderForChaining()
    {
        // Arrange
        var builder = new DataLoaderConfigBuilder();

        // Act
        var result = builder.WithSeed(123);

        // Assert
        Assert.Same(builder, result);
    }

    [Fact]
    public void WithPinMemory_ShouldReturnBuilderForChaining()
    {
        // Arrange
        var builder = new DataLoaderConfigBuilder();

        // Act
        var result = builder.WithPinMemory(false);

        // Assert
        Assert.Same(builder, result);
    }

    [Fact]
    public void Build_WithMethodChaining_ShouldCreateConfigWithAllValues()
    {
        // Arrange
        var builder = new DataLoaderConfigBuilder()
            .WithNumWorkers(8)
            .WithBatchSize(64)
            .WithPrefetchCount(4)
            .WithQueueSize(20)
            .WithShuffle(false)
            .WithSeed(123)
            .WithPinMemory(false);

        // Act
        var config = builder.Build();

        // Assert
        Assert.Equal(8, config.NumWorkers);
        Assert.Equal(64, config.BatchSize);
        Assert.Equal(4, config.PrefetchCount);
        Assert.Equal(20, config.QueueSize);
        Assert.False(config.Shuffle);
        Assert.Equal(123, config.Seed);
        Assert.False(config.PinMemory);
    }

    [Fact]
    public void Build_CanReuseBuilderWithDifferentValues()
    {
        // Arrange
        var builder = new DataLoaderConfigBuilder().WithNumWorkers(8);

        // Act
        var config1 = builder.WithBatchSize(32).Build();
        var config2 = builder.WithBatchSize(64).Build();

        // Assert
        Assert.Equal(32, config1.BatchSize);
        Assert.Equal(64, config2.BatchSize);
    }
}
