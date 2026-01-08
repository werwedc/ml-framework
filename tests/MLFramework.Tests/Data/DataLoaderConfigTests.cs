using Xunit;
using MLFramework.Data;
using System;

namespace MLFramework.Tests.Data;

/// <summary>
/// Unit tests for DataLoaderConfig class.
/// </summary>
public class DataLoaderConfigTests
{
    [Fact]
    public void Constructor_WithDefaultValues_ShouldCreateValidConfig()
    {
        // Act
        var config = new DataLoaderConfig();

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
    public void Constructor_WithNumWorkersZero_ShouldUseProcessorCount()
    {
        // Act
        var config = new DataLoaderConfig(numWorkers: 0);

        // Assert
        Assert.Equal(Environment.ProcessorCount, config.NumWorkers);
    }

    [Fact]
    public void Constructor_WithNegativeNumWorkers_ShouldThrowException()
    {
        // Act & Assert
        var exception = Assert.Throws<ArgumentOutOfRangeException>(() =>
            new DataLoaderConfig(numWorkers: -1));

        Assert.Equal("numWorkers", exception.ParamName);
        Assert.Contains("NumWorkers must be >= 0", exception.Message);
    }

    [Fact]
    public void Constructor_WithInvalidBatchSize_ShouldThrowException()
    {
        // Act & Assert
        var exception = Assert.Throws<ArgumentOutOfRangeException>(() =>
            new DataLoaderConfig(batchSize: 0));

        Assert.Equal("batchSize", exception.ParamName);
        Assert.Contains("BatchSize must be > 0", exception.Message);
    }

    [Fact]
    public void Constructor_WithNegativeBatchSize_ShouldThrowException()
    {
        // Act & Assert
        var exception = Assert.Throws<ArgumentOutOfRangeException>(() =>
            new DataLoaderConfig(batchSize: -1));

        Assert.Equal("batchSize", exception.ParamName);
        Assert.Contains("BatchSize must be > 0", exception.Message);
    }

    [Fact]
    public void Constructor_WithNegativePrefetchCount_ShouldThrowException()
    {
        // Act & Assert
        var exception = Assert.Throws<ArgumentOutOfRangeException>(() =>
            new DataLoaderConfig(prefetchCount: -1));

        Assert.Equal("prefetchCount", exception.ParamName);
        Assert.Contains("PrefetchCount must be >= 0", exception.Message);
    }

    [Fact]
    public void Constructor_WithQueueSizeSmallerThanPrefetchCountPlusOne_ShouldThrowException()
    {
        // Act & Assert
        var exception = Assert.Throws<ArgumentOutOfRangeException>(() =>
            new DataLoaderConfig(prefetchCount: 2, queueSize: 2));

        Assert.Equal("queueSize", exception.ParamName);
        Assert.Contains("QueueSize must be >= PrefetchCount + 1 (>= 3)", exception.Message);
    }

    [Fact]
    public void Constructor_WithQueueSizeEqualPrefetchCountPlusOne_ShouldSucceed()
    {
        // Act & Assert
        var config = new DataLoaderConfig(prefetchCount: 2, queueSize: 3);

        Assert.Equal(2, config.PrefetchCount);
        Assert.Equal(3, config.QueueSize);
    }

    [Fact]
    public void Constructor_WithCustomValues_ShouldSetAllProperties()
    {
        // Act
        var config = new DataLoaderConfig(
            numWorkers: 8,
            batchSize: 64,
            prefetchCount: 4,
            queueSize: 20,
            shuffle: false,
            seed: 123,
            pinMemory: false);

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
    public void Clone_ShouldCreateIndependentCopy()
    {
        // Arrange
        var original = new DataLoaderConfig(
            numWorkers: 8,
            batchSize: 64,
            prefetchCount: 4,
            queueSize: 20,
            shuffle: false,
            seed: 123,
            pinMemory: false);

        // Act
        var cloned = original.Clone();

        // Assert
        Assert.Equal(original.NumWorkers, cloned.NumWorkers);
        Assert.Equal(original.BatchSize, cloned.BatchSize);
        Assert.Equal(original.PrefetchCount, cloned.PrefetchCount);
        Assert.Equal(original.QueueSize, cloned.QueueSize);
        Assert.Equal(original.Shuffle, cloned.Shuffle);
        Assert.Equal(original.Seed, cloned.Seed);
        Assert.Equal(original.PinMemory, cloned.PinMemory);

        // Ensure they are independent objects
        Assert.NotSame(original, cloned);
    }

    [Fact]
    public void WithNumWorkers_ShouldCreateNewInstanceWithUpdatedValue()
    {
        // Arrange
        var original = new DataLoaderConfig(numWorkers: 4);

        // Act
        var modified = original.WithNumWorkers(8);

        // Assert
        Assert.Equal(4, original.NumWorkers);
        Assert.Equal(8, modified.NumWorkers);
        Assert.NotSame(original, modified);
    }

    [Fact]
    public void WithBatchSize_ShouldCreateNewInstanceWithUpdatedValue()
    {
        // Arrange
        var original = new DataLoaderConfig(batchSize: 32);

        // Act
        var modified = original.WithBatchSize(64);

        // Assert
        Assert.Equal(32, original.BatchSize);
        Assert.Equal(64, modified.BatchSize);
        Assert.NotSame(original, modified);
    }

    [Fact]
    public void WithPrefetchCount_ShouldCreateNewInstanceWithUpdatedValue()
    {
        // Arrange
        var original = new DataLoaderConfig(prefetchCount: 2);

        // Act
        var modified = original.WithPrefetchCount(4);

        // Assert
        Assert.Equal(2, original.PrefetchCount);
        Assert.Equal(4, modified.PrefetchCount);
        Assert.NotSame(original, modified);
    }

    [Fact]
    public void WithQueueSize_ShouldCreateNewInstanceWithUpdatedValue()
    {
        // Arrange
        var original = new DataLoaderConfig(queueSize: 10);

        // Act
        var modified = original.WithQueueSize(20);

        // Assert
        Assert.Equal(10, original.QueueSize);
        Assert.Equal(20, modified.QueueSize);
        Assert.NotSame(original, modified);
    }

    [Fact]
    public void WithShuffle_ShouldCreateNewInstanceWithUpdatedValue()
    {
        // Arrange
        var original = new DataLoaderConfig(shuffle: true);

        // Act
        var modified = original.WithShuffle(false);

        // Assert
        Assert.True(original.Shuffle);
        Assert.False(modified.Shuffle);
        Assert.NotSame(original, modified);
    }

    [Fact]
    public void WithSeed_ShouldCreateNewInstanceWithUpdatedValue()
    {
        // Arrange
        var original = new DataLoaderConfig(seed: 42);

        // Act
        var modified = original.WithSeed(123);

        // Assert
        Assert.Equal(42, original.Seed);
        Assert.Equal(123, modified.Seed);
        Assert.NotSame(original, modified);
    }

    [Fact]
    public void WithPinMemory_ShouldCreateNewInstanceWithUpdatedValue()
    {
        // Arrange
        var original = new DataLoaderConfig(pinMemory: true);

        // Act
        var modified = original.WithPinMemory(false);

        // Assert
        Assert.True(original.PinMemory);
        Assert.False(modified.PinMemory);
        Assert.NotSame(original, modified);
    }

    [Fact]
    public void WithMethods_WithInvalidValues_ShouldThrowException()
    {
        var config = new DataLoaderConfig();

        Assert.Throws<ArgumentOutOfRangeException>(() => config.WithNumWorkers(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => config.WithBatchSize(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => config.WithPrefetchCount(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => config.WithQueueSize(1));
    }

    [Fact]
    public void ToString_ShouldReturnFormattedString()
    {
        // Arrange
        var config = new DataLoaderConfig(
            numWorkers: 4,
            batchSize: 32,
            prefetchCount: 2,
            queueSize: 10,
            shuffle: true,
            seed: 42,
            pinMemory: true);

        // Act
        var result = config.ToString();

        // Assert
        Assert.Contains("DataLoaderConfig", result);
        Assert.Contains("NumWorkers: 4", result);
        Assert.Contains("BatchSize: 32", result);
        Assert.Contains("PrefetchCount: 2", result);
        Assert.Contains("QueueSize: 10", result);
        Assert.Contains("Shuffle: True", result);
        Assert.Contains("Seed: 42", result);
        Assert.Contains("PinMemory: True", result);
    }
}
