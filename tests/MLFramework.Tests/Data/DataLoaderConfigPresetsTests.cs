using Xunit;
using MLFramework.Data;
using System;

namespace MLFramework.Tests.Data;

/// <summary>
/// Unit tests for DataLoaderConfigPresets class.
/// </summary>
public class DataLoaderConfigPresetsTests
{
    [Fact]
    public void ForCPUBound_ShouldCreateOptimizedConfig()
    {
        // Act
        var config = DataLoaderConfigPresets.ForCPUBound();

        // Assert
        Assert.Equal(Environment.ProcessorCount, config.NumWorkers);
        Assert.Equal(64, config.BatchSize);
        Assert.Equal(3, config.PrefetchCount);
        Assert.Equal(15, config.QueueSize); // prefetchCount + 12
        Assert.True(config.Shuffle);
        Assert.Equal(42, config.Seed);
        Assert.False(config.PinMemory);
    }

    [Fact]
    public void ForGPUBound_ShouldCreateOptimizedConfig()
    {
        // Act
        var config = DataLoaderConfigPresets.ForGPUBound();

        // Assert
        Assert.Equal(2, config.NumWorkers);
        Assert.Equal(32, config.BatchSize);
        Assert.Equal(2, config.PrefetchCount);
        Assert.Equal(10, config.QueueSize);
        Assert.True(config.Shuffle);
        Assert.Equal(42, config.Seed);
        Assert.True(config.PinMemory);
    }

    [Fact]
    public void ForSmallDataset_ShouldCreateOptimizedConfig()
    {
        // Act
        var config = DataLoaderConfigPresets.ForSmallDataset();

        // Assert
        Assert.Equal(1, config.NumWorkers);
        Assert.Equal(16, config.BatchSize);
        Assert.Equal(1, config.PrefetchCount);
        Assert.Equal(5, config.QueueSize);
        Assert.True(config.Shuffle);
        Assert.Equal(42, config.Seed);
        Assert.False(config.PinMemory);
    }

    [Fact]
    public void ForLargeDataset_ShouldCreateOptimizedConfig()
    {
        // Act
        var config = DataLoaderConfigPresets.ForLargeDataset();

        // Assert
        Assert.Equal(4, config.NumWorkers);
        Assert.Equal(128, config.BatchSize);
        Assert.Equal(4, config.PrefetchCount);
        Assert.Equal(20, config.QueueSize);
        Assert.True(config.Shuffle);
        Assert.Equal(42, config.Seed);
        Assert.True(config.PinMemory);
    }

    [Fact]
    public void ForCPUBound_ConfigShouldBeImmutable()
    {
        // Arrange
        var original = DataLoaderConfigPresets.ForCPUBound();

        // Act
        var modified = original.WithBatchSize(128);

        // Assert
        Assert.NotEqual(original.BatchSize, modified.BatchSize);
        Assert.Equal(64, original.BatchSize);
        Assert.Equal(128, modified.BatchSize);
    }

    [Fact]
    public void ForGPUBound_ConfigShouldBeImmutable()
    {
        // Arrange
        var original = DataLoaderConfigPresets.ForGPUBound();

        // Act
        var modified = original.WithNumWorkers(4);

        // Assert
        Assert.NotEqual(original.NumWorkers, modified.NumWorkers);
        Assert.Equal(2, original.NumWorkers);
        Assert.Equal(4, modified.NumWorkers);
    }

    [Fact]
    public void ForSmallDataset_ConfigShouldBeImmutable()
    {
        // Arrange
        var original = DataLoaderConfigPresets.ForSmallDataset();

        // Act
        var modified = original.WithPinMemory(true);

        // Assert
        Assert.NotEqual(original.PinMemory, modified.PinMemory);
        Assert.False(original.PinMemory);
        Assert.True(modified.PinMemory);
    }

    [Fact]
    public void ForLargeDataset_ConfigShouldBeImmutable()
    {
        // Arrange
        var original = DataLoaderConfigPresets.ForLargeDataset();

        // Act
        var modified = original.WithShuffle(false);

        // Assert
        Assert.NotEqual(original.Shuffle, modified.Shuffle);
        Assert.True(original.Shuffle);
        Assert.False(modified.Shuffle);
    }

    [Fact]
    public void AllPresets_ShouldPassValidation()
    {
        // Act & Assert
        var cpuConfig = DataLoaderConfigPresets.ForCPUBound();
        var gpuConfig = DataLoaderConfigPresets.ForGPUBound();
        var smallConfig = DataLoaderConfigPresets.ForSmallDataset();
        var largeConfig = DataLoaderConfigPresets.ForLargeDataset();

        // Just creating them should validate
        Assert.NotNull(cpuConfig);
        Assert.NotNull(gpuConfig);
        Assert.NotNull(smallConfig);
        Assert.NotNull(largeConfig);
    }

    [Fact]
    public void Presets_ShouldHaveDistinctConfigurations()
    {
        // Act
        var cpuConfig = DataLoaderConfigPresets.ForCPUBound();
        var gpuConfig = DataLoaderConfigPresets.ForGPUBound();
        var smallConfig = DataLoaderConfigPresets.ForSmallDataset();
        var largeConfig = DataLoaderConfigPresets.ForLargeDataset();

        // Assert - Verify each preset has different characteristics
        Assert.NotEqual(cpuConfig.NumWorkers, gpuConfig.NumWorkers);
        Assert.NotEqual(smallConfig.BatchSize, largeConfig.BatchSize);
        Assert.NotEqual(cpuConfig.PrefetchCount, smallConfig.PrefetchCount);
        Assert.NotEqual(gpuConfig.PinMemory, cpuConfig.PinMemory);
    }

    [Fact]
    public void ForCPUBound_ShouldUseMoreWorkersThanGPUBound()
    {
        // Act
        var cpuConfig = DataLoaderConfigPresets.ForCPUBound();
        var gpuConfig = DataLoaderConfigPresets.ForGPUBound();

        // Assert
        Assert.True(cpuConfig.NumWorkers >= gpuConfig.NumWorkers,
            "CPU-bound config should use at least as many workers as GPU-bound config");
    }

    [Fact]
    public void ForSmallDataset_ShouldUseSmallerBatchSizeThanLargeDataset()
    {
        // Act
        var smallConfig = DataLoaderConfigPresets.ForSmallDataset();
        var largeConfig = DataLoaderConfigPresets.ForLargeDataset();

        // Assert
        Assert.True(smallConfig.BatchSize < largeConfig.BatchSize,
            "Small dataset should use smaller batch size than large dataset");
    }

    [Fact]
    public void ForGPUBound_ShouldEnablePinMemory()
    {
        // Act
        var gpuConfig = DataLoaderConfigPresets.ForGPUBound();

        // Assert
        Assert.True(gpuConfig.PinMemory,
            "GPU-bound config should enable pinned memory for faster transfers");
    }
}
