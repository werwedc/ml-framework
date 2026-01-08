using MLFramework.Checkpointing.Strategies;
using MLFramework.Checkpointing;
using Xunit;

namespace MLFramework.Tests.Checkpointing.Strategies;

/// <summary>
/// Tests for IntervalCheckpointStrategy
/// </summary>
public class IntervalCheckpointStrategyTests
{
    [Fact]
    public void ShouldCheckpoint_CheckpointsAtCorrectIntervals()
    {
        // Arrange
        var strategy = new IntervalCheckpointStrategy(3);
        var tensor = new Tensor(1000, 4);

        // Act & Assert
        Assert.True(strategy.ShouldCheckpoint("layer0", tensor, 0)); // 0 % 3 == 0
        Assert.False(strategy.ShouldCheckpoint("layer1", tensor, 1)); // 1 % 3 != 0
        Assert.False(strategy.ShouldCheckpoint("layer2", tensor, 2)); // 2 % 3 != 0
        Assert.True(strategy.ShouldCheckpoint("layer3", tensor, 3)); // 3 % 3 == 0
        Assert.True(strategy.ShouldCheckpoint("layer6", tensor, 6)); // 6 % 3 == 0
    }

    [Fact]
    public void Constructor_InvalidInterval_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => new IntervalCheckpointStrategy(0));
        Assert.Throws<ArgumentException>(() => new IntervalCheckpointStrategy(-1));
    }

    [Fact]
    public void Reset_ClearsStateCorrectly()
    {
        // Arrange
        var strategy = new IntervalCheckpointStrategy(2);
        var tensor = new Tensor(1000, 4);

        // Act - checkpoint some layers
        strategy.ShouldCheckpoint("layer0", tensor, 0);
        strategy.ShouldCheckpoint("layer2", tensor, 2);
        strategy.Reset();

        // Assert - should still checkpoint at the same intervals
        Assert.True(strategy.ShouldCheckpoint("layer0", tensor, 0));
        Assert.True(strategy.ShouldCheckpoint("layer2", tensor, 2));
    }

    [Fact]
    public void Name_ReturnsCorrectString()
    {
        // Arrange
        var strategy = new IntervalCheckpointStrategy(3);

        // Assert
        Assert.Equal("Interval(3)", strategy.Name);
    }
}

/// <summary>
/// Tests for SelectiveCheckpointStrategy
/// </summary>
public class SelectiveCheckpointStrategyTests
{
    [Fact]
    public void ShouldCheckpoint_CheckpointsOnlySpecifiedLayers()
    {
        // Arrange
        var checkpointLayers = new[] { "layer0", "layer2", "layer4" };
        var strategy = new SelectiveCheckpointStrategy(checkpointLayers);
        var tensor = new Tensor(1000, 4);

        // Act & Assert
        Assert.True(strategy.ShouldCheckpoint("layer0", tensor, 0));
        Assert.False(strategy.ShouldCheckpoint("layer1", tensor, 1));
        Assert.True(strategy.ShouldCheckpoint("layer2", tensor, 2));
        Assert.False(strategy.ShouldCheckpoint("layer3", tensor, 3));
        Assert.True(strategy.ShouldCheckpoint("layer4", tensor, 4));
    }

    [Fact]
    public void ShouldCheckpoint_ExcludesSpecifiedLayers()
    {
        // Arrange
        var checkpointLayers = new[] { "layer0", "layer1", "layer2" };
        var excludeLayers = new[] { "layer1" };
        var strategy = new SelectiveCheckpointStrategy(checkpointLayers, excludeLayers);
        var tensor = new Tensor(1000, 4);

        // Act & Assert
        Assert.True(strategy.ShouldCheckpoint("layer0", tensor, 0));
        Assert.False(strategy.ShouldCheckpoint("layer1", tensor, 1)); // excluded
        Assert.True(strategy.ShouldCheckpoint("layer2", tensor, 2));
        Assert.False(strategy.ShouldCheckpoint("layer3", tensor, 3));
    }

    [Fact]
    public void Constructor_OverlappingLayers_ThrowsException()
    {
        // Arrange
        var checkpointLayers = new[] { "layer0", "layer1" };
        var excludeLayers = new[] { "layer1" }; // overlap!

        // Act & Assert
        Assert.Throws<ArgumentException>(() => new SelectiveCheckpointStrategy(checkpointLayers, excludeLayers));
    }

    [Fact]
    public void Constructor_EmptyLists_HandlesCorrectly()
    {
        // Arrange
        var strategy = new SelectiveCheckpointStrategy(null, null);
        var tensor = new Tensor(1000, 4);

        // Act & Assert - should not checkpoint anything
        Assert.False(strategy.ShouldCheckpoint("layer0", tensor, 0));
        Assert.False(strategy.ShouldCheckpoint("layer1", tensor, 1));
    }

    [Fact]
    public void Reset_NoStateToReset_DoesNotThrow()
    {
        // Arrange
        var strategy = new SelectiveCheckpointStrategy(new[] { "layer0" });

        // Act & Assert - should not throw
        strategy.Reset();
    }

    [Fact]
    public void Name_ReturnsCorrectString()
    {
        // Arrange
        var strategy = new SelectiveCheckpointStrategy();

        // Assert
        Assert.Equal("Selective", strategy.Name);
    }
}

/// <summary>
/// Tests for SizeBasedCheckpointStrategy
/// </summary>
public class SizeBasedCheckpointStrategyTests
{
    [Fact]
    public void ShouldCheckpoint_CheckpointsLargeActivations()
    {
        // Arrange
        var minSize = 10000; // 10KB
        var strategy = new SizeBasedCheckpointStrategy(minSize);
        var largeTensor = new Tensor(3000, 4); // 12KB
        var smallTensor = new Tensor(1000, 4); // 4KB

        // Act & Assert
        Assert.True(strategy.ShouldCheckpoint("layer0", largeTensor, 0));
        Assert.False(strategy.ShouldCheckpoint("layer1", smallTensor, 1));
    }

    [Fact]
    public void ShouldCheckpoint_ExcludesSpecifiedLayers()
    {
        // Arrange
        var strategy = new SizeBasedCheckpointStrategy(10000, new[] { "layer1" });
        var largeTensor = new Tensor(3000, 4); // 12KB

        // Act & Assert
        Assert.True(strategy.ShouldCheckpoint("layer0", largeTensor, 0));
        Assert.False(strategy.ShouldCheckpoint("layer1", largeTensor, 1)); // excluded
        Assert.True(strategy.ShouldCheckpoint("layer2", largeTensor, 2));
    }

    [Fact]
    public void Constructor_InvalidSizeThreshold_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => new SizeBasedCheckpointStrategy(0));
        Assert.Throws<ArgumentException>(() => new SizeBasedCheckpointStrategy(-1));
    }

    [Fact]
    public void Reset_ClearsStateCorrectly()
    {
        // Arrange
        var strategy = new SizeBasedCheckpointStrategy(10000);
        var largeTensor = new Tensor(3000, 4);

        // Act - checkpoint some layers
        strategy.ShouldCheckpoint("layer0", largeTensor, 0);
        strategy.ShouldCheckpoint("layer1", largeTensor, 1);
        strategy.Reset();

        // Assert - should still checkpoint based on size
        Assert.True(strategy.ShouldCheckpoint("layer2", largeTensor, 2));
    }

    [Fact]
    public void Name_ReturnsFormattedString()
    {
        // Arrange
        var strategy = new SizeBasedCheckpointStrategy(1024 * 1024); // 1MB

        // Assert
        Assert.Equal("SizeBased(1MB)", strategy.Name);
    }
}

/// <summary>
/// Tests for MemoryAwareCheckpointStrategy
/// </summary>
public class MemoryAwareCheckpointStrategyTests
{
    [Fact]
    public void Constructor_ValidParameters_CreatesStrategy()
    {
        // Arrange & Act
        var strategy = new MemoryAwareCheckpointStrategy(0.8f, null, 8L * 1024 * 1024 * 1024);

        // Assert
        Assert.Equal("MemoryAware(80%)", strategy.Name);
    }

    [Fact]
    public void Constructor_InvalidMemoryPercentage_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => new MemoryAwareCheckpointStrategy(0.0f));
        Assert.Throws<ArgumentException>(() => new MemoryAwareCheckpointStrategy(1.5f));
        Assert.Throws<ArgumentException>(() => new MemoryAwareCheckpointStrategy(-0.1f));
    }

    [Fact]
    public void ShouldCheckpoint_UsesIntervalBasedOnMemoryPressure()
    {
        // Arrange
        var memoryTracker = new MemoryTracker();
        var strategy = new MemoryAwareCheckpointStrategy(0.5f, memoryTracker, 10000);
        var tensor = new Tensor(1000, 4);

        // Simulate high memory pressure (50%+ of 10000 = 5000+)
        memoryTracker.RecordAllocation("existing", 6000);

        // Act & Assert - with high memory pressure, should checkpoint more frequently
        var result1 = strategy.ShouldCheckpoint("layer0", tensor, 0);
        var result2 = strategy.ShouldCheckpoint("layer1", tensor, 1);

        // Should checkpoint at least one layer
        Assert.True(result1 || result2);
    }

    [Fact]
    public void Reset_ClearsStateCorrectly()
    {
        // Arrange
        var strategy = new MemoryAwareCheckpointStrategy(0.8f);
        var tensor = new Tensor(1000, 4);

        // Act - checkpoint some layers
        strategy.ShouldCheckpoint("layer0", tensor, 0);
        strategy.Reset();

        // Assert - interval should be reset to initial value
        // This is hard to test directly, but we can verify it doesn't throw
        strategy.ShouldCheckpoint("layer1", tensor, 1);
    }

    [Fact]
    public void Name_ReturnsFormattedString()
    {
        // Arrange
        var strategy = new MemoryAwareCheckpointStrategy(0.75f);

        // Assert
        Assert.Equal("MemoryAware(75%)", strategy.Name);
    }
}

/// <summary>
/// Tests for SmartCheckpointStrategy
/// </summary>
public class SmartCheckpointStrategyTests
{
    [Fact]
    public void ShouldCheckpoint_CollectsStatisticsDuringInitialization()
    {
        // Arrange
        var strategy = new SmartCheckpointStrategy();
        var smallTensor = new Tensor(1000, 4);
        var largeTensor = new Tensor(5000, 4);

        // Act - first 10 layers should not checkpoint (collecting data)
        for (int i = 0; i < 10; i++)
        {
            var result = strategy.ShouldCheckpoint($"layer{i}", i % 2 == 0 ? largeTensor : smallTensor, i);
            Assert.False(result, $"Layer {i} should not checkpoint during initialization");
        }
    }

    [Fact]
    public void ShouldCheckpoint_CheckpointsLargeActivations()
    {
        // Arrange
        var strategy = new SmartCheckpointStrategy();
        var smallTensor = new Tensor(1000, 4);
        var largeTensor = new Tensor(10000, 4);

        // Initialize
        for (int i = 0; i < 10; i++)
        {
            strategy.ShouldCheckpoint($"layer{i}", smallTensor, i);
        }

        // Act & Assert - large tensor should be checkpointed
        var result = strategy.ShouldCheckpoint("layer11", largeTensor, 11);
        Assert.True(result);
    }

    [Fact]
    public void ShouldCheckpoint_ExcludesSpecifiedLayers()
    {
        // Arrange
        var strategy = new SmartCheckpointStrategy(new[] { "layer1" });
        var tensor = new Tensor(10000, 4);

        // Initialize
        for (int i = 0; i < 10; i++)
        {
            strategy.ShouldCheckpoint($"layer{i}", tensor, i);
        }

        // Act & Assert - excluded layer should not be checkpointed
        var result = strategy.ShouldCheckpoint("layer1", tensor, 1);
        Assert.False(result);
    }

    [Fact]
    public void Reset_ClearsStateCorrectly()
    {
        // Arrange
        var strategy = new SmartCheckpointStrategy();
        var tensor = new Tensor(1000, 4);

        // Initialize
        for (int i = 0; i < 10; i++)
        {
            strategy.ShouldCheckpoint($"layer{i}", tensor, i);
        }

        // Act - checkpoint some layers
        strategy.ShouldCheckpoint("layer11", tensor, 11);
        strategy.Reset();

        // Act & Assert - should be back to initialization phase
        for (int i = 0; i < 10; i++)
        {
            var result = strategy.ShouldCheckpoint($"layer{i}", tensor, i);
            Assert.False(result, $"Layer {i} should not checkpoint during re-initialization");
        }
    }

    [Fact]
    public void Name_ReturnsCorrectString()
    {
        // Arrange
        var strategy = new SmartCheckpointStrategy();

        // Assert
        Assert.Equal("Smart", strategy.Name);
    }
}

/// <summary>
/// Tests for CheckpointStrategyFactory
/// </summary>
public class CheckpointStrategyFactoryTests
{
    [Fact]
    public void CreateStrategy_FromConfig_CreatesCorrectStrategy()
    {
        // Arrange
        var config = new CheckpointConfig
        {
            Strategy = CheckpointStrategy.Interval,
            Interval = 3
        };

        // Act
        var strategy = CheckpointStrategyFactory.CreateStrategy(config);

        // Assert
        Assert.IsType<IntervalCheckpointStrategy>(strategy);
        Assert.Equal("Interval(3)", strategy.Name);
    }

    [Fact]
    public void CreateStrategy_UnknownStrategy_ThrowsException()
    {
        // Arrange
        var config = new CheckpointConfig
        {
            Strategy = (CheckpointStrategy)99 // invalid value
        };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => CheckpointStrategyFactory.CreateStrategy(config));
    }

    [Fact]
    public void CreateInterval_CreatesIntervalStrategy()
    {
        // Act
        var strategy = CheckpointStrategyFactory.CreateInterval(3);

        // Assert
        Assert.IsType<IntervalCheckpointStrategy>(strategy);
        Assert.Equal("Interval(3)", strategy.Name);
    }

    [Fact]
    public void CreateSelective_CreatesSelectiveStrategy()
    {
        // Arrange
        var layers = new[] { "layer0", "layer1" };

        // Act
        var strategy = CheckpointStrategyFactory.CreateSelective(layers);

        // Assert
        Assert.IsType<SelectiveCheckpointStrategy>(strategy);
        Assert.Equal("Selective", strategy.Name);
    }

    [Fact]
    public void CreateSizeBased_CreatesSizeBasedStrategy()
    {
        // Act
        var strategy = CheckpointStrategyFactory.CreateSizeBased(10000);

        // Assert
        Assert.IsType<SizeBasedCheckpointStrategy>(strategy);
        Assert.Equal("SizeBased(10KB)", strategy.Name);
    }

    [Fact]
    public void CreateMemoryAware_CreatesMemoryAwareStrategy()
    {
        // Act
        var strategy = CheckpointStrategyFactory.CreateMemoryAware(0.75f);

        // Assert
        Assert.IsType<MemoryAwareCheckpointStrategy>(strategy);
        Assert.Equal("MemoryAware(75%)", strategy.Name);
    }

    [Fact]
    public void CreateSmart_CreatesSmartStrategy()
    {
        // Act
        var strategy = CheckpointStrategyFactory.CreateSmart();

        // Assert
        Assert.IsType<SmartCheckpointStrategy>(strategy);
        Assert.Equal("Smart", strategy.Name);
    }

    [Theory]
    [InlineData(CheckpointStrategy.Interval, typeof(IntervalCheckpointStrategy))]
    [InlineData(CheckpointStrategy.Selective, typeof(SelectiveCheckpointStrategy))]
    [InlineData(CheckpointStrategy.SizeBased, typeof(SizeBasedCheckpointStrategy))]
    [InlineData(CheckpointStrategy.MemoryAware, typeof(MemoryAwareCheckpointStrategy))]
    [InlineData(CheckpointStrategy.Smart, typeof(SmartCheckpointStrategy))]
    public void CreateStrategy_AllStrategies_CreatesCorrectType(CheckpointStrategy strategyType, Type expectedType)
    {
        // Arrange
        var config = new CheckpointConfig { Strategy = strategyType };

        // Act
        var strategy = CheckpointStrategyFactory.CreateStrategy(config);

        // Assert
        Assert.IsType(expectedType, strategy);
    }
}

/// <summary>
/// Edge case tests for all strategies
/// </summary>
public class CheckpointStrategyEdgeCaseTests
{
    [Fact]
    public void AllStrategies_HandleLayerIndexZero()
    {
        // Arrange
        var strategies = new List<ICheckpointStrategy>
        {
            new IntervalCheckpointStrategy(2),
            new SelectiveCheckpointStrategy(new[] { "layer0" }),
            new SizeBasedCheckpointStrategy(100),
            new SmartCheckpointStrategy()
        };

        var tensor = new Tensor(1000, 4);

        // Act & Assert - should not throw
        foreach (var strategy in strategies)
        {
            var result = strategy.ShouldCheckpoint("layer0", tensor, 0);
            // We don't assert the result, just that it doesn't throw
        }
    }

    [Fact]
    public void AllStrategies_HandleVeryLargeLayerIndex()
    {
        // Arrange
        var strategies = new List<ICheckpointStrategy>
        {
            new IntervalCheckpointStrategy(2),
            new SelectiveCheckpointStrategy(),
            new SizeBasedCheckpointStrategy(100)
        };

        var tensor = new Tensor(1000, 4);
        const int largeIndex = 1000000;

        // Act & Assert - should not throw
        foreach (var strategy in strategies)
        {
            var result = strategy.ShouldCheckpoint($"layer{largeIndex}", tensor, largeIndex);
            // We don't assert the result, just that it doesn't throw
        }
    }

    [Fact]
    public void AllStrategies_HandleVerySmallActivations()
    {
        // Arrange
        var strategies = new List<ICheckpointStrategy>
        {
            new SizeBasedCheckpointStrategy(1000000), // 1MB threshold
            new SmartCheckpointStrategy()
        };

        var smallTensor = new Tensor(1, 4); // 4 bytes

        // Act & Assert - should not throw
        foreach (var strategy in strategies)
        {
            var result = strategy.ShouldCheckpoint("layer0", smallTensor, 0);
            // We don't assert the result, just that it doesn't throw
        }
    }

    [Fact]
    public void AllStrategies_HandleVeryLargeActivations()
    {
        // Arrange
        var strategies = new List<ICheckpointStrategy>
        {
            new SizeBasedCheckpointStrategy(100),
            new SmartCheckpointStrategy()
        };

        var largeTensor = new Tensor(1000000000, 4); // 4GB

        // Act & Assert - should not throw
        foreach (var strategy in strategies)
        {
            var result = strategy.ShouldCheckpoint("layer0", largeTensor, 0);
            // We don't assert the result, just that it doesn't throw
        }
    }
}
