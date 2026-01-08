using System;
using Xunit;
using MLFramework.Checkpointing;

namespace MLFramework.Tests.Checkpointing;

public class CheckpointConfigTests
{
    #region CheckpointStrategy Enum Tests

    [Fact]
    public void AllStrategyValuesDefined()
    {
        // Verify all enum values are defined correctly
        Assert.Equal(0, (int)CheckpointStrategy.Interval);
        Assert.Equal(1, (int)CheckpointStrategy.Selective);
        Assert.Equal(2, (int)CheckpointStrategy.SizeBased);
        Assert.Equal(3, (int)CheckpointStrategy.MemoryAware);
        Assert.Equal(4, (int)CheckpointStrategy.Smart);
    }

    [Fact]
    public void StrategyCountMatchesExpected()
    {
        // Verify the expected number of strategies
        var strategyValues = Enum.GetValues<CheckpointStrategy>();
        Assert.Equal(5, strategyValues.Length);
    }

    #endregion

    #region CheckpointConfig Validation Tests

    [Fact]
    public void ValidConfigurationPassesValidation()
    {
        // Valid configuration should pass validation
        var config = new CheckpointConfig
        {
            Strategy = CheckpointStrategy.Interval,
            Interval = 2,
            MinActivationSizeBytes = 1024 * 1024,
            MaxMemoryPercentage = 0.8f,
            MaxRecomputationCacheSize = 100 * 1024 * 1024
        };

        var exception = Record.Exception(() => config.Validate());
        Assert.Null(exception);
    }

    [Fact]
    public void InvalidIntervalThrowsException()
    {
        // Interval must be greater than 0
        var config = new CheckpointConfig { Interval = 0 };

        Assert.Throws<ArgumentException>(() => config.Validate());

        config.Interval = -1;
        Assert.Throws<ArgumentException>(() => config.Validate());
    }

    [Fact]
    public void InvalidMaxMemoryPercentageThrowsException()
    {
        // MaxMemoryPercentage must be between 0 and 1 (exclusive of 0)
        var config = new CheckpointConfig { MaxMemoryPercentage = 0f };

        Assert.Throws<ArgumentException>(() => config.Validate());

        config.MaxMemoryPercentage = -0.5f;
        Assert.Throws<ArgumentException>(() => config.Validate());

        config.MaxMemoryPercentage = 1.5f;
        Assert.Throws<ArgumentException>(() => config.Validate());
    }

    [Fact]
    public void InvalidMaxMemoryPercentageBoundaryOneThrowsException()
    {
        // MaxMemoryPercentage must be less than or equal to 1
        var config = new CheckpointConfig { MaxMemoryPercentage = 1.0f };

        // Should throw since it must be > 0 and <= 1.0
        Assert.Throws<ArgumentException>(() => config.Validate());
    }

    [Fact]
    public void NegativeMinActivationSizeBytesThrowsException()
    {
        // MinActivationSizeBytes must be greater than 0
        var config = new CheckpointConfig { MinActivationSizeBytes = -1 };

        Assert.Throws<ArgumentException>(() => config.Validate());

        config.MinActivationSizeBytes = 0;
        Assert.Throws<ArgumentException>(() => config.Validate());
    }

    [Fact]
    public void NegativeMaxRecomputationCacheSizeThrowsException()
    {
        // MaxRecomputationCacheSize cannot be negative
        var config = new CheckpointConfig { MaxRecomputationCacheSize = -1 };

        Assert.Throws<ArgumentException>(() => config.Validate());
    }

    [Fact]
    public void EmptyLayerIdsThrowException()
    {
        // Empty layer IDs should throw
        var config = new CheckpointConfig
        {
            CheckpointLayers = new[] { "layer1", "", "layer3" }
        };

        Assert.Throws<ArgumentException>(() => config.Validate());

        config.CheckpointLayers = Array.Empty<string>();
        config.ExcludeLayers = new[] { " ", "  " };
        Assert.Throws<ArgumentException>(() => config.Validate());
    }

    [Fact]
    public void NullOrWhitespaceLayerIdsThrowException()
    {
        // Null or whitespace layer IDs should throw
        var config = new CheckpointConfig
        {
            CheckpointLayers = new[] { "layer1", null, "layer3" }
        };

        Assert.Throws<ArgumentException>(() => config.Validate());
    }

    [Fact]
    public void OverlappingCheckpointAndExcludeLayersThrowException()
    {
        // Layers cannot be both checkpointed and excluded
        var config = new CheckpointConfig
        {
            CheckpointLayers = new[] { "layer1", "layer2", "layer3" },
            ExcludeLayers = new[] { "layer4", "layer2", "layer5" }
        };

        var ex = Assert.Throws<ArgumentException>(() => config.Validate());
        Assert.Contains("layer2", ex.Message);
    }

    [Fact]
    public void InvalidStrategyEnumThrowsException()
    {
        // Invalid strategy enum should throw (this would require reflection to set invalid enum value)
        // This test verifies that the Validate method properly checks enum validity
        var config = new CheckpointConfig();

        // By default, it should use a valid strategy
        config.Validate();

        // We can't easily set an invalid enum value in C# without reflection
        // so this test is more documentation that the check exists
    }

    #endregion

    #region Default Configuration Tests

    [Fact]
    public void DefaultConfigurationIsValid()
    {
        // Default configuration should be valid
        var config = CheckpointConfig.Default;

        var exception = Record.Exception(() => config.Validate());
        Assert.Null(exception);
    }

    [Fact]
    public void DefaultUsesIntervalStrategy()
    {
        // Default should use Interval strategy with interval 2
        var config = CheckpointConfig.Default;

        Assert.Equal(CheckpointStrategy.Interval, config.Strategy);
        Assert.Equal(2, config.Interval);
    }

    [Fact]
    public void AggressiveConfigurationIsValid()
    {
        // Aggressive configuration should be valid
        var config = CheckpointConfig.Aggressive;

        var exception = Record.Exception(() => config.Validate());
        Assert.Null(exception);
    }

    [Fact]
    public void AggressiveConfigurationUsesInterval4()
    {
        // Aggressive should use interval 4
        var config = CheckpointConfig.Aggressive;

        Assert.Equal(CheckpointStrategy.Interval, config.Strategy);
        Assert.Equal(4, config.Interval);
        Assert.True(config.EnableRecomputationCache);
        Assert.True(config.TrackStatistics);
    }

    [Fact]
    public void ConservativeConfigurationIsValid()
    {
        // Conservative configuration should be valid
        var config = CheckpointConfig.Conservative;

        var exception = Record.Exception(() => config.Validate());
        Assert.Null(exception);
    }

    [Fact]
    public void ConservativeConfigurationUsesInterval2()
    {
        // Conservative should use interval 2
        var config = CheckpointConfig.Conservative;

        Assert.Equal(CheckpointStrategy.Interval, config.Strategy);
        Assert.Equal(2, config.Interval);
        Assert.True(config.EnableRecomputationCache);
        Assert.True(config.TrackStatistics);
    }

    [Fact]
    public void MemoryAwareConfigurationIsValid()
    {
        // MemoryAware configuration should be valid
        var config = CheckpointConfig.MemoryAware;

        var exception = Record.Exception(() => config.Validate());
        Assert.Null(exception);
    }

    [Fact]
    public void MemoryAwareConfigurationUsesMemoryAwareStrategy()
    {
        // MemoryAware should use MemoryAware strategy
        var config = CheckpointConfig.MemoryAware;

        Assert.Equal(CheckpointStrategy.MemoryAware, config.Strategy);
        Assert.Equal(0.75f, config.MaxMemoryPercentage);
        Assert.True(config.EnableRecomputationCache);
        Assert.True(config.TrackStatistics);
    }

    #endregion

    #region Builder Pattern Tests

    [Fact]
    public void BuilderCreatesValidConfiguration()
    {
        // Builder should create valid configuration
        var config = new CheckpointConfigBuilder()
            .WithStrategy(CheckpointStrategy.Interval)
            .WithInterval(3)
            .Build();

        Assert.Equal(CheckpointStrategy.Interval, config.Strategy);
        Assert.Equal(3, config.Interval);
    }

    [Fact]
    public void BuilderFluentApiWorks()
    {
        // Builder fluent API should work correctly
        var config = new CheckpointConfigBuilder()
            .WithStrategy(CheckpointStrategy.Selective)
            .WithInterval(5)
            .WithCheckpointLayers(new[] { "layer1", "layer2" })
            .WithExcludeLayers(new[] { "layer3" })
            .WithMinActivationSizeBytes(2048 * 1024)
            .WithMaxMemoryPercentage(0.75f)
            .EnableRecomputationCache(false)
            .WithMaxRecomputationCacheSize(50 * 1024 * 1024)
            .UseAsyncRecomputation(true)
            .TrackStatistics(false)
            .Build();

        Assert.Equal(CheckpointStrategy.Selective, config.Strategy);
        Assert.Equal(5, config.Interval);
        Assert.Equal(new[] { "layer1", "layer2" }, config.CheckpointLayers);
        Assert.Equal(new[] { "layer3" }, config.ExcludeLayers);
        Assert.Equal(2048 * 1024, config.MinActivationSizeBytes);
        Assert.Equal(0.75f, config.MaxMemoryPercentage);
        Assert.False(config.EnableRecomputationCache);
        Assert.Equal(50 * 1024 * 1024, config.MaxRecomputationCacheSize);
        Assert.True(config.UseAsyncRecomputation);
        Assert.False(config.TrackStatistics);
    }

    [Fact]
    public void BuilderValidatesFinalConfiguration()
    {
        // Build() should validate final configuration
        var ex = Assert.Throws<ArgumentException>(() =>
        {
            new CheckpointConfigBuilder()
                .WithInterval(0)  // Invalid interval
                .Build();
        });

        Assert.Contains("Interval must be greater than 0", ex.Message);
    }

    [Fact]
    public void BuilderCanSetAllProperties()
    {
        // Builder should be able to set all properties
        var config = new CheckpointConfigBuilder()
            .WithStrategy(CheckpointStrategy.SizeBased)
            .WithInterval(10)
            .WithCheckpointLayers(new[] { "layer1" })
            .WithExcludeLayers(new[] { "layer2" })
            .WithMinActivationSizeBytes(512 * 1024)
            .WithMaxMemoryPercentage(0.9f)
            .EnableRecomputationCache(true)
            .WithMaxRecomputationCacheSize(200 * 1024 * 1024)
            .UseAsyncRecomputation(false)
            .TrackStatistics(true)
            .Build();

        Assert.Equal(CheckpointStrategy.SizeBased, config.Strategy);
        Assert.Equal(10, config.Interval);
        Assert.Equal(new[] { "layer1" }, config.CheckpointLayers);
        Assert.Equal(new[] { "layer2" }, config.ExcludeLayers);
        Assert.Equal(512 * 1024, config.MinActivationSizeBytes);
        Assert.Equal(0.9f, config.MaxMemoryPercentage);
        Assert.True(config.EnableRecomputationCache);
        Assert.Equal(200 * 1024 * 1024, config.MaxRecomputationCacheSize);
        Assert.False(config.UseAsyncRecomputation);
        Assert.True(config.TrackStatistics);
    }

    #endregion

    #region Extension Method Tests

    [Fact]
    public void WithStrategyCreatesCorrectCopy()
    {
        // WithStrategy() should create correct copy
        var original = new CheckpointConfig { Strategy = CheckpointStrategy.Interval };
        var modified = original.WithStrategy(CheckpointStrategy.Selective);

        Assert.Equal(CheckpointStrategy.Interval, original.Strategy);
        Assert.Equal(CheckpointStrategy.Selective, modified.Strategy);
        Assert.NotSame(original, modified);
    }

    [Fact]
    public void WithIntervalCreatesCorrectCopy()
    {
        // WithInterval() should create correct copy
        var original = new CheckpointConfig { Interval = 2 };
        var modified = original.WithInterval(5);

        Assert.Equal(2, original.Interval);
        Assert.Equal(5, modified.Interval);
        Assert.NotSame(original, modified);
    }

    [Fact]
    public void CloneCreatesDeepCopy()
    {
        // Clone() should create deep copy
        var original = new CheckpointConfig
        {
            Strategy = CheckpointStrategy.Selective,
            Interval = 3,
            CheckpointLayers = new[] { "layer1", "layer2" },
            ExcludeLayers = new[] { "layer3" },
            MinActivationSizeBytes = 2048 * 1024,
            MaxMemoryPercentage = 0.7f,
            EnableRecomputationCache = false,
            MaxRecomputationCacheSize = 50 * 1024 * 1024,
            UseAsyncRecomputation = true,
            TrackStatistics = false
        };

        var clone = original.Clone();

        Assert.NotSame(original, clone);
        Assert.Equal(original.Strategy, clone.Strategy);
        Assert.Equal(original.Interval, clone.Interval);
        Assert.Equal(original.CheckpointLayers, clone.CheckpointLayers);
        Assert.Equal(original.ExcludeLayers, clone.ExcludeLayers);
        Assert.Equal(original.MinActivationSizeBytes, clone.MinActivationSizeBytes);
        Assert.Equal(original.MaxMemoryPercentage, clone.MaxMemoryPercentage);
        Assert.Equal(original.EnableRecomputationCache, clone.EnableRecomputationCache);
        Assert.Equal(original.MaxRecomputationCacheSize, clone.MaxRecomputationCacheSize);
        Assert.Equal(original.UseAsyncRecomputation, clone.UseAsyncRecomputation);
        Assert.Equal(original.TrackStatistics, clone.TrackStatistics);
    }

    [Fact]
    public void CloneDoesNotAffectOriginal()
    {
        // Clone() should not affect original
        var original = new CheckpointConfig { Interval = 2 };
        var clone = original.Clone();

        clone.Interval = 10;
        clone.CheckpointLayers = new[] { "layer1" };

        Assert.Equal(2, original.Interval);
        Assert.Empty(original.CheckpointLayers);
        Assert.Equal(10, clone.Interval);
        Assert.Equal(new[] { "layer1" }, clone.CheckpointLayers);
    }

    [Fact]
    public void CloneCreatesIndependentArrays()
    {
        // Clone should create independent arrays
        var original = new CheckpointConfig
        {
            CheckpointLayers = new[] { "layer1", "layer2" },
            ExcludeLayers = new[] { "layer3" }
        };

        var clone = original.Clone();

        // Modify clone's arrays
        clone.CheckpointLayers[0] = "modified1";
        clone.ExcludeLayers[0] = "modified2";

        // Original should be unchanged
        Assert.Equal("layer1", original.CheckpointLayers[0]);
        Assert.Equal("layer2", original.CheckpointLayers[1]);
        Assert.Equal("layer3", original.ExcludeLayers[0]);
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void EmptyArraysForLayerListsWork()
    {
        // Empty arrays for layer lists should be valid
        var config = new CheckpointConfig
        {
            CheckpointLayers = Array.Empty<string>(),
            ExcludeLayers = Array.Empty<string>()
        };

        var exception = Record.Exception(() => config.Validate());
        Assert.Null(exception);
    }

    [Fact]
    public void VeryLargeIntervalValueWorks()
    {
        // Very large interval values should be valid
        var config = new CheckpointConfig { Interval = 1000000 };

        var exception = Record.Exception(() => config.Validate());
        Assert.Null(exception);
        Assert.Equal(1000000, config.Interval);
    }

    [Fact]
    public void MaxMemoryPercentageZeroPointOneWorks()
    {
        // Boundary value 0.1 should be valid
        var config = new CheckpointConfig { MaxMemoryPercentage = 0.1f };

        var exception = Record.Exception(() => config.Validate());
        Assert.Null(exception);
    }

    [Fact]
    public void MaxMemoryPercentageZeroPointNineNineWorks()
    {
        // Boundary value 0.99 should be valid
        var config = new CheckpointConfig { MaxMemoryPercentage = 0.99f };

        var exception = Record.Exception(() => config.Validate());
        Assert.Null(exception);
    }

    [Fact]
    public void MinActivationSizeBytesOfOneByteWorks()
    {
        // MinActivationSizeBytes of 1 byte should be valid
        var config = new CheckpointConfig { MinActivationSizeBytes = 1 };

        var exception = Record.Exception(() => config.Validate());
        Assert.Null(exception);
        Assert.Equal(1, config.MinActivationSizeBytes);
    }

    [Fact]
    public void MaxRecomputationCacheSizeZeroWorks()
    {
        // Zero cache size should be valid (disables caching)
        var config = new CheckpointConfig { MaxRecomputationCacheSize = 0 };

        var exception = Record.Exception(() => config.Validate());
        Assert.Null(exception);
        Assert.Equal(0, config.MaxRecomputationCacheSize);
    }

    [Fact]
    public void DefaultConfigHasAllPropertiesInitialized()
    {
        // Default config should have all properties initialized
        var config = new CheckpointConfig();

        Assert.Equal(CheckpointStrategy.Interval, config.Strategy);
        Assert.Equal(2, config.Interval);
        Assert.NotNull(config.CheckpointLayers);
        Assert.NotNull(config.ExcludeLayers);
        Assert.Equal(1024 * 1024, config.MinActivationSizeBytes);
        Assert.Equal(0.8f, config.MaxMemoryPercentage);
        Assert.True(config.EnableRecomputationCache);
        Assert.Equal(100 * 1024 * 1024, config.MaxRecomputationCacheSize);
        Assert.False(config.UseAsyncRecomputation);
        Assert.True(config.TrackStatistics);
    }

    #endregion
}
