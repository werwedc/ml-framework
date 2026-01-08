using System;
using MLFramework.Serving;
using Xunit;

namespace MLFramework.Tests.Serving;

/// <summary>
/// Tests for BatchingConfiguration class
/// </summary>
public class BatchingConfigurationTests
{
    [Fact]
    public void DefaultConfiguration_HasValidValues()
    {
        var config = BatchingConfiguration.Default();

        Assert.Equal(32, config.MaxBatchSize);
        Assert.Equal(TimeSpan.FromMilliseconds(5), config.MaxWaitTime);
        Assert.Equal(16, config.PreferBatchSize);
        Assert.Equal(100, config.MaxQueueSize);
        Assert.Equal(TimeoutStrategy.DispatchPartial, config.TimeoutStrategy);
    }

    [Fact]
    public void Validate_WithValidConfig_DoesNotThrow()
    {
        var config = BatchingConfiguration.Default();
        config.Validate(); // Should not throw
    }

    [Fact]
    public void Validate_WithInvalidMaxBatchSize_Throws()
    {
        var config = BatchingConfiguration.Default();
        config.MaxBatchSize = 0;

        Assert.Throws<ArgumentOutOfRangeException>(() => config.Validate());
    }

    [Fact]
    public void Validate_WithMaxBatchSizeTooHigh_Throws()
    {
        var config = BatchingConfiguration.Default();
        config.MaxBatchSize = 1025;

        Assert.Throws<ArgumentOutOfRangeException>(() => config.Validate());
    }

    [Fact]
    public void Validate_WithPreferBatchSizeGreaterThanMax_Throws()
    {
        var config = BatchingConfiguration.Default();
        config.MaxBatchSize = 16;
        config.PreferBatchSize = 32;

        Assert.Throws<ArgumentOutOfRangeException>(() => config.Validate());
    }

    [Fact]
    public void Validate_WithInvalidMaxWaitTimeTooLow_Throws()
    {
        var config = BatchingConfiguration.Default();
        config.MaxWaitTime = TimeSpan.FromMilliseconds(0.5);

        Assert.Throws<ArgumentOutOfRangeException>(() => config.Validate());
    }

    [Fact]
    public void Validate_WithInvalidMaxWaitTimeTooHigh_Throws()
    {
        var config = BatchingConfiguration.Default();
        config.MaxWaitTime = TimeSpan.FromMilliseconds(1001);

        Assert.Throws<ArgumentOutOfRangeException>(() => config.Validate());
    }

    [Fact]
    public void Validate_WithInvalidMaxQueueSizeTooLow_Throws()
    {
        var config = BatchingConfiguration.Default();
        config.MaxQueueSize = 5;

        Assert.Throws<ArgumentOutOfRangeException>(() => config.Validate());
    }

    [Fact]
    public void Validate_WithInvalidMaxQueueSizeTooHigh_Throws()
    {
        var config = BatchingConfiguration.Default();
        config.MaxQueueSize = 10001;

        Assert.Throws<ArgumentOutOfRangeException>(() => config.Validate());
    }

    [Fact]
    public void AllTimeoutStrategyValues_AreAccessible()
    {
        var values = Enum.GetValues(typeof(TimeoutStrategy));
        Assert.Equal(3, values.Length);
    }

    [Fact]
    public void TimeoutStrategy_DispatchPartial_HasCorrectValue()
    {
        Assert.Equal(0, (int)TimeoutStrategy.DispatchPartial);
    }

    [Fact]
    public void TimeoutStrategy_WaitForFull_HasCorrectValue()
    {
        Assert.Equal(1, (int)TimeoutStrategy.WaitForFull);
    }

    [Fact]
    public void TimeoutStrategy_Adaptive_HasCorrectValue()
    {
        Assert.Equal(2, (int)TimeoutStrategy.Adaptive);
    }

    [Fact]
    public void Validate_WithPreferBatchSizeZero_Throws()
    {
        var config = BatchingConfiguration.Default();
        config.PreferBatchSize = 0;

        Assert.Throws<ArgumentOutOfRangeException>(() => config.Validate());
    }

    [Fact]
    public void Validate_WithPreferBatchSizeEqualToMaxBatchSize_DoesNotThrow()
    {
        var config = BatchingConfiguration.Default();
        config.MaxBatchSize = 16;
        config.PreferBatchSize = 16;

        var exception = Record.Exception(() => config.Validate());

        Assert.Null(exception);
    }

    [Fact]
    public void Validate_WithMaxBatchSizeAtBoundary_DoesNotThrow()
    {
        var config = new BatchingConfiguration
        {
            MaxBatchSize = 1,
            MaxWaitTime = TimeSpan.FromMilliseconds(1),
            PreferBatchSize = 1,
            MaxQueueSize = 10,
            TimeoutStrategy = TimeoutStrategy.DispatchPartial
        };

        var exception = Record.Exception(() => config.Validate());

        Assert.Null(exception);
    }

    [Fact]
    public void Validate_WithMaxBatchSizeAtUpperBoundary_DoesNotThrow()
    {
        var config = new BatchingConfiguration
        {
            MaxBatchSize = 1024,
            MaxWaitTime = TimeSpan.FromMilliseconds(1),
            PreferBatchSize = 512,
            MaxQueueSize = 10,
            TimeoutStrategy = TimeoutStrategy.DispatchPartial
        };

        var exception = Record.Exception(() => config.Validate());

        Assert.Null(exception);
    }

    [Fact]
    public void Validate_WithMaxWaitTimeAtBoundary_DoesNotThrow()
    {
        var config = new BatchingConfiguration
        {
            MaxBatchSize = 16,
            MaxWaitTime = TimeSpan.FromMilliseconds(1),
            PreferBatchSize = 8,
            MaxQueueSize = 10,
            TimeoutStrategy = TimeoutStrategy.DispatchPartial
        };

        var exception = Record.Exception(() => config.Validate());

        Assert.Null(exception);
    }

    [Fact]
    public void Validate_WithMaxWaitTimeAtUpperBoundary_DoesNotThrow()
    {
        var config = new BatchingConfiguration
        {
            MaxBatchSize = 16,
            MaxWaitTime = TimeSpan.FromMilliseconds(1000),
            PreferBatchSize = 8,
            MaxQueueSize = 10,
            TimeoutStrategy = TimeoutStrategy.DispatchPartial
        };

        var exception = Record.Exception(() => config.Validate());

        Assert.Null(exception);
    }

    [Fact]
    public void Validate_WithMaxQueueSizeAtBoundary_DoesNotThrow()
    {
        var config = new BatchingConfiguration
        {
            MaxBatchSize = 16,
            MaxWaitTime = TimeSpan.FromMilliseconds(10),
            PreferBatchSize = 8,
            MaxQueueSize = 10,
            TimeoutStrategy = TimeoutStrategy.DispatchPartial
        };

        var exception = Record.Exception(() => config.Validate());

        Assert.Null(exception);
    }

    [Fact]
    public void Validate_WithMaxQueueSizeAtUpperBoundary_DoesNotThrow()
    {
        var config = new BatchingConfiguration
        {
            MaxBatchSize = 16,
            MaxWaitTime = TimeSpan.FromMilliseconds(10),
            PreferBatchSize = 8,
            MaxQueueSize = 10000,
            TimeoutStrategy = TimeoutStrategy.DispatchPartial
        };

        var exception = Record.Exception(() => config.Validate());

        Assert.Null(exception);
    }
}
