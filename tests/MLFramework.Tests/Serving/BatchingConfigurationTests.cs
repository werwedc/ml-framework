using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;

namespace MLFramework.Tests.Serving;

[TestClass]
public class BatchingConfigurationTests
{
    [TestMethod]
    public void DefaultConfiguration_HasValidValues()
    {
        var config = BatchingConfiguration.Default();
        Assert.AreEqual(32, config.MaxBatchSize);
        Assert.AreEqual(TimeSpan.FromMilliseconds(5), config.MaxWaitTime);
        Assert.AreEqual(16, config.PreferBatchSize);
        Assert.AreEqual(100, config.MaxQueueSize);
        Assert.AreEqual(TimeoutStrategy.DispatchPartial, config.TimeoutStrategy);
    }

    [TestMethod]
    public void Validate_WithValidConfig_DoesNotThrow()
    {
        var config = BatchingConfiguration.Default();
        config.Validate(); // Should not throw
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentOutOfRangeException))]
    public void Validate_WithInvalidMaxBatchSize_Throws()
    {
        var config = BatchingConfiguration.Default();
        config.MaxBatchSize = 0;
        config.Validate();
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentOutOfRangeException))]
    public void Validate_WithMaxBatchSizeTooLarge_Throws()
    {
        var config = BatchingConfiguration.Default();
        config.MaxBatchSize = 1025;
        config.Validate();
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentOutOfRangeException))]
    public void Validate_WithPreferBatchSizeGreaterThanMax_Throws()
    {
        var config = BatchingConfiguration.Default();
        config.MaxBatchSize = 16;
        config.PreferBatchSize = 32;
        config.Validate();
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentOutOfRangeException))]
    public void Validate_WithInvalidMaxWaitTime_Throws()
    {
        var config = BatchingConfiguration.Default();
        config.MaxWaitTime = TimeSpan.Zero;
        config.Validate();
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentOutOfRangeException))]
    public void Validate_WithMaxWaitTimeTooLong_Throws()
    {
        var config = BatchingConfiguration.Default();
        config.MaxWaitTime = TimeSpan.FromMilliseconds(1001);
        config.Validate();
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentOutOfRangeException))]
    public void Validate_WithInvalidMaxQueueSize_Throws()
    {
        var config = BatchingConfiguration.Default();
        config.MaxQueueSize = 5;
        config.Validate();
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentOutOfRangeException))]
    public void Validate_WithMaxQueueSizeTooLarge_Throws()
    {
        var config = BatchingConfiguration.Default();
        config.MaxQueueSize = 10001;
        config.Validate();
    }

    [TestMethod]
    public void AllTimeoutStrategyValues_AreAccessible()
    {
        Assert.AreEqual(3, Enum.GetValues(typeof(TimeoutStrategy)).Length);
    }

    [TestMethod]
    public void AllTimeoutStrategyValues_AreUnique()
    {
        var values = Enum.GetValues(typeof(TimeoutStrategy));
        var uniqueValues = new HashSet<int>();

        foreach (TimeoutStrategy value in values)
        {
            Assert.IsTrue(uniqueValues.Add((int)value));
        }
    }

    [TestMethod]
    public void Validate_WithValidMinValues_DoesNotThrow()
    {
        var config = new BatchingConfiguration
        {
            MaxBatchSize = 1,
            MaxWaitTime = TimeSpan.FromMilliseconds(1),
            PreferBatchSize = 1,
            MaxQueueSize = 10,
            TimeoutStrategy = TimeoutStrategy.DispatchPartial
        };

        config.Validate(); // Should not throw
    }

    [TestMethod]
    public void Validate_WithValidMaxValues_DoesNotThrow()
    {
        var config = new BatchingConfiguration
        {
            MaxBatchSize = 1024,
            MaxWaitTime = TimeSpan.FromMilliseconds(1000),
            PreferBatchSize = 1024,
            MaxQueueSize = 10000,
            TimeoutStrategy = TimeoutStrategy.Adaptive
        };

        config.Validate(); // Should not throw
    }

    [TestMethod]
    public void Validate_WithPreferBatchSizeEqualToMax_DoesNotThrow()
    {
        var config = new BatchingConfiguration
        {
            MaxBatchSize = 32,
            PreferBatchSize = 32,
            MaxWaitTime = TimeSpan.FromMilliseconds(50),
            MaxQueueSize = 100,
            TimeoutStrategy = TimeoutStrategy.DispatchPartial
        };

        config.Validate(); // Should not throw
    }

    [TestMethod]
    public void Validate_WithAllTimeoutStrategies_DoesNotThrow()
    {
        var strategies = Enum.GetValues(typeof(TimeoutStrategy));

        foreach (TimeoutStrategy strategy in strategies)
        {
            var config = new BatchingConfiguration
            {
                MaxBatchSize = 32,
                MaxWaitTime = TimeSpan.FromMilliseconds(50),
                PreferBatchSize = 16,
                MaxQueueSize = 100,
                TimeoutStrategy = strategy
            };

            config.Validate(); // Should not throw for any strategy
        }
    }

    [TestMethod]
    public void Default_CanBeModified()
    {
        var config = BatchingConfiguration.Default();
        config.MaxBatchSize = 64;
        Assert.AreEqual(64, config.MaxBatchSize);
    }

    [TestMethod]
    public void Validate_WithPreferBatchSizeZero_Throws()
    {
        var config = BatchingConfiguration.Default();
        config.PreferBatchSize = 0;

        var ex = Assert.ThrowsException<ArgumentOutOfRangeException>(() => config.Validate());
        Assert.IsTrue(ex.Message.Contains("PreferBatchSize"));
    }

    [TestMethod]
    public void Validate_WithNegativeMaxBatchSize_Throws()
    {
        var config = BatchingConfiguration.Default();
        config.MaxBatchSize = -1;

        var ex = Assert.ThrowsException<ArgumentOutOfRangeException>(() => config.Validate());
        Assert.IsTrue(ex.Message.Contains("MaxBatchSize"));
    }
}
