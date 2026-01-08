namespace MLFramework.Communication.Tests;

using MLFramework.Communication.Backends;
using Xunit;

/// <summary>
/// Tests for RCCL configuration
/// </summary>
public class RCCLConfigTests
{
    [Fact]
    public void Config_DefaultValues_AreCorrect()
    {
        // Arrange & Act
        var config = new RCCLConfig();

        // Assert
        Assert.True(config.UseRingAllReduce);
        Assert.False(config.UseTreeAllReduce);
        Assert.Equal(1024 * 1024, config.TreeThresholdBytes); // 1MB
        Assert.Equal(1, config.NumChannels);
        Assert.False(config.EnableDebug);
        Assert.Equal(4194304, config.BufferSize); // 4MB
        Assert.True(config.UseAsyncOps);
        Assert.Equal(300000, config.TimeoutMs); // 5 minutes
    }

    [Fact]
    public void Config_Validate_ReturnsTrueForValidConfig()
    {
        // Arrange
        var config = new RCCLConfig();

        // Act
        var isValid = config.Validate();

        // Assert
        Assert.True(isValid);
    }

    [Fact]
    public void Config_Validate_ReturnsFalseForInvalidBufferSize()
    {
        // Arrange
        var config = new RCCLConfig
        {
            BufferSize = -1
        };

        // Act
        var isValid = config.Validate();

        // Assert
        Assert.False(isValid);
    }

    [Fact]
    public void Config_Validate_ReturnsFalseForInvalidNumChannels()
    {
        // Arrange
        var config = new RCCLConfig
        {
            NumChannels = 0
        };

        // Act
        var isValid = config.Validate();

        // Assert
        Assert.False(isValid);
    }

    [Fact]
    public void Config_Validate_ReturnsFalseForInvalidTimeout()
    {
        // Arrange
        var config = new RCCLConfig
        {
            TimeoutMs = -100
        };

        // Act
        var isValid = config.Validate();

        // Assert
        Assert.False(isValid);
    }

    [Fact]
    public void Config_Validate_ReturnsFalseForNegativeTreeThreshold()
    {
        // Arrange
        var config = new RCCLConfig
        {
            TreeThresholdBytes = -1
        };

        // Act
        var isValid = config.Validate();

        // Assert
        Assert.False(isValid);
    }

    [Fact]
    public void Config_Apply_SetsEnvironmentVariables()
    {
        // Arrange
        var config = new RCCLConfig
        {
            EnableDebug = true,
            BufferSize = 8 * 1024 * 1024,
            NumChannels = 2,
            UseTreeAllReduce = true,
            TimeoutMs = 600000
        };

        // Act
        config.Apply();

        // Assert
        var debug = Environment.GetEnvironmentVariable("RCCL_DEBUG");
        var bufferSize = Environment.GetEnvironmentVariable("RCCL_BUFFSIZE");
        var numChannels = Environment.GetEnvironmentVariable("RCCL_NCHANNELS");
        var algo = Environment.GetEnvironmentVariable("RCCL_ALGO");
        var blockingWait = Environment.GetEnvironmentVariable("RCCL_BLOCKING_WAIT");

        Assert.Equal("INFO", debug);
        Assert.Equal((8 * 1024 * 1024).ToString(), bufferSize);
        Assert.Equal("2", numChannels);
        Assert.Equal("Tree", algo);
        Assert.Equal("600000", blockingWait);
    }

    [Fact]
    public void Config_CreateDefault_ReturnsValidConfig()
    {
        // Act
        var config = RCCLConfig.CreateDefault();

        // Assert
        Assert.NotNull(config);
        Assert.True(config.Validate());
    }

    [Fact]
    public void Config_CreateDefault_HasCorrectValues()
    {
        // Act
        var config = RCCLConfig.CreateDefault();

        // Assert
        Assert.True(config.UseRingAllReduce);
        Assert.False(config.UseTreeAllReduce);
        Assert.Equal(1024 * 1024, config.TreeThresholdBytes); // 1MB
        Assert.Equal(1, config.NumChannels);
        Assert.False(config.EnableDebug);
        Assert.Equal(4194304, config.BufferSize); // 4MB
        Assert.True(config.UseAsyncOps);
        Assert.Equal(300000, config.TimeoutMs); // 5 minutes
    }

    [Fact]
    public void Config_CreateHighPerformance_ReturnsValidConfig()
    {
        // Act
        var config = RCCLConfig.CreateHighPerformance();

        // Assert
        Assert.NotNull(config);
        Assert.True(config.Validate());
    }

    [Fact]
    public void Config_CreateHighPerformance_HasOptimizedValues()
    {
        // Act
        var config = RCCLConfig.CreateHighPerformance();

        // Assert
        Assert.True(config.UseRingAllReduce);
        Assert.True(config.UseTreeAllReduce);
        Assert.Equal(16 * 1024 * 1024, config.TreeThresholdBytes); // 16MB
        Assert.Equal(4, config.NumChannels); // Multi-rail
        Assert.False(config.EnableDebug);
        Assert.Equal(16 * 1024 * 1024, config.BufferSize); // 16MB
        Assert.True(config.UseAsyncOps);
        Assert.Equal(600000, config.TimeoutMs); // 10 minutes
    }

    [Fact]
    public void Config_CreateDebug_ReturnsValidConfig()
    {
        // Act
        var config = RCCLConfig.CreateDebug();

        // Assert
        Assert.NotNull(config);
        Assert.True(config.Validate());
    }

    [Fact]
    public void Config_CreateDebug_HasDebugEnabled()
    {
        // Act
        var config = RCCLConfig.CreateDebug();

        // Assert
        Assert.True(config.EnableDebug);
        Assert.True(config.UseRingAllReduce);
        Assert.False(config.UseTreeAllReduce);
    }

    [Fact]
    public void Config_SetEnvironmentVariable_SetsValue()
    {
        // Arrange
        string testKey = "RCCL_TEST_VAR";
        string testValue = "12345";

        // Act
        RCCLConfig.SetEnvironmentVariable(testKey, testValue);

        // Assert
        var result = Environment.GetEnvironmentVariable(testKey);
        Assert.Equal(testValue, result);

        // Cleanup
        Environment.SetEnvironmentVariable(testKey, null);
    }
}
