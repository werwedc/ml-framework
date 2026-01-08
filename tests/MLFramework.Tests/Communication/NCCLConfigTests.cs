namespace MLFramework.Communication.Tests;

using MLFramework.Communication.Backends;
using Xunit;

/// <summary>
/// Tests for NCCL configuration
/// </summary>
public class NCCLConfigTests
{
    [Fact]
    public void Config_DefaultValues_AreCorrect()
    {
        // Arrange & Act
        var config = new NCCLConfig();

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
        var config = new NCCLConfig();

        // Act
        var isValid = config.Validate();

        // Assert
        Assert.True(isValid);
    }

    [Fact]
    public void Config_Validate_ReturnsFalseForInvalidBufferSize()
    {
        // Arrange
        var config = new NCCLConfig
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
        var config = new NCCLConfig
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
        var config = new NCCLConfig
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
        var config = new NCCLConfig
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
        var config = new NCCLConfig
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
        var debug = Environment.GetEnvironmentVariable("NCCL_DEBUG");
        var bufferSize = Environment.GetEnvironmentVariable("NCCL_BUFFSIZE");
        var numChannels = Environment.GetEnvironmentVariable("NCCL_NCHANNELS");
        var algo = Environment.GetEnvironmentVariable("NCCL_ALGO");
        var blockingWait = Environment.GetEnvironmentVariable("NCCL_BLOCKING_WAIT");

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
        var config = NCCLConfig.CreateDefault();

        // Assert
        Assert.NotNull(config);
        Assert.True(config.Validate());
    }

    [Fact]
    public void Config_CreateDefault_HasCorrectValues()
    {
        // Act
        var config = NCCLConfig.CreateDefault();

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
        var config = NCCLConfig.CreateHighPerformance();

        // Assert
        Assert.NotNull(config);
        Assert.True(config.Validate());
    }

    [Fact]
    public void Config_CreateHighPerformance_HasOptimizedValues()
    {
        // Act
        var config = NCCLConfig.CreateHighPerformance();

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
        var config = NCCLConfig.CreateDebug();

        // Assert
        Assert.NotNull(config);
        Assert.True(config.Validate());
    }

    [Fact]
    public void Config_CreateDebug_HasDebugEnabled()
    {
        // Act
        var config = NCCLConfig.CreateDebug();

        // Assert
        Assert.True(config.EnableDebug);
        Assert.True(config.UseRingAllReduce);
        Assert.False(config.UseTreeAllReduce);
    }

    [Fact]
    public void Config_SetEnvironmentVariable_SetsValue()
    {
        // Arrange
        string testKey = "NCCL_TEST_VAR";
        string testValue = "12345";

        // Act
        NCCLConfig.SetEnvironmentVariable(testKey, testValue);

        // Assert
        var result = Environment.GetEnvironmentVariable(testKey);
        Assert.Equal(testValue, result);

        // Cleanup
        Environment.SetEnvironmentVariable(testKey, null);
    }
}
