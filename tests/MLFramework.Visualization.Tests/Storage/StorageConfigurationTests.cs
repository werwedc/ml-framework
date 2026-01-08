using Xunit;
using MachineLearning.Visualization.Storage;

namespace MLFramework.Visualization.Tests.Storage;

/// <summary>
/// Unit tests for StorageConfiguration
/// </summary>
public class StorageConfigurationTests
{
    [Fact]
    public void Constructor_WithDefaults_CreatesValidConfiguration()
    {
        // Arrange & Act
        var config = new StorageConfiguration();

        // Assert
        Assert.Equal("file", config.BackendType);
        Assert.Equal(string.Empty, config.ConnectionString);
        Assert.Equal(100, config.WriteBufferSize);
        Assert.Equal(TimeSpan.FromSeconds(1), config.FlushInterval);
        Assert.True(config.EnableAsyncWrites);
    }

    [Fact]
    public void IsValid_WithValidConfiguration_ReturnsTrue()
    {
        // Arrange
        var config = new StorageConfiguration
        {
            BackendType = "file",
            ConnectionString = "./logs",
            WriteBufferSize = 50,
            FlushInterval = TimeSpan.FromSeconds(2),
            EnableAsyncWrites = true
        };

        // Act
        var result = config.IsValid();

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void IsValid_WithNullBackendType_ReturnsFalse()
    {
        // Arrange
        var config = new StorageConfiguration
        {
            BackendType = null,
            ConnectionString = "./logs"
        };

        // Act
        var result = config.IsValid();

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void IsValid_WithEmptyBackendType_ReturnsFalse()
    {
        // Arrange
        var config = new StorageConfiguration
        {
            BackendType = "",
            ConnectionString = "./logs"
        };

        // Act
        var result = config.IsValid();

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void IsValid_WithWhitespaceBackendType_ReturnsFalse()
    {
        // Arrange
        var config = new StorageConfiguration
        {
            BackendType = "   ",
            ConnectionString = "./logs"
        };

        // Act
        var result = config.IsValid();

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void IsValid_WithNullConnectionString_ReturnsFalse()
    {
        // Arrange
        var config = new StorageConfiguration
        {
            BackendType = "file",
            ConnectionString = null
        };

        // Act
        var result = config.IsValid();

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void IsValid_WithEmptyConnectionString_ReturnsFalse()
    {
        // Arrange
        var config = new StorageConfiguration
        {
            BackendType = "file",
            ConnectionString = ""
        };

        // Act
        var result = config.IsValid();

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void IsValid_WithZeroWriteBufferSize_ReturnsFalse()
    {
        // Arrange
        var config = new StorageConfiguration
        {
            BackendType = "file",
            ConnectionString = "./logs",
            WriteBufferSize = 0
        };

        // Act
        var result = config.IsValid();

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void IsValid_WithNegativeWriteBufferSize_ReturnsFalse()
    {
        // Arrange
        var config = new StorageConfiguration
        {
            BackendType = "file",
            ConnectionString = "./logs",
            WriteBufferSize = -10
        };

        // Act
        var result = config.IsValid();

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void IsValid_WithZeroFlushInterval_ReturnsFalse()
    {
        // Arrange
        var config = new StorageConfiguration
        {
            BackendType = "file",
            ConnectionString = "./logs",
            FlushInterval = TimeSpan.Zero
        };

        // Act
        var result = config.IsValid();

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void IsValid_WithNegativeFlushInterval_ReturnsFalse()
    {
        // Arrange
        var config = new StorageConfiguration
        {
            BackendType = "file",
            ConnectionString = "./logs",
            FlushInterval = TimeSpan.FromSeconds(-1)
        };

        // Act
        var result = config.IsValid();

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void EnsureValid_WithValidConfiguration_DoesNotThrow()
    {
        // Arrange
        var config = new StorageConfiguration
        {
            BackendType = "file",
            ConnectionString = "./logs",
            WriteBufferSize = 50,
            FlushInterval = TimeSpan.FromSeconds(2)
        };

        // Act & Assert
        config.EnsureValid();
    }

    [Fact]
    public void EnsureValid_WithNullBackendType_ThrowsInvalidOperationException()
    {
        // Arrange
        var config = new StorageConfiguration
        {
            BackendType = null,
            ConnectionString = "./logs"
        };

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => config.EnsureValid());
    }

    [Fact]
    public void EnsureValid_WithEmptyBackendType_ThrowsInvalidOperationException()
    {
        // Arrange
        var config = new StorageConfiguration
        {
            BackendType = "",
            ConnectionString = "./logs"
        };

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => config.EnsureValid());
    }

    [Fact]
    public void EnsureValid_WithNullConnectionString_ThrowsInvalidOperationException()
    {
        // Arrange
        var config = new StorageConfiguration
        {
            BackendType = "file",
            ConnectionString = null
        };

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => config.EnsureValid());
    }

    [Fact]
    public void EnsureValid_WithZeroWriteBufferSize_ThrowsInvalidOperationException()
    {
        // Arrange
        var config = new StorageConfiguration
        {
            BackendType = "file",
            ConnectionString = "./logs",
            WriteBufferSize = 0
        };

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => config.EnsureValid());
    }

    [Fact]
    public void EnsureValid_WithZeroFlushInterval_ThrowsInvalidOperationException()
    {
        // Arrange
        var config = new StorageConfiguration
        {
            BackendType = "file",
            ConnectionString = "./logs",
            FlushInterval = TimeSpan.Zero
        };

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => config.EnsureValid());
    }

    [Fact]
    public void Properties_WithValidValues_CanBeSet()
    {
        // Arrange
        var config = new StorageConfiguration();

        // Act
        config.BackendType = "memory";
        config.ConnectionString = "test-connection";
        config.WriteBufferSize = 200;
        config.FlushInterval = TimeSpan.FromMinutes(5);
        config.EnableAsyncWrites = false;

        // Assert
        Assert.Equal("memory", config.BackendType);
        Assert.Equal("test-connection", config.ConnectionString);
        Assert.Equal(200, config.WriteBufferSize);
        Assert.Equal(TimeSpan.FromMinutes(5), config.FlushInterval);
        Assert.False(config.EnableAsyncWrites);
    }
}
