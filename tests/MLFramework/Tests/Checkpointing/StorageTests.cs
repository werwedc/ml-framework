namespace MachineLearning.Checkpointing.Tests;

using Xunit;

/// <summary>
/// Tests for LocalFileSystemStorage
/// </summary>
public class LocalFileSystemStorageTests
{
    private string _testDirectory = string.Empty;
    private LocalFileSystemStorage _storage = null!;

    public LocalFileSystemStorageTests()
    {
        // Constructor doesn't run [Setup], so we need to initialize in each test
        // or use a constructor pattern
    }

    private void SetUp()
    {
        _testDirectory = Path.Combine(Path.GetTempPath(), $"checkpoint_test_{Guid.NewGuid():N}");
        _storage = new LocalFileSystemStorage(_testDirectory);
    }

    private void TearDown()
    {
        if (!string.IsNullOrEmpty(_testDirectory) && Directory.Exists(_testDirectory))
        {
            try
            {
                Directory.Delete(_testDirectory, recursive: true);
            }
            catch
            {
                // Ignore cleanup errors
            }
        }
    }

    [Fact]
    public async Task WriteAsync_CreatesDirectory_IfNotExists()
    {
        // Arrange
        SetUp();
        try
        {
            var data = System.Text.Encoding.UTF8.GetBytes("test data");
            var path = "subdir/file.txt";

            // Act
            await _storage.WriteAsync(path, data);

            // Assert
            var fullPath = Path.Combine(_testDirectory, path);
            Assert.True(File.Exists(fullPath));
        }
        finally
        {
            TearDown();
        }
    }

    [Fact]
    public async Task WriteAndReadAsync_Roundtrip_Success()
    {
        // Arrange
        SetUp();
        try
        {
            var expectedData = System.Text.Encoding.UTF8.GetBytes("test data");
            var path = "test.txt";

            // Act
            await _storage.WriteAsync(path, expectedData);
            var actualData = await _storage.ReadAsync(path);

            // Assert
            Assert.Equal(expectedData, actualData);
        }
        finally
        {
            TearDown();
        }
    }

    [Fact]
    public async Task ExistsAsync_ReturnsTrue_IfFileExists()
    {
        // Arrange
        SetUp();
        try
        {
            var data = System.Text.Encoding.UTF8.GetBytes("test data");
            var path = "test.txt";
            await _storage.WriteAsync(path, data);

            // Act
            var exists = await _storage.ExistsAsync(path);

            // Assert
            Assert.True(exists);
        }
        finally
        {
            TearDown();
        }
    }

    [Fact]
    public async Task ExistsAsync_ReturnsFalse_IfFileNotExists()
    {
        // Arrange
        SetUp();
        try
        {
            // Act
            var exists = await _storage.ExistsAsync("nonexistent.txt");

            // Assert
            Assert.False(exists);
        }
        finally
        {
            TearDown();
        }
    }

    [Fact]
    public async Task DeleteAsync_RemovesFile()
    {
        // Arrange
        SetUp();
        try
        {
            var data = System.Text.Encoding.UTF8.GetBytes("test data");
            var path = "test.txt";
            await _storage.WriteAsync(path, data);

            // Act
            await _storage.DeleteAsync(path);
            var exists = await _storage.ExistsAsync(path);

            // Assert
            Assert.False(exists);
        }
        finally
        {
            TearDown();
        }
    }

    [Fact]
    public async Task GetMetadataAsync_ReturnsCorrectMetadata()
    {
        // Arrange
        SetUp();
        try
        {
            var data = System.Text.Encoding.UTF8.GetBytes("test data");
            var path = "test.txt";
            await _storage.WriteAsync(path, data);

            // Act
            var metadata = await _storage.GetMetadataAsync(path);

            // Assert
            Assert.Equal(data.Length, metadata.Size);
            Assert.True(metadata.LastModified <= DateTime.UtcNow);
            Assert.NotNull(metadata.AdditionalInfo);
        }
        finally
        {
            TearDown();
        }
    }

    [Fact]
    public void StorageFactory_CreateLocalStorage_ReturnsValidInstance()
    {
        // Arrange
        var options = new StorageOptions
        {
            Provider = "local",
            ConnectionSettings = new Dictionary<string, string>
            {
                ["basePath"] = Path.GetTempPath()
            }
        };

        // Act
        var storage = StorageFactory.Create(options);

        // Assert
        Assert.NotNull(storage);
        Assert.IsType<LocalFileSystemStorage>(storage);
    }

    [Fact]
    public void StorageFactory_WithNullOptions_ThrowsArgumentNullException()
    {
        // Arrange
        StorageOptions options = null!;

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => StorageFactory.Create(options));
    }

    [Fact]
    public void StorageFactory_WithUnknownProvider_ThrowsArgumentException()
    {
        // Arrange
        var options = new StorageOptions
        {
            Provider = "unknown",
            ConnectionSettings = new Dictionary<string, string>()
        };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => StorageFactory.Create(options));
    }

    [Fact]
    public void StorageFactory_WithMissingBasePath_ThrowsArgumentException()
    {
        // Arrange
        var options = new StorageOptions
        {
            Provider = "local",
            ConnectionSettings = new Dictionary<string, string>()
        };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => StorageFactory.Create(options));
    }
}
