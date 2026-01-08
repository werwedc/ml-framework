using Xunit;
using MachineLearning.Visualization.Events;
using MachineLearning.Visualization.Storage;
using MLFramework.Visualization.Storage;

namespace MLFramework.Visualization.Tests.Storage;

/// <summary>
/// Unit tests for FileStorageBackend
/// </summary>
public class FileStorageBackendTests : IDisposable
{
    private readonly string _testDirectory;
    private readonly StorageConfiguration _config;

    public FileStorageBackendTests()
    {
        _testDirectory = Path.Combine(Path.GetTempPath(), $"FileStorageBackendTests_{Guid.NewGuid():N}");
        _config = new StorageConfiguration
        {
            BackendType = "file",
            ConnectionString = _testDirectory,
            WriteBufferSize = 5,
            FlushInterval = TimeSpan.FromMilliseconds(100),
            EnableAsyncWrites = true
        };
        Directory.CreateDirectory(_testDirectory);
    }

    [Fact]
    public void Constructor_WithValidConfig_CreatesBackend()
    {
        // Act
        using var backend = new FileStorageBackend(_config);

        // Assert
        Assert.NotNull(backend);
        Assert.Equal(_testDirectory, backend.LogDirectory);
        Assert.False(backend.IsInitialized);
    }

    [Fact]
    public void Constructor_WithNullConfig_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new FileStorageBackend(null!));
    }

    [Fact]
    public void Initialize_WithValidDirectory_InitializesBackend()
    {
        // Arrange
        using var backend = new FileStorageBackend(_config);

        // Act
        backend.Initialize(_testDirectory);

        // Assert
        Assert.True(backend.IsInitialized);
        Assert.True(Directory.Exists(_testDirectory));
    }

    [Fact]
    public void StoreEvent_WhenNotInitialized_ThrowsInvalidOperationException()
    {
        // Arrange
        using var backend = new FileStorageBackend(_config);
        var testEvent = new ScalarMetricEvent("test", 1.0f, 1);

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => backend.StoreEvent(testEvent));
    }

    [Fact]
    public void StoreEvent_WithValidEvent_IncrementsEventCount()
    {
        // Arrange
        using var backend = new FileStorageBackend(_config);
        backend.Initialize(_testDirectory);
        var testEvent = new ScalarMetricEvent("test", 1.0f, 1);

        // Act
        backend.StoreEvent(testEvent);

        // Assert
        Assert.Equal(1, backend.EventCount);
    }

    [Fact]
    public void StoreEvent_WithMultipleEvents_IncrementsEventCount()
    {
        // Arrange
        using var backend = new FileStorageBackend(_config);
        backend.Initialize(_testDirectory);

        // Act
        for (int i = 0; i < 10; i++)
        {
            backend.StoreEvent(new ScalarMetricEvent($"test{i}", i, i));
        }

        // Assert
        Assert.Equal(10, backend.EventCount);
    }

    [Fact]
    public void StoreEvent_WhenBufferSizeReached_AutoFlushes()
    {
        // Arrange
        using var backend = new FileStorageBackend(_config);
        backend.Initialize(_testDirectory);

        // Act - Write exactly WriteBufferSize events
        for (int i = 0; i < _config.WriteBufferSize; i++)
        {
            backend.StoreEvent(new ScalarMetricEvent($"test{i}", i, i));
        }

        // Assert - Events should be flushed to disk
        var eventFiles = Directory.GetFiles(_testDirectory, "*.events");
        Assert.Single(eventFiles);
    }

    [Fact]
    public void GetEvents_WithValidRange_ReturnsEvents()
    {
        // Arrange
        using var backend = new FileStorageBackend(_config);
        backend.Initialize(_testDirectory);

        // Add events with different steps
        for (int i = 0; i < 10; i++)
        {
            var ev = new ScalarMetricEvent($"test{i}", i, i);
            backend.StoreEvent(ev);
        }
        backend.Flush();

        // Act
        var events = backend.GetEvents(3, 6).ToList();

        // Assert
        Assert.NotNull(events);
        Assert.Equal(4, events.Count); // Should get events with steps 3, 4, 5, 6
    }

    [Fact]
    public void GetEvents_WithNoFiles_ReturnsEmptyList()
    {
        // Arrange
        using var backend = new FileStorageBackend(_config);
        backend.Initialize(_testDirectory);

        // Act
        var events = backend.GetEvents(0, 10).ToList();

        // Assert
        Assert.Empty(events);
    }

    [Fact]
    public void Flush_WithEventsInBuffer_WritesToFile()
    {
        // Arrange
        using var backend = new FileStorageBackend(_config);
        backend.Initialize(_testDirectory);
        backend.StoreEvent(new ScalarMetricEvent("test", 1.0f, 1));

        // Act
        backend.Flush();

        // Assert
        var eventFiles = Directory.GetFiles(_testDirectory, "*.events");
        Assert.Single(eventFiles);
    }

    [Fact]
    public void Clear_WhenInitialized_RemovesAllEvents()
    {
        // Arrange
        using var backend = new FileStorageBackend(_config);
        backend.Initialize(_testDirectory);
        backend.StoreEvent(new ScalarMetricEvent("test", 1.0f, 1));
        backend.Flush();

        // Act
        backend.Clear();

        // Assert
        Assert.Equal(0, backend.EventCount);
        var eventFiles = Directory.GetFiles(_testDirectory, "*.events");
        Assert.Empty(eventFiles);
    }

    [Fact]
    public async Task StoreEventAsync_WithValidEvent_StoresEvent()
    {
        // Arrange
        using var backend = new FileStorageBackend(_config);
        backend.Initialize(_testDirectory);
        var testEvent = new ScalarMetricEvent("test", 1.0f, 1);

        // Act
        await backend.StoreEventAsync(testEvent);

        // Assert
        Assert.Equal(1, backend.EventCount);
    }

    [Fact]
    public void FileRotation_WhenMaxFileSizeReached_CreatesNewFile()
    {
        // Arrange
        using var backend = new FileStorageBackend(_config);
        backend.Initialize(_testDirectory);
        backend.MaxFileSize = 1024; // Small size for testing

        // Act - Write many events to trigger rotation
        for (int i = 0; i < 100; i++)
        {
            backend.StoreEvent(new ScalarMetricEvent($"test{i}", i, i));
        }
        backend.Flush();

        // Assert - Should have created multiple files
        var eventFiles = Directory.GetFiles(_testDirectory, "*.events");
        Assert.True(eventFiles.Length > 1);
    }

    [Fact]
    public void CurrentFile_AfterInitialization_HasValue()
    {
        // Arrange
        using var backend = new FileStorageBackend(_config);

        // Act
        backend.Initialize(_testDirectory);

        // Assert
        Assert.NotNull(backend.CurrentFile);
        Assert.True(File.Exists(backend.CurrentFile));
    }

    [Fact]
    public void Dispose_WhenCalled_FlushesAndShutsDown()
    {
        // Arrange
        var backend = new FileStorageBackend(_config);
        backend.Initialize(_testDirectory);
        backend.StoreEvent(new ScalarMetricEvent("test", 1.0f, 1));

        // Act
        backend.Dispose();

        // Assert - Events should be flushed
        var eventFiles = Directory.GetFiles(_testDirectory, "*.events");
        Assert.Single(eventFiles);
    }

    [Fact]
    public void MaxFileSize_CanBeChanged_AffectsRotation()
    {
        // Arrange
        using var backend = new FileStorageBackend(_config);
        backend.Initialize(_testDirectory);

        // Act
        backend.MaxFileSize = 5 * 1024 * 1024; // 5MB

        // Assert
        Assert.Equal(5 * 1024 * 1024, backend.MaxFileSize);
    }

    public void Dispose()
    {
        // Cleanup
        if (Directory.Exists(_testDirectory))
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
}
