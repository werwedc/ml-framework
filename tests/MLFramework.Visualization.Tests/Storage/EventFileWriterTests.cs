using Xunit;
using MachineLearning.Visualization.Events;
using MachineLearning.Visualization.Storage;
using MLFramework.Visualization.Storage;

namespace MLFramework.Visualization.Tests.Storage;

/// <summary>
/// Unit tests for EventFileWriter
/// </summary>
public class EventFileWriterTests : IDisposable
{
    private readonly string _testDirectory;
    private readonly string _testFilePath;

    public EventFileWriterTests()
    {
        _testDirectory = Path.Combine(Path.GetTempPath(), $"EventFileWriterTests_{Guid.NewGuid():N}");
        _testFilePath = Path.Combine(_testDirectory, "test.events");
        Directory.CreateDirectory(_testDirectory);
    }

    [Fact]
    public void Constructor_WithValidFilePath_CreatesFile()
    {
        // Act
        using var writer = new EventFileWriter(_testFilePath);

        // Assert
        Assert.True(File.Exists(_testFilePath));
        Assert.Equal(_testFilePath, writer.FilePath);
        Assert.Equal(0, writer.FileSize);
    }

    [Fact]
    public void Constructor_WithNullFilePath_ThrowsArgumentException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => new EventFileWriter(null!));
    }

    [Fact]
    public void Constructor_WithEmptyFilePath_ThrowsArgumentException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => new EventFileWriter(string.Empty));
    }

    [Fact]
    public void WriteEvent_WithValidEvent_WritesToFile()
    {
        // Arrange
        using var writer = new EventFileWriter(_testFilePath);
        var testEvent = new ScalarMetricEvent("test", 1.0f, 1);

        // Act
        writer.WriteEvent(testEvent);

        // Assert
        Assert.True(writer.FileSize > 0);
        Assert.True(File.Exists(_testFilePath));
    }

    [Fact]
    public void WriteEvent_WithNullEvent_ThrowsArgumentNullException()
    {
        // Arrange
        using var writer = new EventFileWriter(_testFilePath);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => writer.WriteEvent(null!));
    }

    [Fact]
    public void WriteEvent_AfterDispose_ThrowsObjectDisposedException()
    {
        // Arrange
        var writer = new EventFileWriter(_testFilePath);
        var testEvent = new ScalarMetricEvent("test", 1.0f, 1);
        writer.Dispose();

        // Act & Assert
        Assert.Throws<ObjectDisposedException>(() => writer.WriteEvent(testEvent));
    }

    [Fact]
    public async Task WriteEventAsync_WithValidEvent_WritesToFile()
    {
        // Arrange
        using var writer = new EventFileWriter(_testFilePath);
        var testEvent = new ScalarMetricEvent("test", 1.0f, 1);

        // Act
        await writer.WriteEventAsync(testEvent);

        // Assert
        Assert.True(writer.FileSize > 0);
    }

    [Fact]
    public void Flush_WithWrittenEvents_FlushesToDisk()
    {
        // Arrange
        using var writer = new EventFileWriter(_testFilePath);
        var testEvent = new ScalarMetricEvent("test", 1.0f, 1);
        writer.WriteEvent(testEvent);

        // Act
        writer.Flush();

        // Assert - File size should match writer's reported size
        Assert.Equal(writer.FileSize, new FileInfo(_testFilePath).Length);
    }

    [Fact]
    public void WriteEvent_WithMultipleEvents_IncreasesFileSize()
    {
        // Arrange
        using var writer = new EventFileWriter(_testFilePath);
        long initialSize = writer.FileSize;

        // Act
        for (int i = 0; i < 10; i++)
        {
            writer.WriteEvent(new ScalarMetricEvent($"test{i}", i, i));
        }

        // Assert
        Assert.True(writer.FileSize > initialSize);
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
