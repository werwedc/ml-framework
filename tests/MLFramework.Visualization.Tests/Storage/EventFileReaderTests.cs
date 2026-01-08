using Xunit;
using MachineLearning.Visualization.Events;
using MLFramework.Visualization.Storage;

namespace MLFramework.Visualization.Tests.Storage;

/// <summary>
/// Unit tests for EventFileReader
/// </summary>
public class EventFileReaderTests : IDisposable
{
    private readonly string _testDirectory;
    private readonly string _testFilePath;

    public EventFileReaderTests()
    {
        _testDirectory = Path.Combine(Path.GetTempPath(), $"EventFileReaderTests_{Guid.NewGuid():N}");
        _testFilePath = Path.Combine(_testDirectory, "test.events");
        Directory.CreateDirectory(_testDirectory);
    }

    [Fact]
    public void Constructor_WithValidFilePath_CreatesReader()
    {
        // Arrange - Create test file
        using (var writer = new EventFileWriter(_testFilePath))
        {
            writer.WriteEvent(new ScalarMetricEvent("test", 1.0f, 1));
        }

        // Act
        using var reader = new EventFileReader(_testFilePath);

        // Assert
        Assert.Equal(_testFilePath, reader.FilePath);
    }

    [Fact]
    public void Constructor_WithNonexistentFilePath_ThrowsFileNotFoundException()
    {
        // Act & Assert
        Assert.Throws<FileNotFoundException>(() => new EventFileReader(_testFilePath));
    }

    [Fact]
    public void Constructor_WithNullFilePath_ThrowsArgumentException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => new EventFileReader(null!));
    }

    [Fact]
    public void ReadEvents_WithWrittenEvents_ReturnsEvents()
    {
        // Arrange - Write test events
        using (var writer = new EventFileWriter(_testFilePath))
        {
            writer.WriteEvent(new ScalarMetricEvent("test1", 1.0f, 1));
            writer.WriteEvent(new ScalarMetricEvent("test2", 2.0f, 2));
            writer.WriteEvent(new ScalarMetricEvent("test3", 3.0f, 3));
        }

        // Act
        using var reader = new EventFileReader(_testFilePath);
        var events = reader.ReadEvents().ToList();

        // Assert
        Assert.Equal(3, events.Count);
    }

    [Fact]
    public void ReadEvents_WithEmptyFile_ReturnsEmptyList()
    {
        // Arrange - Create empty file
        using (var writer = new EventFileWriter(_testFilePath))
        {
            // Don't write any events
        }

        // Act
        using var reader = new EventFileReader(_testFilePath);
        var events = reader.ReadEvents().ToList();

        // Assert
        Assert.Empty(events);
    }

    [Fact]
    public void ReadEvents_AfterDispose_ThrowsObjectDisposedException()
    {
        // Arrange - Write test events
        using (var writer = new EventFileWriter(_testFilePath))
        {
            writer.WriteEvent(new ScalarMetricEvent("test", 1.0f, 1));
        }

        var reader = new EventFileReader(_testFilePath);
        reader.Dispose();

        // Act & Assert
        Assert.Throws<ObjectDisposedException>(() => reader.ReadEvents());
    }

    [Fact]
    public async Task ReadEventsAsync_WithWrittenEvents_ReturnsEvents()
    {
        // Arrange - Write test events
        using (var writer = new EventFileWriter(_testFilePath))
        {
            writer.WriteEvent(new ScalarMetricEvent("test1", 1.0f, 1));
            writer.WriteEvent(new ScalarMetricEvent("test2", 2.0f, 2));
        }

        // Act
        using var reader = new EventFileReader(_testFilePath);
        var events = (await reader.ReadEventsAsync()).ToList();

        // Assert
        Assert.Equal(2, events.Count);
    }

    [Fact]
    public void ReadEvents_WithDifferentEventTypes_ReturnsAllEvents()
    {
        // Arrange - Write different event types
        using (var writer = new EventFileWriter(_testFilePath))
        {
            writer.WriteEvent(new ScalarMetricEvent("scalar", 1.0f, 1));
            writer.WriteEvent(new HistogramEvent("histogram", new float[] { 1, 2, 3 }, 2));
            writer.WriteEvent(new ProfilingStartEvent("profiling", 3));
        }

        // Act
        using var reader = new EventFileReader(_testFilePath);
        var events = reader.ReadEvents().ToList();

        // Assert
        Assert.Equal(3, events.Count);
        Assert.Contains(events, e => e is ScalarMetricEvent);
        Assert.Contains(events, e => e is HistogramEvent);
        Assert.Contains(events, e => e is ProfilingStartEvent);
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
