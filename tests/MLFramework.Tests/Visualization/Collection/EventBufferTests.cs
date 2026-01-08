using Xunit;
using MachineLearning.Visualization.Collection;
using MachineLearning.Visualization.Tests;

namespace MLFramework.Visualization.Tests;

/// <summary>
/// Unit tests for EventBuffer
/// </summary>
public class EventBufferTests
{
    [Fact]
    public void Constructor_WithValidCapacity_CreatesBuffer()
    {
        // Arrange & Act
        var buffer = new EventBuffer(100);

        // Assert
        Assert.Equal(100, buffer.Capacity);
        Assert.Equal(0, buffer.Count);
        Assert.Equal(0, buffer.DroppedEvents);
    }

    [Fact]
    public void Constructor_WithInvalidCapacity_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => new EventBuffer(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new EventBuffer(-1));
    }

    [Fact]
    public void Enqueue_WithValidEvent_AddsToBuffer()
    {
        // Arrange
        var buffer = new EventBuffer(10);
        var testEvent = new TestEvent("test", 42);

        // Act
        buffer.Enqueue(testEvent);

        // Assert
        Assert.Equal(1, buffer.Count);
    }

    [Fact]
    public void Enqueue_WithNullEvent_ThrowsException()
    {
        // Arrange
        var buffer = new EventBuffer(10);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => buffer.Enqueue(null!));
    }

    [Fact]
    public void Enqueue_WhenFull_DropsOldestEvents()
    {
        // Arrange
        var buffer = new EventBuffer(2);
        var event1 = new TestEvent("event1", 1);
        var event2 = new TestEvent("event2", 2);
        var event3 = new TestEvent("event3", 3);

        // Act
        buffer.Enqueue(event1);
        buffer.Enqueue(event2);
        buffer.Enqueue(event3);

        // Assert
        Assert.Equal(2, buffer.Count);
        Assert.Equal(1, buffer.DroppedEvents);
    }

    [Fact]
    public void TryDequeue_WithEvents_ReturnsEvent()
    {
        // Arrange
        var buffer = new EventBuffer(10);
        var testEvent = new TestEvent("test", 42);
        buffer.Enqueue(testEvent);

        // Act
        var result = buffer.TryDequeue(out var dequeuedEvent);

        // Assert
        Assert.True(result);
        Assert.NotNull(dequeuedEvent);
        Assert.Equal("test", ((TestEvent)dequeuedEvent!).Message);
        Assert.Equal(0, buffer.Count);
    }

    [Fact]
    public void TryDequeue_WithoutEvents_ReturnsFalse()
    {
        // Arrange
        var buffer = new EventBuffer(10);

        // Act
        var result = buffer.TryDequeue(out var dequeuedEvent);

        // Assert
        Assert.False(result);
        Assert.Null(dequeuedEvent);
    }

    [Fact]
    public void DequeueBatch_WithMultipleEvents_ReturnsBatch()
    {
        // Arrange
        var buffer = new EventBuffer(10);
        for (int i = 0; i < 5; i++)
        {
            buffer.Enqueue(new TestEvent($"event{i}", i));
        }

        // Act
        var batch = buffer.DequeueBatch(3);

        // Assert
        Assert.Equal(3, batch.Length);
        Assert.Equal(2, buffer.Count);
    }

    [Fact]
    public void DequeueBatch_WithMoreRequestedThanAvailable_ReturnsAllEvents()
    {
        // Arrange
        var buffer = new EventBuffer(10);
        for (int i = 0; i < 3; i++)
        {
            buffer.Enqueue(new TestEvent($"event{i}", i));
        }

        // Act
        var batch = buffer.DequeueBatch(10);

        // Assert
        Assert.Equal(3, batch.Length);
        Assert.Equal(0, buffer.Count);
    }

    [Fact]
    public void Clear_WithEvents_RemovesAllEvents()
    {
        // Arrange
        var buffer = new EventBuffer(10);
        for (int i = 0; i < 5; i++)
        {
            buffer.Enqueue(new TestEvent($"event{i}", i));
        }

        // Act
        var cleared = buffer.Clear();

        // Assert
        Assert.Equal(5, cleared);
        Assert.Equal(0, buffer.Count);
    }

    [Fact]
    public void PeakSize_TracksMaximumSize()
    {
        // Arrange
        var buffer = new EventBuffer(10);

        // Act
        for (int i = 0; i < 5; i++)
        {
            buffer.Enqueue(new TestEvent($"event{i}", i));
        }
        buffer.TryDequeue(out _);

        // Assert
        Assert.Equal(5, buffer.PeakSize);
        Assert.Equal(4, buffer.Count);
    }

    [Fact]
    public async Task DequeueBatchAsync_WithEvents_ReturnsBatchImmediately()
    {
        // Arrange
        var buffer = new EventBuffer(10);
        for (int i = 0; i < 5; i++)
        {
            buffer.Enqueue(new TestEvent($"event{i}", i));
        }

        // Act
        var batch = await buffer.DequeueBatchAsync(3, TimeSpan.FromSeconds(1));

        // Assert
        Assert.Equal(3, batch.Length);
    }

    [Fact]
    public async Task DequeueBatchAsync_WithoutEvents_WaitsAndReturnsEmpty()
    {
        // Arrange
        var buffer = new EventBuffer(10);

        // Act
        var batch = await buffer.DequeueBatchAsync(3, TimeSpan.FromMilliseconds(100));

        // Assert
        Assert.Empty(batch);
    }
}
