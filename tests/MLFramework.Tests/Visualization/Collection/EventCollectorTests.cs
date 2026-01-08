using Xunit;
using MachineLearning.Visualization.Collection;
using MachineLearning.Visualization.Collection.Configuration;
using MachineLearning.Visualization.Tests;

namespace MLFramework.Visualization.Tests;

/// <summary>
/// Unit tests for EventCollector
/// </summary>
public class EventCollectorTests
{
    [Fact]
    public async Task Collect_WhenNotStarted_ThrowsException()
    {
        // Arrange
        var storage = new TestStorageBackend();
        var collector = new TestEventCollector(storage);
        var testEvent = new TestEvent("test", 42);

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => collector.Collect(testEvent));
    }

    [Fact]
    public async Task CollectAsync_WhenNotStarted_ThrowsException()
    {
        // Arrange
        var storage = new TestStorageBackend();
        var collector = new TestEventCollector(storage);
        var testEvent = new TestEvent("test", 42);

        // Act & Assert
        await Assert.ThrowsAsync<InvalidOperationException>(() => collector.CollectAsync(testEvent));
    }

    [Fact]
    public async Task Collect_WhenStarted_AcceptsEvents()
    {
        // Arrange
        var storage = new TestStorageBackend();
        var collector = new TestEventCollector(storage);
        var testEvent = new TestEvent("test", 42);

        // Act
        collector.Start();
        collector.Collect(testEvent);

        // Assert
        Assert.Equal(1, collector.PendingEventCount);
        collector.Stop();
    }

    [Fact]
    public async Task CollectAsync_WhenStarted_AcceptsEvents()
    {
        // Arrange
        var storage = new TestStorageBackend();
        var collector = new TestEventCollector(storage);
        var testEvent = new TestEvent("test", 42);

        // Act
        collector.Start();
        await collector.CollectAsync(testEvent);

        // Assert
        Assert.Equal(1, collector.PendingEventCount);
        collector.Stop();
    }

    [Fact]
    public async Task Flush_WithPendingEvents_ProcessesEvents()
    {
        // Arrange
        var storage = new TestStorageBackend();
        var collector = new TestEventCollector(storage);
        collector.Start();

        for (int i = 0; i < 5; i++)
        {
            collector.Collect(new TestEvent($"event{i}", i));
        }

        // Act
        collector.Flush();

        // Assert
        Assert.Equal(0, collector.PendingEventCount);
        Assert.Equal(5, storage.TotalEventsStored);
        collector.Stop();
    }

    [Fact]
    public async Task FlushAsync_WithPendingEvents_ProcessesEvents()
    {
        // Arrange
        var storage = new TestStorageBackend();
        var collector = new TestEventCollector(storage);
        collector.Start();

        for (int i = 0; i < 5; i++)
        {
            await collector.CollectAsync(new TestEvent($"event{i}", i));
        }

        // Act
        await collector.FlushAsync();

        // Assert
        Assert.Equal(0, collector.PendingEventCount);
        Assert.Equal(5, storage.TotalEventsStored);
        collector.Stop();
    }

    [Fact]
    public async Task GetStatistics_ReturnsCorrectStatistics()
    {
        // Arrange
        var storage = new TestStorageBackend();
        var config = new EventCollectorConfig { BufferCapacity = 10, BatchSize = 5 };
        var collector = new TestEventCollector(storage, config);
        collector.Start();

        for (int i = 0; i < 7; i++)
        {
            await collector.CollectAsync(new TestEvent($"event{i}", i));
        }

        // Act
        var stats = collector.GetStatistics();

        // Assert
        Assert.Equal(7, stats.EventsCollected);
        Assert.Equal(7, stats.PendingEvents);
        Assert.Equal(7, stats.PeakBufferSize);
        collector.Stop();
    }

    [Fact]
    public async Task Start_AlreadyStarted_DoesNothing()
    {
        // Arrange
        var storage = new TestStorageBackend();
        var collector = new TestEventCollector(storage);
        collector.Start();

        // Act
        collector.Start();

        // Assert
        Assert.True(collector.IsRunning);
        collector.Stop();
    }

    [Fact]
    public async Task Stop_WhenRunning_StopsProcessing()
    {
        // Arrange
        var storage = new TestStorageBackend();
        var collector = new TestEventCollector(storage);
        collector.Start();
        collector.Collect(new TestEvent("test", 42));

        // Act
        collector.Stop();

        // Assert
        Assert.False(collector.IsRunning);
        Assert.Equal(0, collector.PendingEventCount);
    }

    [Fact]
    public async Task Stop_NotRunning_DoesNothing()
    {
        // Arrange
        var storage = new TestStorageBackend();
        var collector = new TestEventCollector(storage);

        // Act
        collector.Stop();

        // Assert
        Assert.False(collector.IsRunning);
    }

    [Fact]
    public async Task Backpressure_WhenEnabledAndFull_DropsOldestEvents()
    {
        // Arrange
        var storage = new TestStorageBackend();
        var config = new EventCollectorConfig
        {
            BufferCapacity = 2,
            EnableBackpressure = true,
            BackpressureAction = BackpressureAction.DropOldest
        };
        var collector = new TestEventCollector(storage, config);
        collector.Start();

        // Act
        collector.Collect(new TestEvent("event1", 1));
        collector.Collect(new TestEvent("event2", 2));
        collector.Collect(new TestEvent("event3", 3));

        // Assert
        var stats = collector.GetStatistics();
        Assert.True(stats.EventsDropped > 0);
        collector.Stop();
    }

    [Fact]
    public async Task Backpressure_WhenDisabled_AcceptsEvents()
    {
        // Arrange
        var storage = new TestStorageBackend();
        var config = new EventCollectorConfig
        {
            BufferCapacity = 2,
            EnableBackpressure = false
        };
        var collector = new TestEventCollector(storage, config);
        collector.Start();

        // Act
        collector.Collect(new TestEvent("event1", 1));
        collector.Collect(new TestEvent("event2", 2));
        collector.Collect(new TestEvent("event3", 3));

        // Assert
        var stats = collector.GetStatistics();
        Assert.Equal(3, stats.EventsCollected);
        collector.Stop();
    }

    /// <summary>
    /// Test implementation of EventCollector for testing purposes
    /// </summary>
    private class TestEventCollector : EventCollector
    {
        public TestEventCollector(TestStorageBackend storage, EventCollectorConfig? config = null)
            : base(storage, config)
        {
        }

        protected override Task ProcessEventsAsync()
        {
            var events = Buffer.DequeueBatch(Config.BatchSize);

            if (!events.IsDefaultOrEmpty)
            {
                Interlocked.Add(ref _eventsProcessed, events.Length);

                if (_storageBackend != null)
                {
                    return _storageBackend.StoreBatchAsync(events, _cancellationTokenSource.Token);
                }
            }

            return Task.CompletedTask;
        }
    }
}
