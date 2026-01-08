using Xunit;
using MachineLearning.Visualization.Collection;
using MachineLearning.Visualization.Collection.Configuration;
using MachineLearning.Visualization.Tests;

namespace MLFramework.Visualization.Tests;

/// <summary>
/// Unit tests for AsyncEventCollector
/// </summary>
public class AsyncEventCollectorTests
{
    [Fact]
    public async Task Constructor_WithStorageBackend_CreatesCollector()
    {
        // Arrange
        var storage = new TestStorageBackend();

        // Act
        var collector = new AsyncEventCollector(storage);

        // Assert
        Assert.NotNull(collector);
        Assert.False(collector.IsRunning);
    }

    [Fact]
    public async Task Constructor_WithSubscribers_CreatesCollector()
    {
        // Arrange
        var subscriber = new TestEventSubscriber();

        // Act
        var collector = new AsyncEventCollector(new[] { subscriber });

        // Assert
        Assert.NotNull(collector);
        Assert.Single(collector.GetSubscribers());
    }

    [Fact]
    public async Task Constructor_WithNullSubscribers_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new AsyncEventCollector(null!));
    }

    [Fact]
    public async Task CollectAsync_WithStorageBackend_StoresEvents()
    {
        // Arrange
        var storage = new TestStorageBackend();
        var collector = new AsyncEventCollector(storage);
        collector.Start();

        // Act
        for (int i = 0; i < 5; i++)
        {
            await collector.CollectAsync(new TestEvent($"event{i}", i));
        }
        await collector.FlushAsync();

        // Assert
        Assert.Equal(5, storage.TotalEventsStored);
        collector.Stop();
    }

    [Fact]
    public async Task CollectAsync_WithSubscribers_NotifiesSubscribers()
    {
        // Arrange
        var subscriber = new TestEventSubscriber();
        var collector = new AsyncEventCollector(new[] { subscriber });
        collector.Start();

        // Act
        for (int i = 0; i < 5; i++)
        {
            await collector.CollectAsync(new TestEvent($"event{i}", i));
        }
        await collector.FlushAsync();

        // Assert
        Assert.Equal(5, subscriber.TotalEventsProcessed);
        collector.Stop();
    }

    [Fact]
    public async Task AddSubscriber_WhenRunning_AddsSubscriber()
    {
        // Arrange
        var subscriber1 = new TestEventSubscriber();
        var collector = new AsyncEventCollector(new[] { subscriber1 });
        collector.Start();

        // Act
        var subscriber2 = new TestEventSubscriber();
        collector.AddSubscriber(subscriber2);
        await collector.CollectAsync(new TestEvent("test", 42));
        await collector.FlushAsync();

        // Assert
        Assert.Equal(2, collector.GetSubscribers().Count);
        Assert.Equal(1, subscriber2.TotalEventsProcessed);
        collector.Stop();
    }

    [Fact]
    public async Task AddSubscriber_WithNullSubscriber_ThrowsException()
    {
        // Arrange
        var subscriber = new TestEventSubscriber();
        var collector = new AsyncEventCollector(new[] { subscriber });

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => collector.AddSubscriber(null!));
    }

    [Fact]
    public async Task RemoveSubscriber_RemovesSubscriber()
    {
        // Arrange
        var subscriber = new TestEventSubscriber();
        var collector = new AsyncEventCollector(new[] { subscriber });
        collector.Start();

        // Act
        var removed = collector.RemoveSubscriber(subscriber);
        await collector.CollectAsync(new TestEvent("test", 42));
        await collector.FlushAsync();

        // Assert
        Assert.True(removed);
        Assert.Empty(collector.GetSubscribers());
        Assert.Equal(0, subscriber.TotalEventsProcessed);
        collector.Stop();
    }

    [Fact]
    public async Task RemoveSubscriber_WithNonExistentSubscriber_ReturnsFalse()
    {
        // Arrange
        var subscriber1 = new TestEventSubscriber();
        var collector = new AsyncEventCollector(new[] { subscriber1 });
        var subscriber2 = new TestEventSubscriber();

        // Act
        var removed = collector.RemoveSubscriber(subscriber2);

        // Assert
        Assert.False(removed);
    }

    [Fact]
    public async Task GetSubscribers_ReturnsReadOnlyList()
    {
        // Arrange
        var subscribers = new[]
        {
            new TestEventSubscriber(),
            new TestEventSubscriber(),
            new TestEventSubscriber()
        };
        var collector = new AsyncEventCollector(subscribers);

        // Act
        var retrievedSubscribers = collector.GetSubscribers();

        // Assert
        Assert.Equal(3, retrievedSubscribers.Count);
        Assert.IsAssignableFrom<IReadOnlyList<IEventSubscriber>>(retrievedSubscribers);
    }

    [Fact]
    public async Task MultipleSubscribers_AllReceiveEvents()
    {
        // Arrange
        var subscriber1 = new TestEventSubscriber();
        var subscriber2 = new TestEventSubscriber();
        var collector = new AsyncEventCollector(new[] { subscriber1, subscriber2 });
        collector.Start();

        // Act
        for (int i = 0; i < 3; i++)
        {
            await collector.CollectAsync(new TestEvent($"event{i}", i));
        }
        await collector.FlushAsync();

        // Assert
        Assert.Equal(3, subscriber1.TotalEventsProcessed);
        Assert.Equal(3, subscriber2.TotalEventsProcessed);
        collector.Stop();
    }

    [Fact]
    public async Task TimeBasedFlush_AutoFlushesOnInterval()
    {
        // Arrange
        var storage = new TestStorageBackend();
        var config = new EventCollectorConfig
        {
            Strategy = FlushStrategy.TimeBased,
            FlushInterval = TimeSpan.FromMilliseconds(100),
            BatchSize = 1000
        };
        var collector = new AsyncEventCollector(storage, config);
        collector.Start();

        // Act
        await collector.CollectAsync(new TestEvent("test", 42));

        // Wait for auto-flush
        await Task.Delay(200);

        // Assert
        Assert.Equal(1, storage.TotalEventsStored);
        collector.Stop();
    }

    [Fact]
    public async Task SizeBasedFlush_AutoFlushesWhenBatchFull()
    {
        // Arrange
        var storage = new TestStorageBackend();
        var config = new EventCollectorConfig
        {
            Strategy = FlushStrategy.SizeBased,
            BatchSize = 5
        };
        var collector = new AsyncEventCollector(storage, config);
        collector.Start();

        // Act
        for (int i = 0; i < 10; i++)
        {
            await collector.CollectAsync(new TestEvent($"event{i}", i));
        }

        // Wait for async processing
        await Task.Delay(100);

        // Assert
        Assert.Equal(10, storage.TotalEventsStored);
        collector.Stop();
    }

    [Fact]
    public async Task HybridFlush_FlushesOnSizeAndTime()
    {
        // Arrange
        var storage = new TestStorageBackend();
        var config = new EventCollectorConfig
        {
            Strategy = FlushStrategy.Hybrid,
            BatchSize = 10,
            FlushInterval = TimeSpan.FromMilliseconds(100)
        };
        var collector = new AsyncEventCollector(storage, config);
        collector.Start();

        // Act - Flush on size
        for (int i = 0; i < 10; i++)
        {
            await collector.CollectAsync(new TestEvent($"event{i}", i));
        }
        await Task.Delay(50);

        var statsAfterSizeFlush = storage.TotalEventsStored;

        // Add one more event and wait for time-based flush
        await collector.CollectAsync(new TestEvent("event10", 10));
        await Task.Delay(200);

        // Assert
        Assert.Equal(10, statsAfterSizeFlush);
        Assert.Equal(11, storage.TotalEventsStored);
        collector.Stop();
    }

    [Fact]
    public async Task ManualFlush_OnlyFlushesWhenRequested()
    {
        // Arrange
        var storage = new TestStorageBackend();
        var config = new EventCollectorConfig
        {
            Strategy = FlushStrategy.Manual,
            FlushInterval = TimeSpan.FromMilliseconds(100)
        };
        var collector = new AsyncEventCollector(storage, config);
        collector.Start();

        // Act
        await collector.CollectAsync(new TestEvent("test", 42));

        // Wait longer than flush interval - should NOT auto-flush
        await Task.Delay(200);
        var eventsBeforeManualFlush = storage.TotalEventsStored;

        // Manual flush
        await collector.FlushAsync();

        // Assert
        Assert.Equal(0, eventsBeforeManualFlush);
        Assert.Equal(1, storage.TotalEventsStored);
        collector.Stop();
    }

    [Fact]
    public async Task GetStatistics_ReturnsAccurateStats()
    {
        // Arrange
        var storage = new TestStorageBackend();
        var collector = new AsyncEventCollector(storage);
        collector.Start();

        for (int i = 0; i < 5; i++)
        {
            await collector.CollectAsync(new TestEvent($"event{i}", i));
        }
        await collector.FlushAsync();

        // Act
        var stats = collector.GetStatistics();

        // Assert
        Assert.Equal(5, stats.EventsCollected);
        Assert.Equal(5, stats.EventsProcessed);
        Assert.Equal(0, stats.EventsDropped);
        Assert.Equal(0, stats.PendingEvents);
        Assert.True(stats.UptimeSeconds > 0);
        collector.Stop();
    }

    [Fact]
    public async Task Dispose_StopsCollectorAndCleansUp()
    {
        // Arrange
        var storage = new TestStorageBackend();
        var collector = new AsyncEventCollector(storage);
        collector.Start();
        await collector.CollectAsync(new TestEvent("test", 42));

        // Act
        collector.Dispose();

        // Assert
        Assert.False(collector.IsRunning);
    }
}
