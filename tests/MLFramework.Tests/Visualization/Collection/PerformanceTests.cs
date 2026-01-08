using Xunit;
using System.Diagnostics;
using MachineLearning.Visualization.Collection;
using MachineLearning.Visualization.Collection.Configuration;
using MachineLearning.Visualization.Tests;

namespace MLFramework.Visualization.Tests;

/// <summary>
/// Performance tests for event collection
/// </summary>
public class EventCollectorPerformanceTests
{
    [Fact(Skip = "Performance test - run manually")]
    public async Task EventBuffer_HighThroughput_LessThan1msPerEvent()
    {
        // Arrange
        var buffer = new EventBuffer(10000);
        var stopwatch = new Stopwatch();
        var eventCount = 10000;

        // Act
        stopwatch.Start();
        for (int i = 0; i < eventCount; i++)
        {
            buffer.Enqueue(new TestEvent($"event{i}", i));
        }
        stopwatch.Stop();

        // Assert
        var avgTimePerEvent = stopwatch.ElapsedMilliseconds / (double)eventCount;
        Console.WriteLine($"Average time per event: {avgTimePerEvent}ms");
        Assert.True(avgTimePerEvent < 1.0, $"Average time per event {avgTimePerEvent}ms exceeds 1ms threshold");
    }

    [Fact(Skip = "Performance test - run manually")]
    public async Task AsyncEventCollector_Collect10000EventsPerSecond_LessThan1msOverhead()
    {
        // Arrange
        var storage = new TestStorageBackend();
        var config = new EventCollectorConfig
        {
            BufferCapacity = 10000,
            BatchSize = 1000,
            Strategy = FlushStrategy.SizeBased
        };
        var collector = new AsyncEventCollector(storage, config);
        collector.Start();

        var stopwatch = new Stopwatch();
        var eventCount = 10000;

        // Act
        stopwatch.Start();
        for (int i = 0; i < eventCount; i++)
        {
            await collector.CollectAsync(new TestEvent($"event{i}", i));
        }
        stopwatch.Stop();

        // Wait for processing to complete
        await collector.FlushAsync();

        // Assert
        var avgTimePerEvent = stopwatch.ElapsedMilliseconds / (double)eventCount;
        var throughput = eventCount / stopwatch.Elapsed.TotalSeconds;

        Console.WriteLine($"Average time per event: {avgTimePerEvent}ms");
        Console.WriteLine($"Throughput: {throughput:F0} events/second");

        Assert.True(avgTimePerEvent < 1.0, $"Average time per event {avgTimePerEvent}ms exceeds 1ms threshold");
        Assert.True(throughput >= 10000, $"Throughput {throughput} events/second below 10000 threshold");

        collector.Stop();
    }

    [Fact(Skip = "Performance test - run manually")]
    public async Task AsyncEventCollector_GracefulShutdown_CompletesIn100ms()
    {
        // Arrange
        var storage = new TestStorageBackend();
        var config = new EventCollectorConfig
        {
            BufferCapacity = 1000,
            BatchSize = 100,
            Strategy = FlushStrategy.Manual
        };
        var collector = new AsyncEventCollector(storage, config);
        collector.Start();

        // Add some events
        for (int i = 0; i < 500; i++)
        {
            await collector.CollectAsync(new TestEvent($"event{i}", i));
        }

        // Act
        var stopwatch = new Stopwatch();
        stopwatch.Start();
        collector.Stop();
        stopwatch.Stop();

        // Assert
        Console.WriteLine($"Shutdown time: {stopwatch.ElapsedMilliseconds}ms");
        Assert.True(stopwatch.ElapsedMilliseconds < 100, $"Shutdown time {stopwatch.ElapsedMilliseconds}ms exceeds 100ms threshold");
    }

    [Fact(Skip = "Performance test - run manually")]
    public async Task BatchProcessing_EfficientBatching_ReducesStorageCalls()
    {
        // Arrange
        var storage = new TestStorageBackend();
        var config = new EventCollectorConfig
        {
            BufferCapacity = 1000,
            BatchSize = 100,
            Strategy = FlushStrategy.SizeBased
        };
        var collector = new AsyncEventCollector(storage, config);
        collector.Start();

        var eventCount = 1000;

        // Act
        for (int i = 0; i < eventCount; i++)
        {
            await collector.CollectAsync(new TestEvent($"event{i}", i));
        }

        // Wait for all batches to be processed
        await Task.Delay(500);

        // Assert
        var storageCalls = storage.TotalBatchesStored;
        Console.WriteLine($"Storage calls for {eventCount} events: {storageCalls}");

        // With batch size 100 and 1000 events, we should have approximately 10 storage calls
        Assert.True(storageCalls <= 10, $"Storage calls {storageCalls} exceed expected 10");
        Assert.Equal(eventCount, storage.TotalEventsStored);

        collector.Stop();
    }

    [Fact(Skip = "Performance test - run manually")]
    public async Task ConcurrentAccess_ThreadSafe_BlocksCorrectly()
    {
        // Arrange
        var storage = new TestStorageBackend();
        var config = new EventCollectorConfig
        {
            BufferCapacity = 10000,
            BatchSize = 500,
            Strategy = FlushStrategy.SizeBased
        };
        var collector = new AsyncEventCollector(storage, config);
        collector.Start();

        var threadCount = 10;
        var eventsPerThread = 1000;
        var tasks = new List<Task>();
        var stopwatch = new Stopwatch();

        // Act
        stopwatch.Start();

        for (int t = 0; t < threadCount; t++)
        {
            var threadId = t;
            tasks.Add(Task.Run(async () =>
            {
                for (int i = 0; i < eventsPerThread; i++)
                {
                    await collector.CollectAsync(new TestEvent($"thread{threadId}_event{i}", i));
                }
            }));
        }

        await Task.WhenAll(tasks);
        await collector.FlushAsync();

        stopwatch.Stop();

        // Assert
        var totalEvents = threadCount * eventsPerThread;
        var avgTimePerEvent = stopwatch.ElapsedMilliseconds / (double)totalEvents;

        Console.WriteLine($"Total events: {totalEvents}");
        Console.WriteLine($"Average time per event: {avgTimePerEvent}ms");
        Console.WriteLine($"Events processed: {storage.TotalEventsStored}");

        Assert.Equal(totalEvents, storage.TotalEventsStored);
        Assert.True(avgTimePerEvent < 1.0, $"Average time per event {avgTimePerEvent}ms exceeds 1ms threshold");

        collector.Stop();
    }

    [Fact(Skip = "Performance test - run manually")]
    public async Task Backpressure_PreventsMemoryBloat_DropsEventsWhenFull()
    {
        // Arrange
        var storage = new TestStorageBackend();
        storage.ProcessingDelay = TimeSpan.FromMilliseconds(10); // Slow processing

        var config = new EventCollectorConfig
        {
            BufferCapacity = 100,
            BatchSize = 10,
            Strategy = FlushStrategy.Manual,
            EnableBackpressure = true,
            BackpressureAction = BackpressureAction.DropOldest
        };
        var collector = new AsyncEventCollector(storage, config);
        collector.Start();

        // Act - Flood the collector with events
        var stopwatch = new Stopwatch();
        stopwatch.Start();

        for (int i = 0; i < 10000; i++)
        {
            collector.Collect(new TestEvent($"event{i}", i));
        }

        stopwatch.Stop();

        var stats = collector.GetStatistics();

        // Assert
        Console.WriteLine($"Time to enqueue 10000 events: {stopwatch.ElapsedMilliseconds}ms");
        Console.WriteLine($"Events dropped: {stats.EventsDropped}");
        Console.WriteLine($"Events collected: {stats.EventsCollected}");
        Console.WriteLine($"Current buffer size: {stats.CurrentBufferSize}");

        // With backpressure enabled, we should have dropped events
        Assert.True(stats.EventsDropped > 0, "No events were dropped despite backpressure being enabled");
        Assert.True(stats.CurrentBufferSize <= config.BufferCapacity, $"Buffer size {stats.CurrentBufferSize} exceeds capacity {config.BufferCapacity}");

        collector.Stop();
    }

    [Fact(Skip = "Performance test - run manually")]
    public async Task ObjectPool_ReuseEvents_ReducesAllocations()
    {
        // Arrange
        var pool = new ObjectPool<TestEvent>(100);
        var stopwatch = new Stopwatch();

        // Act - Warm up the pool
        for (int i = 0; i < 100; i++)
        {
            var evt = pool.Rent();
            pool.Return(evt);
        }

        // Measure reuse performance
        stopwatch.Start();
        for (int i = 0; i < 10000; i++)
        {
            var evt = pool.Rent();
            evt.Message = $"event{i}";
            evt.Value = i;
            pool.Return(evt);
        }
        stopwatch.Stop();

        // Assert
        var reuseRate = pool.GetReuseRate();
        Console.WriteLine($"Reuse rate: {reuseRate:F2}%");
        Console.WriteLine($"Time for 10000 operations: {stopwatch.ElapsedMilliseconds}ms");

        Assert.True(reuseRate > 90, $"Reuse rate {reuseRate}% below 90% threshold");
    }

    [Fact(Skip = "Performance test - run manually")]
    public async Task MultipleSubscribers_ScaleLinearly_ProcessesEfficiently()
    {
        // Arrange
        var subscribers = new List<TestEventSubscriber>();
        for (int i = 0; i < 10; i++)
        {
            subscribers.Add(new TestEventSubscriber());
        }

        var collector = new AsyncEventCollector(subscribers);
        collector.Start();

        var eventCount = 1000;
        var stopwatch = new Stopwatch();

        // Act
        stopwatch.Start();
        for (int i = 0; i < eventCount; i++)
        {
            await collector.CollectAsync(new TestEvent($"event{i}", i));
        }
        await collector.FlushAsync();
        stopwatch.Stop();

        // Assert
        var avgTimePerEvent = stopwatch.ElapsedMilliseconds / (double)eventCount;

        Console.WriteLine($"Average time per event with {subscribers.Count} subscribers: {avgTimePerEvent}ms");
        Console.WriteLine($"Total time: {stopwatch.ElapsedMilliseconds}ms");

        // Verify all subscribers received all events
        foreach (var subscriber in subscribers)
        {
            Assert.Equal(eventCount, subscriber.TotalEventsProcessed);
        }

        collector.Stop();
    }
}
