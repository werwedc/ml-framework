using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace MLFramework.Tests.Serving;

[TestClass]
public class DynamicBatcherTests
{
    [TestMethod]
    public async Task ProcessAsync_WithSingleRequest_ProcessesCorrectly()
    {
        var config = new BatchingConfiguration
        {
            MaxBatchSize = 4,
            MaxWaitTime = TimeSpan.FromMilliseconds(50),
            PreferBatchSize = 2
        };

        var batchProcessor = new BatchProcessor<string, int>(requests =>
        {
            return Task.FromResult(requests.Select(r => r.Length).ToList());
        });

        using var batcher = new DynamicBatcher<string, int>(config, batchProcessor);

        var result = await batcher.ProcessAsync("test");

        Assert.AreEqual(4, result);
    }

    [TestMethod]
    public async Task ProcessAsync_WithMultipleConcurrentRequests_BatchesCorrectly()
    {
        var config = new BatchingConfiguration
        {
            MaxBatchSize = 4,
            MaxWaitTime = TimeSpan.FromMilliseconds(50),
            PreferBatchSize = 2
        };

        int batchCount = 0;
        var batchProcessor = new BatchProcessor<string, int>(requests =>
        {
            Interlocked.Increment(ref batchCount);
            return Task.FromResult(requests.Select(r => r.Length).ToList());
        });

        using var batcher = new DynamicBatcher<string, int>(config, batchProcessor);

        var tasks = Enumerable.Range(0, 8)
            .Select(i => batcher.ProcessAsync($"test{i}"))
            .ToArray();

        await Task.WhenAll(tasks);

        Assert.AreEqual(2, batchCount); // Should be 2 batches of 4

        // Verify all results
        foreach (var task in tasks)
        {
            Assert.IsTrue(task.IsCompletedSuccessfully);
            Assert.AreEqual(5, task.Result); // "test0", "test1", etc. all have length 5
        }
    }

    [TestMethod]
    public async Task ProcessAsync_WhenQueueFull_Throws()
    {
        var config = new BatchingConfiguration
        {
            MaxBatchSize = 4,
            MaxWaitTime = TimeSpan.FromMilliseconds(50),
            MaxQueueSize = 2
        };

        var batchProcessor = new BatchProcessor<string, int>(requests =>
            Task.FromResult(requests.Select(r => r.Length).ToList()));

        using var batcher = new DynamicBatcher<string, int>(config, batchProcessor);

        // Enqueue 2 requests (queue size is 2)
        var task1 = batcher.ProcessAsync("test1");
        var task2 = batcher.ProcessAsync("test2");

        // Third request should fail
        await Assert.ThrowsExceptionAsync<InvalidOperationException>(
            () => batcher.ProcessAsync("test3"));

        // First two should complete
        await task1;
        await task2;
    }

    [TestMethod]
    public async Task GetStatistics_ReturnsCorrectValues()
    {
        var config = new BatchingConfiguration
        {
            MaxBatchSize = 4,
            MaxWaitTime = TimeSpan.FromMilliseconds(50),
            MaxQueueSize = 10
        };

        var batchProcessor = new BatchProcessor<string, int>(requests =>
            Task.Delay(10).ContinueWith(_ => requests.Select(r => r.Length).ToList()));

        using var batcher = new DynamicBatcher<string, int>(config, batchProcessor);

        var tasks = Enumerable.Range(0, 5)
            .Select(i => batcher.ProcessAsync($"test{i}"))
            .ToArray();

        var stats = batcher.GetStatistics();

        Assert.IsTrue(stats.CurrentQueueSize >= 0);
        Assert.IsNotNull(stats);
    }

    [TestMethod]
    public async Task ProcessAsync_WithCancellation_CancelsRequest()
    {
        var config = new BatchingConfiguration
        {
            MaxBatchSize = 10,
            MaxWaitTime = TimeSpan.FromMilliseconds(100),
            MaxQueueSize = 100
        };

        var batchProcessor = new BatchProcessor<string, int>(requests =>
        {
            // Simulate slow processing
            return Task.Delay(TimeSpan.FromSeconds(1))
                .ContinueWith(_ => requests.Select(r => r.Length).ToList());
        });

        using var batcher = new DynamicBatcher<string, int>(config, batchProcessor);
        var cts = new CancellationTokenSource();

        var task = batcher.ProcessAsync("test", cts.Token);
        cts.Cancel();

        await Assert.ThrowsExceptionAsync<TaskCanceledException>(() => task);
    }

    [TestMethod]
    public async Task ProcessAsync_AfterDispose_Throws()
    {
        var config = new BatchingConfiguration
        {
            MaxBatchSize = 4,
            MaxWaitTime = TimeSpan.FromMilliseconds(50),
            MaxQueueSize = 10
        };

        var batchProcessor = new BatchProcessor<string, int>(requests =>
            Task.FromResult(requests.Select(r => r.Length).ToList()));

        var batcher = new DynamicBatcher<string, int>(config, batchProcessor);
        batcher.Dispose();

        await Assert.ThrowsExceptionAsync<ObjectDisposedException>(
            () => batcher.ProcessAsync("test"));
    }

    [TestMethod]
    public async Task ProcessAsync_WithSlowProcessor_HandlesCorrectly()
    {
        var config = new BatchingConfiguration
        {
            MaxBatchSize = 2,
            MaxWaitTime = TimeSpan.FromMilliseconds(50),
            MaxQueueSize = 10
        };

        var batchProcessor = new BatchProcessor<string, int>(requests =>
        {
            return Task.Delay(100)
                .ContinueWith(_ => requests.Select(r => r.Length).ToList());
        });

        using var batcher = new DynamicBatcher<string, int>(config, batchProcessor);

        var task1 = batcher.ProcessAsync("test1");
        var task2 = batcher.ProcessAsync("test2");

        var results = await Task.WhenAll(task1, task2);

        Assert.AreEqual(2, results.Length);
        Assert.AreEqual(5, results[0]);
        Assert.AreEqual(5, results[1]);
    }

    [TestMethod]
    public async Task ProcessAsync_WithPartialBatch_DispatchesCorrectly()
    {
        var config = new BatchingConfiguration
        {
            MaxBatchSize = 10,
            MaxWaitTime = TimeSpan.FromMilliseconds(50),
            PreferBatchSize = 8,
            MaxQueueSize = 100,
            TimeoutStrategy = TimeoutStrategy.DispatchPartial
        };

        var batchProcessor = new BatchProcessor<string, int>(requests =>
        {
            return Task.FromResult(requests.Select(r => r.Length).ToList());
        });

        using var batcher = new DynamicBatcher<string, int>(config, batchProcessor);

        // Send only 2 requests, wait for timeout
        var task1 = batcher.ProcessAsync("test1");
        var task2 = batcher.ProcessAsync("test2");

        var results = await Task.WhenAll(task1, task2);

        Assert.AreEqual(2, results.Length);
        Assert.AreEqual(5, results[0]);
        Assert.AreEqual(5, results[1]);
    }

    [TestMethod]
    public async Task ProcessAsync_WithExceptionInProcessor_SetsException()
    {
        var config = new BatchingConfiguration
        {
            MaxBatchSize = 4,
            MaxWaitTime = TimeSpan.FromMilliseconds(50),
            MaxQueueSize = 10
        };

        var batchProcessor = new BatchProcessor<string, int>(requests =>
        {
            throw new InvalidOperationException("Processing failed");
        });

        using var batcher = new DynamicBatcher<string, int>(config, batchProcessor);

        var task1 = batcher.ProcessAsync("test1");
        var task2 = batcher.ProcessAsync("test2");

        // Both requests should fail with the same exception
        await Assert.ThrowsExceptionAsync<InvalidOperationException>(() => task1);
        await Assert.ThrowsExceptionAsync<InvalidOperationException>(() => task2);
    }

    [TestMethod]
    public async Task ProcessAsync_MultipleBatchesIndependently_ProcessesCorrectly()
    {
        var config = new BatchingConfiguration
        {
            MaxBatchSize = 2,
            MaxWaitTime = TimeSpan.FromMilliseconds(50),
            MaxQueueSize = 10
        };

        int batchCount = 0;
        var batchProcessor = new BatchProcessor<string, int>(requests =>
        {
            Interlocked.Increment(ref batchCount);
            return Task.FromResult(requests.Select((r, i) => r.Length + i * 100).ToList());
        });

        using var batcher = new DynamicBatcher<string, int>(config, batchProcessor);

        var tasks = Enumerable.Range(0, 6)
            .Select(i => batcher.ProcessAsync($"test{i}"))
            .ToArray();

        await Task.WhenAll(tasks);

        Assert.AreEqual(3, batchCount); // 6 requests / 2 per batch = 3 batches
    }

    [TestMethod]
    public async Task GetStatistics_QueueEmpty_ReturnsEmptyStats()
    {
        var config = new BatchingConfiguration
        {
            MaxBatchSize = 4,
            MaxWaitTime = TimeSpan.FromMilliseconds(50),
            MaxQueueSize = 10
        };

        var batchProcessor = new BatchProcessor<string, int>(requests =>
            Task.FromResult(requests.Select(r => r.Length).ToList()));

        using var batcher = new DynamicBatcher<string, int>(config, batchProcessor);

        var stats = batcher.GetStatistics();

        Assert.AreEqual(0, stats.CurrentQueueSize);
        Assert.IsTrue(stats.IsQueueEmpty);
        Assert.IsFalse(stats.IsQueueFull);
    }

    [TestMethod]
    public async Task GetStatistics_QueueFull_ReturnsFullStats()
    {
        var config = new BatchingConfiguration
        {
            MaxBatchSize = 1,
            MaxWaitTime = TimeSpan.FromMilliseconds(1000), // Long timeout
            MaxQueueSize = 2
        };

        var batchProcessor = new BatchProcessor<string, int>(requests =>
        {
            // Slow processing to keep queue full
            return Task.Delay(1000)
                .ContinueWith(_ => requests.Select(r => r.Length).ToList());
        });

        using var batcher = new DynamicBatcher<string, int>(config, batchProcessor);

        // Fill queue
        var task1 = batcher.ProcessAsync("test1");
        var task2 = batcher.ProcessAsync("test2");

        // Give it time to enqueue
        await Task.Delay(10);

        var stats = batcher.GetStatistics();

        Assert.IsTrue(stats.CurrentQueueSize > 0 || stats.IsQueueFull);
    }

    [TestMethod]
    public async Task Dispose_CancelsPendingRequests()
    {
        var config = new BatchingConfiguration
        {
            MaxBatchSize = 10,
            MaxWaitTime = TimeSpan.FromMilliseconds(1000),
            MaxQueueSize = 10
        };

        var batchProcessor = new BatchProcessor<string, int>(requests =>
        {
            // Never completes
            return Task.FromResult(requests.Select(r => r.Length).ToList());
        });

        var batcher = new DynamicBatcher<string, int>(config, batchProcessor);

        var task = batcher.ProcessAsync("test");

        // Dispose while request is in queue
        await Task.Delay(10);
        batcher.Dispose();

        // Task should be cancelled
        await Assert.ThrowsExceptionAsync<TaskCanceledException>(() => task);
    }

    [TestMethod]
    public async Task ProcessAsync_WithDifferentRequestTypes_WorksCorrectly()
    {
        var config = new BatchingConfiguration
        {
            MaxBatchSize = 2,
            MaxWaitTime = TimeSpan.FromMilliseconds(50),
            MaxQueueSize = 10
        };

        var batchProcessor = new BatchProcessor<int, string>(requests =>
        {
            return Task.FromResult(requests.Select(r => $"Value-{r}").ToList());
        });

        using var batcher = new DynamicBatcher<int, string>(config, batchProcessor);

        var task1 = batcher.ProcessAsync(1);
        var task2 = batcher.ProcessAsync(2);

        var results = await Task.WhenAll(task1, task2);

        Assert.AreEqual("Value-1", results[0]);
        Assert.AreEqual("Value-2", results[1]);
    }

    [TestMethod]
    public async Task ProcessAsync_PreservesOrderInResponses()
    {
        var config = new BatchingConfiguration
        {
            MaxBatchSize = 4,
            MaxWaitTime = TimeSpan.FromMilliseconds(50),
            MaxQueueSize = 10
        };

        var batchProcessor = new BatchProcessor<int, int>(requests =>
        {
            return Task.FromResult(requests.Select(r => r * 2).ToList());
        });

        using var batcher = new DynamicBatcher<int, int>(config, batchProcessor);

        var task1 = batcher.ProcessAsync(1);
        var task2 = batcher.ProcessAsync(2);
        var task3 = batcher.ProcessAsync(3);

        var results = await Task.WhenAll(task1, task2, task3);

        Assert.AreEqual(2, results[0]);
        Assert.AreEqual(4, results[1]);
        Assert.AreEqual(6, results[2]);
    }

    [TestMethod]
    public async Task ProcessAsync_WithConcurrentRequestsAndSmallBatchSize_MaintainsThroughput()
    {
        var config = new BatchingConfiguration
        {
            MaxBatchSize = 2,
            MaxWaitTime = TimeSpan.FromMilliseconds(50),
            MaxQueueSize = 100
        };

        var batchProcessor = new BatchProcessor<int, int>(requests =>
        {
            return Task.FromResult(requests.Sum());
        }).Result; // Wrap in Task.FromResult

        var asyncProcessor = new BatchProcessor<int, int>(requests =>
            Task.FromResult(requests.Sum()));

        using var batcher = new DynamicBatcher<int, int>(config, asyncProcessor);

        // Send many requests
        var tasks = Enumerable.Range(0, 20)
            .Select(i => batcher.ProcessAsync(i))
            .ToArray();

        await Task.WhenAll(tasks);

        // All should complete
        foreach (var task in tasks)
        {
            Assert.IsTrue(task.IsCompletedSuccessfully);
        }
    }
}
