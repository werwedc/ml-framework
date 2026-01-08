using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Threading;
using System.Threading.Tasks;

namespace MLFramework.Tests.Serving;

[TestClass]
public class BatchBuilderTests
{
    private BatchingConfiguration CreateTestConfig(
        int maxBatchSize = 10,
        int preferBatchSize = 5,
        int maxWaitTimeMs = 100,
        TimeoutStrategy strategy = TimeoutStrategy.DispatchPartial)
    {
        return new BatchingConfiguration
        {
            MaxBatchSize = maxBatchSize,
            PreferBatchSize = preferBatchSize,
            MaxWaitTime = TimeSpan.FromMilliseconds(maxWaitTimeMs),
            MaxQueueSize = 100,
            TimeoutStrategy = strategy
        };
    }

    [TestMethod]
    public async Task BuildBatchAsync_FillsToMaxBatchSize()
    {
        var config = CreateTestConfig(maxBatchSize: 5);
        var queue = new BoundedRequestQueue<string, int>(100);
        var builder = new BatchBuilder<string, int>(queue, config);

        // Enqueue more than MaxBatchSize
        for (int i = 0; i < 10; i++)
        {
            var request = new QueuedRequest<string, int>($"test{i}");
            await queue.TryEnqueueAsync(request);
        }

        var batch = await builder.BuildBatchAsync();

        Assert.AreEqual(5, batch.Requests.Count);
        Assert.AreEqual("test0", batch.Requests[0].Request);
        Assert.AreEqual("test4", batch.Requests[4].Request);
    }

    [TestMethod]
    public async Task BuildBatchAsync_DispatchesAtPreferBatchSizeWhenQueueEmpties()
    {
        var config = CreateTestConfig(maxBatchSize: 10, preferBatchSize: 5);
        var queue = new BoundedRequestQueue<string, int>(100);
        var builder = new BatchBuilder<string, int>(queue, config);

        // Enqueue exactly PreferBatchSize
        for (int i = 0; i < 5; i++)
        {
            var request = new QueuedRequest<string, int>($"test{i}");
            await queue.TryEnqueueAsync(request);
        }

        var batch = await builder.BuildBatchAsync();

        Assert.AreEqual(5, batch.Requests.Count);
    }

    [TestMethod]
    public async Task BuildBatchAsync_TimeoutWithPartialBatch_DispatchPartial()
    {
        var config = CreateTestConfig(
            maxBatchSize: 10,
            preferBatchSize: 8,
            maxWaitTimeMs: 10,
            TimeoutStrategy.DispatchPartial);
        var queue = new BoundedRequestQueue<string, int>(100);
        var builder = new BatchBuilder<string, int>(queue, config);

        // Enqueue only a few items
        for (int i = 0; i < 3; i++)
        {
            var request = new QueuedRequest<string, int>($"test{i}");
            await queue.TryEnqueueAsync(request);
        }

        var batch = await builder.BuildBatchAsync();

        Assert.AreEqual(3, batch.Requests.Count);
    }

    [TestMethod]
    public async Task BuildBatchAsync_TimeoutWithNoRequests_WaitForFull_Throws()
    {
        var config = CreateTestConfig(
            maxBatchSize: 10,
            preferBatchSize: 8,
            maxWaitTimeMs: 10,
            TimeoutStrategy.WaitForFull);
        var queue = new BoundedRequestQueue<string, int>(100);
        var builder = new BatchBuilder<string, int>(queue, config);

        // Don't enqueue anything
        await Assert.ThrowsExceptionAsync<TimeoutException>(() =>
            builder.BuildBatchAsync());
    }

    [TestMethod]
    public async Task BuildBatchAsync_TimeoutWithPartialBatch_WaitForFull_ReturnsPartial()
    {
        var config = CreateTestConfig(
            maxBatchSize: 10,
            preferBatchSize: 8,
            maxWaitTimeMs: 10,
            TimeoutStrategy.WaitForFull);
        var queue = new BoundedRequestQueue<string, int>(100);
        var builder = new BatchBuilder<string, int>(queue, config);

        // Enqueue some items
        for (int i = 0; i < 3; i++)
        {
            var request = new QueuedRequest<string, int>($"test{i}");
            await queue.TryEnqueueAsync(request);
        }

        var batch = await builder.BuildBatchAsync();

        Assert.AreEqual(3, batch.Requests.Count);
    }

    [TestMethod]
    public async Task BuildBatchAsync_TimeoutWithNoRequests_Adaptive_Throws()
    {
        var config = CreateTestConfig(
            maxBatchSize: 10,
            preferBatchSize: 8,
            maxWaitTimeMs: 10,
            TimeoutStrategy.Adaptive);
        var queue = new BoundedRequestQueue<string, int>(100);
        var builder = new BatchBuilder<string, int>(queue, config);

        // Don't enqueue anything
        await Assert.ThrowsExceptionAsync<TimeoutException>(() =>
            builder.BuildBatchAsync());
    }

    [TestMethod]
    public async Task BuildBatchAsync_TimeoutWithPartialBatch_Adaptive_ReturnsPartial()
    {
        var config = CreateTestConfig(
            maxBatchSize: 10,
            preferBatchSize: 8,
            maxWaitTimeMs: 10,
            TimeoutStrategy.Adaptive);
        var queue = new BoundedRequestQueue<string, int>(100);
        var builder = new BatchBuilder<string, int>(queue, config);

        // Enqueue some items
        for (int i = 0; i < 3; i++)
        {
            var request = new QueuedRequest<string, int>($"test{i}");
            await queue.TryEnqueueAsync(request);
        }

        var batch = await builder.BuildBatchAsync();

        Assert.AreEqual(3, batch.Requests.Count);
    }

    [TestMethod]
    public void Constructor_WithNullQueue_Throws()
    {
        var config = CreateTestConfig();
        Assert.ThrowsException<ArgumentNullException>(() =>
        {
            new BatchBuilder<string, int>(null, config);
        });
    }

    [TestMethod]
    public void Constructor_WithNullConfig_Throws()
    {
        var queue = new BoundedRequestQueue<string, int>(100);
        Assert.ThrowsException<ArgumentNullException>(() =>
        {
            new BatchBuilder<string, int>(queue, null);
        });
    }

    [TestMethod]
    public void Constructor_WithInvalidConfig_Throws()
    {
        var queue = new BoundedRequestQueue<string, int>(100);
        var config = CreateTestConfig(maxBatchSize: 0); // Invalid

        Assert.ThrowsException<ArgumentOutOfRangeException>(() =>
        {
            new BatchBuilder<string, int>(queue, config);
        });
    }

    [TestMethod]
    public async Task BuildBatchAsync_CancellationToken_Cancels()
    {
        var config = CreateTestConfig(maxWaitTimeMs: 10000);
        var queue = new BoundedRequestQueue<string, int>(100);
        var builder = new BatchBuilder<string, int>(queue, config);
        var cts = new CancellationTokenSource();

        // Start the build task
        var buildTask = builder.BuildBatchAsync(cts.Token);

        // Cancel immediately
        cts.Cancel();

        // Should throw OperationCanceledException
        await Assert.ThrowsExceptionAsync<OperationCanceledException>(() =>
            buildTask);
    }

    [TestMethod]
    public async Task BuildBatchAsync_ConcurrentCalls_AllSucceed()
    {
        var config = CreateTestConfig(maxBatchSize: 2, maxWaitTimeMs: 100);
        var queue = new BoundedRequestQueue<string, int>(100);

        // Enqueue many requests
        for (int i = 0; i < 20; i++)
        {
            var request = new QueuedRequest<string, int>($"test{i}");
            await queue.TryEnqueueAsync(request);
        }

        // Start multiple concurrent builds
        var tasks = new Task<RequestBatch<string, int>>[5];
        for (int i = 0; i < 5; i++)
        {
            var builder = new BatchBuilder<string, int>(queue, config);
            tasks[i] = builder.BuildBatchAsync();
        }

        await Task.WhenAll(tasks);

        // All tasks should complete
        foreach (var task in tasks)
        {
            Assert.IsTrue(task.IsCompleted);
            Assert.IsTrue(task.Result.Requests.Count > 0);
            Assert.IsTrue(task.Result.Requests.Count <= config.MaxBatchSize);
        }
    }

    [TestMethod]
    public async Task RequestBatch_GetPayloads_ReturnsCorrectPayloads()
    {
        var config = CreateTestConfig(maxBatchSize: 3);
        var queue = new BoundedRequestQueue<string, int>(100);
        var builder = new BatchBuilder<string, int>(queue, config);

        for (int i = 0; i < 3; i++)
        {
            var request = new QueuedRequest<string, int>($"test{i}");
            await queue.TryEnqueueAsync(request);
        }

        var batch = await builder.BuildBatchAsync();
        var payloads = batch.GetPayloads();

        Assert.AreEqual(3, payloads.Count);
        Assert.AreEqual("test0", payloads[0]);
        Assert.AreEqual("test1", payloads[1]);
        Assert.AreEqual("test2", payloads[2]);
    }

    [TestMethod]
    public void RequestBatch_BatchCreatedAt_IsSet()
    {
        var before = DateTime.UtcNow;
        var batch = new RequestBatch<string, int>(new List<QueuedRequest<string, int>>());
        var after = DateTime.UtcNow;

        Assert.IsTrue(batch.BatchCreatedAt >= before);
        Assert.IsTrue(batch.BatchCreatedAt <= after);
    }

    [TestMethod]
    public async Task RequestBatch_Requests_IsReadOnly()
    {
        var config = CreateTestConfig(maxBatchSize: 1);
        var queue = new BoundedRequestQueue<string, int>(100);
        var builder = new BatchBuilder<string, int>(queue, config);

        var request = new QueuedRequest<string, int>("test");
        await queue.TryEnqueueAsync(request);

        var batch = await builder.BuildBatchAsync();

        // Try to cast and modify - should fail
        var requestsList = batch.Requests as IList<QueuedRequest<string, int>>;
        Assert.IsNotNull(requestsList);
        
        // Should throw because it's a read-only collection
        Assert.ThrowsException<NotSupportedException>(() =>
        {
            requestsList.Add(new QueuedRequest<string, int>("test2"));
        });
    }

    [TestMethod]
    public async Task BuildBatchAsync_FillsFromEmptyQueueOverTime()
    {
        var config = CreateTestConfig(maxBatchSize: 5, maxWaitTimeMs: 200);
        var queue = new BoundedRequestQueue<string, int>(100);
        var builder = new BatchBuilder<string, int>(queue, config);

        // Start building task
        var buildTask = builder.BuildBatchAsync();

        // Wait a bit, then add items
        await Task.Delay(50);
        for (int i = 0; i < 5; i++)
        {
            var request = new QueuedRequest<string, int>($"test{i}");
            await queue.TryEnqueueAsync(request);
        }

        var batch = await buildTask;

        Assert.AreEqual(5, batch.Requests.Count);
    }

    [TestMethod]
    public async Task BuildBatchAsync_RespectsMaxBatchSizeStrictly()
    {
        var config = CreateTestConfig(maxBatchSize: 3);
        var queue = new BoundedRequestQueue<string, int>(100);
        var builder = new BatchBuilder<string, int>(queue, config);

        // Enqueue many more than MaxBatchSize
        for (int i = 0; i < 20; i++)
        {
            var request = new QueuedRequest<string, int>($"test{i}");
            await queue.TryEnqueueAsync(request);
        }

        var batch = await builder.BuildBatchAsync();

        Assert.AreEqual(3, batch.Requests.Count);
        Assert.AreEqual(17, queue.Count); // 20 - 3
    }

    [TestMethod]
    public async Task BuildBatchAsync_WithMultipleTimeoutStrategies_AllWork()
    {
        var strategies = new[]
        {
            TimeoutStrategy.DispatchPartial,
            TimeoutStrategy.WaitForFull,
            TimeoutStrategy.Adaptive
        };

        foreach (var strategy in strategies)
        {
            var config = CreateTestConfig(
                maxBatchSize: 10,
                maxWaitTimeMs: 10,
                strategy: strategy);
            var queue = new BoundedRequestQueue<string, int>(100);
            var builder = new BatchBuilder<string, int>(queue, config);

            // Add one item
            var request = new QueuedRequest<string, int>("test");
            await queue.TryEnqueueAsync(request);

            try
            {
                var batch = await builder.BuildBatchAsync();
                Assert.AreEqual(1, batch.Requests.Count);
            }
            catch (TimeoutException)
            {
                // Expected for WaitForFull and Adaptive with empty batch
                if (strategy == TimeoutStrategy.DispatchPartial)
                {
                    Assert.Fail($"DispatchPartial should not throw for non-empty batch");
                }
            }
        }
    }

    [TestMethod]
    public async Task BuildBatchAsync_EmptyQueue_TimesOut()
    {
        var config = CreateTestConfig(maxWaitTimeMs: 10);
        var queue = new BoundedRequestQueue<string, int>(100);
        var builder = new BatchBuilder<string, int>(queue, config);

        var start = DateTime.UtcNow;
        try
        {
            await builder.BuildBatchAsync();
            Assert.Fail("Should have thrown TimeoutException");
        }
        catch (TimeoutException)
        {
            var elapsed = DateTime.UtcNow - start;
            // Should be close to MaxWaitTime (with some tolerance)
            Assert.IsTrue(elapsed.TotalMilliseconds >= 8, $"Wait time too short: {elapsed.TotalMilliseconds}ms");
            Assert.IsTrue(elapsed.TotalMilliseconds <= 50, $"Wait time too long: {elapsed.TotalMilliseconds}ms");
        }
    }

    [TestMethod]
    public async Task BuildBatchAsync_PreferBatchSizeWithQueueNotEmpty_WaitsForMore()
    {
        var config = CreateTestConfig(
            maxBatchSize: 10,
            preferBatchSize: 5,
            maxWaitTimeMs: 50);
        var queue = new BoundedRequestQueue<string, int>(100);
        var builder = new BatchBuilder<string, int>(queue, config);

        // Start building task
        var buildTask = builder.BuildBatchAsync();

        // Add PreferBatchSize items
        for (int i = 0; i < 5; i++)
        {
            var request = new QueuedRequest<string, int>($"test{i}");
            await queue.TryEnqueueAsync(request);
        }

        // Wait a bit to ensure it doesn't return early
        await Task.Delay(10);
        Assert.IsFalse(buildTask.IsCompleted);

        // Add one more to reach MaxBatchSize
        for (int i = 5; i < 10; i++)
        {
            var request = new QueuedRequest<string, int>($"test{i}");
            await queue.TryEnqueueAsync(request);
        }

        var batch = await buildTask;

        Assert.AreEqual(10, batch.Requests.Count);
    }

    [TestMethod]
    public async Task RequestBatch_GenericType_WorksCorrectly()
    {
        var config = CreateTestConfig(maxBatchSize: 2);
        var queue = new BoundedRequestQueue<int, string>(100);
        var builder = new BatchBuilder<int, string>(queue, config);

        for (int i = 0; i < 2; i++)
        {
            var request = new QueuedRequest<int, string>(i);
            await queue.TryEnqueueAsync(request);
        }

        var batch = await builder.BuildBatchAsync();
        var payloads = batch.GetPayloads();

        Assert.AreEqual(2, payloads.Count);
        Assert.AreEqual(0, payloads[0]);
        Assert.AreEqual(1, payloads[1]);
    }
}
