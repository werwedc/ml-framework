using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;

namespace MLFramework.Tests.Serving;

[TestClass]
public class BatchingMetricsTests
{
    [TestMethod]
    public void RecordBatch_UpdatesMetricsCorrectly()
    {
        var collector = new BatchingMetricsCollector();

        collector.RecordBatch(batchSize: 4, queueWaitMs: 10.5, processingMs: 25.3);

        var snapshot = collector.GetSnapshot(currentQueueDepth: 0);

        Assert.AreEqual(1, snapshot.TotalBatches);
        Assert.AreEqual(4, snapshot.AverageBatchSize);
        Assert.AreEqual(10.5, snapshot.AverageQueueWaitMs);
        Assert.AreEqual(25.3, snapshot.AverageBatchProcessingMs);
    }

    [TestMethod]
    public void RecordBatch_WithMultipleBatches_CalculatesAveragesCorrectly()
    {
        var collector = new BatchingMetricsCollector();

        collector.RecordBatch(4, 10, 20);
        collector.RecordBatch(6, 15, 30);
        collector.RecordBatch(8, 20, 40);

        var snapshot = collector.GetSnapshot(0);

        Assert.AreEqual(3, snapshot.TotalBatches);
        Assert.AreEqual(6.0, snapshot.AverageBatchSize, 0.001); // (4+6+8)/3
        Assert.AreEqual(15.0, snapshot.AverageQueueWaitMs, 0.001); // (10+15+20)/3
        Assert.AreEqual(30.0, snapshot.AverageBatchProcessingMs, 0.001); // (20+30+40)/3
    }

    [TestMethod]
    public void RecordRequestEnqueued_IncrementsTotalRequests()
    {
        var collector = new BatchingMetricsCollector();

        collector.RecordRequestEnqueued(currentQueueDepth: 0);
        collector.RecordRequestEnqueued(currentQueueDepth: 1);

        var snapshot = collector.GetSnapshot(0);

        Assert.AreEqual(2, snapshot.TotalRequests);
    }

    [TestMethod]
    public void RecordRequestEnqueued_UpdatesMaxQueueDepth()
    {
        var collector = new BatchingMetricsCollector();

        collector.RecordRequestEnqueued(5);
        collector.RecordRequestEnqueued(10);
        collector.RecordRequestEnqueued(3);
        collector.RecordRequestEnqueued(15);
        collector.RecordRequestEnqueued(8);

        var snapshot = collector.GetSnapshot(0);

        Assert.AreEqual(15, snapshot.MaxQueueDepth);
    }

    [TestMethod]
    public void RecordQueueRejection_IncrementsRejectionCount()
    {
        var collector = new BatchingMetricsCollector();

        collector.RecordQueueRejection();
        collector.RecordQueueRejection();
        collector.RecordQueueRejection();

        var snapshot = collector.GetSnapshot(0);

        Assert.AreEqual(3, snapshot.QueueFullRejections);
    }

    [TestMethod]
    public void GetBatchSizeDistribution_CategorizesCorrectly()
    {
        var collector = new BatchingMetricsCollector();
        collector.RecordBatch(3, 10, 20);  // Very small (1-5)
        collector.RecordBatch(10, 10, 20); // Small (6-15)
        collector.RecordBatch(20, 10, 20); // Medium (16-30)
        collector.RecordBatch(40, 10, 20); // Large (31-63)
        collector.RecordBatch(70, 10, 20); // Very large (64+)

        var dist = collector.GetBatchSizeDistribution();

        Assert.AreEqual(1, dist.VerySmall);
        Assert.AreEqual(1, dist.Small);
        Assert.AreEqual(1, dist.Medium);
        Assert.AreEqual(1, dist.Large);
        Assert.AreEqual(1, dist.VeryLarge);
    }

    [TestMethod]
    public void GetBatchSizeDistribution_WithMultipleBatchesInSameCategory()
    {
        var collector = new BatchingMetricsCollector();

        collector.RecordBatch(2, 10, 20);  // Very small
        collector.RecordBatch(4, 10, 20);  // Very small
        collector.RecordBatch(5, 10, 20);  // Very small
        collector.RecordBatch(12, 10, 20); // Small
        collector.RecordBatch(14, 10, 20); // Small

        var dist = collector.GetBatchSizeDistribution();

        Assert.AreEqual(3, dist.VerySmall);
        Assert.AreEqual(2, dist.Small);
        Assert.AreEqual(0, dist.Medium);
        Assert.AreEqual(0, dist.Large);
        Assert.AreEqual(0, dist.VeryLarge);
    }

    [TestMethod]
    public void GetBatchSizeDistribution_BoundaryValues_CategorizedCorrectly()
    {
        var collector = new BatchingMetricsCollector();

        collector.RecordBatch(1, 10, 20);   // Very small boundary
        collector.RecordBatch(5, 10, 20);   // Very small boundary
        collector.RecordBatch(6, 10, 20);   // Small boundary
        collector.RecordBatch(15, 10, 20);  // Small boundary
        collector.RecordBatch(16, 10, 20);  // Medium boundary
        collector.RecordBatch(30, 10, 20);  // Medium boundary
        collector.RecordBatch(31, 10, 20);  // Large boundary
        collector.RecordBatch(63, 10, 20);  // Large boundary
        collector.RecordBatch(64, 10, 20);  // Very large boundary

        var dist = collector.GetBatchSizeDistribution();

        Assert.AreEqual(2, dist.VerySmall);
        Assert.AreEqual(2, dist.Small);
        Assert.AreEqual(2, dist.Medium);
        Assert.AreEqual(2, dist.Large);
        Assert.AreEqual(1, dist.VeryLarge);
    }

    [TestMethod]
    public void Reset_ClearsAllMetrics()
    {
        var collector = new BatchingMetricsCollector();
        collector.RecordBatch(4, 10, 20);
        collector.RecordRequestEnqueued(5);
        collector.RecordQueueRejection();

        collector.Reset();

        var snapshot = collector.GetSnapshot(0);
        Assert.AreEqual(0, snapshot.TotalBatches);
        Assert.AreEqual(0, snapshot.TotalRequests);
        Assert.AreEqual(0, snapshot.AverageBatchSize);
        Assert.AreEqual(0, snapshot.MaxQueueDepth);
        Assert.AreEqual(0, snapshot.QueueFullRejections);
        Assert.AreEqual(0, snapshot.AverageQueueWaitMs);
        Assert.AreEqual(0, snapshot.AverageBatchProcessingMs);
    }

    [TestMethod]
    public void Reset_ClearsDistribution()
    {
        var collector = new BatchingMetricsCollector();

        collector.RecordBatch(3, 10, 20);
        collector.RecordBatch(40, 10, 20);
        collector.RecordBatch(70, 10, 20);

        collector.Reset();

        var dist = collector.GetBatchSizeDistribution();

        Assert.AreEqual(0, dist.VerySmall);
        Assert.AreEqual(0, dist.Small);
        Assert.AreEqual(0, dist.Medium);
        Assert.AreEqual(0, dist.Large);
        Assert.AreEqual(0, dist.VeryLarge);
    }

    [TestMethod]
    public void GetSnapshot_WithCurrentQueueDepth_ReturnsCorrectDepth()
    {
        var collector = new BatchingMetricsCollector();

        var snapshot1 = collector.GetSnapshot(5);
        Assert.AreEqual(5, snapshot1.CurrentQueueDepth);

        var snapshot2 = collector.GetSnapshot(10);
        Assert.AreEqual(10, snapshot2.CurrentQueueDepth);
    }

    [TestMethod]
    public void GetSnapshot_CapturedAt_IsSet()
    {
        var collector = new BatchingMetricsCollector();
        var before = DateTime.UtcNow;

        var snapshot = collector.GetSnapshot(0);
        var after = DateTime.UtcNow;

        Assert.IsTrue(snapshot.CapturedAt >= before);
        Assert.IsTrue(snapshot.CapturedAt <= after);
    }

    [TestMethod]
    public void GetSnapshot_WithNoBatches_ReturnsZeroAverages()
    {
        var collector = new BatchingMetricsCollector();

        var snapshot = collector.GetSnapshot(0);

        Assert.AreEqual(0, snapshot.TotalBatches);
        Assert.AreEqual(0, snapshot.AverageBatchSize);
        Assert.AreEqual(0, snapshot.AverageQueueWaitMs);
        Assert.AreEqual(0, snapshot.AverageBatchProcessingMs);
    }

    [TestMethod]
    public void RecordBatch_ThreadSafe()
    {
        var collector = new BatchingMetricsCollector();
        var random = new Random();

        // Simulate concurrent batch recording
        var tasks = new System.Threading.Tasks.Task[100];
        for (int i = 0; i < tasks.Length; i++)
        {
            tasks[i] = System.Threading.Tasks.Task.Run(() =>
            {
                collector.RecordBatch(
                    batchSize: random.Next(1, 100),
                    queueWaitMs: random.NextDouble() * 100,
                    processingMs: random.NextDouble() * 100);
            });
        }

        System.Threading.Tasks.Task.WaitAll(tasks);

        var snapshot = collector.GetSnapshot(0);
        Assert.AreEqual(100, snapshot.TotalBatches);
    }

    [TestMethod]
    public void RecordQueueRejection_ThreadSafe()
    {
        var collector = new BatchingMetricsCollector();

        // Simulate concurrent rejections
        var tasks = new System.Threading.Tasks.Task[50];
        for (int i = 0; i < tasks.Length; i++)
        {
            tasks[i] = System.Threading.Tasks.Task.Run(() =>
            {
                collector.RecordQueueRejection();
            });
        }

        System.Threading.Tasks.Task.WaitAll(tasks);

        var snapshot = collector.GetSnapshot(0);
        Assert.AreEqual(50, snapshot.QueueFullRejections);
    }

    [TestMethod]
    public void RecordRequestEnqueued_ThreadSafe()
    {
        var collector = new BatchingMetricsCollector();

        // Simulate concurrent request enqueue
        var tasks = new System.Threading.Tasks.Task[50];
        for (int i = 0; i < tasks.Length; i++)
        {
            tasks[i] = System.Threading.Tasks.Task.Run(() =>
            {
                collector.RecordRequestEnqueued(currentQueueDepth: i);
            });
        }

        System.Threading.Tasks.Task.WaitAll(tasks);

        var snapshot = collector.GetSnapshot(0);
        Assert.AreEqual(50, snapshot.TotalRequests);
    }

    [TestMethod]
    public void GetSnapshot_ThreadSafe()
    {
        var collector = new BatchingMetricsCollector();

        // Simulate concurrent reads
        var tasks = new System.Threading.Tasks.Task[100];
        for (int i = 0; i < tasks.Length; i++)
        {
            tasks[i] = System.Threading.Tasks.Task.Run(() =>
            {
                var snapshot = collector.GetSnapshot(0);
                Assert.IsNotNull(snapshot);
            });
        }

        System.Threading.Tasks.Task.WaitAll(tasks);
    }

    [TestMethod]
    public void Metrics_AreImmutableSnapshots()
    {
        var collector = new BatchingMetricsCollector();
        collector.RecordBatch(10, 20, 30);

        var snapshot = collector.GetSnapshot(0);

        // Modify the original collector
        collector.RecordBatch(20, 40, 60);

        // Snapshot should remain unchanged
        Assert.AreEqual(1, snapshot.TotalBatches);
        Assert.AreEqual(10, snapshot.AverageBatchSize);
    }

    [TestMethod]
    public void BatchSizeDistribution_AreImmutableSnapshots()
    {
        var collector = new BatchingMetricsCollector();
        collector.RecordBatch(3, 10, 20);

        var dist = collector.GetBatchSizeDistribution();

        // Modify the original collector
        collector.RecordBatch(40, 10, 20);

        // Distribution should remain unchanged
        Assert.AreEqual(1, dist.VerySmall);
        Assert.AreEqual(0, dist.Large);
    }

    [TestMethod]
    public void RecordBatch_WithLargeValues_DoesNotOverflow()
    {
        var collector = new BatchingMetricsCollector();

        collector.RecordBatch(1000, 10000, 10000);
        collector.RecordBatch(1000, 10000, 10000);
        collector.RecordBatch(1000, 10000, 10000);

        var snapshot = collector.GetSnapshot(0);

        Assert.AreEqual(3, snapshot.TotalBatches);
        Assert.AreEqual(1000, snapshot.AverageBatchSize);
        Assert.AreEqual(10000, snapshot.AverageQueueWaitMs);
        Assert.AreEqual(10000, snapshot.AverageBatchProcessingMs);
    }

    [TestMethod]
    public void RecordBatch_WithZeroValues_HandlesCorrectly()
    {
        var collector = new BatchingMetricsCollector();

        collector.RecordBatch(1, 0, 0);
        collector.RecordBatch(1, 0, 0);

        var snapshot = collector.GetSnapshot(0);

        Assert.AreEqual(0, snapshot.AverageQueueWaitMs);
        Assert.AreEqual(0, snapshot.AverageBatchProcessingMs);
    }
}
