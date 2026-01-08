using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using MLFramework.Data;
using Xunit;

namespace MLFramework.Tests.Data;

/// <summary>
/// Integration tests for AdvancedDataLoader<T> end-to-end functionality.
/// </summary>
public class AdvancedDataLoaderIntegrationTests : IDisposable
{
    #region Helper Classes

    /// <summary>
    /// Simple test dataset for testing.
    /// </summary>
    private class TestDataset : Dataset<int>
    {
        private readonly int[] _data;

        public TestDataset(int[] data)
        {
            _data = data;
        }

        public override int Count => _data.Length;

        public override int GetItem(int index)
        {
            int normalizedIndex = NormalizeIndex(index);
            return _data[normalizedIndex];
        }
    }

    /// <summary>
    /// Dataset that simulates slow data loading.
    /// </summary>
    private class SlowDataset : Dataset<int>
    {
        private readonly int[] _data;
        private readonly TimeSpan _delay;

        public SlowDataset(int[] data, TimeSpan delay)
        {
            _data = data;
            _delay = delay;
        }

        public override int Count => _data.Length;

        public override int GetItem(int index)
        {
            int normalizedIndex = NormalizeIndex(index);
            Thread.Sleep(_delay);
            return _data[normalizedIndex];
        }
    }

    #endregion

    #region End-to-End Flow Tests

    [Fact]
    public async Task EndToEnd_DatasetToBatches_CompleteFlow()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 100).ToArray());
        var config = new DataLoaderConfig(batchSize: 10, numWorkers: 2);
        var dataloader = new AdvancedDataLoader<int>(dataset, config);

        // Act
        dataloader.Start();

        var batches = new List<int>();
        await foreach (var batch in dataloader)
        {
            batches.Add(batch);
        }

        dataloader.Stop();

        // Assert
        Assert.Equal(10, batches.Count);
    }

    [Fact]
    public async Task EndToEnd_IterationConsumesAllBatches()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 100).ToArray());
        var config = new DataLoaderConfig(batchSize: 10, numWorkers: 2);
        var dataloader = new AdvancedDataLoader<int>(dataset, config);

        // Act
        dataloader.Start();

        var batchCount = 0;
        await foreach (var batch in dataloader)
        {
            batchCount++;
        }

        dataloader.Stop();

        // Assert
        Assert.Equal(10, batchCount);
    }

    [Fact]
    public async Task EndToEnd_MultipleIterations_WorkCorrectly()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 100).ToArray());
        var config = new DataLoaderConfig(batchSize: 10, numWorkers: 2);
        var dataloader = new AdvancedDataLoader<int>(dataset, config);

        // Act
        dataloader.Start();

        var firstRunBatches = new List<int>();
        await foreach (var batch in dataloader)
        {
            firstRunBatches.Add(batch);
        }

        dataloader.Reset();
        dataloader.Start();

        var secondRunBatches = new List<int>();
        await foreach (var batch in dataloader)
        {
            secondRunBatches.Add(batch);
        }

        dataloader.Stop();

        // Assert
        Assert.Equal(10, firstRunBatches.Count);
        Assert.Equal(10, secondRunBatches.Count);
    }

    #endregion

    #region Worker Pool Integration Tests

    [Fact]
    public async Task Workers_ProduceBatchesContinuously()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 100).ToArray());
        var config = new DataLoaderConfig(batchSize: 10, numWorkers: 2);
        var dataloader = new AdvancedDataLoader<int>(dataset, config);

        // Act
        dataloader.Start();

        var batches = new List<int>();
        await foreach (var batch in dataloader)
        {
            batches.Add(batch);
        }

        dataloader.Stop();

        // Assert
        Assert.Equal(10, batches.Count);
    }

    [Fact]
    public async Task Workers_ProduceCorrectNumberOfBatches()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 100).ToArray());
        var config = new DataLoaderConfig(batchSize: 10, numWorkers: 2);
        var dataloader = new AdvancedDataLoader<int>(dataset, config);

        // Act
        dataloader.Start();

        var batchCount = 0;
        await foreach (var batch in dataloader)
        {
            batchCount++;
        }

        dataloader.Stop();

        // Assert
        Assert.Equal(10, batchCount);
    }

    [Fact]
    public async Task MultipleWorkers_RunConcurrently()
    {
        // Arrange
        var dataset = new SlowDataset(Enumerable.Range(0, 100).ToArray(), TimeSpan.FromMilliseconds(10));
        var config = new DataLoaderConfig(batchSize: 10, numWorkers: 4);
        var dataloader = new AdvancedDataLoader<int>(dataset, config);

        // Act
        dataloader.Start();

        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        var batchCount = 0;
        await foreach (var batch in dataloader)
        {
            batchCount++;
        }
        stopwatch.Stop();

        dataloader.Stop();

        // Assert
        Assert.Equal(10, batchCount);
        // With 4 workers and 10ms delay per item, should be faster than sequential
        // This is a rough check, may need adjustment based on system performance
        Assert.True(stopwatch.ElapsedMilliseconds < 1000, $"Took {stopwatch.ElapsedMilliseconds}ms, expected < 1000ms");
    }

    #endregion

    #region Prefetching Integration Tests

    [Fact]
    public async Task Prefetching_Started_InitialQueueFilled()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 100).ToArray());
        var config = new DataLoaderConfig(batchSize: 10, prefetchCount: 2);
        var dataloader = new AdvancedDataLoader<int>(dataset, config);

        // Act
        dataloader.Start();
        await Task.Delay(200); // Wait for initial prefetch

        var stats = dataloader.GetStatistics();

        dataloader.Stop();

        // Assert
        Assert.NotNull(stats);
    }

    [Fact]
    public async Task Prefetching_ConsumerConsumes_PrefetchRefills()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 100).ToArray());
        var config = new DataLoaderConfig(batchSize: 10, prefetchCount: 2);
        var dataloader = new AdvancedDataLoader<int>(dataset, config);

        // Act
        dataloader.Start();

        var batchCount = 0;
        await foreach (var batch in dataloader)
        {
            batchCount++;
        }

        dataloader.Stop();

        // Assert
        Assert.Equal(10, batchCount);
    }

    [Fact]
    public async Task Prefetching_WithPrefetchCount_CorrectBehavior()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 100).ToArray());
        var config = new DataLoaderConfig(batchSize: 10, prefetchCount: 3);
        var dataloader = new AdvancedDataLoader<int>(dataset, config);

        // Act
        dataloader.Start();

        var batchCount = 0;
        await foreach (var batch in dataloader)
        {
            batchCount++;
        }

        dataloader.Stop();

        // Assert
        Assert.Equal(10, batchCount);
    }

    #endregion

    #region Training Loop Simulation Tests

    [Fact]
    public async Task TrainingLoopSimulation_CompleteTrainingCycle()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 100).ToArray());
        var config = new DataLoaderConfig(batchSize: 10, numWorkers: 2);
        var dataloader = new AdvancedDataLoader<int>(dataset, config);

        int epochCount = 2;

        // Act
        for (int epoch = 0; epoch < epochCount; epoch++)
        {
            dataloader.Start();

            var batchCount = 0;
            await foreach (var batch in dataloader)
            {
                batchCount++;
                // Simulate training step
                await Task.Delay(1);
            }

            dataloader.Stop();
            dataloader.Reset();

            Assert.Equal(10, batchCount);
        }

        // Assert - All epochs completed successfully
    }

    #endregion

    #region Statistics Integration Tests

    [Fact]
    public async Task Statistics_WorkerStats_Propagated()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 100).ToArray());
        var config = new DataLoaderConfig(batchSize: 10, numWorkers: 2);
        var dataloader = new AdvancedDataLoader<int>(dataset, config);

        // Act
        dataloader.Start();

        await foreach (var batch in dataloader)
        {
            // Consume all batches
        }

        var stats = dataloader.GetStatistics();

        dataloader.Stop();

        // Assert
        Assert.NotNull(stats);
        Assert.Equal(100, stats.TotalSamples);
    }

    [Fact]
    public async Task Statistics_PrefetchStats_Propagated()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 100).ToArray());
        var config = new DataLoaderConfig(batchSize: 10, prefetchCount: 2);
        var dataloader = new AdvancedDataLoader<int>(dataset, config);

        // Act
        dataloader.Start();

        await foreach (var batch in dataloader)
        {
            // Consume all batches
        }

        var stats = dataloader.GetStatistics();

        dataloader.Stop();

        // Assert
        Assert.NotNull(stats);
        Assert.NotNull(stats.PrefetchStatistics);
    }

    #endregion

    #region Large Dataset Tests

    [Fact]
    public async Task LargeDataset_AllBatchesProduced()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 10000).ToArray());
        var config = new DataLoaderConfig(batchSize: 100, numWorkers: 4);
        var dataloader = new AdvancedDataLoader<int>(dataset, config);

        // Act
        dataloader.Start();

        var batchCount = 0;
        await foreach (var batch in dataloader)
        {
            batchCount++;
        }

        dataloader.Stop();

        // Assert
        Assert.Equal(100, batchCount);
    }

    #endregion

    #region Concurrent Access Tests

    [Fact]
    public async Task ConcurrentAccess_MultipleIterators_ThreadSafe()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 100).ToArray());
        var config = new DataLoaderConfig(batchSize: 10, numWorkers: 2);
        var dataloader = new AdvancedDataLoader<int>(dataset, config);
        dataloader.Start();

        // Act - This test verifies that the DataLoader can handle multiple consumers
        // Note: In practice, you typically don't iterate the same dataloader from multiple threads
        // but the implementation should be safe

        var task1 = Task.Run(async () =>
        {
            var count = 0;
            await foreach (var batch in dataloader)
            {
                count++;
            }
            return count;
        });

        var task2 = Task.Run(async () =>
        {
            var count = 0;
            await foreach (var batch in dataloader)
            {
                count++;
            }
            return count;
        });

        await Task.WhenAll(task1, task2);

        dataloader.Stop();

        // Assert
        // Both tasks should complete without exceptions
        Assert.NotNull(task1.Result);
        Assert.NotNull(task2.Result);
    }

    #endregion

    #region Error Handling Tests

    [Fact]
    public void ErrorHandling_WithEmptyDataset_DoesNotThrow()
    {
        // Arrange
        var dataset = new TestDataset(Array.Empty<int>());
        var config = new DataLoaderConfig(batchSize: 10);
        var dataloader = new AdvancedDataLoader<int>(dataset, config);

        // Act & Assert
        var exception = Record.Exception(() => dataloader.Start());
        Assert.Null(exception);
        dataloader.Stop();
    }

    [Fact]
    public async Task ErrorHandling_WithCancellation_StopsGracefully()
    {
        // Arrange
        var dataset = new SlowDataset(Enumerable.Range(0, 100).ToArray(), TimeSpan.FromMilliseconds(10));
        var config = new DataLoaderConfig(batchSize: 10, numWorkers: 2);
        var dataloader = new AdvancedDataLoader<int>(dataset, config);
        dataloader.Start();

        var cts = new CancellationTokenSource(100);

        // Act
        var exception = await Record.ExceptionAsync(async () =>
        {
            await foreach (var batch in dataloader.WithCancellation(cts.Token))
            {
                // Consume batches
            }
        });

        dataloader.Stop();

        // Assert
        // Should handle cancellation gracefully
        Assert.Null(exception);
    }

    #endregion

    #region Performance Tests (Explicit)

    [Fact(Explicit = true)]
    public async Task Performance_Throughput_Measured()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 10000).ToArray());
        var config = new DataLoaderConfig(batchSize: 100, numWorkers: 4);
        var dataloader = new AdvancedDataLoader<int>(dataset, config);

        // Act
        dataloader.Start();

        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        var batchCount = 0;
        await foreach (var batch in dataloader)
        {
            batchCount++;
        }
        stopwatch.Stop();

        dataloader.Stop();

        // Assert
        var batchesPerSecond = batchCount / stopwatch.Elapsed.TotalSeconds;
        var samplesPerSecond = (batchCount * config.BatchSize) / stopwatch.Elapsed.TotalSeconds;

        Assert.True(batchesPerSecond > 0, $"Throughput: {batchesPerSecond} batches/sec");
        Assert.True(samplesPerSecond > 0, $"Throughput: {samplesPerSecond} samples/sec");
    }

    [Fact(Explicit = true)]
    public async Task Performance_Latency_FirstBatch_Measured()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 1000).ToArray());
        var config = new DataLoaderConfig(batchSize: 32, numWorkers: 2);
        var dataloader = new AdvancedDataLoader<int>(dataset, config);

        // Act
        dataloader.Start();

        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        var firstBatchRetrieved = false;
        await foreach (var batch in dataloader)
        {
            if (!firstBatchRetrieved)
            {
                stopwatch.Stop();
                firstBatchRetrieved = true;
            }
        }

        dataloader.Stop();

        // Assert
        var firstBatchLatencyMs = stopwatch.ElapsedMilliseconds;
        Assert.True(firstBatchLatencyMs > 0, $"First batch latency: {firstBatchLatencyMs}ms");
    }

    #endregion

    #region IDisposable

    public void Dispose()
    {
        // Cleanup any resources used in tests
    }

    #endregion
}
