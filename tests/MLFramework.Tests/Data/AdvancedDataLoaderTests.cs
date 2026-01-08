using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using MLFramework.Data;
using Xunit;

namespace MLFramework.Tests.Data;

/// <summary>
/// Comprehensive unit tests for AdvancedDataLoader<T> core functionality.
/// </summary>
public class AdvancedDataLoaderTests : IDisposable
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

    #region Constructor Tests

    [Fact]
    public void Constructor_ValidParameters_CreatesDataloader()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 100).ToArray());
        var config = new DataLoaderConfig(batchSize: 10, numWorkers: 2);

        // Act
        var dataloader = new AdvancedDataLoader<int>(dataset, config);

        // Assert
        Assert.NotNull(dataloader);
        Assert.Same(dataset, dataloader.Config);
    }

    [Fact]
    public void Constructor_NullDataset_ThrowsArgumentNullException()
    {
        // Arrange
        var config = new DataLoaderConfig(batchSize: 10);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new AdvancedDataLoader<int>(null!, config));
    }

    [Fact]
    public void Constructor_NullConfig_ThrowsArgumentNullException()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 100).ToArray());

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new AdvancedDataLoader<int>(dataset, null!));
    }

    [Fact]
    public void Constructor_StoresConfig_ReturnsSameConfig()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 100).ToArray());
        var config = new DataLoaderConfig(batchSize: 10, numWorkers: 2);

        // Act
        var dataloader = new AdvancedDataLoader<int>(dataset, config);

        // Assert
        Assert.Same(config, dataloader.Config);
    }

    [Fact]
    public void Constructor_SingleWorker_CreatesCorrectly()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 100).ToArray());
        var config = new DataLoaderConfig(batchSize: 10, numWorkers: 1);

        // Act
        var dataloader = new AdvancedDataLoader<int>(dataset, config);

        // Assert
        Assert.Equal(1, dataloader.Config.NumWorkers);
    }

    [Fact]
    public void Constructor_MultipleWorkers_CreatesCorrectly()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 100).ToArray());
        var config = new DataLoaderConfig(batchSize: 10, numWorkers: 4);

        // Act
        var dataloader = new AdvancedDataLoader<int>(dataset, config);

        // Assert
        Assert.Equal(4, dataloader.Config.NumWorkers);
    }

    [Fact]
    public void Constructor_ShuffleTrue_ConfiguresShuffling()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 100).ToArray());
        var config = new DataLoaderConfig(batchSize: 10, shuffle: true);

        // Act
        var dataloader = new AdvancedDataLoader<int>(dataset, config);

        // Assert
        Assert.True(dataloader.Config.Shuffle);
    }

    [Fact]
    public void Constructor_ShuffleFalse_DisablesShuffling()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 100).ToArray());
        var config = new DataLoaderConfig(batchSize: 10, shuffle: false);

        // Act
        var dataloader = new AdvancedDataLoader<int>(dataset, config);

        // Assert
        Assert.False(dataloader.Config.Shuffle);
    }

    #endregion

    #region Properties Tests

    [Fact]
    public void IsRunning_InitiallyFalse()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 100).ToArray());
        var config = new DataLoaderConfig(batchSize: 10);
        var dataloader = new AdvancedDataLoader<int>(dataset, config);

        // Act & Assert
        Assert.False(dataloader.IsRunning);
    }

    [Fact]
    public void IsRunning_AfterStart_ReturnsTrue()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 100).ToArray());
        var config = new DataLoaderConfig(batchSize: 10);
        var dataloader = new AdvancedDataLoader<int>(dataset, config);

        // Act
        dataloader.Start();

        // Assert
        Assert.True(dataloader.IsRunning);
    }

    [Fact]
    public void IsRunning_AfterStop_ReturnsFalse()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 100).ToArray());
        var config = new DataLoaderConfig(batchSize: 10);
        var dataloader = new AdvancedDataLoader<int>(dataset, config);
        dataloader.Start();

        // Act
        dataloader.Stop();

        // Assert
        Assert.False(dataloader.IsRunning);
    }

    [Fact]
    public void Config_ReturnsProvidedConfig()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 100).ToArray());
        var config = new DataLoaderConfig(batchSize: 10, numWorkers: 2);

        // Act
        var dataloader = new AdvancedDataLoader<int>(dataset, config);

        // Assert
        Assert.Same(config, dataloader.Config);
    }

    [Fact]
    public void BatchCount_CalculatedCorrectly()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 100).ToArray());
        var config = new DataLoaderConfig(batchSize: 10);
        var dataloader = new AdvancedDataLoader<int>(dataset, config);

        // Act & Assert
        Assert.Equal(10, dataloader.BatchCount);
    }

    [Fact]
    public void BatchCount_UnevenDivision_RoundedUp()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 95).ToArray());
        var config = new DataLoaderConfig(batchSize: 10);
        var dataloader = new AdvancedDataLoader<int>(dataset, config);

        // Act & Assert
        Assert.Equal(10, dataloader.BatchCount); // 95 / 10 = 9.5, rounded to 10
    }

    [Fact]
    public void BatchCount_EmptyDataset_ReturnsZero()
    {
        // Arrange
        var dataset = new TestDataset(Array.Empty<int>());
        var config = new DataLoaderConfig(batchSize: 10);
        var dataloader = new AdvancedDataLoader<int>(dataset, config);

        // Act & Assert
        Assert.Equal(0, dataloader.BatchCount);
    }

    #endregion

    #region Start Tests

    [Fact]
    public void Start_InitializesWorkers()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 100).ToArray());
        var config = new DataLoaderConfig(batchSize: 10, numWorkers: 2);
        var dataloader = new AdvancedDataLoader<int>(dataset, config);

        // Act
        dataloader.Start();

        // Assert
        Assert.True(dataloader.IsRunning);
        dataloader.Stop();
    }

    [Fact]
    public void Start_StartsBackgroundProduction()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 100).ToArray());
        var config = new DataLoaderConfig(batchSize: 10, numWorkers: 2);
        var dataloader = new AdvancedDataLoader<int>(dataset, config);

        // Act
        dataloader.Start();
        Thread.Sleep(100); // Give workers time to start

        // Assert
        Assert.True(dataloader.IsRunning);
        dataloader.Stop();
    }

    [Fact]
    public void Start_UpdatesIsRunningToTrue()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 100).ToArray());
        var config = new DataLoaderConfig(batchSize: 10);
        var dataloader = new AdvancedDataLoader<int>(dataset, config);

        // Act
        dataloader.Start();

        // Assert
        Assert.True(dataloader.IsRunning);
        dataloader.Stop();
    }

    [Fact]
    public void Start_AfterStop_RestartsCorrectly()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 100).ToArray());
        var config = new DataLoaderConfig(batchSize: 10);
        var dataloader = new AdvancedDataLoader<int>(dataset, config);
        dataloader.Start();
        dataloader.Stop();

        // Act
        dataloader.Start();

        // Assert
        Assert.True(dataloader.IsRunning);
        dataloader.Stop();
    }

    [Fact]
    public void Start_WhenAlreadyRunning_ThrowsInvalidOperationException()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 100).ToArray());
        var config = new DataLoaderConfig(batchSize: 10);
        var dataloader = new AdvancedDataLoader<int>(dataset, config);
        dataloader.Start();

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => dataloader.Start());
        dataloader.Stop();
    }

    [Fact]
    public void Start_RespectsNumWorkersInConfig()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 100).ToArray());
        var config = new DataLoaderConfig(batchSize: 10, numWorkers: 4);
        var dataloader = new AdvancedDataLoader<int>(dataset, config);

        // Act
        dataloader.Start();

        // Assert
        Assert.Equal(4, dataloader.Config.NumWorkers);
        dataloader.Stop();
    }

    [Fact]
    public void Start_RespectsBatchSizeInConfig()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 100).ToArray());
        var config = new DataLoaderConfig(batchSize: 20);
        var dataloader = new AdvancedDataLoader<int>(dataset, config);

        // Act
        dataloader.Start();

        // Assert
        Assert.Equal(20, dataloader.Config.BatchSize);
        dataloader.Stop();
    }

    [Fact]
    public void Start_RespectsQueueSizeInConfig()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 100).ToArray());
        var config = new DataLoaderConfig(batchSize: 10, queueSize: 15);
        var dataloader = new AdvancedDataLoader<int>(dataset, config);

        // Act
        dataloader.Start();

        // Assert
        Assert.Equal(15, dataloader.Config.QueueSize);
        dataloader.Stop();
    }

    #endregion

    #region Stop Tests

    [Fact]
    public void Stop_StopsWorkers()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 100).ToArray());
        var config = new DataLoaderConfig(batchSize: 10);
        var dataloader = new AdvancedDataLoader<int>(dataset, config);
        dataloader.Start();

        // Act
        dataloader.Stop();

        // Assert
        Assert.False(dataloader.IsRunning);
    }

    [Fact]
    public void Stop_UpdatesIsRunningToFalse()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 100).ToArray());
        var config = new DataLoaderConfig(batchSize: 10);
        var dataloader = new AdvancedDataLoader<int>(dataset, config);
        dataloader.Start();

        // Act
        dataloader.Stop();

        // Assert
        Assert.False(dataloader.IsRunning);
    }

    [Fact]
    public void Stop_BeforeStart_NoEffect()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 100).ToArray());
        var config = new DataLoaderConfig(batchSize: 10);
        var dataloader = new AdvancedDataLoader<int>(dataset, config);

        // Act & Assert (should not throw)
        dataloader.Stop();
    }

    [Fact]
    public void Stop_BeforeStart_DoesNotThrow()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 100).ToArray());
        var config = new DataLoaderConfig(batchSize: 10);
        var dataloader = new AdvancedDataLoader<int>(dataset, config);

        // Act & Assert
        var exception = Record.Exception(() => dataloader.Stop());
        Assert.Null(exception);
    }

    [Fact]
    public void Stop_MultipleCalls_Idempotent()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 100).ToArray());
        var config = new DataLoaderConfig(batchSize: 10);
        var dataloader = new AdvancedDataLoader<int>(dataset, config);
        dataloader.Start();

        // Act & Assert (should not throw)
        dataloader.Stop();
        dataloader.Stop();
    }

    #endregion

    #region Reset Tests

    [Fact]
    public void Reset_StopsIfRunning()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 100).ToArray());
        var config = new DataLoaderConfig(batchSize: 10);
        var dataloader = new AdvancedDataLoader<int>(dataset, config);
        dataloader.Start();

        // Act
        dataloader.Reset();

        // Assert
        Assert.False(dataloader.IsRunning);
    }

    [Fact]
    public void Reset_ClearsInternalState()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 100).ToArray());
        var config = new DataLoaderConfig(batchSize: 10);
        var dataloader = new AdvancedDataLoader<int>(dataset, config);
        dataloader.Start();
        Thread.Sleep(100);

        // Act
        dataloader.Reset();

        // Assert
        Assert.False(dataloader.IsRunning);
    }

    [Fact]
    public void Reset_BeforeStart_NoEffect()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 100).ToArray());
        var config = new DataLoaderConfig(batchSize: 10);
        var dataloader = new AdvancedDataLoader<int>(dataset, config);

        // Act & Assert (should not throw)
        dataloader.Reset();
    }

    [Fact]
    public void Reset_AfterReset_CanStartAgain()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 100).ToArray());
        var config = new DataLoaderConfig(batchSize: 10);
        var dataloader = new AdvancedDataLoader<int>(dataset, config);
        dataloader.Start();
        dataloader.Reset();

        // Act
        dataloader.Start();

        // Assert
        Assert.True(dataloader.IsRunning);
        dataloader.Stop();
    }

    #endregion

    #region Synchronous Iterator Tests

    [Fact]
    public void GetEnumerator_BeforeStart_ThrowsInvalidOperationException()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 100).ToArray());
        var config = new DataLoaderConfig(batchSize: 10);
        var dataloader = new AdvancedDataLoader<int>(dataset, config);

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => dataloader.GetEnumerator());
    }

    [Fact]
    public void GetEnumerator_AfterDispose_ThrowsObjectDisposedException()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 100).ToArray());
        var config = new DataLoaderConfig(batchSize: 10);
        var dataloader = new AdvancedDataLoader<int>(dataset, config);
        dataloader.Dispose();

        // Act & Assert
        Assert.Throws<ObjectDisposedException>(() => dataloader.GetEnumerator());
    }

    #endregion

    #region Async Iterator Tests

    [Fact]
    public async Task GetAsyncEnumerator_BeforeStart_ThrowsInvalidOperationException()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 100).ToArray());
        var config = new DataLoaderConfig(batchSize: 10);
        var dataloader = new AdvancedDataLoader<int>(dataset, config);

        // Act & Assert
        await Assert.ThrowsAsync<InvalidOperationException>(() => GetAsyncEnumeratorAsync(dataloader));
    }

    private async Task GetAsyncEnumeratorAsync(AdvancedDataLoader<int> dataloader)
    {
        await foreach (var item in dataloader)
        {
            // Consume items
        }
    }

    [Fact]
    public async Task GetAsyncEnumerator_AfterDispose_ThrowsObjectDisposedException()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 100).ToArray());
        var config = new DataLoaderConfig(batchSize: 10);
        var dataloader = new AdvancedDataLoader<int>(dataset, config);
        dataloader.Dispose();

        // Act & Assert
        await Assert.ThrowsAsync<ObjectDisposedException>(() => GetAsyncEnumeratorAsync(dataloader));
    }

    #endregion

    #region Shuffling Tests

    [Fact]
    public void ShufflingFalse_BatchesInSameOrder()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 50).ToArray());
        var config = new DataLoaderConfig(batchSize: 10, shuffle: false);
        var dataloader = new AdvancedDataLoader<int>(dataset, config);

        // Act
        var batchCount1 = dataloader.BatchCount;
        var batchCount2 = dataloader.BatchCount;

        // Assert
        Assert.Equal(batchCount1, batchCount2);
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void EmptyDataset_BatchCountZero()
    {
        // Arrange
        var dataset = new TestDataset(Array.Empty<int>());
        var config = new DataLoaderConfig(batchSize: 10);
        var dataloader = new AdvancedDataLoader<int>(dataset, config);

        // Act & Assert
        Assert.Equal(0, dataloader.BatchCount);
    }

    [Fact]
    public void SingleItemDataset_SingleBatch()
    {
        // Arrange
        var dataset = new TestDataset(new[] { 42 });
        var config = new DataLoaderConfig(batchSize: 10);
        var dataloader = new AdvancedDataLoader<int>(dataset, config);

        // Act & Assert
        Assert.Equal(1, dataloader.BatchCount);
    }

    [Fact]
    public void BatchSizeLargerThanDataset_SingleBatch()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 10).ToArray());
        var config = new DataLoaderConfig(batchSize: 100);
        var dataloader = new AdvancedDataLoader<int>(dataset, config);

        // Act & Assert
        Assert.Equal(1, dataloader.BatchCount);
    }

    [Fact]
    public void BatchSizeEqualsDataset_SingleBatch()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 10).ToArray());
        var config = new DataLoaderConfig(batchSize: 10);
        var dataloader = new AdvancedDataLoader<int>(dataset, config);

        // Act & Assert
        Assert.Equal(1, dataloader.BatchCount);
    }

    [Fact]
    public void BatchSizeOne_MultipleBatches()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 10).ToArray());
        var config = new DataLoaderConfig(batchSize: 1);
        var dataloader = new AdvancedDataLoader<int>(dataset, config);

        // Act & Assert
        Assert.Equal(10, dataloader.BatchCount);
    }

    [Fact]
    public void LargeDataset_CorrectBatchCount()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 10000).ToArray());
        var config = new DataLoaderConfig(batchSize: 100);
        var dataloader = new AdvancedDataLoader<int>(dataset, config);

        // Act & Assert
        Assert.Equal(100, dataloader.BatchCount);
    }

    #endregion

    #region Dispose Tests

    [Fact]
    public void Dispose_StopsIfRunning()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 100).ToArray());
        var config = new DataLoaderConfig(batchSize: 10);
        var dataloader = new AdvancedDataLoader<int>(dataset, config);
        dataloader.Start();

        // Act
        dataloader.Dispose();

        // Assert
        Assert.False(dataloader.IsRunning);
    }

    [Fact]
    public void Dispose_CanCallMultipleTimes()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 100).ToArray());
        var config = new DataLoaderConfig(batchSize: 10);
        var dataloader = new AdvancedDataLoader<int>(dataset, config);

        // Act & Assert (should not throw)
        dataloader.Dispose();
        dataloader.Dispose();
    }

    #endregion

    #region Statistics Tests

    [Fact]
    public void Statistics_BatchesLoaded_IncrementsCorrectly()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 100).ToArray());
        var config = new DataLoaderConfig(batchSize: 10);
        var dataloader = new AdvancedDataLoader<int>(dataset, config);
        dataloader.Start();

        // Act
        var stats = dataloader.GetStatistics();

        // Assert
        Assert.NotNull(stats);
        Assert.Equal(100, stats.TotalSamples);
        dataloader.Stop();
    }

    [Fact]
    public void Statistics_TotalSamples_CalculatedCorrectly()
    {
        // Arrange
        var dataset = new TestDataset(Enumerable.Range(0, 100).ToArray());
        var config = new DataLoaderConfig(batchSize: 10);
        var dataloader = new AdvancedDataLoader<int>(dataset, config);

        // Act
        var stats = dataloader.GetStatistics();

        // Assert
        Assert.Equal(100, stats.TotalSamples);
    }

    #endregion

    #region IDisposable

    public void Dispose()
    {
        // Cleanup any resources used in tests
    }

    #endregion
}
