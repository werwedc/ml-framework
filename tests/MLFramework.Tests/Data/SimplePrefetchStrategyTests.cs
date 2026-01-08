using Xunit;
using System.Threading;
using MLFramework.Data;

namespace MLFramework.Tests.Data
{
    public class SimplePrefetchStrategyTests
    {
        [Fact]
        public void Constructor_ValidParameters_CreatesStrategy()
        {
            // Arrange
            var queue = new SharedQueue<int>(10);

            // Act
            var strategy = new SimplePrefetchStrategy<int>(queue, 5);

            // Assert
            Assert.Equal(5, strategy.BufferCapacity);
            Assert.Equal(0, strategy.BufferCount);
            Assert.False(strategy.IsAvailable);
        }

        [Fact]
        public void Constructor_NullQueue_ThrowsArgumentNullException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentNullException>(() => new SimplePrefetchStrategy<int>(null!, 5));
        }

        [Fact]
        public void Constructor_ZeroPrefetchCount_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var queue = new SharedQueue<int>(10);

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() => new SimplePrefetchStrategy<int>(queue, 0));
        }

        [Fact]
        public void Constructor_NegativePrefetchCount_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var queue = new SharedQueue<int>(10);

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() => new SimplePrefetchStrategy<int>(queue, -1));
        }

        [Fact]
        public async Task GetNextAsync_ItemInBuffer_ReturnsImmediately()
        {
            // Arrange
            var queue = new SharedQueue<int>(10);
            var strategy = new SimplePrefetchStrategy<int>(queue, 5);

            // Prefetch items
            queue.Enqueue(1);
            queue.Enqueue(2);
            queue.Enqueue(3);
            await strategy.PrefetchAsync(3, CancellationToken.None);

            // Act
            var item = await strategy.GetNextAsync(CancellationToken.None);

            // Assert
            Assert.Equal(1, item);
        }

        [Fact]
        public async Task GetNextAsync_ItemNotInBuffer_WaitsForQueue()
        {
            // Arrange
            var queue = new SharedQueue<int>(10);
            var strategy = new SimplePrefetchStrategy<int>(queue, 5);

            // Enqueue item after delay
            var enqueueTask = Task.Run(async () =>
            {
                await Task.Delay(100);
                queue.Enqueue(42);
            });

            // Act
            var item = await strategy.GetNextAsync(CancellationToken.None);

            // Assert
            Assert.Equal(42, item);
            await enqueueTask; // Wait for enqueue task to complete
        }

        [Fact]
        public async Task GetNextAsync_CancellationToken_ThrowsOperationCanceledException()
        {
            // Arrange
            var queue = new SharedQueue<int>(10);
            var cts = new CancellationTokenSource();
            var strategy = new SimplePrefetchStrategy<int>(queue, 5);

            // Act & Assert
            cts.Cancel();
            await Assert.ThrowsAsync<OperationCanceledException>(() =>
                strategy.GetNextAsync(cts.Token));
        }

        [Fact]
        public async Task GetNextAsync_MultipleItems_ReturnsInFIFOOrder()
        {
            // Arrange
            var queue = new SharedQueue<int>(10);
            var strategy = new SimplePrefetchStrategy<int>(queue, 5);

            queue.Enqueue(1);
            queue.Enqueue(2);
            queue.Enqueue(3);
            await strategy.PrefetchAsync(3, CancellationToken.None);

            // Act
            var item1 = await strategy.GetNextAsync(CancellationToken.None);
            var item2 = await strategy.GetNextAsync(CancellationToken.None);
            var item3 = await strategy.GetNextAsync(CancellationToken.None);

            // Assert
            Assert.Equal(1, item1);
            Assert.Equal(2, item2);
            Assert.Equal(3, item3);
        }

        [Fact]
        public async Task PrefetchAsync_EnqueuesItemsToBuffer()
        {
            // Arrange
            var queue = new SharedQueue<int>(10);
            var strategy = new SimplePrefetchStrategy<int>(queue, 5);

            queue.Enqueue(1);
            queue.Enqueue(2);
            queue.Enqueue(3);

            // Act
            await strategy.PrefetchAsync(3, CancellationToken.None);

            // Assert
            Assert.True(strategy.IsAvailable);
            Assert.Equal(3, strategy.BufferCount);
        }

        [Fact]
        public async Task PrefetchAsync_BufferFull_StopsFilling()
        {
            // Arrange
            var queue = new SharedQueue<int>(10);
            var strategy = new SimplePrefetchStrategy<int>(queue, 3);

            queue.Enqueue(1);
            queue.Enqueue(2);
            queue.Enqueue(3);
            queue.Enqueue(4);
            queue.Enqueue(5);

            // Act
            await strategy.PrefetchAsync(5, CancellationToken.None);

            // Assert
            Assert.Equal(3, strategy.BufferCount); // Should only fill to capacity
            Assert.True(strategy.IsAvailable);
        }

        [Fact]
        public async Task PrefetchAsync_CancellationToken_ThrowsOperationCanceledException()
        {
            // Arrange
            var queue = new SharedQueue<int>(10);
            var cts = new CancellationTokenSource();
            var strategy = new SimplePrefetchStrategy<int>(queue, 5);

            // Act & Assert
            cts.Cancel();
            await Assert.ThrowsAnyAsync<OperationCanceledException>(() =>
                strategy.PrefetchAsync(3, cts.Token));
        }

        [Fact]
        public void Reset_ClearsBuffer()
        {
            // Arrange
            var queue = new SharedQueue<int>(10);
            var strategy = new SimplePrefetchStrategy<int>(queue, 5);

            queue.Enqueue(1);
            queue.Enqueue(2);
            queue.Enqueue(3);
            var task = strategy.PrefetchAsync(3, CancellationToken.None);
            task.Wait();

            // Act
            strategy.Reset();

            // Assert
            Assert.Equal(0, strategy.BufferCount);
            Assert.False(strategy.IsAvailable);
        }

        [Fact]
        public async Task Reset_AfterGetNext_ResetsCorrectly()
        {
            // Arrange
            var queue = new SharedQueue<int>(10);
            var strategy = new SimplePrefetchStrategy<int>(queue, 5);

            queue.Enqueue(1);
            queue.Enqueue(2);
            queue.Enqueue(3);
            await strategy.PrefetchAsync(3, CancellationToken.None);
            await strategy.GetNextAsync(CancellationToken.None);

            // Act
            strategy.Reset();

            // Assert
            Assert.Equal(0, strategy.BufferCount);
            Assert.False(strategy.IsAvailable);
        }

        [Fact]
        public async Task IsAvailable_WithItems_ReturnsTrue()
        {
            // Arrange
            var queue = new SharedQueue<int>(10);
            var strategy = new SimplePrefetchStrategy<int>(queue, 5);

            queue.Enqueue(1);
            queue.Enqueue(2);
            await strategy.PrefetchAsync(2, CancellationToken.None);

            // Act & Assert
            Assert.True(strategy.IsAvailable);
        }

        [Fact]
        public void IsAvailable_WithoutItems_ReturnsFalse()
        {
            // Arrange
            var queue = new SharedQueue<int>(10);
            var strategy = new SimplePrefetchStrategy<int>(queue, 5);

            // Act & Assert
            Assert.False(strategy.IsAvailable);
        }

        [Fact]
        public async Task GetStatistics_AfterOperations_ReturnsCorrectStats()
        {
            // Arrange
            var queue = new SharedQueue<int>(10);
            var strategy = new SimplePrefetchStrategy<int>(queue, 5);

            queue.Enqueue(1);
            queue.Enqueue(2);
            await strategy.PrefetchAsync(2, CancellationToken.None);

            // Act
            var stats = strategy.GetStatistics();

            // Assert
            Assert.NotNull(stats);
            Assert.Equal(0, stats.CacheHits); // No items retrieved yet
            Assert.Equal(0, stats.CacheMisses);
        }

        [Fact]
        public async Task GetStatistics_AfterGetNext_RecordsCacheHit()
        {
            // Arrange
            var queue = new SharedQueue<int>(10);
            var strategy = new SimplePrefetchStrategy<int>(queue, 5);

            queue.Enqueue(1);
            queue.Enqueue(2);
            await strategy.PrefetchAsync(2, CancellationToken.None);

            // Act
            await strategy.GetNextAsync(CancellationToken.None);
            var stats = strategy.GetStatistics();

            // Assert
            Assert.Equal(1, stats.CacheHits);
            Assert.Equal(0, stats.CacheMisses);
        }

        [Fact]
        public async Task GetStatistics_AfterGetNextWithoutPrefetch_RecordsCacheMiss()
        {
            // Arrange
            var queue = new SharedQueue<int>(10);
            var strategy = new SimplePrefetchStrategy<int>(queue, 5);

            queue.Enqueue(1);

            // Act
            await strategy.GetNextAsync(CancellationToken.None);
            var stats = strategy.GetStatistics();

            // Assert
            Assert.Equal(0, stats.CacheHits);
            Assert.Equal(1, stats.CacheMisses);
            Assert.Equal(1, stats.StarvationCount);
        }

        [Fact]
        public async Task Reset_ClearsStatistics()
        {
            // Arrange
            var queue = new SharedQueue<int>(10);
            var strategy = new SimplePrefetchStrategy<int>(queue, 5);

            queue.Enqueue(1);
            queue.Enqueue(2);
            await strategy.PrefetchAsync(2, CancellationToken.None);
            await strategy.GetNextAsync(CancellationToken.None);

            // Act
            strategy.Reset();
            var stats = strategy.GetStatistics();

            // Assert
            Assert.Equal(0, stats.CacheHits);
            Assert.Equal(0, stats.CacheMisses);
            Assert.Equal(0, stats.RefillCount);
            Assert.Equal(0, stats.StarvationCount);
        }

        [Fact]
        public async Task Dispose_AfterDispose_ThrowsObjectDisposedException()
        {
            // Arrange
            var queue = new SharedQueue<int>(10);
            var strategy = new SimplePrefetchStrategy<int>(queue, 5);

            // Act
            strategy.Dispose();

            // Assert
            await Assert.ThrowsAsync<ObjectDisposedException>(() =>
                strategy.GetNextAsync(CancellationToken.None));
        }

        [Fact]
        public void BufferCapacity_ReturnsCorrectValue()
        {
            // Arrange
            var queue = new SharedQueue<int>(10);
            var strategy = new SimplePrefetchStrategy<int>(queue, 7);

            // Act & Assert
            Assert.Equal(7, strategy.BufferCapacity);
        }

        [Fact]
        public async Task BufferCount_AfterPrefetch_ReturnsCorrectCount()
        {
            // Arrange
            var queue = new SharedQueue<int>(10);
            var strategy = new SimplePrefetchStrategy<int>(queue, 5);

            queue.Enqueue(1);
            queue.Enqueue(2);
            queue.Enqueue(3);

            // Act
            await strategy.PrefetchAsync(3, CancellationToken.None);

            // Assert
            Assert.Equal(3, strategy.BufferCount);
        }

        [Fact]
        public async Task BufferCount_AfterGetNext_Decreases()
        {
            // Arrange
            var queue = new SharedQueue<int>(10);
            var strategy = new SimplePrefetchStrategy<int>(queue, 5);

            queue.Enqueue(1);
            queue.Enqueue(2);
            queue.Enqueue(3);
            await strategy.PrefetchAsync(3, CancellationToken.None);

            // Act
            await strategy.GetNextAsync(CancellationToken.None);

            // Assert
            Assert.Equal(2, strategy.BufferCount);
        }

        [Fact]
        public async Task MultipleOperations_CompleteFlow_WorksCorrectly()
        {
            // Arrange
            var queue = new SharedQueue<int>(10);
            var strategy = new SimplePrefetchStrategy<int>(queue, 3);

            queue.Enqueue(1);
            queue.Enqueue(2);
            queue.Enqueue(3);
            queue.Enqueue(4);
            queue.Enqueue(5);

            // Act
            await strategy.PrefetchAsync(3, CancellationToken.None);
            var item1 = await strategy.GetNextAsync(CancellationToken.None);
            var item2 = await strategy.GetNextAsync(CancellationToken.None);
            var item3 = await strategy.GetNextAsync(CancellationToken.None);
            var item4 = await strategy.GetNextAsync(CancellationToken.None); // Will hit cache miss
            var item5 = await strategy.GetNextAsync(CancellationToken.None);

            // Assert
            Assert.Equal(1, item1);
            Assert.Equal(2, item2);
            Assert.Equal(3, item3);
            Assert.Equal(4, item4);
            Assert.Equal(5, item5);

            var stats = strategy.GetStatistics();
            Assert.Equal(2, stats.CacheHits); // item1, item2
            Assert.Equal(3, stats.CacheMisses); // item3 (buffer empty), item4, item5
        }
    }
}
