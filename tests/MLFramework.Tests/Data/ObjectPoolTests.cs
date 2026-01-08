using System;
using System.Threading.Tasks;
using Xunit;
using MLFramework.Data;

namespace MLFramework.Tests.Data
{
    /// <summary>
    /// Unit tests for ObjectPool.
    /// </summary>
    public class ObjectPoolTests
    {
        [Fact]
        public void Constructor_WithValidParameters_InitializesPool()
        {
            // Arrange & Act
            var pool = new ObjectPool<int>(() => 42);

            // Assert
            Assert.NotNull(pool);
            Assert.Equal(0, pool.AvailableCount);
            Assert.Equal(0, pool.TotalCount);
            Assert.Equal(100, pool.MaxSize);
        }

        [Fact]
        public void Constructor_WithInitialSize_PreAllocatesItems()
        {
            // Arrange & Act
            var pool = new ObjectPool<int>(() => 42, initialSize: 5);

            // Assert
            Assert.Equal(5, pool.AvailableCount);
            Assert.Equal(5, pool.TotalCount);
        }

        [Fact]
        public void Constructor_WithNullFactory_ThrowsArgumentNullException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentNullException>(() => new ObjectPool<int>(null!));
        }

        [Fact]
        public void Constructor_WithInvalidMaxSize_ThrowsArgumentOutOfRangeException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() => new ObjectPool<int>(() => 42, maxSize: 0));
            Assert.Throws<ArgumentOutOfRangeException>(() => new ObjectPool<int>(() => 42, maxSize: -1));
        }

        [Fact]
        public void Rent_WhenPoolEmpty_CreatesNewItem()
        {
            // Arrange
            var pool = new ObjectPool<int>(() => 42);

            // Act
            var item = pool.Rent();

            // Assert
            Assert.Equal(42, item);
            Assert.Equal(0, pool.AvailableCount);
            Assert.Equal(1, pool.TotalCount);
        }

        [Fact]
        public void Rent_WhenPoolHasItems_ReturnsAvailableItem()
        {
            // Arrange
            var pool = new ObjectPool<int>(() => 42, initialSize: 2);
            Assert.Equal(2, pool.AvailableCount);

            // Act
            var item = pool.Rent();

            // Assert
            Assert.Equal(42, item);
            Assert.Equal(1, pool.AvailableCount);
            Assert.Equal(2, pool.TotalCount);
        }

        [Fact]
        public void Return_WhenPoolHasSpace_ReturnsItemToPool()
        {
            // Arrange
            var pool = new ObjectPool<int>(() => 42);
            var item = pool.Rent();

            // Act
            pool.Return(item);

            // Assert
            Assert.Equal(1, pool.AvailableCount);
            Assert.Equal(1, pool.TotalCount);
        }

        [Fact]
        public void Return_WhenResetActionProvided_CallsReset()
        {
            // Arrange
            var resetCalled = false;
            var pool = new ObjectPool<int>(
                () => 42,
                reset: _ => resetCalled = true);
            var item = pool.Rent();

            // Act
            pool.Return(item);

            // Assert
            Assert.True(resetCalled);
        }

        [Fact]
        public void Return_WhenPoolFull_DiscardsItem()
        {
            // Arrange
            var pool = new ObjectPool<int>(() => 42, maxSize: 2);
            for (int i = 0; i < 2; i++)
            {
                var item = pool.Rent();
                pool.Return(item);
            }
            Assert.Equal(2, pool.AvailableCount);
            var stats = pool.GetStatistics();
            var initialDiscardCount = stats.DiscardCount;

            // Act
            var extraItem = pool.Rent();
            pool.Return(extraItem);

            // Assert
            Assert.Equal(2, pool.AvailableCount);
            Assert.Equal(3, pool.TotalCount);
            Assert.Equal(initialDiscardCount + 1, pool.GetStatistics().DiscardCount);
        }

        [Fact]
        public void Clear_RemovesAllAvailableItems()
        {
            // Arrange
            var pool = new ObjectPool<int>(() => 42, initialSize: 5);

            // Act
            pool.Clear();

            // Assert
            Assert.Equal(0, pool.AvailableCount);
            Assert.Equal(5, pool.TotalCount); // Total count remains unchanged
        }

        [Fact]
        public void Statistics_TrackRentCount()
        {
            // Arrange
            var pool = new ObjectPool<int>(() => 42, initialSize: 2);

            // Act
            pool.Rent();
            pool.Rent();
            pool.Rent(); // Creates new item

            // Assert
            var stats = pool.GetStatistics();
            Assert.Equal(3, stats.RentCount);
        }

        [Fact]
        public void Statistics_TrackReturnCount()
        {
            // Arrange
            var pool = new ObjectPool<int>(() => 42);
            var item1 = pool.Rent();
            var item2 = pool.Rent();

            // Act
            pool.Return(item1);
            pool.Return(item2);

            // Assert
            var stats = pool.GetStatistics();
            Assert.Equal(2, stats.ReturnCount);
        }

        [Fact]
        public void Statistics_TrackMissCount()
        {
            // Arrange
            var pool = new ObjectPool<int>(() => 42, initialSize: 2);

            // Act
            pool.Rent();
            pool.Rent();
            pool.Rent(); // Miss - creates new item

            // Assert
            var stats = pool.GetStatistics();
            Assert.Equal(1, stats.MissCount);
        }

        [Fact]
        public void Statistics_HitRate_IsCorrect()
        {
            // Arrange
            var pool = new ObjectPool<int>(() => 42, initialSize: 2);

            // Act
            pool.Rent(); // Hit
            pool.Rent(); // Hit
            pool.Rent(); // Miss

            // Assert
            var stats = pool.GetStatistics();
            Assert.Equal(2.0 / 3.0, stats.HitRate, 4); // 4 decimal precision
        }

        [Fact]
        public void ResetStatistics_ResetsAllCounters()
        {
            // Arrange
            var pool = new ObjectPool<int>(() => 42, initialSize: 2);
            pool.Rent();
            pool.Return(pool.Rent());

            // Act
            pool.ResetStatistics();

            // Assert
            var stats = pool.GetStatistics();
            Assert.Equal(0, stats.RentCount);
            Assert.Equal(0, stats.ReturnCount);
            Assert.Equal(0, stats.MissCount);
            Assert.Equal(0, stats.DiscardCount);
        }

        [Fact]
        public async Task ConcurrentRentAndReturn_ThreadSafe()
        {
            // Arrange
            var pool = new ObjectPool<int>(() => 42, maxSize: 1000);
            var tasks = new Task[100];

            // Act
            for (int i = 0; i < 100; i++)
            {
                tasks[i] = Task.Run(() =>
                {
                    for (int j = 0; j < 100; j++)
                    {
                        var item = pool.Rent();
                        Task.Delay(1).Wait();
                        pool.Return(item);
                    }
                });
            }

            await Task.WhenAll(tasks);

            // Assert
            Assert.Equal(100, pool.AvailableCount); // 100 items returned to pool
            var stats = pool.GetStatistics();
            Assert.Equal(10000, stats.RentCount);
            Assert.Equal(10000, stats.ReturnCount);
        }

        [Fact]
        public void Dispose_ThrowsOnRentAfterDispose()
        {
            // Arrange
            var pool = new ObjectPool<int>(() => 42);
            pool.Dispose();

            // Act & Assert
            Assert.Throws<ObjectDisposedException>(() => pool.Rent());
        }

        [Fact]
        public void Dispose_ThrowsOnReturnAfterDispose()
        {
            // Arrange
            var pool = new ObjectPool<int>(() => 42);
            var item = pool.Rent();
            pool.Dispose();

            // Act & Assert
            Assert.Throws<ObjectDisposedException>(() => pool.Return(item));
        }

        [Fact]
        public void RentAfterClear_CreatesNewItems()
        {
            // Arrange
            var pool = new ObjectPool<int>(() => 42, initialSize: 5);
            pool.Clear();
            Assert.Equal(0, pool.AvailableCount);
            Assert.Equal(5, pool.TotalCount);

            // Act
            var item = pool.Rent();

            // Assert
            Assert.NotNull(item);
            Assert.Equal(6, pool.TotalCount); // New item created
        }
    }
}
