using System;
using System.Threading.Tasks;
using Xunit;
using MLFramework.Data;

namespace MLFramework.Tests.Data
{
    /// <summary>
    /// Unit tests for ArrayPool.
    /// </summary>
    public class ArrayPoolTests
    {
        [Fact]
        public void Constructor_WithValidParameters_InitializesPool()
        {
            // Arrange & Act
            var pool = new ArrayPool<int>(10);

            // Assert
            Assert.NotNull(pool);
            Assert.Equal(0, pool.AvailableCount);
            Assert.Equal(0, pool.TotalCount);
            Assert.Equal(10, pool.ArrayLength);
            Assert.Equal(50, pool.MaxSize);
        }

        [Fact]
        public void Constructor_WithInitialSize_PreAllocatesArrays()
        {
            // Arrange & Act
            var pool = new ArrayPool<int>(10, initialSize: 5);

            // Assert
            Assert.Equal(5, pool.AvailableCount);
            Assert.Equal(5, pool.TotalCount);
        }

        [Fact]
        public void Constructor_WithInvalidArrayLength_ThrowsArgumentOutOfRangeException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() => new ArrayPool<int>(0));
            Assert.Throws<ArgumentOutOfRangeException>(() => new ArrayPool<int>(-1));
        }

        [Fact]
        public void Rent_WhenPoolEmpty_CreatesNewArray()
        {
            // Arrange
            var pool = new ArrayPool<int>(10);

            // Act
            var array = pool.Rent();

            // Assert
            Assert.NotNull(array);
            Assert.Equal(10, array.Length);
            Assert.Equal(0, pool.AvailableCount);
            Assert.Equal(1, pool.TotalCount);
        }

        [Fact]
        public void Rent_WhenPoolHasArrays_ReturnsAvailableArray()
        {
            // Arrange
            var pool = new ArrayPool<int>(10, initialSize: 2);

            // Act
            var array = pool.Rent();

            // Assert
            Assert.NotNull(array);
            Assert.Equal(10, array.Length);
            Assert.Equal(1, pool.AvailableCount);
        }

        [Fact]
        public void Return_WithNullArray_ThrowsArgumentNullException()
        {
            // Arrange
            var pool = new ArrayPool<int>(10);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => pool.Return(null!));
        }

        [Fact]
        public void Return_WithWrongLengthArray_ThrowsArgumentException()
        {
            // Arrange
            var pool = new ArrayPool<int>(10);
            var array = new int[5];

            // Act & Assert
            Assert.Throws<ArgumentException>(() => pool.Return(array));
        }

        [Fact]
        public void Return_WithCorrectLengthArray_ReturnsToPool()
        {
            // Arrange
            var pool = new ArrayPool<int>(10);
            var array = pool.Rent();

            // Act
            pool.Return(array);

            // Assert
            Assert.Equal(1, pool.AvailableCount);
        }

        [Fact]
        public void Return_WithClearOnSet_ClearsArray()
        {
            // Arrange
            var pool = new ArrayPool<int>(10, clearOnReturn: true);
            var array = pool.Rent();
            for (int i = 0; i < array.Length; i++)
            {
                array[i] = 42;
            }

            // Act
            pool.Return(array);

            // Assert
            // Array should be cleared (all zeros)
            Assert.All(array, value => Assert.Equal(0, value));
        }

        [Fact]
        public void Return_WithoutClearOnSet_DoesNotClearArray()
        {
            // Arrange
            var pool = new ArrayPool<int>(10, clearOnReturn: false);
            var array = pool.Rent();
            for (int i = 0; i < array.Length; i++)
            {
                array[i] = 42;
            }

            // Act
            pool.Return(array);
            var returnedArray = pool.Rent();

            // Assert
            // Array should retain its values
            Assert.All(returnedArray, value => Assert.Equal(42, value));
        }

        [Fact]
        public void Clear_RemovesAllAvailableArrays()
        {
            // Arrange
            var pool = new ArrayPool<int>(10, initialSize: 5);

            // Act
            pool.Clear();

            // Assert
            Assert.Equal(0, pool.AvailableCount);
            Assert.Equal(5, pool.TotalCount);
        }

        [Fact]
        public void Statistics_TrackRentCount()
        {
            // Arrange
            var pool = new ArrayPool<int>(10, initialSize: 2);

            // Act
            pool.Rent();
            pool.Rent();
            pool.Rent();

            // Assert
            var stats = pool.GetStatistics();
            Assert.Equal(3, stats.RentCount);
        }

        [Fact]
        public void Statistics_TrackReturnCount()
        {
            // Arrange
            var pool = new ArrayPool<int>(10);
            var array1 = pool.Rent();
            var array2 = pool.Rent();

            // Act
            pool.Return(array1);
            pool.Return(array2);

            // Assert
            var stats = pool.GetStatistics();
            Assert.Equal(2, stats.ReturnCount);
        }

        [Fact]
        public async Task ConcurrentRentAndReturn_ThreadSafe()
        {
            // Arrange
            var pool = new ArrayPool<int>(100, maxSize: 1000);
            var tasks = new Task[100];

            // Act
            for (int i = 0; i < 100; i++)
            {
                tasks[i] = Task.Run(() =>
                {
                    for (int j = 0; j < 100; j++)
                    {
                        var array = pool.Rent();
                        array[0] = j; // Modify array
                        Task.Delay(1).Wait();
                        pool.Return(array);
                    }
                });
            }

            await Task.WhenAll(tasks);

            // Assert
            Assert.Equal(100, pool.AvailableCount);
            var stats = pool.GetStatistics();
            Assert.Equal(10000, stats.RentCount);
            Assert.Equal(10000, stats.ReturnCount);
        }

        [Fact]
        public void Dispose_ThrowsOnRentAfterDispose()
        {
            // Arrange
            var pool = new ArrayPool<int>(10);
            pool.Dispose();

            // Act & Assert
            Assert.Throws<ObjectDisposedException>(() => pool.Rent());
        }

        [Fact]
        public void Dispose_ThrowsOnReturnAfterDispose()
        {
            // Arrange
            var pool = new ArrayPool<int>(10);
            var array = pool.Rent();
            pool.Dispose();

            // Act & Assert
            Assert.Throws<ObjectDisposedException>(() => pool.Return(array));
        }

        [Fact]
        public void ResetStatistics_ResetsAllCounters()
        {
            // Arrange
            var pool = new ArrayPool<int>(10);
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
        public void PoolReducesAllocations()
        {
            // Arrange
            var pool = new ArrayPool<int>(100, initialSize: 10, maxSize: 10);
            var initialTotalCount = pool.TotalCount;

            // Act
            // Rent and return many times without exceeding pool size
            for (int i = 0; i < 100; i++)
            {
                var array = pool.Rent();
                pool.Return(array);
            }

            // Assert
            // Should not create any new arrays beyond the initial 10
            Assert.Equal(10, pool.TotalCount);
            Assert.Equal(initialTotalCount, pool.TotalCount);
        }
    }
}
