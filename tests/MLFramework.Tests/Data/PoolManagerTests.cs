using System;
using System.Threading.Tasks;
using Xunit;
using MLFramework.Data;

namespace MLFramework.Tests.Data
{
    /// <summary>
    /// Unit tests for PoolManager.
    /// </summary>
    public class PoolManagerTests : IDisposable
    {
        private PoolManager _manager;

        public PoolManagerTests()
        {
            _manager = new PoolManager();
        }

        public void Dispose()
        {
            _manager.DisposeAll();
        }

        [Fact]
        public void GetPool_WithExistingKey_ReturnsPool()
        {
            // Arrange
            var pool = _manager.CreatePool<int>("test-pool", () => 42);

            // Act
            var retrievedPool = _manager.GetPool<int>("test-pool");

            // Assert
            Assert.Same(pool, retrievedPool);
        }

        [Fact]
        public void GetPool_WithNonExistentKey_ThrowsKeyNotFoundException()
        {
            // Arrange, Act & Assert
            Assert.Throws<KeyNotFoundException>(() => _manager.GetPool<int>("non-existent"));
        }

        [Fact]
        public void GetPool_WithInvalidKey_ThrowsArgumentException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentException>(() => _manager.GetPool<int>(null!));
            Assert.Throws<ArgumentException>(() => _manager.GetPool<int>(""));
            Assert.Throws<ArgumentException>(() => _manager.GetPool<int>("   "));
        }

        [Fact]
        public void GetOrCreatePool_WithNonExistentKey_CreatesNewPool()
        {
            // Arrange
            var initialPoolCount = _manager.GetStatistics().PoolCount;

            // Act
            var pool = _manager.GetOrCreatePool<int>("new-pool", () => 42);

            // Assert
            Assert.NotNull(pool);
            Assert.Equal(initialPoolCount + 1, _manager.GetStatistics().PoolCount);
        }

        [Fact]
        public void GetOrCreatePool_WithExistingKey_ReturnsExistingPool()
        {
            // Arrange
            var pool1 = _manager.CreatePool<int>("test-pool", () => 42);
            var initialPoolCount = _manager.GetStatistics().PoolCount;

            // Act
            var pool2 = _manager.GetOrCreatePool<int>("test-pool", () => 99);

            // Assert
            Assert.Same(pool1, pool2);
            Assert.Equal(initialPoolCount, _manager.GetStatistics().PoolCount);
        }

        [Fact]
        public void CreatePool_WithValidParameters_CreatesPool()
        {
            // Arrange
            var initialPoolCount = _manager.GetStatistics().PoolCount;

            // Act
            var pool = _manager.CreatePool<int>("test-pool", () => 42, initialSize: 5);

            // Assert
            Assert.NotNull(pool);
            Assert.Equal(5, pool.AvailableCount);
            Assert.Equal(initialPoolCount + 1, _manager.GetStatistics().PoolCount);
        }

        [Fact]
        public void CreatePool_WithExistingKey_ReplacesPool()
        {
            // Arrange
            var pool1 = _manager.CreatePool<int>("test-pool", () => 42, initialSize: 5);

            // Act
            var pool2 = _manager.CreatePool<int>("test-pool", () => 99, initialSize: 10);

            // Assert
            Assert.NotSame(pool1, pool2);
            Assert.Equal(10, pool2.AvailableCount);
        }

        [Fact]
        public void CreatePool_WithInvalidKey_ThrowsArgumentException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentException>(() => _manager.CreatePool<int>(null!, () => 42));
            Assert.Throws<ArgumentException>(() => _manager.CreatePool<int>("", () => 42));
        }

        [Fact]
        public void RegisterPool_WithValidParameters_RegistersPool()
        {
            // Arrange
            var pool = new ObjectPool<int>(() => 42);
            var initialPoolCount = _manager.GetStatistics().PoolCount;

            // Act
            _manager.RegisterPool<int>("test-pool", pool);

            // Assert
            Assert.Same(pool, _manager.GetPool<int>("test-pool"));
            Assert.Equal(initialPoolCount + 1, _manager.GetStatistics().PoolCount);
        }

        [Fact]
        public void RegisterPool_WithExistingKey_ReplacesPool()
        {
            // Arrange
            var pool1 = _manager.CreatePool<int>("test-pool", () => 42);
            var pool2 = new ObjectPool<int>(() => 99);

            // Act
            _manager.RegisterPool<int>("test-pool", pool2);

            // Assert
            Assert.Same(pool2, _manager.GetPool<int>("test-pool"));
        }

        [Fact]
        public void HasPool_WithExistingKey_ReturnsTrue()
        {
            // Arrange
            _manager.CreatePool<int>("test-pool", () => 42);

            // Act
            var hasPool = _manager.HasPool("test-pool");

            // Assert
            Assert.True(hasPool);
        }

        [Fact]
        public void HasPool_WithNonExistentKey_ReturnsFalse()
        {
            // Act
            var hasPool = _manager.HasPool("non-existent");

            // Assert
            Assert.False(hasPool);
        }

        [Fact]
        public void HasPool_WithInvalidKey_ReturnsFalse()
        {
            // Act
            var hasPool1 = _manager.HasPool(null);
            var hasPool2 = _manager.HasPool("");
            var hasPool3 = _manager.HasPool("   ");

            // Assert
            Assert.False(hasPool1);
            Assert.False(hasPool2);
            Assert.False(hasPool3);
        }

        [Fact]
        public void RemovePool_WithExistingKey_RemovesAndDisposesPool()
        {
            // Arrange
            _manager.CreatePool<int>("test-pool", () => 42);
            var initialPoolCount = _manager.GetStatistics().PoolCount;

            // Act
            var removed = _manager.RemovePool("test-pool");

            // Assert
            Assert.True(removed);
            Assert.False(_manager.HasPool("test-pool"));
            Assert.Equal(initialPoolCount - 1, _manager.GetStatistics().PoolCount);
        }

        [Fact]
        public void RemovePool_WithNonExistentKey_ReturnsFalse()
        {
            // Act
            var removed = _manager.RemovePool("non-existent");

            // Assert
            Assert.False(removed);
        }

        [Fact]
        public void ClearAll_ClearsAllManagedPools()
        {
            // Arrange
            var pool1 = _manager.CreatePool<int>("pool1", () => 42, initialSize: 5);
            var pool2 = _manager.CreatePool<int>("pool2", () => 99, initialSize: 10);
            Assert.Equal(15, _manager.GetStatistics().TotalAvailableItems);

            // Act
            _manager.ClearAll();

            // Assert
            Assert.Equal(0, _manager.GetStatistics().TotalAvailableItems);
            Assert.True(_manager.HasPool("pool1"));
            Assert.True(_manager.HasPool("pool2"));
        }

        [Fact]
        public void DisposeAll_DisposesAllManagedPools()
        {
            // Arrange
            _manager.CreatePool<int>("pool1", () => 42);
            _manager.CreatePool<int>("pool2", () => 99);
            Assert.Equal(2, _manager.GetStatistics().PoolCount);

            // Act
            _manager.DisposeAll();

            // Assert
            Assert.Equal(0, _manager.GetStatistics().PoolCount);
            Assert.False(_manager.HasPool("pool1"));
            Assert.False(_manager.HasPool("pool2"));
        }

        [Fact]
        public void GetStatistics_ReturnsCorrectStatistics()
        {
            // Arrange
            var pool1 = _manager.CreatePool<int>("pool1", () => 42, initialSize: 5);
            var pool2 = _manager.CreatePool<int>("pool2", () => 99, initialSize: 10);

            // Act
            var stats = _manager.GetStatistics();

            // Assert
            Assert.Equal(2, stats.PoolCount);
            Assert.Equal(15, stats.TotalAvailableItems);
            Assert.Equal(15, stats.TotalCreatedItems);
            Assert.True(stats.PoolSizes.ContainsKey("pool1"));
            Assert.True(stats.PoolSizes.ContainsKey("pool2"));
            Assert.Equal(5, stats.PoolSizes["pool1"]);
            Assert.Equal(10, stats.PoolSizes["pool2"]);
        }

        [Fact]
        public void GetPool_WithWrongType_ThrowsInvalidCastException()
        {
            // Arrange
            _manager.CreatePool<int>("test-pool", () => 42);

            // Act & Assert
            Assert.Throws<InvalidCastException>(() => _manager.GetPool<string>("test-pool"));
        }

        [Fact]
        public void GetOrCreatePool_WithCustomReset_ResetsOnReturn()
        {
            // Arrange
            var resetCalled = false;
            var pool = _manager.GetOrCreatePool<int>(
                "test-pool",
                () => 42,
                reset: _ => resetCalled = true);
            var item = pool.Rent();

            // Act
            pool.Return(item);

            // Assert
            Assert.True(resetCalled);
        }

        [Fact]
        public void GetOrCreatePool_WithCustomMaxSize_RespectsMaxSize()
        {
            // Arrange
            var pool = _manager.GetOrCreatePool<int>("test-pool", () => 42, maxSize: 5);

            // Act
            for (int i = 0; i < 10; i++)
            {
                pool.Return(pool.Rent());
            }

            // Assert
            Assert.Equal(5, pool.AvailableCount);
        }

        [Fact]
        public async Task ConcurrentAccess_ThreadSafe()
        {
            // Arrange
            var tasks = new Task[10];

            // Act
            for (int i = 0; i < 10; i++)
            {
                tasks[i] = Task.Run(() =>
                {
                    for (int j = 0; j < 100; j++)
                    {
                        var pool = _manager.GetOrCreatePool<int>($"pool-{j % 3}", () => j);
                        var item = pool.Rent();
                        Task.Delay(1).Wait();
                        pool.Return(item);
                    }
                });
            }

            await Task.WhenAll(tasks);

            // Assert
            var stats = _manager.GetStatistics();
            Assert.Equal(3, stats.PoolCount);
        }

        [Fact]
        public void Instance_ReturnsSingleton()
        {
            // Arrange & Act
            var instance1 = PoolManager.Instance;
            var instance2 = PoolManager.Instance;

            // Assert
            Assert.Same(instance1, instance2);
        }
    }
}
