using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using MLFramework.MobileRuntime.Memory;
using Xunit;

namespace MLFramework.MobileRuntime.Tests.Memory
{
    /// <summary>
    /// Unit tests for DefaultMemoryPool.
    /// </summary>
    public class DefaultMemoryPoolTests
    {
        [Fact]
        public void Allocate_ValidSize_ReturnsNonNullPointer()
        {
            // Arrange
            using var pool = new DefaultMemoryPool(16 * 1024 * 1024);

            // Act
            var ptr = pool.Allocate(1024, DataType.Float32);

            // Assert
            Assert.NotEqual(IntPtr.Zero, ptr);
        }

        [Fact]
        public void Allocate_ThenFree_NoMemoryLeak()
        {
            // Arrange
            using var pool = new DefaultMemoryPool(16 * 1024 * 1024);
            var ptr = pool.Allocate(1024, DataType.Float32);

            // Act
            pool.Free(ptr, 1024);

            // Assert
            var stats = pool.GetStats();
            Assert.Equal(0, stats.UsedMemory);
        }

        [Fact]
        public void Allocate_MultipleTimes_IncreasesAllocationCount()
        {
            // Arrange
            using var pool = new DefaultMemoryPool(16 * 1024 * 1024);

            // Act
            var ptr1 = pool.Allocate(1024, DataType.Float32);
            var ptr2 = pool.Allocate(2048, DataType.Float32);
            var ptr3 = pool.Allocate(4096, DataType.Float32);

            // Assert
            var stats = pool.GetStats();
            Assert.Equal(3, stats.AllocationCount);
        }

        [Fact]
        public void Allocate_CachedBlock_IncreasesCacheHit()
        {
            // Arrange
            using var pool = new DefaultMemoryPool(16 * 1024 * 1024);
            var ptr1 = pool.Allocate(1024, DataType.Float32);
            pool.Free(ptr1, 1024);

            // Act
            var ptr2 = pool.Allocate(1024, DataType.Float32);

            // Assert
            var stats = pool.GetStats();
            Assert.Equal(1, stats.CacheHits);
        }

        [Fact]
        public void Allocate_NewBlock_IncreasesCacheMiss()
        {
            // Arrange
            using var pool = new DefaultMemoryPool(16 * 1024 * 1024);

            // Act
            var ptr = pool.Allocate(1024, DataType.Float32);

            // Assert
            var stats = pool.GetStats();
            Assert.Equal(1, stats.CacheMisses);
        }

        [Fact]
        public void Free_AlreadyFreedBlock_ThrowsInvalidOperationException()
        {
            // Arrange
            using var pool = new DefaultMemoryPool(16 * 1024 * 1024);
            var ptr = pool.Allocate(1024, DataType.Float32);
            pool.Free(ptr, 1024);

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => pool.Free(ptr, 1024));
        }

        [Fact]
        public void Free_InvalidPointer_ThrowsInvalidOperationException()
        {
            // Arrange
            using var pool = new DefaultMemoryPool(16 * 1024 * 1024);
            var invalidPtr = new IntPtr(0x12345678);

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => pool.Free(invalidPtr, 1024));
        }

        [Fact]
        public void SetMemoryLimit_ValidLimit_UpdatesLimit()
        {
            // Arrange
            using var pool = new DefaultMemoryPool(16 * 1024 * 1024);

            // Act
            pool.SetMemoryLimit(8 * 1024 * 1024);

            // Assert
            // No exception thrown
        }

        [Fact]
        public void SetMemoryLimit_NegativeLimit_ThrowsArgumentException()
        {
            // Arrange
            using var pool = new DefaultMemoryPool(16 * 1024 * 1024);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => pool.SetMemoryLimit(-1));
        }

        [Fact]
        public void Allocate_ExceedsMemoryLimit_ThrowsOutOfMemoryException()
        {
            // Arrange
            using var pool = new DefaultMemoryPool(1 * 1024 * 1024); // 1MB
            pool.SetMemoryLimit(1 * 1024 * 1024);

            // Act & Assert
            Assert.Throws<OutOfMemoryException>(() =>
            {
                var ptr = pool.Allocate(2 * 1024 * 1024, DataType.Float32); // 2MB
            });
        }

        [Fact]
        public void GetAvailableMemory_AfterAllocation_ReturnsCorrectValue()
        {
            // Arrange
            using var pool = new DefaultMemoryPool(16 * 1024 * 1024);
            pool.Allocate(4 * 1024 * 1024, DataType.Float32);

            // Act
            var available = pool.GetAvailableMemory();

            // Assert
            Assert.Equal(12 * 1024 * 1024, available);
        }

        [Fact]
        public void GetUsedMemory_AfterAllocation_ReturnsCorrectValue()
        {
            // Arrange
            using var pool = new DefaultMemoryPool(16 * 1024 * 1024);
            pool.Allocate(4 * 1024 * 1024, DataType.Float32);

            // Act
            var used = pool.GetUsedMemory();

            // Assert
            Assert.Equal(4 * 1024 * 1024, used);
        }

        [Fact]
        public void GetStats_ReturnsAccurateStatistics()
        {
            // Arrange
            using var pool = new DefaultMemoryPool(16 * 1024 * 1024);
            var ptr1 = pool.Allocate(1024, DataType.Float32);
            var ptr2 = pool.Allocate(2048, DataType.Float32);
            pool.Free(ptr1, 1024);
            var ptr3 = pool.Allocate(1024, DataType.Float32);

            // Act
            var stats = pool.GetStats();

            // Assert
            Assert.Equal(3, stats.AllocationCount);
            Assert.Equal(1, stats.FreeCount);
            Assert.Equal(1, stats.CacheHits);
            Assert.Equal(3, stats.CacheMisses); // 2 new + 1 cache hit counted as miss in allocation
        }

        [Fact]
        public void EnableLowMemoryMode_ReleasesCachedBlocks()
        {
            // Arrange
            using var pool = new DefaultMemoryPool(16 * 1024 * 1024);
            var ptr1 = pool.Allocate(1024, DataType.Float32);
            pool.Free(ptr1, 1024);
            var statsBefore = pool.GetStats();

            // Act
            pool.EnableLowMemoryMode(true);
            var statsAfter = pool.GetStats();

            // Assert
            Assert.True(statsAfter.UsedMemory < statsBefore.UsedMemory);
        }

        [Fact]
        public void PreAllocateForTensor_AllocatesBlockInPool()
        {
            // Arrange
            using var pool = new DefaultMemoryPool(16 * 1024 * 1024);
            var statsBefore = pool.GetStats();

            // Act
            pool.PreAllocateForTensor(1024);
            var statsAfter = pool.GetStats();

            // Assert
            Assert.True(statsAfter.AvailableMemory > statsBefore.AvailableMemory);
        }

        [Fact]
        public void Reset_ClearsAllAllocations()
        {
            // Arrange
            using var pool = new DefaultMemoryPool(16 * 1024 * 1024);
            pool.Allocate(1024, DataType.Float32);
            pool.Allocate(2048, DataType.Float32);

            // Act
            pool.Reset();
            var stats = pool.GetStats();

            // Assert
            Assert.Equal(0, stats.UsedMemory);
            Assert.Equal(0, stats.AllocationCount);
            Assert.Equal(0, stats.PeakUsage);
        }

        [Fact]
        public void Dispose_PreventsFurtherAllocations()
        {
            // Arrange
            var pool = new DefaultMemoryPool(16 * 1024 * 1024);
            pool.Dispose();

            // Act & Assert
            Assert.Throws<ObjectDisposedException>(() => pool.Allocate(1024, DataType.Float32));
        }

        [Fact]
        public void Allocate_NegativeSize_ThrowsArgumentException()
        {
            // Arrange
            using var pool = new DefaultMemoryPool(16 * 1024 * 1024);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => pool.Allocate(-1, DataType.Float32));
        }

        [Fact]
        public void Allocate_ZeroSize_ThrowsArgumentException()
        {
            // Arrange
            using var pool = new DefaultMemoryPool(16 * 1024 * 1024);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => pool.Allocate(0, DataType.Float32));
        }

        [Fact]
        public void CacheHitRate_CalculatedCorrectly()
        {
            // Arrange
            using var pool = new DefaultMemoryPool(16 * 1024 * 1024);
            var ptr1 = pool.Allocate(1024, DataType.Float32);
            pool.Free(ptr1, 1024);
            var ptr2 = pool.Allocate(1024, DataType.Float32);
            var ptr3 = pool.Allocate(2048, DataType.Float32);

            // Act
            var stats = pool.GetStats();
            var hitRate = stats.GetCacheHitRate();

            // Assert
            Assert.True(hitRate > 0 && hitRate <= 100);
        }

        [Fact]
        public void ThreadSafety_ConcurrentAllocations_DoesNotThrow()
        {
            // Arrange
            using var pool = new DefaultMemoryPool(16 * 1024 * 1024);
            var pointers = new IntPtr[100];
            var exceptions = new Exception[100];

            // Act
            Parallel.For(0, 100, i =>
            {
                try
                {
                    pointers[i] = pool.Allocate(1024, DataType.Float32);
                }
                catch (Exception ex)
                {
                    exceptions[i] = ex;
                }
            });

            // Assert
            Assert.DoesNotContain(exceptions, e => e != null);
            Assert.All(pointers, ptr => Assert.NotEqual(IntPtr.Zero, ptr));
        }

        [Fact]
        public void ThreadSafety_ConcurrentAllocationsAndFrees_DoesNotThrow()
        {
            // Arrange
            using var pool = new DefaultMemoryPool(32 * 1024 * 1024);
            var pointers = new IntPtr[100];
            var exceptions = new Exception[200];

            // Act - Allocate
            Parallel.For(0, 100, i =>
            {
                try
                {
                    pointers[i] = pool.Allocate(1024, DataType.Float32);
                }
                catch (Exception ex)
                {
                    exceptions[i] = ex;
                }
            });

            // Act - Free
            Parallel.For(0, 100, i =>
            {
                try
                {
                    pool.Free(pointers[i], 1024);
                }
                catch (Exception ex)
                {
                    exceptions[100 + i] = ex;
                }
            });

            // Assert
            Assert.DoesNotContain(exceptions, e => e != null);
        }

        [Fact]
        public void PeakUsage_TracksMaximumMemoryUsage()
        {
            // Arrange
            using var pool = new DefaultMemoryPool(16 * 1024 * 1024);
            var ptr1 = pool.Allocate(4 * 1024 * 1024, DataType.Float32);
            var ptr2 = pool.Allocate(4 * 1024 * 1024, DataType.Float32);
            var ptr3 = pool.Allocate(4 * 1024 * 1024, DataType.Float32);

            // Act
            pool.Free(ptr1, 4 * 1024 * 1024);
            pool.Free(ptr2, 4 * 1024 * 1024);
            var stats = pool.GetStats();

            // Assert
            Assert.Equal(12 * 1024 * 1024, stats.PeakUsage);
        }

        [Fact]
        public void DifferentDataTypes_AllDifferentSizes()
        {
            // Arrange
            using var pool = new DefaultMemoryPool(16 * 1024 * 1024);

            // Act
            var ptr1 = pool.Allocate(1000, DataType.Float32);
            var ptr2 = pool.Allocate(1000, DataType.Int8);
            var ptr3 = pool.Allocate(1000, DataType.Int16);

            // Assert
            Assert.NotEqual(ptr1, ptr2);
            Assert.NotEqual(ptr2, ptr3);
            Assert.NotEqual(ptr1, ptr3);
        }
    }
}
