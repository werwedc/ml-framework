using System;
using MLFramework.MobileRuntime.Memory;
using Xunit;

namespace MLFramework.MobileRuntime.Tests.Memory
{
    /// <summary>
    /// Unit tests for PreallocatedMemoryPool.
    /// </summary>
    public class PreallocatedMemoryPoolTests
    {
        [Fact]
        public void Allocate_ValidSize_ReturnsNonNullPointer()
        {
            // Arrange
            using var pool = new PreallocatedMemoryPool(16 * 1024 * 1024);

            // Act
            var ptr = pool.Allocate(1024, DataType.Float32);

            // Assert
            Assert.NotEqual(IntPtr.Zero, ptr);
        }

        [Fact]
        public void Allocate_SequentialAllocation_ReturnsContiguousMemory()
        {
            // Arrange
            using var pool = new PreallocatedMemoryPool(16 * 1024 * 1024);

            // Act
            var ptr1 = pool.Allocate(1024, DataType.Float32);
            var ptr2 = pool.Allocate(1024, DataType.Float32);

            // Assert
            Assert.True(ptr2.ToInt64() > ptr1.ToInt64());
        }

        [Fact]
        public void Free_MarksBlockAsFree()
        {
            // Arrange
            using var pool = new PreallocatedMemoryPool(16 * 1024 * 1024);
            var ptr = pool.Allocate(1024, DataType.Float32);

            // Act
            pool.Free(ptr, 1024);
            var stats = pool.GetStats();

            // Assert
            Assert.Equal(0, stats.UsedMemory);
        }

        [Fact]
        public void Free_AlreadyFreedBlock_ThrowsInvalidOperationException()
        {
            // Arrange
            using var pool = new PreallocatedMemoryPool(16 * 1024 * 1024);
            var ptr = pool.Allocate(1024, DataType.Float32);
            pool.Free(ptr, 1024);

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => pool.Free(ptr, 1024));
        }

        [Fact]
        public void Free_InvalidPointer_ThrowsInvalidOperationException()
        {
            // Arrange
            using var pool = new PreallocatedMemoryPool(16 * 1024 * 1024);
            var invalidPtr = new IntPtr(0x12345678);

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => pool.Free(invalidPtr, 1024));
        }

        [Fact]
        public void Allocate_ExceedsPoolSize_ThrowsOutOfMemoryException()
        {
            // Arrange
            using var pool = new PreallocatedMemoryPool(1 * 1024 * 1024);

            // Act & Assert
            Assert.Throws<OutOfMemoryException>(() =>
            {
                var ptr = pool.Allocate(2 * 1024 * 1024, DataType.Float32);
            });
        }

        [Fact]
        public void GetAvailableMemory_ReturnsRemainingSpace()
        {
            // Arrange
            using var pool = new PreallocatedMemoryPool(16 * 1024 * 1024);
            pool.Allocate(4 * 1024 * 1024, DataType.Float32);

            // Act
            var available = pool.GetAvailableMemory();

            // Assert
            Assert.True(available > 0 && available < 16 * 1024 * 1024);
        }

        [Fact]
        public void GetUsedMemory_ReturnsUsedSpace()
        {
            // Arrange
            using var pool = new PreallocatedMemoryPool(16 * 1024 * 1024);
            pool.Allocate(4 * 1024 * 1024, DataType.Float32);

            // Act
            var used = pool.GetUsedMemory();

            // Assert
            Assert.True(used >= 4 * 1024 * 1024);
        }

        [Fact]
        public void GetStats_ReturnsAccurateStatistics()
        {
            // Arrange
            using var pool = new PreallocatedMemoryPool(16 * 1024 * 1024);
            var ptr1 = pool.Allocate(1024, DataType.Float32);
            var ptr2 = pool.Allocate(2048, DataType.Float32);
            pool.Free(ptr1, 1024);

            // Act
            var stats = pool.GetStats();

            // Assert
            Assert.Equal(2, stats.AllocationCount);
            Assert.Equal(1, stats.FreeCount);
            Assert.Equal(16 * 1024 * 1024, stats.TotalMemory);
        }

        [Fact]
        public void GetStats_NoCacheHits()
        {
            // Arrange
            using var pool = new PreallocatedMemoryPool(16 * 1024 * 1024);

            // Act
            var stats = pool.GetStats();

            // Assert
            Assert.Equal(0, stats.CacheHits);
        }

        [Fact]
        public void PreAllocateForTensor_DoesNothing()
        {
            // Arrange
            using var pool = new PreallocatedMemoryPool(16 * 1024 * 1024);
            var statsBefore = pool.GetStats();

            // Act
            pool.PreAllocateForTensor(1024);
            var statsAfter = pool.GetStats();

            // Assert
            Assert.Equal(statsBefore.AllocationCount, statsAfter.AllocationCount);
        }

        [Fact]
        public void Reset_ClearsAllAllocations()
        {
            // Arrange
            using var pool = new PreallocatedMemoryPool(16 * 1024 * 1024);
            pool.Allocate(1024, DataType.Float32);
            pool.Allocate(2048, DataType.Float32);

            // Act
            pool.Reset();
            var stats = pool.GetStats();

            // Assert
            Assert.Equal(0, stats.UsedMemory);
            Assert.Equal(0, stats.AllocationCount);
        }

        [Fact]
        public void Dispose_PreventsFurtherAllocations()
        {
            // Arrange
            var pool = new PreallocatedMemoryPool(16 * 1024 * 1024);
            pool.Dispose();

            // Act & Assert
            Assert.Throws<ObjectDisposedException>(() => pool.Allocate(1024, DataType.Float32));
        }

        [Fact]
        public void Allocate_NegativeSize_ThrowsArgumentException()
        {
            // Arrange
            using var pool = new PreallocatedMemoryPool(16 * 1024 * 1024);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => pool.Allocate(-1, DataType.Float32));
        }

        [Fact]
        public void Allocate_ZeroSize_ThrowsArgumentException()
        {
            // Arrange
            using var pool = new PreallocatedMemoryPool(16 * 1024 * 1024);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => pool.Allocate(0, DataType.Float32));
        }

        [Fact]
        public void SetMemoryLimit_LargerThanTotalSize_ThrowsInvalidOperationException()
        {
            // Arrange
            using var pool = new PreallocatedMemoryPool(16 * 1024 * 1024);

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() =>
                pool.SetMemoryLimit(32 * 1024 * 1024));
        }

        [Fact]
        public void SetMemoryLimit_ValidSize_DoesNotThrow()
        {
            // Arrange
            using var pool = new PreallocatedMemoryPool(16 * 1024 * 1024);

            // Act & Assert
            pool.SetMemoryLimit(8 * 1024 * 1024); // No exception
        }

        [Fact]
        public void Constructor_NegativeSize_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => new PreallocatedMemoryPool(-1));
        }

        [Fact]
        public void Constructor_ZeroSize_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => new PreallocatedMemoryPool(0));
        }

        [Fact]
        public void Allocate_AlignedTo16Bytes()
        {
            // Arrange
            using var pool = new PreallocatedMemoryPool(16 * 1024 * 1024);

            // Act
            var ptr1 = pool.Allocate(100, DataType.Float32);
            var ptr2 = pool.Allocate(1, DataType.Float32);

            // Assert
            Assert.Equal(0, (ptr2.ToInt64() - ptr1.ToInt64()) % 16);
        }
    }
}
