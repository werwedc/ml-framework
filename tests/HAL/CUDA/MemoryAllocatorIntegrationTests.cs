using MLFramework.HAL.CUDA;
using System;
using System.Threading.Tasks;
using Xunit;

namespace MLFramework.HAL.CUDA.Tests
{
    /// <summary>
    /// Unit tests for the graph-compatible memory allocator.
    /// </summary>
    public class GraphCompatibleMemoryAllocatorTests
    {
        [Fact]
        public void Constructor_InitializesCorrectly()
        {
            // Arrange & Act
            var allocator = new GraphCompatibleMemoryAllocator();

            // Assert
            Assert.NotNull(allocator);
            Assert.False(allocator.IsGraphMode);
            Assert.Null(allocator.GraphPool);
        }

        [Fact]
        public void SetGraphMode_WithNullPool_ThrowsInvalidOperationException()
        {
            // Arrange
            var allocator = new GraphCompatibleMemoryAllocator();

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => allocator.SetGraphMode(true));
        }

        [Fact]
        public void SetGraphMode_WithValidPool_Succeeds()
        {
            // Arrange
            var allocator = new GraphCompatibleMemoryAllocator();
            var pool = new CUDAGraphMemoryPool(initialCapacity: 16 * 1024 * 1024); // 16MB

            // Act
            allocator.GraphPool = pool;
            allocator.SetGraphMode(true);

            // Assert
            Assert.True(allocator.IsGraphMode);
        }

        [Fact]
        public void SetGraphMode_CanBeDisabled()
        {
            // Arrange
            var allocator = new GraphCompatibleMemoryAllocator();
            var pool = new CUDAGraphMemoryPool(initialCapacity: 16 * 1024 * 1024);
            allocator.GraphPool = pool;
            allocator.SetGraphMode(true);

            // Act
            allocator.SetGraphMode(false);

            // Assert
            Assert.False(allocator.IsGraphMode);
        }

        [Fact]
        public void Allocate_InRegularMode_Works()
        {
            // Arrange
            var allocator = new GraphCompatibleMemoryAllocator();

            // Act
            var ptr = allocator.Allocate(1024);

            // Assert
            Assert.NotEqual(IntPtr.Zero, ptr);

            // Cleanup
            allocator.Free(ptr);
            allocator.Dispose();
        }

        [Fact]
        public void Allocate_InGraphMode_Works()
        {
            // Arrange
            var allocator = new GraphCompatibleMemoryAllocator();
            var pool = new CUDAGraphMemoryPool(initialCapacity: 16 * 1024 * 1024);
            allocator.GraphPool = pool;
            allocator.SetGraphMode(true);

            // Act
            var ptr = allocator.Allocate(1024);

            // Assert
            Assert.NotEqual(IntPtr.Zero, ptr);
            Assert.True(allocator.IsGraphMode);

            // Cleanup
            allocator.Free(ptr);
            allocator.Dispose();
            pool.Dispose();
        }

        [Fact]
        public void GetStats_InRegularMode_ReturnsCorrectStats()
        {
            // Arrange
            var allocator = new GraphCompatibleMemoryAllocator();
            var ptr = allocator.Allocate(1024);

            // Act
            var stats = allocator.GetStats();

            // Assert
            Assert.NotNull(stats);
            Assert.False(stats.IsGraphMode);
            Assert.True(stats.TotalAllocated >= 0);

            // Cleanup
            allocator.Free(ptr);
            allocator.Dispose();
        }

        [Fact]
        public void GetStats_InGraphMode_ReturnsCorrectStats()
        {
            // Arrange
            var allocator = new GraphCompatibleMemoryAllocator();
            var pool = new CUDAGraphMemoryPool(initialCapacity: 16 * 1024 * 1024);
            allocator.GraphPool = pool;
            allocator.SetGraphMode(true);
            var ptr = allocator.Allocate(1024);

            // Act
            var stats = allocator.GetStats();

            // Assert
            Assert.NotNull(stats);
            Assert.True(stats.IsGraphMode);
            Assert.True(stats.BlockCount > 0);

            // Cleanup
            allocator.Free(ptr);
            allocator.Dispose();
            pool.Dispose();
        }

        [Fact]
        public void Dispose_MultipleCalls_DoesNotThrow()
        {
            // Arrange
            var allocator = new GraphCompatibleMemoryAllocator();

            // Act & Assert
            allocator.Dispose();
            allocator.Dispose(); // Should not throw
        }
    }

    /// <summary>
    /// Unit tests for the CUDAGraphMemoryPool.
    /// </summary>
    public class CUDAGraphMemoryPoolTests
    {
        [Fact]
        public void Constructor_InitializesCorrectly()
        {
            // Arrange & Act
            var pool = new CUDAGraphMemoryPool(initialCapacity: 16 * 1024 * 1024);

            // Assert
            Assert.NotNull(pool);
            Assert.Equal(0, pool.BlockCount);
            Assert.Equal(0, pool.AllocatedBytes);

            // Cleanup
            pool.Dispose();
        }

        [Fact]
        public void Allocate_IncrementsBlockCount()
        {
            // Arrange
            var pool = new CUDAGraphMemoryPool(initialCapacity: 16 * 1024 * 1024);

            // Act
            var block = pool.Allocate(1024);

            // Assert
            Assert.Equal(1, pool.BlockCount);
            Assert.True(pool.AllocatedBytes >= 1024);
            Assert.NotNull(block);
            Assert.NotEqual(IntPtr.Zero, block.Ptr);

            // Cleanup
            pool.Dispose();
        }

        [Fact]
        public void GetBlock_ReturnsCorrectBlock()
        {
            // Arrange
            var pool = new CUDAGraphMemoryPool(initialCapacity: 16 * 1024 * 1024);
            var block = pool.Allocate(1024);

            // Act
            var retrievedBlock = pool.GetBlock(block.BlockId);

            // Assert
            Assert.NotNull(retrievedBlock);
            Assert.Equal(block.BlockId, retrievedBlock.BlockId);
            Assert.Equal(block.Ptr, retrievedBlock.Ptr);
            Assert.Equal(block.Size, retrievedBlock.Size);

            // Cleanup
            pool.Dispose();
        }

        [Fact]
        public void ReturnBlock_MarksBlockAsAvailable()
        {
            // Arrange
            var pool = new CUDAGraphMemoryPool(initialCapacity: 16 * 1024 * 1024);
            var block = pool.Allocate(1024);

            // Act
            pool.ReturnBlock(block.BlockId);
            var retrievedBlock = pool.GetBlock(block.BlockId);

            // Assert
            Assert.NotNull(retrievedBlock);
            Assert.False(retrievedBlock.InUse);

            // Cleanup
            pool.Dispose();
        }

        [Fact]
        public void Reset_MarksAllBlocksAsAvailable()
        {
            // Arrange
            var pool = new CUDAGraphMemoryPool(initialCapacity: 16 * 1024 * 1024);
            var block1 = pool.Allocate(1024);
            var block2 = pool.Allocate(2048);
            var block3 = pool.Allocate(4096);

            // Act
            pool.Reset();
            var retrievedBlock1 = pool.GetBlock(block1.BlockId);
            var retrievedBlock2 = pool.GetBlock(block2.BlockId);
            var retrievedBlock3 = pool.GetBlock(block3.BlockId);

            // Assert
            Assert.NotNull(retrievedBlock1);
            Assert.NotNull(retrievedBlock2);
            Assert.NotNull(retrievedBlock3);
            Assert.False(retrievedBlock1.InUse);
            Assert.False(retrievedBlock2.InUse);
            Assert.False(retrievedBlock3.InUse);

            // Cleanup
            pool.Dispose();
        }

        [Fact]
        public void Dispose_MultipleCalls_DoesNotThrow()
        {
            // Arrange
            var pool = new CUDAGraphMemoryPool(initialCapacity: 16 * 1024 * 1024);

            // Act & Assert
            pool.Dispose();
            pool.Dispose(); // Should not throw
        }

        [Fact]
        public void Allocate_AfterDispose_ThrowsObjectDisposedException()
        {
            // Arrange
            var pool = new CUDAGraphMemoryPool(initialCapacity: 16 * 1024 * 1024);
            pool.Dispose();

            // Act & Assert
            Assert.Throws<ObjectDisposedException>(() => pool.Allocate(1024));
        }

        [Fact]
        public void GetBlock_AfterDispose_ThrowsObjectDisposedException()
        {
            // Arrange
            var pool = new CUDAGraphMemoryPool(initialCapacity: 16 * 1024 * 1024);
            pool.Dispose();

            // Act & Assert
            Assert.Throws<ObjectDisposedException>(() => pool.GetBlock(1));
        }

        [Fact]
        public void BlockId_IsUnique()
        {
            // Arrange
            var pool = new CUDAGraphMemoryPool(initialCapacity: 16 * 1024 * 1024);

            // Act
            var block1 = pool.Allocate(1024);
            var block2 = pool.Allocate(1024);
            var block3 = pool.Allocate(1024);

            // Assert
            Assert.NotEqual(block1.BlockId, block2.BlockId);
            Assert.NotEqual(block2.BlockId, block3.BlockId);
            Assert.NotEqual(block1.BlockId, block3.BlockId);

            // Cleanup
            pool.Dispose();
        }
    }

    /// <summary>
    /// Unit tests for the memory manager extensions.
    /// </summary>
    public class MemoryManagerExtensionsTests
    {
        [Fact]
        public void ConfigureForGraph_SetsGraphPoolAndMode()
        {
            // Arrange
            var allocator = new GraphCompatibleMemoryAllocator();
            var pool = new CUDAGraphMemoryPool(initialCapacity: 16 * 1024 * 1024);

            // Act
            allocator.ConfigureForGraph(pool);

            // Assert
            Assert.Same(pool, allocator.GraphPool);
            Assert.True(allocator.IsGraphMode);

            // Cleanup
            allocator.Dispose();
            pool.Dispose();
        }

        [Fact]
        public void EnableGraphMode_EnableThenDisable_Works()
        {
            // Arrange
            var allocator = new GraphCompatibleMemoryAllocator();
            var pool = new CUDAGraphMemoryPool(initialCapacity: 16 * 1024 * 1024);
            allocator.GraphPool = pool;

            // Act
            allocator.EnableGraphMode();
            var enabled = allocator.IsGraphMode;
            allocator.DisableGraphMode();
            var disabled = allocator.IsGraphMode;

            // Assert
            Assert.True(enabled);
            Assert.False(disabled);

            // Cleanup
            allocator.Dispose();
            pool.Dispose();
        }

        [Fact]
        public void WithGraphMode_ExecutesActionAndRestoresState()
        {
            // Arrange
            var allocator = new GraphCompatibleMemoryAllocator();
            var pool = new CUDAGraphMemoryPool(initialCapacity: 16 * 1024 * 1024);
            allocator.GraphPool = pool;
            var wasGraphModeBefore = allocator.IsGraphMode;

            // Act
            var executed = false;
            allocator.WithGraphMode(() =>
            {
                executed = true;
                Assert.True(allocator.IsGraphMode);
            });
            var wasGraphModeAfter = allocator.IsGraphMode;

            // Assert
            Assert.True(executed);
            Assert.Equal(wasGraphModeBefore, wasGraphModeAfter);

            // Cleanup
            allocator.Dispose();
            pool.Dispose();
        }

        [Fact]
        public void WithGraphMode_WithReturnValue_Works()
        {
            // Arrange
            var allocator = new GraphCompatibleMemoryAllocator();
            var pool = new CUDAGraphMemoryPool(initialCapacity: 16 * 1024 * 1024);
            allocator.GraphPool = pool;

            // Act
            var result = allocator.WithGraphMode(() => 42);

            // Assert
            Assert.Equal(42, result);

            // Cleanup
            allocator.Dispose();
            pool.Dispose();
        }

        [Fact]
        public void WithGraphMode_RestoresStateOnException()
        {
            // Arrange
            var allocator = new GraphCompatibleMemoryAllocator();
            var pool = new CUDAGraphMemoryPool(initialCapacity: 16 * 1024 * 1024);
            allocator.GraphPool = pool;
            var wasGraphModeBefore = allocator.IsGraphMode;

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => allocator.WithGraphMode(() =>
            {
                throw new InvalidOperationException();
            }));

            var wasGraphModeAfter = allocator.IsGraphMode;
            Assert.Equal(wasGraphModeBefore, wasGraphModeAfter);

            // Cleanup
            allocator.Dispose();
            pool.Dispose();
        }
    }
}
