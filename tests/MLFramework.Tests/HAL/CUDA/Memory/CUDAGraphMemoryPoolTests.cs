using Microsoft.VisualStudio.TestTools.UnitTesting;
using MLFramework.HAL.CUDA;
using System;
using System.Threading.Tasks;

namespace MLFramework.Tests.HAL.CUDA.Memory;

/// <summary>
/// Unit tests for CUDAGraphMemoryPool
/// </summary>
[TestClass]
public class CUDAGraphMemoryPoolTests
{
    private CUDAGraphMemoryPool? _pool;

    [TestInitialize]
    public void Setup()
    {
        // Create a small pool for tests (16MB initial, 32MB max)
        _pool = new CUDAGraphMemoryPool(16 * 1024 * 1024, 32 * 1024 * 1024);
    }

    [TestCleanup]
    public void Cleanup()
    {
        _pool?.Dispose();
    }

    [TestMethod]
    public void CUDAGraphMemoryPool_Constructor_InitializesCorrectly()
    {
        // Act
        var pool = new CUDAGraphMemoryPool(1024 * 1024, 2 * 1024 * 1024);

        // Assert
        Assert.IsNotNull(pool);
        Assert.AreEqual(0, pool.BlockCount);
        Assert.IsTrue(pool.AllocatedBytes >= 0);
    }

    [TestMethod]
    public void CUDAGraphMemoryPool_Allocate_SmallBlock_ReturnsBlock()
    {
        // Skip this test if GPU is not available
        if (!CudaDevice.Available)
        {
            Assert.Inconclusive("GPU not available for this test");
            return;
        }

        // Act
        var block = _pool!.Allocate(1024);

        // Assert
        Assert.IsNotNull(block);
        Assert.IsNotNull(block.Ptr);
        Assert.AreNotEqual(IntPtr.Zero, block.Ptr);
        Assert.AreEqual(1024ul, block.Size);
        Assert.IsTrue(block.BlockId > 0);
    }

    [TestMethod]
    public void CUDAGraphMemoryPool_Allocate_LargeBlock_ReturnsBlock()
    {
        // Skip this test if GPU is not available
        if (!CudaDevice.Available)
        {
            Assert.Inconclusive("GPU not available for this test");
            return;
        }

        // Act
        var block = _pool!.Allocate(8 * 1024 * 1024); // 8MB

        // Assert
        Assert.IsNotNull(block);
        Assert.IsNotNull(block.Ptr);
        Assert.AreNotEqual(IntPtr.Zero, block.Ptr);
        Assert.AreEqual(8ul * 1024 * 1024, block.Size);
    }

    [TestMethod]
    public void CUDAGraphMemoryPool_Allocate_WithAlignment_ReturnsAlignedBlock()
    {
        // Skip this test if GPU is not available
        if (!CudaDevice.Available)
        {
            Assert.Inconclusive("GPU not available for this test");
            return;
        }

        // Act
        var block = _pool!.Allocate(1024, 512);

        // Assert
        Assert.IsNotNull(block);
        Assert.IsNotNull(block.Ptr);
        Assert.AreNotEqual(IntPtr.Zero, block.Ptr);
        // Check alignment (ptr % 512 == 0)
        Assert.AreEqual(0, block.Ptr.ToInt64() % 512);
    }

    [TestMethod]
    [ExpectedException(typeof(OutOfMemoryException))]
    public void CUDAGraphMemoryPool_Allocate_ExceedsCapacity_ThrowsException()
    {
        // Skip this test if GPU is not available
        if (!CudaDevice.Available)
        {
            Assert.Inconclusive("GPU not available for this test");
            return;
        }

        // Create a pool with small max capacity
        var pool = new CUDAGraphMemoryPool(1024, 2048);

        // Act - try to allocate more than max capacity
        pool.Allocate(4096);

        // Cleanup
        pool.Dispose();
    }

    [TestMethod]
    public void CUDAGraphMemoryPool_Allocate_IncrementsBlockCount()
    {
        // Skip this test if GPU is not available
        if (!CudaDevice.Available)
        {
            Assert.Inconclusive("GPU not available for this test");
            return;
        }

        // Act
        var initialCount = _pool!.BlockCount;
        _pool.Allocate(1024);
        var newCount = _pool.BlockCount;

        // Assert
        Assert.AreEqual(initialCount + 1, newCount);
    }

    [TestMethod]
    public void CUDAGraphMemoryPool_Allocate_IncrementsAllocatedBytes()
    {
        // Skip this test if GPU is not available
        if (!CudaDevice.Available)
        {
            Assert.Inconclusive("GPU not available for this test");
            return;
        }

        // Act
        var initialBytes = _pool!.AllocatedBytes;
        _pool.Allocate(1024);
        var newBytes = _pool.AllocatedBytes;

        // Assert
        Assert.AreEqual(initialBytes + 1024, newBytes);
    }

    [TestMethod]
    public void CUDAGraphMemoryPool_GetBlock_ExistingId_ReturnsBlock()
    {
        // Skip this test if GPU is not available
        if (!CudaDevice.Available)
        {
            Assert.Inconclusive("GPU not available for this test");
            return;
        }

        // Arrange
        var block = _pool!.Allocate(1024);
        var blockId = block.BlockId;

        // Act
        var retrievedBlock = _pool.GetBlock(blockId);

        // Assert
        Assert.IsNotNull(retrievedBlock);
        Assert.AreEqual(blockId, retrievedBlock.BlockId);
        Assert.AreEqual(block.Ptr, retrievedBlock.Ptr);
        Assert.AreEqual(block.Size, retrievedBlock.Size);
    }

    [TestMethod]
    public void CUDAGraphMemoryPool_GetBlock_NonExistingId_ReturnsNull()
    {
        // Act
        var block = _pool!.GetBlock(999999);

        // Assert
        Assert.IsNull(block);
    }

    [TestMethod]
    public void CUDAGraphMemoryPool_ReturnBlock_SetsBlockToNotInUse()
    {
        // Skip this test if GPU is not available
        if (!CudaDevice.Available)
        {
            Assert.Inconclusive("GPU not available for this test");
            return;
        }

        // Arrange
        var block = _pool!.Allocate(1024);
        var blockId = block.BlockId;

        // Act
        _pool.ReturnBlock(blockId);
        var retrievedBlock = _pool.GetBlock(blockId);

        // Assert
        Assert.IsNotNull(retrievedBlock);
        Assert.IsFalse(retrievedBlock.InUse);
    }

    [TestMethod]
    public void CUDAGraphMemoryPool_ReturnBlock_NonExistingId_DoesNotThrow()
    {
        // Act & Assert - should not throw
        _pool!.ReturnBlock(999999);
    }

    [TestMethod]
    public void CUDAGraphMemoryPool_Reset_SetsAllBlocksToNotInUse()
    {
        // Skip this test if GPU is not available
        if (!CudaDevice.Available)
        {
            Assert.Inconclusive("GPU not available for this test");
            return;
        }

        // Arrange
        var block1 = _pool!.Allocate(1024);
        var block2 = _pool.Allocate(2048);
        var block3 = _pool.Allocate(4096);

        // Act
        _pool.Reset();

        // Assert
        Assert.IsFalse(_pool.GetBlock(block1.BlockId)!.InUse);
        Assert.IsFalse(_pool.GetBlock(block2.BlockId)!.InUse);
        Assert.IsFalse(_pool.GetBlock(block3.BlockId)!.InUse);
    }

    [TestMethod]
    public void CUDAGraphMemoryPool_Reset_DoesNotFreeMemory()
    {
        // Skip this test if GPU is not available
        if (!CudaDevice.Available)
        {
            Assert.Inconclusive("GPU not available for this test");
            return;
        }

        // Arrange
        var block1 = _pool!.Allocate(1024);
        var block2 = _pool.Allocate(2048);
        var allocatedBytesBeforeReset = _pool.AllocatedBytes;

        // Act
        _pool.Reset();

        // Assert
        Assert.AreEqual(allocatedBytesBeforeReset, _pool.AllocatedBytes);
        Assert.IsNotNull(_pool.GetBlock(block1.BlockId));
        Assert.IsNotNull(_pool.GetBlock(block2.BlockId));
    }

    [TestMethod]
    public void CUDAGraphMemoryPool_Allocate_AfterReset_MaintainsFixedAddresses()
    {
        // Skip this test if GPU is not available
        if (!CudaDevice.Available)
        {
            Assert.Inconclusive("GPU not available for this test");
            return;
        }

        // Arrange
        var block1 = _pool!.Allocate(1024);
        var block2 = _pool.Allocate(2048);
        var ptr1Before = block1.Ptr;
        var ptr2Before = block2.Ptr;

        // Act
        _pool.Reset();
        var block1After = _pool.GetBlock(block1.BlockId);
        var block2After = _pool.GetBlock(block2.BlockId);

        // Assert
        Assert.IsNotNull(block1After);
        Assert.IsNotNull(block2After);
        Assert.AreEqual(ptr1Before, block1After.Ptr);
        Assert.AreEqual(ptr2Before, block2After.Ptr);
    }

    [TestMethod]
    public void CUDAGraphMemoryPool_Dispose_FreesAllMemory()
    {
        // Skip this test if GPU is not available
        if (!CudaDevice.Available)
        {
            Assert.Inconclusive("GPU not available for this test");
            return;
        }

        // Arrange
        var pool = new CUDAGraphMemoryPool(16 * 1024 * 1024, 32 * 1024 * 1024);
        pool.Allocate(1024);
        pool.Allocate(2048);

        // Act
        pool.Dispose();

        // Assert
        Assert.AreEqual(0, pool.BlockCount);
        Assert.AreEqual(0, pool.AllocatedBytes);
    }

    [TestMethod]
    [ExpectedException(typeof(ObjectDisposedException))]
    public void CUDAGraphMemoryPool_Allocate_AfterDispose_ThrowsException()
    {
        // Skip this test if GPU is not available
        if (!CudaDevice.Available)
        {
            Assert.Inconclusive("GPU not available for this test");
            return;
        }

        // Arrange
        var pool = new CUDAGraphMemoryPool(16 * 1024 * 1024, 32 * 1024 * 1024);
        pool.Dispose();

        // Act
        pool.Allocate(1024);
    }

    [TestMethod]
    [ExpectedException(typeof(ObjectDisposedException))]
    public void CUDAGraphMemoryPool_GetBlock_AfterDispose_ThrowsException()
    {
        // Arrange
        var pool = new CUDAGraphMemoryPool(16 * 1024 * 1024, 32 * 1024 * 1024);
        pool.Dispose();

        // Act
        pool.GetBlock(1);
    }

    [TestMethod]
    [ExpectedException(typeof(ObjectDisposedException))]
    public void CUDAGraphMemoryPool_Reset_AfterDispose_ThrowsException()
    {
        // Arrange
        var pool = new CUDAGraphMemoryPool(16 * 1024 * 1024, 32 * 1024 * 1024);
        pool.Dispose();

        // Act
        pool.Reset();
    }

    [TestMethod]
    public void CUDAGraphMemoryPool_ConcurrentAllocations_ThreadSafe()
    {
        // Skip this test if GPU is not available
        if (!CudaDevice.Available)
        {
            Assert.Inconclusive("GPU not available for this test");
            return;
        }

        // Arrange
        var pool = new CUDAGraphMemoryPool(16 * 1024 * 1024, 32 * 1024 * 1024);
        const int threadCount = 10;
        const int allocationsPerThread = 5;
        var tasks = new Task[threadCount];

        // Act
        for (int i = 0; i < threadCount; i++)
        {
            tasks[i] = Task.Run(() =>
            {
                for (int j = 0; j < allocationsPerThread; j++)
                {
                    var block = pool.Allocate(1024);
                    Thread.Sleep(1); // Small delay to increase chance of race conditions
                    pool.ReturnBlock(block.BlockId);
                }
            });
        }

        Task.WaitAll(tasks);

        // Assert
        Assert.AreEqual(threadCount * allocationsPerThread, pool.BlockCount);
        Assert.IsTrue(pool.AllocatedBytes > 0);

        // Cleanup
        pool.Dispose();
    }

    [TestMethod]
    public void CUDAGraphMemoryPool_Dispose_CalledMultipleTimes_DoesNotThrow()
    {
        // Act & Assert - should not throw
        _pool!.Dispose();
        _pool.Dispose();
    }

    [TestMethod]
    public void CUDAGraphMemoryPool_BlockReuse_AfterReset()
    {
        // Skip this test if GPU is not available
        if (!CudaDevice.Available)
        {
            Assert.Inconclusive("GPU not available for this test");
            return;
        }

        // Arrange
        var block1 = _pool!.Allocate(1024);
        var block2 = _pool.Allocate(2048);
        _pool.Reset();

        // Act - blocks should be reusable after reset
        var retrieved1 = _pool.GetBlock(block1.BlockId);
        var retrieved2 = _pool.GetBlock(block2.BlockId);

        // Assert
        Assert.IsNotNull(retrieved1);
        Assert.IsNotNull(retrieved2);
        Assert.IsFalse(retrieved1.InUse);
        Assert.IsFalse(retrieved2.InUse);
        Assert.AreEqual(block1.Ptr, retrieved1.Ptr);
        Assert.AreEqual(block2.Ptr, retrieved2.Ptr);
    }
}
