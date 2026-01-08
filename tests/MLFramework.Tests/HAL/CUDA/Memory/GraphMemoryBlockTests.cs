using Microsoft.VisualStudio.TestTools.UnitTesting;
using MLFramework.HAL.CUDA;
using System;

namespace MLFramework.Tests.HAL.CUDA.Memory;

/// <summary>
/// Unit tests for GraphMemoryBlock
/// </summary>
[TestClass]
public class GraphMemoryBlockTests
{
    [TestMethod]
    public void GraphMemoryBlock_Constructor_InitializesCorrectly()
    {
        // Arrange
        var ptr = new IntPtr(0x12345678);
        var size = 1024ul;

        // Act
        var block = CreateGraphMemoryBlock(ptr, size);

        // Assert
        Assert.IsNotNull(block);
        Assert.IsTrue(block.BlockId > 0);
        Assert.AreEqual(ptr, block.Ptr);
        Assert.AreEqual(size, block.Size);
        Assert.IsTrue(block.InUse);
    }

    [TestMethod]
    public void GraphMemoryBlock_BlockId_UniqueForEachInstance()
    {
        // Arrange
        var ptr1 = new IntPtr(0x12345678);
        var ptr2 = new IntPtr(0x87654321);

        // Act
        var block1 = CreateGraphMemoryBlock(ptr1, 1024);
        var block2 = CreateGraphMemoryBlock(ptr2, 2048);

        // Assert
        Assert.AreNotEqual(block1.BlockId, block2.BlockId);
    }

    [TestMethod]
    public void GraphMemoryBlock_BlockId_IncrementsMonotonically()
    {
        // Arrange
        var ptr = new IntPtr(0x12345678);

        // Act
        var block1 = CreateGraphMemoryBlock(ptr, 1024);
        var block2 = CreateGraphMemoryBlock(ptr, 2048);
        var block3 = CreateGraphMemoryBlock(ptr, 4096);

        // Assert
        Assert.IsTrue(block2.BlockId > block1.BlockId);
        Assert.IsTrue(block3.BlockId > block2.BlockId);
    }

    [TestMethod]
    public void GraphMemoryBlock_InUse_InitiallyTrue()
    {
        // Arrange
        var ptr = new IntPtr(0x12345678);

        // Act
        var block = CreateGraphMemoryBlock(ptr, 1024);

        // Assert
        Assert.IsTrue(block.InUse);
    }

    [TestMethod]
    public void GraphMemoryBlock_InUse_CanBeModified()
    {
        // Arrange
        var ptr = new IntPtr(0x12345678);
        var block = CreateGraphMemoryBlock(ptr, 1024);

        // Act
        block.InUse = false;

        // Assert
        Assert.IsFalse(block.InUse);
    }

    [TestMethod]
    public void GraphMemoryBlock_Dispose_DoesNotThrow()
    {
        // Arrange
        var ptr = new IntPtr(0x12345678);
        var block = CreateGraphMemoryBlock(ptr, 1024);

        // Act & Assert - should not throw
        block.Dispose();
    }

    [TestMethod]
    public void GraphMemoryBlock_Dispose_CalledMultipleTimes_DoesNotThrow()
    {
        // Arrange
        var ptr = new IntPtr(0x12345678);
        var block = CreateGraphMemoryBlock(ptr, 1024);

        // Act & Assert - should not throw
        block.Dispose();
        block.Dispose();
        block.Dispose();
    }

    [TestMethod]
    public void GraphMemoryBlock_AfterDispose_CanAccessProperties()
    {
        // Arrange
        var ptr = new IntPtr(0x12345678);
        var size = 1024ul;
        var block = CreateGraphMemoryBlock(ptr, size);

        // Act
        block.Dispose();

        // Assert - should still be able to read properties
        Assert.IsTrue(block.BlockId > 0);
        Assert.AreEqual(ptr, block.Ptr);
        Assert.AreEqual(size, block.Size);
    }

    [TestMethod]
    public void GraphMemoryBlock_ConcurrentCreation_ThreadSafe()
    {
        // Arrange
        const int threadCount = 10;
        var tasks = new Task[threadCount];
        var blockIds = new ulong[threadCount];

        // Act
        for (int i = 0; i < threadCount; i++)
        {
            int index = i;
            tasks[i] = Task.Run(() =>
            {
                var ptr = new IntPtr(0x12345678 + index);
                var block = CreateGraphMemoryBlock(ptr, 1024);
                blockIds[index] = block.BlockId;
            });
        }

        Task.WaitAll(tasks);

        // Assert
        for (int i = 0; i < threadCount; i++)
        {
            for (int j = i + 1; j < threadCount; j++)
            {
                Assert.AreNotEqual(blockIds[i], blockIds[j],
                    $"Block IDs {i} and {j} are not unique");
            }
        }
    }

    // Helper method to create GraphMemoryBlock since the constructor is internal
    private GraphMemoryBlock CreateGraphMemoryBlock(IntPtr ptr, ulong size)
    {
        // Use reflection to access internal constructor
        var constructor = typeof(GraphMemoryBlock).GetConstructor(
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance,
            null,
            new[] { typeof(IntPtr), typeof(ulong) },
            null);

        if (constructor == null)
        {
            throw new InvalidOperationException("Could not find internal constructor for GraphMemoryBlock");
        }

        return (GraphMemoryBlock)constructor.Invoke(new object[] { ptr, size });
    }
}
