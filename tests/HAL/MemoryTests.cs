using NUnit.Framework;
using MLFramework.HAL;
using System.Runtime.InteropServices;

namespace MLFramework.Tests.HAL;

/// <summary>
/// Memory management tests for the HAL system
/// Tests allocator correctness, caching behavior, and memory leak detection
/// </summary>
[TestFixture]
public class MemoryTests
{
    [SetUp]
    public void Setup()
    {
        // Clear registry before each test to ensure clean state
        BackendRegistry.Clear();
    }

    [Test]
    public void SimpleAllocator_Allocate_WorksCorrectly()
    {
        var device = Device.CPU;
        var allocator = new SimpleAllocator(device);

        var buffer = allocator.Allocate(1024);

        Assert.IsNotNull(buffer);
        Assert.IsNotNull(buffer.Pointer);
        Assert.AreEqual(1024, buffer.Size);
        Assert.AreEqual(device, buffer.Device);
        Assert.IsTrue(buffer.IsValid);

        buffer.Dispose();
    }

    [Test]
    public void SimpleAllocator_Free_WorksCorrectly()
    {
        var device = Device.CPU;
        var allocator = new SimpleAllocator(device);

        var buffer = allocator.Allocate(1024);
        buffer.Dispose();

        // After disposing, buffer should be invalid
        Assert.IsFalse(buffer.IsValid);
    }

    [Test]
    public void SimpleAllocator_AllocatedSize_TrackedCorrectly()
    {
        var device = Device.CPU;
        var allocator = new SimpleAllocator(device);

        Assert.AreEqual(0, allocator.AllocatedSize);

        var buffer1 = allocator.Allocate(1024);
        Assert.AreEqual(1024, allocator.AllocatedSize);

        var buffer2 = allocator.Allocate(2048);
        Assert.AreEqual(3072, allocator.AllocatedSize);

        buffer1.Dispose();
        Assert.AreEqual(2048, allocator.AllocatedSize);

        buffer2.Dispose();
        Assert.AreEqual(0, allocator.AllocatedSize);
    }

    [Test]
    public void SimpleAllocator_CacheSize_IsAlwaysZero()
    {
        var device = Device.CPU;
        var allocator = new SimpleAllocator(device);

        var buffer = allocator.Allocate(1024);
        buffer.Dispose();

        Assert.AreEqual(0, allocator.CacheSize);
    }

    [Test]
    public void SimpleAllocator_EmptyCache_NoOp()
    {
        var device = Device.CPU;
        var allocator = new SimpleAllocator(device);

        var buffer = allocator.Allocate(1024);
        buffer.Dispose();

        Assert.DoesNotThrow(() => allocator.EmptyCache());
        Assert.AreEqual(0, allocator.CacheSize);
    }

    [Test]
    public void CachingAllocator_Allocate_WorksCorrectly()
    {
        var device = Device.CPU;
        var allocator = new CachingAllocator(device);

        var buffer = allocator.Allocate(1024);

        Assert.IsNotNull(buffer);
        Assert.IsNotNull(buffer.Pointer);
        Assert.AreNotEqual(IntPtr.Zero, buffer.Pointer);
        Assert.AreEqual(1024, buffer.Size);
        Assert.AreEqual(device, buffer.Device);
        Assert.IsTrue(buffer.IsValid);

        buffer.Dispose();
    }

    [Test]
    public void CachingAllocator_Free_WorksCorrectly()
    {
        var device = Device.CPU;
        var allocator = new CachingAllocator(device);

        var buffer = allocator.Allocate(1024);
        buffer.Dispose();

        Assert.IsFalse(buffer.IsValid);
    }

    [Test]
    public void CachingAllocator_MemoryReuse()
    {
        var device = Device.CPU;
        var allocator = new CachingAllocator(device);

        var buffer1 = allocator.Allocate(1024);
        var ptr1 = buffer1.Pointer;
        buffer1.Dispose();

        var buffer2 = allocator.Allocate(1024);
        var ptr2 = buffer2.Pointer;
        buffer2.Dispose();

        // The same memory should be reused (aligned size)
        Assert.AreEqual(ptr1, ptr2);

        allocator.EmptyCache();
        allocator.Dispose();
    }

    [Test]
    public void CachingAllocator_BlockSplitting()
    {
        var device = Device.CPU;
        var allocator = new CachingAllocator(device);

        // Allocate a large block
        var largeBlock = allocator.Allocate(2048);
        largeBlock.Dispose();

        // Now allocate two smaller blocks from the freed large block
        var smallBlock1 = allocator.Allocate(1024);
        var smallBlock2 = allocator.Allocate(1024);

        Assert.AreNotEqual(smallBlock1.Pointer, smallBlock2.Pointer);
        Assert.AreEqual(1024, smallBlock1.Size);
        Assert.AreEqual(1024, smallBlock2.Size);

        smallBlock1.Dispose();
        smallBlock2.Dispose();

        allocator.EmptyCache();
        allocator.Dispose();
    }

    [Test]
    public void CachingAllocator_BlockMerging()
    {
        var device = Device.CPU;
        var allocator = new CachingAllocator(device);

        // Allocate two adjacent blocks
        var block1 = allocator.Allocate(1024);
        var block2 = allocator.Allocate(1024);

        block1.Dispose();
        block2.Dispose();

        // These should be merged into a 2048-byte block
        var mergedBlock = allocator.Allocate(2048);
        Assert.IsNotNull(mergedBlock);

        mergedBlock.Dispose();

        allocator.EmptyCache();
        allocator.Dispose();
    }

    [Test]
    public void CachingAllocator_CacheSize_TrackedCorrectly()
    {
        var device = Device.CPU;
        var allocator = new CachingAllocator(device);

        Assert.AreEqual(0, allocator.CacheSize);

        var buffer = allocator.Allocate(1024);
        Assert.AreNotEqual(0, allocator.CacheSize); // Some cache should exist

        buffer.Dispose();
        Assert.AreNotEqual(0, allocator.CacheSize); // Cache should retain freed memory

        allocator.EmptyCache();
        Assert.AreEqual(0, allocator.CacheSize); // Cache should be empty

        allocator.Dispose();
    }

    [Test]
    public void CachingAllocator_AllocatedSize_TrackedCorrectly()
    {
        var device = Device.CPU;
        var allocator = new CachingAllocator(device);

        Assert.AreEqual(0, allocator.AllocatedSize);

        var buffer1 = allocator.Allocate(1024);
        Assert.Greater(allocator.AllocatedSize, 0);

        var buffer2 = allocator.Allocate(2048);
        Assert.Greater(allocator.AllocatedSize, 1024);

        buffer1.Dispose();
        Assert.AreEqual(allocator.AllocatedSize, buffer2.Size);

        buffer2.Dispose();
        Assert.AreEqual(0, allocator.AllocatedSize);

        allocator.EmptyCache();
        allocator.Dispose();
    }

    [Test]
    public void CachingAllocator_EmptyCache_ReleasesMemory()
    {
        var device = Device.CPU;
        var allocator = new CachingAllocator(device);

        // Allocate and free several buffers to build up cache
        for (int i = 0; i < 10; i++)
        {
            var buffer = allocator.Allocate(1024);
            buffer.Dispose();
        }

        var cacheSizeBefore = allocator.CacheSize;
        Assert.Greater(cacheSizeBefore, 0);

        allocator.EmptyCache();

        var cacheSizeAfter = allocator.CacheSize;
        Assert.Less(cacheSizeAfter, cacheSizeBefore);
        Assert.AreEqual(0, cacheSizeAfter);

        allocator.Dispose();
    }

    [Test]
    public void CachingAllocator_MultipleSizes_HandledCorrectly()
    {
        var device = Device.CPU;
        var allocator = new CachingAllocator(device);

        // Allocate buffers of various sizes
        var buffers = new IMemoryBuffer[5];
        var sizes = new[] { 512, 1024, 2048, 4096, 8192 };

        for (int i = 0; i < buffers.Length; i++)
        {
            buffers[i] = allocator.Allocate(sizes[i]);
        }

        // Verify all buffers have correct sizes
        for (int i = 0; i < buffers.Length; i++)
        {
            Assert.AreEqual(sizes[i], buffers[i].Size);
        }

        // Free all buffers
        foreach (var buffer in buffers)
        {
            buffer.Dispose();
        }

        allocator.EmptyCache();
        allocator.Dispose();
    }

    [Test]
    public void CachingAllocator_Dispose_CleansUp()
    {
        var device = Device.CPU;
        var allocator = new CachingAllocator(device);

        var buffer = allocator.Allocate(1024);
        buffer.Dispose();

        Assert.Greater(allocator.CacheSize, 0);

        // Dispose should clean up cached memory
        allocator.Dispose();
        // Note: We can't check CacheSize after dispose as the object is disposed
    }

    [Test]
    public void CachingAllocator_Alignment_SizesAreAligned()
    {
        var device = Device.CPU;
        var allocator = new CachingAllocator(device);

        // Allocate unaligned sizes (should be aligned to 16 bytes)
        var buffer1 = allocator.Allocate(100);
        var buffer2 = allocator.Allocate(101);

        Assert.AreEqual(112, buffer1.Size); // 100 aligned to 112 (next 16-byte boundary)
        Assert.AreEqual(112, buffer2.Size); // 101 aligned to 112

        buffer1.Dispose();
        buffer2.Dispose();

        allocator.EmptyCache();
        allocator.Dispose();
    }

    [Test]
    public void NoMemoryLeaks_StressTest()
    {
        var device = Device.CPU;
        var allocator = new CachingAllocator(device);

        // Perform many allocate/free cycles
        for (int i = 0; i < 10000; i++)
        {
            var buffer = allocator.Allocate(1024);
            buffer.Dispose();
        }

        allocator.EmptyCache();

        // Check that allocated size is 0 (no leaked allocations)
        Assert.AreEqual(0, allocator.AllocatedSize);

        allocator.Dispose();
    }

    [Test]
    public void NoMemoryLeaks_MixedSizes_StressTest()
    {
        var device = Device.CPU;
        var allocator = new CachingAllocator(device);

        var random = new Random(42);

        // Perform many allocate/free cycles with random sizes
        for (int i = 0; i < 5000; i++)
        {
            var size = random.Next(512, 8192);
            var buffer = allocator.Allocate(size);
            buffer.Dispose();
        }

        allocator.EmptyCache();

        // Check that allocated size is 0
        Assert.AreEqual(0, allocator.AllocatedSize);

        allocator.Dispose();
    }

    [Test]
    public void SimpleAllocator_InvalidSize_ThrowsException()
    {
        var device = Device.CPU;
        var allocator = new SimpleAllocator(device);

        Assert.Throws<ArgumentException>(() => allocator.Allocate(0));
        Assert.Throws<ArgumentException>(() => allocator.Allocate(-100));

        allocator.Dispose();
    }

    [Test]
    public void SimpleAllocator_NullBuffer_ThrowsExceptionOnFree()
    {
        var device = Device.CPU;
        var allocator = new SimpleAllocator(device);

        Assert.Throws<ArgumentNullException>(() => allocator.Free(null!));

        allocator.Dispose();
    }

    [Test]
    public void CachingAllocator_NullBuffer_ThrowsExceptionOnFree()
    {
        var device = Device.CPU;
        var allocator = new CachingAllocator(device);

        Assert.Throws<ArgumentNullException>(() => allocator.Free(null!));

        allocator.Dispose();
    }
}
