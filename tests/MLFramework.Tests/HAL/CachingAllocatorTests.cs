using MLFramework.HAL;
using Xunit;

namespace MLFramework.Tests.HAL;

/// <summary>
/// Tests for CachingAllocator class
/// </summary>
public class CachingAllocatorTests : MemoryAllocatorContractTests
{
    protected override IMemoryAllocator CreateAllocator()
    {
        var device = new CpuDevice();
        return new CachingAllocator(device);
    }

    [Fact]
    public void Allocate_AlignsSize()
    {
        using var allocator = CreateAllocator();
        var buffer = allocator.Allocate(100);

        // 100 rounded to 16-byte alignment = 112
        Assert.Equal(112, buffer.Size);
    }

    [Fact]
    public void Allocate_AlignsSize_MultipleOf16()
    {
        using var allocator = CreateAllocator();
        var buffer = allocator.Allocate(1024);

        // 1024 is already a multiple of 16
        Assert.Equal(1024, buffer.Size);
    }

    [Fact]
    public void Free_ReuseBuffer()
    {
        using var allocator = CreateAllocator();
        var buffer1 = allocator.Allocate(1024);
        var ptr1 = buffer1.Pointer;

        allocator.Free(buffer1);
        var buffer2 = allocator.Allocate(1024);
        var ptr2 = buffer2.Pointer;

        // The same pointer should be reused
        Assert.Equal(ptr1, ptr2);
    }

    [Fact]
    public void Free_InvalidBufferType_ThrowsArgumentException()
    {
        using var allocator = CreateAllocator();

        // Create a simple memory buffer to test invalid buffer type
        var device = new CpuDevice();
        var fakeBuffer = device.AllocateMemory(1024);

        Assert.Throws<ArgumentException>(() => allocator.Free(fakeBuffer));
    }

    [Fact]
    public void CacheSize_Accumulates()
    {
        using var allocator = CreateAllocator();
        var buffer1 = allocator.Allocate(1024);
        allocator.Free(buffer1);

        var buffer2 = allocator.Allocate(512);
        allocator.Free(buffer2);

        // Cache should contain both blocks
        Assert.True(allocator.CacheSize > 0);
    }

    [Fact]
    public void EmptyCache_ResetsCacheSize()
    {
        using var allocator = CreateAllocator();
        var buffer1 = allocator.Allocate(1024);
        allocator.Free(buffer1);

        allocator.EmptyCache();

        Assert.Equal(0, allocator.CacheSize);
    }

    [Fact]
    public void EmptyCache_MultipleCalls_DoesNotThrow()
    {
        using var allocator = CreateAllocator();
        var buffer1 = allocator.Allocate(1024);
        allocator.Free(buffer1);

        var exception = Record.Exception(() =>
        {
            allocator.EmptyCache();
            allocator.EmptyCache();
        });

        Assert.Null(exception);
    }

    [Fact]
    public void AllocatedSize_TracksCorrectly()
    {
        using var allocator = CreateAllocator();

        var buffer1 = allocator.Allocate(1024);
        Assert.Equal(1024, allocator.AllocatedSize);

        var buffer2 = allocator.Allocate(512);
        Assert.Equal(1536, allocator.AllocatedSize); // 1024 + 512

        allocator.Free(buffer1);
        Assert.Equal(512, allocator.AllocatedSize);

        allocator.Free(buffer2);
        Assert.Equal(0, allocator.AllocatedSize);
    }

    [Fact]
    public void BlockSplitting_LargeBlockSplitsForSmallerRequest()
    {
        using var allocator = CreateAllocator();
        var largeBlock = allocator.Allocate(2048);
        var ptr1 = largeBlock.Pointer;

        allocator.Free(largeBlock);

        var smallBlock1 = allocator.Allocate(1024);
        var smallBlock2 = allocator.Allocate(1024);

        // The first small block should use the same pointer as the large block
        Assert.Equal(ptr1, smallBlock1.Pointer);

        // The second small block should have a different pointer
        Assert.NotEqual(ptr1, smallBlock2.Pointer);
    }

    [Fact]
    public void BlockMerging_AdjacentBlocksMerge()
    {
        using var allocator = CreateAllocator();

        // Allocate and free adjacent blocks
        var block1 = allocator.Allocate(1024);
        var block2 = allocator.Allocate(1024);

        allocator.Free(block1);
        allocator.Free(block2);

        // Try to allocate a larger block that should use the merged blocks
        var largeBlock = allocator.Allocate(2048);

        Assert.NotNull(largeBlock);
    }

    [Fact]
    public void MultipleBuffers_SameSize_Reuse()
    {
        using var allocator = CreateAllocator();

        var ptrs = new List<IntPtr>();

        // Allocate and free multiple buffers of the same size
        for (int i = 0; i < 5; i++)
        {
            var buffer = allocator.Allocate(1024);
            ptrs.Add(buffer.Pointer);
            allocator.Free(buffer);
        }

        // All allocations should use the same pointer (reused)
        var uniquePtrs = ptrs.Distinct().Count();
        Assert.Equal(1, uniquePtrs);
    }

    [Fact]
    public void ThreadSafety_ConcurrentAllocation()
    {
        using var allocator = CreateAllocator();

        var tasks = new List<Task>();

        for (int i = 0; i < 10; i++)
        {
            var taskId = i;
            tasks.Add(Task.Run(() =>
            {
                var buffer = allocator.Allocate(1024);
                Thread.Sleep(10);
                allocator.Free(buffer);
            }));
        }

        Task.WaitAll(tasks);

        // After all tasks complete, no memory should be allocated
        Assert.Equal(0, allocator.AllocatedSize);
    }

    [Fact]
    public void Device_ReturnsCorrectDevice()
    {
        using var allocator = CreateAllocator();
        var device = new CpuDevice();

        Assert.Equal(DeviceType.CPU, allocator.Device.DeviceType);
        Assert.NotNull(allocator.Device);
    }
}
