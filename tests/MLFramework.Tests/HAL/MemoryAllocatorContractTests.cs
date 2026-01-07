using System;
using Xunit;

namespace MLFramework.HAL.Tests;

/// <summary>
/// Abstract base class for testing IMemoryAllocator implementations
/// </summary>
public abstract class MemoryAllocatorContractTests
{
    protected abstract IMemoryAllocator CreateAllocator();

    [Fact]
    public void Allocate_ReturnsValidBuffer()
    {
        using var allocator = CreateAllocator();
        var buffer = allocator.Allocate(1024);

        Assert.NotNull(buffer);
        Assert.Equal(1024, buffer.Size);
        Assert.True(buffer.IsValid);
        buffer.Dispose();
    }

    [Fact]
    public void Allocate_ZeroSize_ThrowsException()
    {
        using var allocator = CreateAllocator();

        Assert.Throws<ArgumentException>(() =>
        {
            allocator.Allocate(0);
        });
    }

    [Fact]
    public void Allocate_NegativeSize_ThrowsException()
    {
        using var allocator = CreateAllocator();

        Assert.Throws<ArgumentException>(() =>
        {
            allocator.Allocate(-1);
        });
    }

    [Fact]
    public void Free_AllowsReuse()
    {
        using var allocator = CreateAllocator();
        var buffer1 = allocator.Allocate(1024);

        allocator.Free(buffer1);

        var buffer2 = allocator.Allocate(1024);

        Assert.NotNull(buffer2);
    }

    [Fact]
    public void Free_AfterDispose_ThrowsObjectDisposedException()
    {
        var allocator = CreateAllocator();
        var buffer = allocator.Allocate(1024);
        allocator.Dispose();

        Assert.Throws<ObjectDisposedException>(() =>
        {
            allocator.Free(buffer);
        });
    }

    [Fact]
    public void EmptyCache_ReducesCacheSize()
    {
        using var allocator = CreateAllocator();
        var buffer = allocator.Allocate(1024);
        allocator.Free(buffer);

        var sizeBefore = allocator.CacheSize;
        allocator.EmptyCache();
        var sizeAfter = allocator.CacheSize;

        Assert.LessOrEqual(sizeAfter, sizeBefore);
    }

    [Fact]
    public void EmptyCache_DoesNotThrow()
    {
        using var allocator = CreateAllocator();

        var exception = Record.Exception(() => allocator.EmptyCache());
        Assert.Null(exception);
    }

    [Fact]
    public void Device_ReturnsValidDevice()
    {
        using var allocator = CreateAllocator();

        Assert.NotNull(allocator.Device);
    }

    [Fact]
    public void AllocatedSize_ReturnsNonNegativeValue()
    {
        using var allocator = CreateAllocator();

        Assert.True(allocator.AllocatedSize >= 0);
    }

    [Fact]
    public void CacheSize_ReturnsNonNegativeValue()
    {
        using var allocator = CreateAllocator();

        Assert.True(allocator.CacheSize >= 0);
    }

    [Fact]
    public void Dispose_CleansUpResources()
    {
        var allocator = CreateAllocator();
        allocator.Dispose();

        Assert.Throws<ObjectDisposedException>(() =>
        {
            allocator.Allocate(1024);
        });
    }

    [Fact]
    public void Dispose_MultipleCallsDoesNotThrow()
    {
        var allocator = CreateAllocator();
        allocator.Dispose();

        var exception = Record.Exception(() => allocator.Dispose());
        Assert.Null(exception);
    }

    [Fact]
    public void Allocate_AfterDispose_ThrowsObjectDisposedException()
    {
        var allocator = CreateAllocator();
        allocator.Dispose();

        Assert.Throws<ObjectDisposedException>(() =>
        {
            allocator.Allocate(1024);
        });
    }

    [Fact]
    public void CacheSize_AfterDispose_ThrowsObjectDisposedException()
    {
        var allocator = CreateAllocator();
        allocator.Dispose();

        Assert.Throws<ObjectDisposedException>(() =>
        {
            var _ = allocator.CacheSize;
        });
    }

    [Fact]
    public void AllocatedSize_AfterDispose_ThrowsObjectDisposedException()
    {
        var allocator = CreateAllocator();
        allocator.Dispose();

        Assert.Throws<ObjectDisposedException>(() =>
        {
            var _ = allocator.AllocatedSize;
        });
    }

    [Fact]
    public void Device_AfterDispose_ThrowsObjectDisposedException()
    {
        var allocator = CreateAllocator();
        allocator.Dispose();

        Assert.Throws<ObjectDisposedException>(() =>
        {
            var _ = allocator.Device;
        });
    }
}
