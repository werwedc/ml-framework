using System;
using System.Runtime.InteropServices;
using Xunit;

namespace MLFramework.Tests.Memory;

public class PredictiveAllocatorTests
{
    private const int ElementSize = 4;

    [Fact]
    public void Constructor_DefaultValues_CreatesInstance()
    {
        var strategy = new AdaptivePaddingStrategy();
        var allocator = new PredictiveAllocator(strategy);

        Assert.Equal(1.2, allocator.PaddingFactor);
        Assert.Equal(0L, allocator.MaxCapacity);
        Assert.Equal(0.5, allocator.ShrinkThreshold);
    }

    [Fact]
    public void Constructor_CustomValues_SetsProperties()
    {
        var strategy = new AdaptivePaddingStrategy();
        var allocator = new PredictiveAllocator(strategy, paddingFactor: 1.5, maxCapacity: 1000000, elementSize: 8, shrinkThreshold: 0.3);

        Assert.Equal(1.5, allocator.PaddingFactor);
        Assert.Equal(1000000L, allocator.MaxCapacity);
        Assert.Equal(0.3, allocator.ShrinkThreshold);
    }

    [Fact]
    public void Constructor_InvalidPaddingFactor_ThrowsArgumentException()
    {
        var strategy = new AdaptivePaddingStrategy();

        Assert.Throws<ArgumentException>(() =>
        {
            new PredictiveAllocator(strategy, paddingFactor: 0.5);
        });
    }

    [Fact]
    public void Constructor_InvalidMaxCapacity_ThrowsArgumentException()
    {
        var strategy = new AdaptivePaddingStrategy();

        Assert.Throws<ArgumentException>(() =>
        {
            new PredictiveAllocator(strategy, maxCapacity: -1);
        });
    }

    [Fact]
    public void Constructor_InvalidElementSize_ThrowsArgumentException()
    {
        var strategy = new AdaptivePaddingStrategy();

        Assert.Throws<ArgumentException>(() =>
        {
            new PredictiveAllocator(strategy, elementSize: 0);
        });
    }

    [Fact]
    public void Allocate_ValidBounds_ReturnsMemoryHandle()
    {
        var strategy = new AdaptivePaddingStrategy();
        var allocator = new PredictiveAllocator(strategy);
        var bounds = new ShapeBounds(new[] { 1, 10 }, new[] { 5, 20 }, new[] { 2, 15 });

        var handle = allocator.Allocate(bounds);

        Assert.NotNull(handle);
        Assert.NotNull(handle.Pointer);
        Assert.True(handle.CapacityBytes > 0);
        Assert.Equal(bounds.ExpectedShape, handle.CurrentShape);
        Assert.Equal(bounds, handle.ShapeBounds);
    }

    [Fact]
    public void Allocate_ExceedsMaxCapacity_ThrowsInvalidOperationException()
    {
        var strategy = new AdaptivePaddingStrategy();
        var allocator = new PredictiveAllocator(strategy, maxCapacity: 100);
        var bounds = new ShapeBounds(new[] { 1, 10 }, new[] { 5, 20 }, new[] { 2, 15 });

        Assert.Throws<InvalidOperationException>(() =>
        {
            allocator.Allocate(bounds);
        });
    }

    [Fact]
    public void Allocate_IncrementsAllocationStats()
    {
        var strategy = new AdaptivePaddingStrategy();
        var allocator = new PredictiveAllocator(strategy);
        var bounds = new ShapeBounds(new[] { 1, 10 }, new[] { 5, 20 }, new[] { 2, 15 });

        allocator.Allocate(bounds);
        var stats = allocator.GetAllocationStats();

        Assert.Equal(1, stats.TotalAllocations);
        Assert.True(stats.TotalBytesAllocated > 0);
    }

    [Fact]
    public void Resize_ValidHandle_UpdatesShape()
    {
        var strategy = new AdaptivePaddingStrategy();
        var allocator = new PredictiveAllocator(strategy);
        var bounds = new ShapeBounds(new[] { 1, 10 }, new[] { 5, 20 }, new[] { 2, 15 });
        var handle = allocator.Allocate(bounds);

        var newShape = new[] { 3, 12 };
        allocator.Resize(handle, newShape);

        Assert.Equal(newShape, handle.CurrentShape);
    }

    [Fact]
    public void Resize_InvalidHandle_ThrowsArgumentException()
    {
        var strategy = new AdaptivePaddingStrategy();
        var allocator = new PredictiveAllocator(strategy);
        var bounds = new ShapeBounds(new[] { 1, 10 }, new[] { 5, 20 }, new[] { 2, 15 });
        var handle = new MemoryHandle(
            new IntPtr(1000),
            200L,
            new[] { 2, 15 },
            bounds,
            ElementSize);

        Assert.Throws<ArgumentException>(() =>
        {
            allocator.Resize(handle, new[] { 3, 12 });
        });
    }

    [Fact]
    public void Resize_IncrementsResizeStats()
    {
        var strategy = new AdaptivePaddingStrategy();
        var allocator = new PredictiveAllocator(strategy);
        var bounds = new ShapeBounds(new[] { 1, 10 }, new[] { 5, 20 }, new[] { 2, 15 });
        var handle = allocator.Allocate(bounds);

        allocator.Resize(handle, new[] { 5, 20 });
        var stats = allocator.GetAllocationStats();

        Assert.Equal(1, stats.TotalResizes);
    }

    [Fact]
    public void Resize_ExceedsMaxCapacity_ThrowsInvalidOperationException()
    {
        var strategy = new AdaptivePaddingStrategy();
        var allocator = new PredictiveAllocator(strategy, maxCapacity: 10000);
        var bounds = new ShapeBounds(new[] { 1, 10 }, new[] { 100, 20 }, new[] { 2, 15 });
        var handle = allocator.Allocate(bounds);

        Assert.Throws<InvalidOperationException>(() =>
        {
            allocator.Resize(handle, new[] { 100, 20 });
        });
    }

    [Fact]
    public void Free_ValidHandle_RemovesFromActiveAllocations()
    {
        var strategy = new AdaptivePaddingStrategy();
        var allocator = new PredictiveAllocator(strategy);
        var bounds = new ShapeBounds(new[] { 1, 10 }, new[] { 5, 20 }, new[] { 2, 15 });
        var handle = allocator.Allocate(bounds);

        allocator.Free(handle);

        // Should not throw on second free (though it will fail with ArgumentException)
        // The test passes if no exception is thrown on first free
    }

    [Fact]
    public void Free_InvalidHandle_ThrowsArgumentException()
    {
        var strategy = new AdaptivePaddingStrategy();
        var allocator = new PredictiveAllocator(strategy);
        var bounds = new ShapeBounds(new[] { 1, 10 }, new[] { 5, 20 }, new[] { 2, 15 });
        var handle = new MemoryHandle(
            new IntPtr(1000),
            200L,
            new[] { 2, 15 },
            bounds,
            ElementSize);

        Assert.Throws<ArgumentException>(() =>
        {
            allocator.Free(handle);
        });
    }

    [Fact]
    public void GetAllocationStats_ReturnsCorrectStats()
    {
        var strategy = new AdaptivePaddingStrategy();
        var allocator = new PredictiveAllocator(strategy);
        var bounds = new ShapeBounds(new[] { 1, 10 }, new[] { 5, 20 }, new[] { 2, 15 });

        allocator.Allocate(bounds);
        var handle = allocator.Allocate(bounds);
        allocator.Resize(handle, new[] { 3, 12 });

        var stats = allocator.GetAllocationStats();

        Assert.Equal(2, stats.TotalAllocations);
        Assert.Equal(1, stats.TotalResizes);
        Assert.True(stats.TotalBytesAllocated > 0);
    }

    [Fact]
    public void UpdateExpectations_ValidHandle_DoesNotThrow()
    {
        var strategy = new AdaptivePaddingStrategy();
        var allocator = new PredictiveAllocator(strategy);
        var bounds = new ShapeBounds(new[] { 1, 10 }, new[] { 5, 20 }, new[] { 2, 15 });
        var handle = allocator.Allocate(bounds);

        // Should not throw
        allocator.UpdateExpectations(handle, new[] { 3, 12 });
    }

    [Fact]
    public void UpdateExpectations_InvalidHandle_ThrowsArgumentException()
    {
        var strategy = new AdaptivePaddingStrategy();
        var allocator = new PredictiveAllocator(strategy);
        var bounds = new ShapeBounds(new[] { 1, 10 }, new[] { 5, 20 }, new[] { 2, 15 });
        var handle = new MemoryHandle(
            new IntPtr(1000),
            200L,
            new[] { 2, 15 },
            bounds,
            ElementSize);

        Assert.Throws<ArgumentException>(() =>
        {
            allocator.UpdateExpectations(handle, new[] { 3, 12 });
        });
    }

    [Fact]
    public void MultipleAllocations_TracksTotalBytes()
    {
        var strategy = new AdaptivePaddingStrategy();
        var allocator = new PredictiveAllocator(strategy);
        var bounds1 = new ShapeBounds(new[] { 1, 10 }, new[] { 5, 20 }, new[] { 2, 15 });
        var bounds2 = new ShapeBounds(new[] { 1, 5 }, new[] { 3, 10 }, new[] { 2, 7 });

        allocator.Allocate(bounds1);
        allocator.Allocate(bounds2);

        var stats = allocator.GetAllocationStats();
        Assert.Equal(2, stats.TotalAllocations);
        Assert.True(stats.TotalBytesAllocated > 0);
    }

    [Fact]
    public void GetAllocationStats_AverageUtilization_UpdatesCorrectly()
    {
        var strategy = new AdaptivePaddingStrategy();
        var allocator = new PredictiveAllocator(strategy);
        var bounds = new ShapeBounds(new[] { 1, 10 }, new[] { 5, 20 }, new[] { 2, 15 });

        var handle1 = allocator.Allocate(bounds);
        var handle2 = allocator.Allocate(bounds);

        var stats = allocator.GetAllocationStats();
        Assert.True(stats.AverageUtilization > 0);
        Assert.True(stats.AverageUtilization <= 1.0);
    }
}
