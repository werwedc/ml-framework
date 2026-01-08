using System;
using Xunit;

namespace MLFramework.Tests.Memory;

public class MemoryHandleTests
{
    private readonly ShapeBounds _bounds;
    private const int ElementSize = 4; // float32

    public MemoryHandleTests()
    {
        _bounds = new ShapeBounds(new[] { 1, 10 }, new[] { 5, 20 }, new[] { 2, 15 });
    }

    [Fact]
    public void Constructor_InitializesProperties()
    {
        var pointer = new IntPtr(1000);
        var capacity = 1000L;
        var initialShape = new[] { 2, 15 };

        var handle = new MemoryHandle(pointer, capacity, initialShape, _bounds, ElementSize);

        Assert.Equal(pointer, handle.Pointer);
        Assert.Equal(capacity, handle.CapacityBytes);
        Assert.Equal(initialShape, handle.CurrentShape);
        Assert.Equal(_bounds, handle.ShapeBounds);
        Assert.True(handle.AllocationTime <= DateTime.UtcNow);
    }

    [Fact]
    public void GetEffectiveSize_ReturnsCorrectSize()
    {
        var handle = new MemoryHandle(
            new IntPtr(1000),
            1000L,
            new[] { 2, 15 },
            _bounds,
            ElementSize);

        long effectiveSize = handle.GetEffectiveSize();

        // 2 * 15 * 4 = 120 bytes
        Assert.Equal(120, effectiveSize);
    }

    [Fact]
    public void GetUtilization_ReturnsCorrectUtilization()
    {
        var capacity = 200L;
        var handle = new MemoryHandle(
            new IntPtr(1000),
            capacity,
            new[] { 2, 15 },
            _bounds,
            ElementSize);

        double utilization = handle.GetUtilization();

        // Effective size: 2 * 15 * 4 = 120 bytes
        // Utilization: 120 / 200 = 0.6
        Assert.Equal(0.6, utilization);
    }

    [Fact]
    public void Resize_ValidShape_UpdatesCurrentShape()
    {
        var handle = new MemoryHandle(
            new IntPtr(1000),
            1000L,
            new[] { 2, 15 },
            _bounds,
            ElementSize);

        var newShape = new[] { 3, 12 };
        handle.Resize(newShape);

        Assert.Equal(newShape, handle.CurrentShape);
    }

    [Fact]
    public void Resize_ShapeOutOfBounds_ThrowsArgumentException()
    {
        var handle = new MemoryHandle(
            new IntPtr(1000),
            1000L,
            new[] { 2, 15 },
            _bounds,
            ElementSize);

        Assert.Throws<ArgumentException>(() =>
        {
            handle.Resize(new[] { 6, 15 }); // Above max (5, 20)
        });
    }

    [Fact]
    public void Resize_SmallerShape_UpdatesUtilization()
    {
        var capacity = 1000L;
        var handle = new MemoryHandle(
            new IntPtr(1000),
            capacity,
            new[] { 2, 15 },
            _bounds,
            ElementSize);

        var newShape = new[] { 1, 10 };
        handle.Resize(newShape);

        double utilization = handle.GetUtilization();
        // Effective size: 1 * 10 * 4 = 40 bytes
        // Utilization: 40 / 1000 = 0.04
        Assert.Equal(0.04, utilization);
    }

    [Fact]
    public void GetUtilization_ZeroCapacity_ReturnsZero()
    {
        var handle = new MemoryHandle(
            new IntPtr(1000),
            0L,
            new[] { 2, 15 },
            _bounds,
            ElementSize);

        double utilization = handle.GetUtilization();

        Assert.Equal(0.0, utilization);
    }
}
