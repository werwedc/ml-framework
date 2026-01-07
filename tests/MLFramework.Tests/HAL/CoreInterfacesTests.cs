using System;
using Xunit;

namespace MLFramework.HAL.Tests;

/// <summary>
/// Abstract base class for testing IDevice implementations
/// </summary>
public abstract class DeviceContractTests
{
    protected abstract IDevice CreateDevice();

    [Fact]
    public void DeviceType_ReturnsValidType()
    {
        var device = CreateDevice();
        Assert.True(Enum.IsDefined(typeof(DeviceType), device.DeviceType));
    }

    [Fact]
    public void DeviceId_ReturnsNonNegativeId()
    {
        var device = CreateDevice();
        Assert.True(device.DeviceId >= 0);
    }

    [Fact]
    public void AllocateMemory_WithValidSize_ReturnsValidBuffer()
    {
        var device = CreateDevice();
        var buffer = device.AllocateMemory(1024);

        Assert.NotNull(buffer);
        Assert.True(buffer.IsValid);
        Assert.Equal(1024, buffer.Size);
        Assert.Equal(device, buffer.Device);
    }

    [Fact]
    public void AllocateMemory_WithZeroSize_ThrowsArgumentException()
    {
        var device = CreateDevice();
        Assert.Throws<ArgumentException>(() => device.AllocateMemory(0));
    }

    [Fact]
    public void AllocateMemory_WithNegativeSize_ThrowsArgumentException()
    {
        var device = CreateDevice();
        Assert.Throws<ArgumentException>(() => device.AllocateMemory(-1));
    }

    [Fact]
    public void CreateStream_ReturnsValidStream()
    {
        var device = CreateDevice();
        var stream = device.CreateStream();

        Assert.NotNull(stream);
        Assert.Equal(device, stream.Device);
    }

    [Fact]
    public void RecordEvent_ReturnsValidEvent()
    {
        var device = CreateDevice();
        var @event = device.RecordEvent();

        Assert.NotNull(@event);
        Assert.NotNull(@event.Stream);
        Assert.Equal(device, @event.Stream.Device);
    }
}

/// <summary>
/// Abstract base class for testing IMemoryBuffer implementations
/// </summary>
public abstract class MemoryBufferContractTests
{
    protected abstract IMemoryBuffer CreateBuffer();

    [Fact]
    public void Pointer_ReturnsValidPointer()
    {
        var buffer = CreateBuffer();
        Assert.NotEqual(IntPtr.Zero, buffer.Pointer);
    }

    [Fact]
    public void Size_ReturnsPositiveSize()
    {
        var buffer = CreateBuffer();
        Assert.True(buffer.Size > 0);
    }

    [Fact]
    public void Device_ReturnsValidDevice()
    {
        var buffer = CreateBuffer();
        Assert.NotNull(buffer.Device);
    }

    [Fact]
    public void IsValid_ReturnsTrueInitially()
    {
        var buffer = CreateBuffer();
        Assert.True(buffer.IsValid);
    }

    [Fact]
    public void Dispose_SetsIsValidToFalse()
    {
        var buffer = CreateBuffer();
        buffer.Dispose();
        Assert.False(buffer.IsValid);
    }

    [Fact]
    public void Dispose_MultipleCallsDoesNotThrow()
    {
        var buffer = CreateBuffer();
        buffer.Dispose();
        var exception = Record.Exception(() => buffer.Dispose());
        Assert.Null(exception);
    }

    [Fact]
    public void Pointer_AfterDispose_ThrowsObjectDisposedException()
    {
        var buffer = CreateBuffer();
        buffer.Dispose();
        Assert.Throws<ObjectDisposedException>(() => { var _ = buffer.Pointer; });
    }
}

/// <summary>
/// Abstract base class for testing IStream implementations
/// </summary>
public abstract class StreamContractTests
{
    protected abstract IStream CreateStream();

    [Fact]
    public void Device_ReturnsValidDevice()
    {
        var stream = CreateStream();
        Assert.NotNull(stream.Device);
    }

    [Fact]
    public void Enqueue_WithValidOperation_DoesNotThrow()
    {
        var stream = CreateStream();
        bool executed = false;
        var exception = Record.Exception(() => stream.Enqueue(() => executed = true));
        Assert.Null(exception);
    }

    [Fact]
    public void Enqueue_WithNullOperation_ThrowsArgumentNullException()
    {
        var stream = CreateStream();
        Assert.Throws<ArgumentNullException>(() => stream.Enqueue(null!));
    }

    [Fact]
    public void RecordEvent_ReturnsValidEvent()
    {
        var stream = CreateStream();
        var @event = stream.RecordEvent();

        Assert.NotNull(@event);
        Assert.Equal(stream, @event.Stream);
    }

    [Fact]
    public void Dispose_SetsStreamAsDisposed()
    {
        var stream = CreateStream();
        stream.Dispose();
        Assert.Throws<ObjectDisposedException>(() => stream.Enqueue(() => { }));
    }
}

/// <summary>
/// Abstract base class for testing IEvent implementations
/// </summary>
public abstract class EventContractTests
{
    protected abstract IEvent CreateEvent();

    [Fact]
    public void Stream_ReturnsValidStream()
    {
        var @event = CreateEvent();
        Assert.NotNull(@event.Stream);
    }

    [Fact]
    public void IsCompleted_ReturnsBoolean()
    {
        var @event = CreateEvent();
        Assert.IsType<bool>(@event.IsCompleted);
    }

    [Fact]
    public void Synchronize_DoesNotThrow()
    {
        var @event = CreateEvent();
        var exception = Record.Exception(() => @event.Synchronize());
        Assert.Null(exception);
    }

    [Fact]
    public void Dispose_MultipleCallsDoesNotThrow()
    {
        var @event = CreateEvent();
        @event.Dispose();
        var exception = Record.Exception(() => @event.Dispose());
        Assert.Null(exception);
    }

    [Fact]
    public void IsCompleted_AfterDispose_ThrowsObjectDisposedException()
    {
        var @event = CreateEvent();
        @event.Dispose();
        Assert.Throws<ObjectDisposedException>(() => { var _ = @event.IsCompleted; });
    }

    [Fact]
    public void Synchronize_AfterDispose_ThrowsObjectDisposedException()
    {
        var @event = CreateEvent();
        @event.Dispose();
        Assert.Throws<ObjectDisposedException>(() => @event.Synchronize());
    }
}
