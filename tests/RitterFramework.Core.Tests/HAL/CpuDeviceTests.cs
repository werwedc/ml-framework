using MLFramework.HAL;

namespace MLFramework.HAL.Tests;

public class CpuDeviceTests
{
    [Fact]
    public void CreateStream_ReturnsCpuStream()
    {
        var device = new CpuDevice();
        var stream = device.CreateStream();

        Assert.IsType<CpuStream>(stream);
    }

    [Fact]
    public void AllocateMemory_ReturnsValidBuffer()
    {
        var device = new CpuDevice();
        var buffer = device.AllocateMemory(1024);

        Assert.NotNull(buffer);
        Assert.Equal(1024, buffer.Size);
        Assert.True(buffer.IsValid);
        Assert.Equal(DeviceType.CPU, buffer.Device.DeviceType);
    }

    [Fact]
    public void FreeMemory_ReleasesBuffer()
    {
        var device = new CpuDevice();
        var buffer = device.AllocateMemory(1024);

        var exception = Record.Exception(() => device.FreeMemory(buffer));
        Assert.Null(exception);
    }

    [Fact]
    public void SimpleAllocator_NoCache()
    {
        var device = new CpuDevice();
        var allocator = new SimpleAllocator(device);

        var buffer = allocator.Allocate(1024);
        allocator.Free(buffer);

        Assert.Equal(0, allocator.CacheSize);
    }

    [Fact]
    public void DeviceType_ReturnsCPU()
    {
        var device = new CpuDevice();

        Assert.Equal(DeviceType.CPU, device.DeviceType);
    }

    [Fact]
    public void DeviceId_ReturnsCorrectId()
    {
        var device = new CpuDevice(5);

        Assert.Equal(5, device.DeviceId);
    }

    [Fact]
    public void AllocateMemory_InvalidSize_ThrowsException()
    {
        var device = new CpuDevice();

        Assert.Throws<ArgumentException>(() => device.AllocateMemory(0));
        Assert.Throws<ArgumentException>(() => device.AllocateMemory(-1));
    }

    [Fact]
    public void Stream_EnqueueOperation_ExecutesImmediately()
    {
        var device = new CpuDevice();
        var stream = device.CreateStream();
        var executed = false;

        stream.Enqueue(() => { executed = true; });

        Assert.True(executed);
    }

    [Fact]
    public void Stream_RecordEvent_ReturnsCompletedEvent()
    {
        var device = new CpuDevice();
        var stream = device.CreateStream();
        var @event = stream.RecordEvent();

        Assert.True(@event.IsCompleted);
    }

    [Fact]
    public void Event_Synchronize_CompletesImmediately()
    {
        var device = new CpuDevice();
        var @event = device.RecordEvent();

        var exception = Record.Exception(() => @event.Synchronize());
        Assert.Null(exception);
        Assert.True(@event.IsCompleted);
    }
}
