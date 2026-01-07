namespace MLFramework.HAL;

/// <summary>
/// CPU device implementation using managed and unmanaged memory
/// </summary>
public class CpuDevice : IDevice
{
    private readonly IMemoryAllocator _allocator;
    private bool _disposed;

    public DeviceType DeviceType => DeviceType.CPU;
    public int DeviceId { get; }

    public CpuDevice(int deviceId = 0)
    {
        DeviceId = deviceId;
        _allocator = new SimpleAllocator(this);
    }

    public IMemoryBuffer AllocateMemory(long size)
    {
        return _allocator.Allocate(size);
    }

    public void FreeMemory(IMemoryBuffer buffer)
    {
        _allocator.Free(buffer);
    }

    public IStream CreateStream()
    {
        return new CpuStream(this);
    }

    public void Synchronize()
    {
        // CPU is always synchronized
    }

    public IEvent RecordEvent()
    {
        return new CpuEvent(null);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _allocator?.Dispose();
            _disposed = true;
        }
    }
}
