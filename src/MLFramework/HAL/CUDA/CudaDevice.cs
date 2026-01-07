namespace MLFramework.HAL.CUDA;

/// <summary>
/// CUDA device implementation
/// </summary>
public class CudaDevice : IDevice
{
    private readonly IMemoryAllocator _allocator;
    private readonly int _deviceId;
    private bool _disposed;

    public DeviceType DeviceType => DeviceType.CUDA;
    public int DeviceId => _deviceId;

    public CudaDevice(int deviceId = 0)
    {
        _deviceId = deviceId;

        // Set the CUDA device
        CudaException.CheckError(
            CudaApi.CudaSetDevice(deviceId));

        _allocator = new CudaAllocator(this);
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
        return new CudaStream(this);
    }

    public void Synchronize()
    {
        CudaException.CheckError(
            CudaApi.CudaDeviceSynchronize());
    }

    public IEvent RecordEvent()
    {
        // Record event in default stream
        CudaException.CheckError(
            CudaApi.CudaEventCreate(out IntPtr eventHandle));

        var eventHandleWrapper = new CudaEventHandle(eventHandle);

        CudaException.CheckError(
            CudaApi.CudaEventRecord(eventHandle, IntPtr.Zero));

        return new CudaEvent(null, eventHandleWrapper);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _allocator.Dispose();
            _disposed = true;
        }
    }
}
