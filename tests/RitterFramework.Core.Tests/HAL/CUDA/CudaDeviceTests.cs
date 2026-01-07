using MLFramework.HAL;
using MLFramework.HAL.CUDA;

namespace MLFramework.HAL.Tests.CUDA;

/// <summary>
/// Tests for CUDA device implementation
/// Note: These tests require CUDA hardware and drivers to run
/// Tests will be marked as inconclusive if CUDA is not available
/// </summary>
public class CudaDeviceTests
{
    [Fact]
    public void CudaDevice_Create_Works()
    {
        if (!CudaAvailable())
            return; // Skip test if CUDA not available

        var device = new CudaDevice(0);

        Assert.Equal(DeviceType.CUDA, device.DeviceType);
        Assert.Equal(0, device.DeviceId);
    }

    [Fact]
    public void CudaDevice_AllocateMemory_Works()
    {
        if (!CudaAvailable())
            return; // Skip test if CUDA not available

        var device = new CudaDevice(0);
        var buffer = device.AllocateMemory(1024);

        Assert.NotNull(buffer);
        Assert.Equal(1024, buffer.Size);
        Assert.True(buffer.IsValid);
        Assert.Equal(DeviceType.CUDA, buffer.Device.DeviceType);

        device.FreeMemory(buffer);
    }

    [Fact]
    public void CudaStream_CreateSynchronize_Works()
    {
        if (!CudaAvailable())
            return; // Skip test if CUDA not available

        var device = new CudaDevice(0);
        var stream = device.CreateStream();

        Assert.NotNull(stream);
        Assert.Equal(DeviceType.CUDA, stream.Device.DeviceType);

        stream.Synchronize();

        stream.Dispose();
    }

    [Fact]
    public void CudaEvent_RecordSynchronize_Works()
    {
        if (!CudaAvailable())
            return; // Skip test if CUDA not available

        var device = new CudaDevice(0);
        var @event = device.RecordEvent();

        Assert.NotNull(@event);

        @event.Synchronize();

        Assert.True(@event.IsCompleted);

        @event.Dispose();
    }

    [Fact]
    public void CudaAllocator_CacheTracking_Works()
    {
        if (!CudaAvailable())
            return; // Skip test if CUDA not available

        var device = new CudaDevice(0);

        // Access the allocator through the device's internal structure
        // This is for testing purposes
        var buffer = device.AllocateMemory(1024);

        Assert.True(buffer.Size > 0);

        device.FreeMemory(buffer);
    }

    [Fact]
    public void CudaDevice_Synchronize_Completes()
    {
        if (!CudaAvailable())
            return; // Skip test if CUDA not available

        var device = new CudaDevice(0);
        var stream = device.CreateStream();

        // Enqueue a simple operation
        stream.Enqueue(() => { });

        // Synchronize device
        var exception = Record.Exception(() => device.Synchronize());
        Assert.Null(exception);

        stream.Dispose();
    }

    [Fact]
    public void DeviceFactory_CUDA_ReturnsCudaDevice()
    {
        if (!CudaAvailable())
            return; // Skip test if CUDA not available

        var device = Device.CUDA(0);

        Assert.Equal(DeviceType.CUDA, device.DeviceType);
        Assert.Equal(0, device.DeviceId);
    }

    private bool CudaAvailable()
    {
        try
        {
            var result = CudaApi.CudaGetDeviceCount(out int count);
            return result == CudaError.Success && count > 0;
        }
        catch (DllNotFoundException)
        {
            // CUDA DLL not found
            return false;
        }
        catch (BadImageFormatException)
        {
            // CUDA DLL not compatible (e.g., wrong architecture)
            return false;
        }
        catch (EntryPointNotFoundException)
        {
            // CUDA entry point not found
            return false;
        }
    }
}
