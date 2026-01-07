# Spec: HAL CUDA Device Implementation

## Overview
Implement the CUDA device using the CUDA interop layer.

## Responsibilities
- Create CudaDevice class implementing IDevice
- Create CudaAllocator implementing IMemoryAllocator
- Create CudaStream implementing IStream
- Create CudaEvent implementing IEvent
- Integrate CUDA backend into Device factory

## Files to Create/Modify
- `src/HAL/CUDA/CudaDevice.cs` - CUDA device implementation
- `src/HAL/CUDA/CudaAllocator.cs` - CUDA memory allocator
- `src/HAL/CUDA/CudaStream.cs` - CUDA stream implementation
- `src/HAL/CUDA/CudaEvent.cs` - CUDA event implementation
- `src/HAL/Device.cs` - Update CreateDevice for CUDA
- `tests/HAL/CUDA/CudaDeviceTests.cs` - CUDA device tests

## API Design

### CudaAllocator.cs
```csharp
namespace MLFramework.HAL.CUDA;

/// <summary>
/// Memory allocator for CUDA devices
/// </summary>
public class CudaAllocator : IMemoryAllocator
{
    private readonly CudaDevice _device;
    private readonly CachingAllocator _cachingAllocator;

    public IDevice Device => _device;
    public long CacheSize => _cachingAllocator.CacheSize;
    public long AllocatedSize => _cachingAllocator.AllocatedSize;

    public CudaAllocator(CudaDevice device)
    {
        _device = device;
        _cachingAllocator = new CachingAllocator(device);
    }

    public IMemoryBuffer Allocate(long size)
    {
        // Use caching allocator but allocate via CUDA
        var buffer = _cachingAllocator.Allocate(size);

        // Set memory to zero (CUDA memset)
        CudaException.CheckError(
            CudaApi.CudaMemset(buffer.Pointer, 0, (ulong)size));

        return buffer;
    }

    public void Free(IMemoryBuffer buffer)
    {
        _cachingAllocator.Free(buffer);
    }

    public void EmptyCache()
    {
        _cachingAllocator.EmptyCache();
    }

    public void Dispose()
    {
        _cachingAllocator.Dispose();
    }
}
```

### CudaEvent.cs
```csharp
namespace MLFramework.HAL.CUDA;

/// <summary>
/// CUDA event implementation
/// </summary>
public class CudaEvent : IEvent
{
    private readonly CudaEventHandle _handle;

    public IStream Stream { get; }
    public bool IsCompleted { get; private set; }

    public CudaEvent(IStream stream, CudaEventHandle handle)
    {
        Stream = stream;
        _handle = handle;
        IsCompleted = false;
    }

    public void Synchronize()
    {
        if (IsCompleted)
            return;

        var result = CudaApi.CudaEventQuery(_handle.DangerousGetHandle());

        if (result == CudaError.Success)
        {
            IsCompleted = true;
        }
        else if (result == CudaError.NotReady)
        {
            // Block until event completes
            CudaException.CheckError(
                CudaApi.CudaEventQuery(_handle.DangerousGetHandle()));
            IsCompleted = true;
        }
        else
        {
            CudaException.CheckError(result);
        }
    }

    public void Dispose()
    {
        _handle.Dispose();
    }
}
```

### CudaStream.cs
```csharp
namespace MLFramework.HAL.CUDA;

/// <summary>
/// CUDA stream implementation
/// </summary>
public class CudaStream : IStream
{
    private readonly CudaStreamHandle _handle;
    private readonly CudaDevice _device;
    private readonly Queue<Action> _pendingOperations;

    public IDevice Device => _device;

    public CudaStream(CudaDevice device)
    {
        _device = device;

        CudaException.CheckError(
            CudaApi.CudaStreamCreate(out IntPtr streamHandle));

        _handle = new CudaStreamHandle(streamHandle);
        _pendingOperations = new Queue<Action>();
    }

    public void Enqueue(Action operation)
    {
        if (operation == null)
            throw new ArgumentNullException(nameof(operation));

        // Execute the operation (should launch CUDA kernel)
        operation();
    }

    public IEvent RecordEvent()
    {
        CudaException.CheckError(
            CudaApi.CudaEventCreate(out IntPtr eventHandle));

        var eventHandle = new CudaEventHandle(eventHandle);

        CudaException.CheckError(
            CudaApi.CudaEventRecord(
                eventHandle,
                _handle.DangerousGetHandle()));

        return new CudaEvent(this, eventHandle);
    }

    public void Wait(IEvent @event)
    {
        if (@event == null)
            throw new ArgumentNullException(nameof(event));

        if (@event is CudaEvent cudaEvent)
        {
            CudaException.CheckError(
                CudaApi.CudaStreamWaitEvent(
                    _handle.DangerousGetHandle(),
                    cudaEvent._handle.DangerousGetHandle(),
                    0));
        }
        else
        {
            throw new ArgumentException("Invalid event type");
        }
    }

    public void Synchronize()
    {
        CudaException.CheckError(
            CudaApi.CudaStreamSynchronize(_handle.DangerousGetHandle()));
    }

    public void Dispose()
    {
        _handle.Dispose();
    }
}
```

### CudaDevice.cs
```csharp
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

        var eventHandle = new CudaEventHandle(eventHandle);

        CudaException.CheckError(
            CudaApi.CudaEventRecord(eventHandle, IntPtr.Zero));

        return new CudaEvent(null, eventHandle);
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
```

### Device.cs Update
```csharp
// Update the CreateDevice method:
private static IDevice CreateDevice(DeviceType type, int deviceId)
{
    return type switch
    {
        DeviceType.CPU => new CpuDevice(deviceId),
        DeviceType.CUDA => new CudaDevice(deviceId),
        _ => throw new NotImplementedException($"Device type {type} not yet implemented")
    };
}
```

## Testing Requirements
```csharp
public class CudaDeviceTests
{
    [Test]
    public void CudaDevice_Create_Works()
    {
        if (!CudaAvailable())
            Assert.Inconclusive("CUDA not available");

        var device = new CudaDevice(0);

        Assert.AreEqual(DeviceType.CUDA, device.DeviceType);
        Assert.AreEqual(0, device.DeviceId);
    }

    [Test]
    public void CudaDevice_AllocateMemory_Works()
    {
        if (!CudaAvailable())
            Assert.Inconclusive("CUDA not available");

        var device = new CudaDevice(0);
        var buffer = device.AllocateMemory(1024);

        Assert.NotNull(buffer);
        Assert.AreEqual(1024, buffer.Size);
        Assert.IsTrue(buffer.IsValid);

        device.FreeMemory(buffer);
    }

    [Test]
    public void CudaStream_CreateSynchronize_Works()
    {
        if (!CudaAvailable())
            Assert.Inconclusive("CUDA not available");

        var device = new CudaDevice(0);
        var stream = device.CreateStream();

        stream.Synchronize();

        stream.Dispose();
    }

    private bool CudaAvailable()
    {
        var result = CudaApi.CudaGetDeviceCount(out int count);
        return result == CudaError.Success && count > 0;
    }
}
```

## Acceptance Criteria
- [ ] CudaDevice implements IDevice
- [ ] CudaAllocator uses CUDA APIs for memory management
- [ ] CudaStream implements async CUDA streams
- [ ] CudaEvent implements CUDA event synchronization
- [ ] Device factory can create CUDA devices
- [ ] All tests pass (when CUDA hardware available)

## Notes for Coder
- Requires CUDA hardware and drivers to run tests
- CudaDeviceSynchronize P/Invoke not yet defined - add to CudaApi.cs
- CudaEvent should expose _handle for use by CudaStream (make internal)
- Memory management uses CachingAllocator as a wrapper
- Ensure proper error handling with CudaException
