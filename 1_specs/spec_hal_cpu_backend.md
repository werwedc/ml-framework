# Spec: HAL CPU Backend Implementation

## Overview
Implement the CPU backend using managed arrays and basic .NET libraries.

## Responsibilities
- Create CpuDevice class implementing IDevice
- Create SimpleAllocator implementing IMemoryAllocator (no caching)
- Create CpuStream implementing IStream
- Create CpuEvent implementing IEvent
- Integrate CPU backend into Device factory

## Files to Create/Modify
- `src/HAL/CpuDevice.cs` - CPU device implementation
- `src/HAL/SimpleAllocator.cs` - Simple memory allocator
- `src/HAL/CpuStream.cs` - CPU stream implementation
- `src/HAL/CpuEvent.cs` - CPU event implementation
- `src/HAL/Device.cs` - Update CreateDevice method
- `tests/HAL/CpuDeviceTests.cs` - CPU device tests

## API Design

### SimpleAllocator.cs
```csharp
namespace MLFramework.HAL;

/// <summary>
/// Simple allocator that allocates memory directly without caching
/// </summary>
public class SimpleAllocator : IMemoryAllocator
{
    private readonly IDevice _device;
    private long _allocatedSize;

    public IDevice Device => _device;
    public long CacheSize => 0; // No caching
    public long AllocatedSize => _allocatedSize;

    public SimpleAllocator(IDevice device)
    {
        _device = device;
    }

    public IMemoryBuffer Allocate(long size)
    {
        if (size <= 0)
            throw new ArgumentException("Size must be positive", nameof(size));

        var pointer = Marshal.AllocHGlobal((IntPtr)size);
        _allocatedSize += size;

        return new SimpleMemoryBuffer(pointer, size, _device, this);
    }

    public void Free(IMemoryBuffer buffer)
    {
        if (buffer == null)
            throw new ArgumentNullException(nameof(buffer));

        if (buffer is SimpleMemoryBuffer simpleBuffer)
        {
            _allocatedSize -= simpleBuffer.Size;
            Marshal.FreeHGlobal(simpleBuffer.Pointer);
        }
        else
        {
            throw new ArgumentException("Invalid buffer type");
        }
    }

    public void EmptyCache()
    {
        // No-op for simple allocator
    }

    public void Dispose()
    {
        // Any remaining buffers should be freed
        // In a real implementation, track all buffers
    }

    private class SimpleMemoryBuffer : IMemoryBuffer
    {
        private readonly SimpleAllocator _allocator;
        private bool _disposed;

        public IntPtr Pointer { get; }
        public long Size { get; }
        public IDevice Device { get; }
        public bool IsValid => !_disposed;

        public SimpleMemoryBuffer(IntPtr pointer, long size, IDevice device, SimpleAllocator allocator)
        {
            Pointer = pointer;
            Size = size;
            Device = device;
            _allocator = allocator;
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _allocator.Free(this);
                _disposed = true;
            }
        }
    }
}
```

### CpuStream.cs
```csharp
namespace MLFramework.HAL;

/// <summary>
/// CPU stream that executes operations synchronously
/// </summary>
public class CpuStream : IStream
{
    private readonly Queue<Action> _pendingOperations = new();
    private bool _disposed;

    public IDevice Device { get; }

    public CpuStream(IDevice device)
    {
        Device = device;
    }

    public void Enqueue(Action operation)
    {
        if (operation == null)
            throw new ArgumentNullException(nameof(operation));

        // CPU executes immediately (no true async)
        operation();
    }

    public IEvent RecordEvent()
    {
        // CPU events complete immediately
        return new CpuEvent(this, completed: true);
    }

    public void Wait(IEvent @event)
    {
        // CPU events are always complete, so this is a no-op
        if (@event == null)
            throw new ArgumentNullException(nameof(event));
    }

    public void Synchronize()
    {
        // No pending operations on CPU
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
        }
    }
}
```

### CpuEvent.cs
```csharp
namespace MLFramework.HAL;

/// <summary>
/// CPU event that is always complete (synchronous execution)
/// </summary>
public class CpuEvent : IEvent
{
    private bool _disposed;

    public IStream Stream { get; }
    public bool IsCompleted { get; private set; }

    public CpuEvent(IStream stream, bool completed = false)
    {
        Stream = stream;
        IsCompleted = completed;
    }

    public void Synchronize()
    {
        // Always complete for CPU
        IsCompleted = true;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
        }
    }
}
```

### CpuDevice.cs
```csharp
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
```

### Device.cs Update
```csharp
// Update the CreateDevice method:
private static IDevice CreateDevice(DeviceType type, int deviceId)
{
    return type switch
    {
        DeviceType.CPU => new CpuDevice(deviceId),
        _ => throw new NotImplementedException($"Device type {type} not yet implemented")
    };
}
```

## Testing Requirements
```csharp
public class CpuDeviceTests
{
    [Test]
    public void CreateStream_ReturnsCpuStream()
    {
        var device = new CpuDevice();
        var stream = device.CreateStream();

        Assert.IsInstanceOf<CpuStream>(stream);
    }

    [Test]
    public void AllocateMemory_ReturnsValidBuffer()
    {
        var device = new CpuDevice();
        var buffer = device.AllocateMemory(1024);

        Assert.NotNull(buffer);
        Assert.AreEqual(1024, buffer.Size);
        Assert.IsTrue(buffer.IsValid);
    }

    [Test]
    public void FreeMemory_ReleasesBuffer()
    {
        var device = new CpuDevice();
        var buffer = device.AllocateMemory(1024);

        Assert.DoesNotThrow(() => device.FreeMemory(buffer));
    }

    [Test]
    public void SimpleAllocator_NoCache()
    {
        using var device = new CpuDevice();
        var allocator = new SimpleAllocator(device);

        var buffer = allocator.Allocate(1024);
        allocator.Free(buffer);

        Assert.AreEqual(0, allocator.CacheSize);
    }
}
```

## Acceptance Criteria
- [ ] CpuDevice implements IDevice correctly
- [ ] SimpleAllocator implements IMemoryAllocator (no caching)
- [ ] CpuStream implements IStream (synchronous execution)
- [ ] CpuEvent implements IEvent (always complete)
- [ ] Device factory can create CPU devices
- [ ] All memory allocations use Marshal.AllocHGlobal
- [ ] All tests pass
- [ ] Proper disposal pattern implemented

## Notes for Coder
- CPU operations are inherently synchronous
- Streams and events on CPU are simplified (no true async)
- Memory is allocated using Marshal.AllocHGlobal (unmanaged)
- Focus on correctness - performance optimization comes later
- Ensure all IDisposable objects are properly disposed
