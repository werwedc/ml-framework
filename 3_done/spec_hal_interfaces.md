# Spec: HAL Core Interfaces

## Overview
Define the foundational interfaces for the Hardware Abstraction Layer (HAL).

## Responsibilities
- Create base interfaces that all device implementations must support
- Establish contracts for memory management, streams, and events

## Files to Create/Modify
- `src/HAL/IDevice.cs` - Main device interface
- `src/HAL/IMemoryBuffer.cs` - Memory buffer interface
- `src/HAL/IStream.cs` - Async stream interface
- `src/HAL/IEvent.cs` - Synchronization event interface
- `tests/HAL/CoreInterfacesTests.cs` - Interface contract tests

## API Design

### IDevice.cs
```csharp
namespace MLFramework.HAL;

/// <summary>
/// Represents a compute device (CPU, GPU, etc.)
/// </summary>
public interface IDevice
{
    /// <summary>
    /// Device type identifier (CPU, CUDA, ROCm, etc.)
    /// </summary>
    DeviceType DeviceType { get; }

    /// <summary>
    /// Unique device ID within the device type
    /// </summary>
    int DeviceId { get; }

    /// <summary>
    /// Allocate memory on this device
    /// </summary>
    IMemoryBuffer AllocateMemory(long size);

    /// <summary>
    /// Free memory allocated on this device
    /// </summary>
    void FreeMemory(IMemoryBuffer buffer);

    /// <summary>
    /// Create a compute stream for async operations
    /// </summary>
    IStream CreateStream();

    /// <summary>
    /// Block until all operations on this device complete
    /// </summary>
    void Synchronize();

    /// <summary>
    /// Record an event at the current point in the default stream
    /// </summary>
    IEvent RecordEvent();
}
```

### IMemoryBuffer.cs
```csharp
namespace MLFramework.HAL;

/// <summary>
/// Represents a block of memory allocated on a device
/// </summary>
public interface IMemoryBuffer : IDisposable
{
    /// <summary>
    /// Pointer to the memory (unmanaged)
    /// </summary>
    IntPtr Pointer { get; }

    /// <summary>
    /// Size in bytes
    /// </summary>
    long Size { get; }

    /// <summary>
    /// Device this buffer is allocated on
    /// </summary>
    IDevice Device { get; }

    /// <summary>
    /// Check if buffer is valid (not disposed)
    /// </summary>
    bool IsValid { get; }
}
```

### IStream.cs
```csharp
namespace MLFramework.HAL;

/// <summary>
/// Represents a command stream for async operations
/// </summary>
public interface IStream : IDisposable
{
    /// <summary>
    /// Device this stream belongs to
    /// </summary>
    IDevice Device { get; }

    /// <summary>
    /// Enqueue an operation to be executed on this stream
    /// </summary>
    void Enqueue(Action operation);

    /// <summary>
    /// Record an event at the current point in this stream
    /// </summary>
    IEvent RecordEvent();

    /// <summary>
    /// Wait for an event to complete before continuing
    /// </summary>
    void Wait(IEvent event);

    /// <summary>
    /// Synchronize this stream (block until all operations complete)
    /// </summary>
    void Synchronize();
}
```

### IEvent.cs
```csharp
namespace MLFramework.HAL;

/// <summary>
/// Represents a synchronization point in a stream
/// </summary>
public interface IEvent : IDisposable
{
    /// <summary>
    /// Stream that recorded this event
    /// </summary>
    IStream Stream { get; }

    /// <summary>
    /// Check if the event has completed (non-blocking)
    /// </summary>
    bool IsCompleted { get; }

    /// <summary>
    /// Block until this event completes
    /// </summary>
    void Synchronize();
}
```

## Testing Requirements
- Create abstract test class verifying interface contracts
- Test that Dispose() cleans up resources correctly
- Test that invalid buffers throw appropriate exceptions

## Acceptance Criteria
- [ ] All interfaces compile without errors
- [ ] Interface contracts documented with XML comments
- [ ] Basic unit tests for interface behavior
- [ ] Namespace structure: MLFramework.HAL

## Notes for Coder
- These are pure interfaces - no implementation required
- Focus on clean API design and comprehensive documentation
- Use XML documentation for all public members
- Ensure IDisposable pattern is correctly defined
