# Spec: HAL CUDA Interop Layer

## Overview
Create P/Invoke bindings and interop layer for CUDA APIs.

## Responsibilities
- Define P/Invoke signatures for core CUDA APIs
- Create SafeHandle wrappers for CUDA resources
- Implement error checking and exception handling

## Files to Create/Modify
- `src/HAL/CUDA/CudaApi.cs` - CUDA API P/Invoke declarations
- `src/HAL/CUDA/CudaError.cs` - CUDA error codes and exceptions
- `src/HAL/CUDA/CudaSafeHandles.cs` - SafeHandle wrappers
- `tests/HAL/CUDA/CudaApiTests.cs` - API tests (requires CUDA hardware)

## API Design

### CudaError.cs
```csharp
namespace MLFramework.HAL.CUDA;

/// <summary>
/// CUDA error codes
/// </summary>
public enum CudaError
{
    Success = 0,
    InvalidValue = 1,
    OutOfMemory = 2,
    NotInitialized = 3,
    Deinitialized = 4,
    ProfilerDisabled = 5,
    ProfilerNotInitialized = 6,
    ProfilerAlreadyStarted = 7,
    ProfilerAlreadyStopped = 8,
    NoDevice = 38,
    InvalidDevice = 101,
    InvalidDeviceFunction = 98,
    InvalidConfiguration = 9,
    InvalidMemcpyDirection = 11,
    InvalidTexture = 21,
    InvalidTextureBinding = 22,
    InvalidChannelDescriptor = 23,
    InvalidFilterSetting = 24,
    // ... more error codes
}

/// <summary>
/// CUDA exception
/// </summary>
public class CudaException : Exception
{
    public CudaError Error { get; }

    public CudaException(CudaError error)
        : base($"CUDA error: {error}")
    {
        Error = error;
    }

    public CudaException(CudaError error, string message)
        : base($"CUDA error: {error} - {message}")
    {
        Error = error;
    }

    /// <summary>
    /// Check CUDA error and throw exception if not success
    /// </summary>
    public static void CheckError(CudaError error)
    {
        if (error != CudaError.Success)
        {
            throw new CudaException(error);
        }
    }

    /// <summary>
    /// Check CUDA error with custom message
    /// </summary>
    public static void CheckError(CudaError error, string message)
    {
        if (error != CudaError.Success)
        {
            throw new CudaException(error, message);
        }
    }
}
```

### CudaSafeHandles.cs
```csharp
using Microsoft.Win32.SafeHandles;

namespace MLFramework.HAL.CUDA;

/// <summary>
/// Safe handle for CUDA device pointers
/// </summary>
public sealed class CudaDevicePointerHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    private CudaDevicePointerHandle()
        : base(true)
    {
    }

    public CudaDevicePointerHandle(IntPtr pointer)
        : base(true)
    {
        SetHandle(pointer);
    }

    protected override bool ReleaseHandle()
    {
        if (IsInvalid)
            return false;

        try
        {
            CudaApi.CudaFree(handle);
            return true;
        }
        catch
        {
            return false;
        }
    }
}

/// <summary>
/// Safe handle for CUDA streams
/// </summary>
public sealed class CudaStreamHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    private CudaStreamHandle()
        : base(true)
    {
    }

    public CudaStreamHandle(IntPtr stream)
        : base(true)
    {
        SetHandle(stream);
    }

    protected override bool ReleaseHandle()
    {
        if (IsInvalid)
            return false;

        try
        {
            CudaApi.CudaStreamDestroy(handle);
            return true;
        }
        catch
        {
            return false;
        }
    }
}

/// <summary>
/// Safe handle for CUDA events
/// </summary>
public sealed class CudaEventHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    private CudaEventHandle()
        : base(true)
    {
    }

    public CudaEventHandle(IntPtr evt)
        : base(true)
    {
        SetHandle(evt);
    }

    protected override bool ReleaseHandle()
    {
        if (IsInvalid)
            return false;

        try
        {
            CudaApi.CudaEventDestroy(handle);
            return true;
        }
        catch
        {
            return false;
        }
    }
}
```

### CudaApi.cs
```csharp
using System.Runtime.InteropServices;

namespace MLFramework.HAL.CUDA;

/// <summary>
/// P/Invoke declarations for CUDA API
/// </summary>
public static class CudaApi
{
    private const string CudaLibrary = "nvcuda.dll";

    #region Device Management

    /// <summary>
    /// Get the number of available CUDA devices
    /// </summary>
    [DllImport(CudaLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern CudaError CudaGetDeviceCount(out int count);

    /// <summary>
    /// Set the current device
    /// </summary>
    [DllImport(CudaLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern CudaError CudaSetDevice(int device);

    /// <summary>
    /// Get properties of a CUDA device
    /// </summary>
    [DllImport(CudaLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern CudaError CudaGetDeviceProperties(ref CudaDeviceProperties prop, int device);

    #endregion

    #region Memory Management

    /// <summary>
    /// Allocate memory on the device
    /// </summary>
    [DllImport(CudaLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern CudaError CudaMalloc(out IntPtr devPtr, ulong size);

    /// <summary>
    /// Free device memory
    /// </summary>
    [DllImport(CudaLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern CudaError CudaFree(IntPtr devPtr);

    /// <summary>
    /// Copy memory from host to device
    /// </summary>
    [DllImport(CudaLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern CudaError CudaMemcpy(
        IntPtr dst,
        IntPtr src,
        ulong count,
        CudaMemcpyKind kind);

    /// <summary>
    /// Set device memory to a value
    /// </summary>
    [DllImport(CudaLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern CudaError CudaMemset(
        IntPtr devPtr,
        int value,
        ulong count);

    #endregion

    #region Stream Management

    /// <summary>
    /// Create a new CUDA stream
    /// </summary>
    [DllImport(CudaLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern CudaError CudaStreamCreate(out IntPtr pStream);

    /// <summary>
    /// Destroy a CUDA stream
    /// </summary>
    [DllImport(CudaLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern CudaError CudaStreamDestroy(IntPtr stream);

    /// <summary>
    /// Wait for a stream to complete
    /// </summary>
    [DllImport(CudaLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern CudaError CudaStreamSynchronize(IntPtr stream);

    #endregion

    #region Event Management

    /// <summary>
    /// Create a new CUDA event
    /// </summary>
    [DllImport(CudaLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern CudaError CudaEventCreate(out IntPtr pEvent);

    /// <summary>
    /// Destroy a CUDA event
    /// </summary>
    [DllImport(CudaLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern CudaError CudaEventDestroy(IntPtr event);

    /// <summary>
    /// Record an event in a stream
    /// </summary>
    [DllImport(CudaLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern CudaError CudaEventRecord(IntPtr event, IntPtr stream);

    /// <summary>
    /// Wait for an event
    /// </summary>
    [DllImport(CudaLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern CudaError CudaStreamWaitEvent(IntPtr stream, IntPtr event, uint flags);

    /// <summary>
    /// Query if an event has completed
    /// </summary>
    [DllImport(CudaLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern CudaError CudaEventQuery(IntPtr event);

    #endregion
}

/// <summary>
/// CUDA memory copy direction
/// </summary>
public enum CudaMemcpyKind
{
    HostToHost = 0,
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3
}

/// <summary>
/// CUDA device properties
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct CudaDeviceProperties
{
    public char Name[256];
    public uint TotalGlobalMem;
    public uint SharedMemPerBlock;
    public uint RegsPerBlock;
    public uint WarpSize;
    public uint MemPitch;
    public uint MaxThreadsPerBlock;
    public uint MaxThreadsDim0;
    public uint MaxThreadsDim1;
    public uint MaxThreadsDim2;
    public uint MaxGridSize0;
    public uint MaxGridSize1;
    public uint MaxGridSize2;
    public int ClockRate;
    public int DeviceOverlap;
    public int MultiProcessorCount;
    public int KernelExecTimeoutEnabled;
    public int Integrated;
    public int CanMapHostMemory;
    public int ComputeMode;
    public int ConcurrentKernels;
    public int EccEnabled;
    public int pciBusID;
    public int pciDeviceID;
    public int tccDriver;
    // ... more fields
}
```

## Testing Requirements
```csharp
public class CudaApiTests
{
    [Test]
    public void CudaGetDeviceCount_ReturnsNonNegative()
    {
        if (!CudaAvailable())
            Assert.Inconclusive("CUDA not available");

        var result = CudaApi.CudaGetDeviceCount(out int count);
        CudaException.CheckError(result);

        Assert.GreaterOrEqual(count, 0);
    }

    [Test]
    public void CudaMalloc_Free_Works()
    {
        if (!CudaAvailable())
            Assert.Inconclusive("CUDA not available");

        var allocResult = CudaApi.CudaMalloc(out IntPtr ptr, 1024);
        CudaException.CheckError(allocResult);

        Assert.AreNotEqual(IntPtr.Zero, ptr);

        var freeResult = CudaApi.CudaFree(ptr);
        CudaException.CheckError(freeResult);
    }

    private bool CudaAvailable()
    {
        var result = CudaApi.CudaGetDeviceCount(out int count);
        return result == CudaError.Success && count > 0;
    }
}
```

## Acceptance Criteria
- [ ] CudaError enum defined with common error codes
- [ ] CudaException class with error checking methods
- [ ] SafeHandle wrappers for CUDA resources
- [ ] P/Invoke declarations for core CUDA APIs
- [ ] CudaDeviceProperties struct defined
- [ ] Tests for CUDA API (when CUDA hardware available)

## Notes for Coder
- These are P/Invoke declarations - no actual CUDA code written yet
- SafeHandle ensures proper cleanup of CUDA resources
- Tests should be conditional on CUDA availability
- Library name "nvcuda.dll" - may need adjustment for different platforms
- C# char[256] in struct - actual implementation needs proper marshaling
