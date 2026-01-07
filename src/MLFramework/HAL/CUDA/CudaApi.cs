using System.Runtime.InteropServices;
using System.Text;

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

    /// <summary>
    /// Synchronize the current device
    /// </summary>
    [DllImport(CudaLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern CudaError CudaDeviceSynchronize();

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
    public static extern CudaError CudaEventDestroy(IntPtr @event);

    /// <summary>
    /// Record an event in a stream
    /// </summary>
    [DllImport(CudaLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern CudaError CudaEventRecord(IntPtr @event, IntPtr stream);

    /// <summary>
    /// Wait for an event
    /// </summary>
    [DllImport(CudaLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern CudaError CudaStreamWaitEvent(IntPtr stream, IntPtr @event, uint flags);

    /// <summary>
    /// Query if an event has completed
    /// </summary>
    [DllImport(CudaLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern CudaError CudaEventQuery(IntPtr @event);

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
    // char Name[256] needs special marshaling
    [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 256)]
    public string Name;

    public uint TotalGlobalMem;
    public uint SharedMemPerBlock;
    public uint RegsPerBlock;
    public uint WarpSize;
    public uint MemPitch;
    public uint MaxThreadsPerBlock;
    public int MaxThreadsDim0;
    public int MaxThreadsDim1;
    public int MaxThreadsDim2;
    public int MaxGridSize0;
    public int MaxGridSize1;
    public int MaxGridSize2;
    public int ClockRate;
    public int DeviceOverlap;
    public int MultiProcessorCount;
    public int KernelExecTimeoutEnabled;
    public int Integrated;
    public int CanMapHostMemory;
    public int ComputeMode;
    public int ConcurrentKernels;
    public int EccEnabled;
    public int PciBusId;
    public int PciDeviceId;
    public int TccDriver;
    public int AsyncEngineCount;
    public int UnifiedAddressing;
    public int MemoryClockRate;
    public int MemoryBusWidth;
    public int L2CacheSize;
    public int MaxThreadsPerMultiProcessor;
    public int ComputeCapabilityMajor;
    public int ComputeCapabilityMinor;
    public int ConcurrentManagedAccess;
    public int PageableMemoryAccess;
    public int PageableMemoryAccessUsesHostPageTables;
}
