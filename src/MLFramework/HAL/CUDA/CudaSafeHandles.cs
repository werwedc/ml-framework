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
