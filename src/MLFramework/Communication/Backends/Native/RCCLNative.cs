namespace MLFramework.Communication.Backends.Native;

using System.Runtime.InteropServices;
using System.Text;

/// <summary>
/// P/Invoke declarations for RCCL library
/// </summary>
internal static class RCCLNative
{
    private const string RCCL_LIBRARY = "rccl";
    private const string HIP_LIBRARY = "hiprtc";

    // RCCL result codes
    public const int RCCL_SUCCESS = 0;

    // RCCL reduce operations
    public const int RCCL_SUM = 0;
    public const int RCCL_PROD = 1;
    public const int RCCL_MAX = 2;
    public const int RCCL_MIN = 3;
    public const int RCCL_AVG = 4;

    [DllImport(RCCL_LIBRARY, CallingConvention = CallingConvention.Cdecl)]
    public static extern int rcclCommInitRank(
        out IntPtr comm,
        int nranks,
        IntPtr commId,
        int rank);

    [DllImport(RCCL_LIBRARY, CallingConvention = CallingConvention.Cdecl)]
    public static extern int rcclCommDestroy(IntPtr comm);

    [DllImport(RCCL_LIBRARY, CallingConvention = CallingConvention.Cdecl)]
    public static extern int rcclBroadcast(
        IntPtr sendbuf,
        IntPtr recvbuf,
        long count,
        int datatype,
        int root,
        IntPtr comm,
        IntPtr stream);

    [DllImport(RCCL_LIBRARY, CallingConvention = CallingConvention.Cdecl)]
    public static extern int rcclReduce(
        IntPtr sendbuf,
        IntPtr recvbuf,
        long count,
        int datatype,
        int op,
        int root,
        IntPtr comm,
        IntPtr stream);

    [DllImport(RCCL_LIBRARY, CallingConvention = CallingConvention.Cdecl)]
    public static extern int rcclAllReduce(
        IntPtr sendbuf,
        IntPtr recvbuf,
        long count,
        int datatype,
        int op,
        IntPtr comm,
        IntPtr stream);

    [DllImport(RCCL_LIBRARY, CallingConvention = CallingConvention.Cdecl)]
    public static extern int rcclAllGather(
        IntPtr sendbuf,
        IntPtr recvbuf,
        long count,
        int datatype,
        IntPtr comm,
        IntPtr stream);

    [DllImport(RCCL_LIBRARY, CallingConvention = CallingConvention.Cdecl)]
    public static extern int rcclReduceScatter(
        IntPtr sendbuf,
        IntPtr recvbuf,
        long count,
        int datatype,
        int op,
        IntPtr comm,
        IntPtr stream);

    [DllImport(RCCL_LIBRARY, CallingConvention = CallingConvention.Cdecl)]
    public static extern int rcclBarrier(
        IntPtr comm,
        IntPtr stream);

    [DllImport(RCCL_LIBRARY, CallingConvention = CallingConvention.Cdecl)]
    public static extern int rcclGetErrorString(int error, IntPtr buf, int len);

    // RCCL data types
    public const int rcclInt8 = 0;
    public const int rcclChar = 0;
    public const int rcclUint8 = 1;
    public const int rcclInt32 = 2;
    public const int rcclUint32 = 3;
    public const int rcclInt64 = 4;
    public const int rcclUint64 = 5;
    public const int rcclFloat16 = 6;
    public const int rcclFloat32 = 7;
    public const int rcclFloat64 = 8;

    // HIP stream operations (simplified)
    [DllImport(HIP_LIBRARY, CallingConvention = CallingConvention.Cdecl)]
    public static extern int hipStreamCreate(out IntPtr stream);

    [DllImport(HIP_LIBRARY, CallingConvention = CallingConvention.Cdecl)]
    public static extern int hipStreamDestroy(IntPtr stream);

    [DllImport(HIP_LIBRARY, CallingConvention = CallingConvention.Cdecl)]
    public static extern int hipStreamSynchronize(IntPtr stream);

    /// <summary>
    /// Create a HIP stream
    /// </summary>
    public static IntPtr CreateHipStream()
    {
        try
        {
            int result = hipStreamCreate(out IntPtr stream);
            if (result != 0)
            {
                throw new System.ComponentModel.Win32Exception(result, "Failed to create HIP stream");
            }
            return stream;
        }
        catch (DllNotFoundException)
        {
            // HIP runtime not available, return zero stream (default stream)
            return IntPtr.Zero;
        }
    }

    /// <summary>
    /// Destroy a HIP stream
    /// </summary>
    public static void DestroyHipStream(IntPtr stream)
    {
        if (stream != IntPtr.Zero)
        {
            try
            {
                hipStreamDestroy(stream);
            }
            catch (DllNotFoundException)
            {
                // HIP runtime not available, ignore
            }
        }
    }

    /// <summary>
    /// Synchronize a HIP stream
    /// </summary>
    public static void HipStreamSynchronize(IntPtr stream)
    {
        if (stream != IntPtr.Zero)
        {
            try
            {
                int result = hipStreamSynchronize(stream);
                if (result != 0)
                {
                    throw new System.ComponentModel.Win32Exception(result, "Failed to synchronize HIP stream");
                }
            }
            catch (DllNotFoundException)
            {
                // HIP runtime not available, ignore
            }
        }
    }

    /// <summary>
    /// Get RCCL datatype from Tensor dtype
    /// </summary>
    public static int GetRCCLDataType(RitterFramework.Core.DataType dtype)
    {
        return dtype switch
        {
            RitterFramework.Core.DataType.Int8 => rcclInt8,
            RitterFramework.Core.DataType.UInt8 => rcclUint8,
            RitterFramework.Core.DataType.Int32 => rcclInt32,
            RitterFramework.Core.DataType.Int64 => rcclInt64,
            RitterFramework.Core.DataType.Float16 => rcclFloat16,
            RitterFramework.Core.DataType.Float32 => rcclFloat32,
            RitterFramework.Core.DataType.Float64 => rcclFloat64,
            _ => throw new ArgumentException($"Unsupported tensor data type: {dtype}")
        };
    }

    /// <summary>
    /// Get RCCL error string
    /// </summary>
    public static string GetErrorString(int error)
    {
        try
        {
            byte[] buffer = new byte[1024];
            GCHandle handle = GCHandle.Alloc(buffer, GCHandleType.Pinned);
            IntPtr bufferPtr = handle.AddrOfPinnedObject();

            int result = rcclGetErrorString(error, bufferPtr, buffer.Length);
            if (result == RCCL_SUCCESS)
            {
                string errorStr = System.Text.Encoding.ASCII.GetString(buffer).TrimEnd('\0');
                handle.Free();
                return errorStr;
            }

            handle.Free();
        }
        catch
        {
            // Ignore errors
        }

        return $"RCCL error code: {error}";
    }
}
