namespace MLFramework.Communication.Backends.Native;

using System.Runtime.InteropServices;
using System.Text;

/// <summary>
/// P/Invoke declarations for NCCL library
/// </summary>
internal static class NCCLNative
{
    private const string NCCL_LIBRARY = "nccl";
    private const string CUDA_LIBRARY = "cuda";

    // NCCL result codes
    public const int NCCL_SUCCESS = 0;

    // NCCL reduce operations
    public const int NCCL_SUM = 0;
    public const int NCCL_PROD = 1;
    public const int NCCL_MAX = 2;
    public const int NCCL_MIN = 3;

    // NCCL data types
    public const int ncclInt8 = 0;
    public const int ncclChar = 0;
    public const int ncclUint8 = 1;
    public const int ncclInt32 = 2;
    public const int ncclUint32 = 3;
    public const int ncclInt64 = 4;
    public const int ncclUint64 = 5;
    public const int ncclFloat16 = 6;
    public const int ncclFloat32 = 7;
    public const int ncclFloat64 = 8;

    [DllImport(NCCL_LIBRARY, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ncclCommInitRank(
        out IntPtr comm,
        int nranks,
        IntPtr commId,
        int rank);

    [DllImport(NCCL_LIBRARY, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ncclCommDestroy(IntPtr comm);

    [DllImport(NCCL_LIBRARY, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ncclBroadcast(
        IntPtr sendbuf,
        IntPtr recvbuf,
        long count,
        int datatype,
        int root,
        IntPtr comm,
        IntPtr stream);

    [DllImport(NCCL_LIBRARY, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ncclReduce(
        IntPtr sendbuf,
        IntPtr recvbuf,
        long count,
        int datatype,
        int op,
        int root,
        IntPtr comm,
        IntPtr stream);

    [DllImport(NCCL_LIBRARY, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ncclAllReduce(
        IntPtr sendbuf,
        IntPtr recvbuf,
        long count,
        int datatype,
        int op,
        IntPtr comm,
        IntPtr stream);

    [DllImport(NCCL_LIBRARY, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ncclAllGather(
        IntPtr sendbuf,
        IntPtr recvbuf,
        long count,
        int datatype,
        IntPtr comm,
        IntPtr stream);

    [DllImport(NCCL_LIBRARY, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ncclReduceScatter(
        IntPtr sendbuf,
        IntPtr recvbuf,
        long count,
        int datatype,
        int op,
        IntPtr comm,
        IntPtr stream);

    [DllImport(NCCL_LIBRARY, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ncclBarrier(
        IntPtr comm,
        IntPtr stream);

    [DllImport(NCCL_LIBRARY, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ncclGetErrorString(int error, IntPtr buf, int len);

    // CUDA stream operations (simplified)
    [DllImport(CUDA_LIBRARY, CallingConvention = CallingConvention.Cdecl)]
    public static extern int cudaStreamCreate(out IntPtr stream);

    [DllImport(CUDA_LIBRARY, CallingConvention = CallingConvention.Cdecl)]
    public static extern int cudaStreamDestroy(IntPtr stream);

    [DllImport(CUDA_LIBRARY, CallingConvention = CallingConvention.Cdecl)]
    public static extern int cudaStreamSynchronize(IntPtr stream);

    /// <summary>
    /// Create a CUDA stream
    /// </summary>
    public static IntPtr CreateCudaStream()
    {
        try
        {
            int result = cudaStreamCreate(out IntPtr stream);
            if (result != 0)
            {
                throw new System.ComponentModel.Win32Exception(result, "Failed to create CUDA stream");
            }
            return stream;
        }
        catch (DllNotFoundException)
        {
            // CUDA runtime not available, return zero stream (default stream)
            return IntPtr.Zero;
        }
    }

    /// <summary>
    /// Destroy a CUDA stream
    /// </summary>
    public static void DestroyCudaStream(IntPtr stream)
    {
        if (stream != IntPtr.Zero)
        {
            try
            {
                cudaStreamDestroy(stream);
            }
            catch (DllNotFoundException)
            {
                // CUDA runtime not available, ignore
            }
        }
    }

    /// <summary>
    /// Synchronize a CUDA stream
    /// </summary>
    public static void CudaStreamSynchronize(IntPtr stream)
    {
        if (stream != IntPtr.Zero)
        {
            try
            {
                int result = cudaStreamSynchronize(stream);
                if (result != 0)
                {
                    throw new System.ComponentModel.Win32Exception(result, "Failed to synchronize CUDA stream");
                }
            }
            catch (DllNotFoundException)
            {
                // CUDA runtime not available, ignore
            }
        }
    }

    /// <summary>
    /// Get NCCL datatype from Tensor dtype
    /// </summary>
    public static int GetNCCLDataType(RitterFramework.Core.DataType dtype)
    {
        return dtype switch
        {
            RitterFramework.Core.DataType.Int8 => ncclInt8,
            RitterFramework.Core.DataType.UInt8 => ncclUint8,
            RitterFramework.Core.DataType.Int32 => ncclInt32,
            RitterFramework.Core.DataType.Int64 => ncclInt64,
            RitterFramework.Core.DataType.Float16 => ncclFloat16,
            RitterFramework.Core.DataType.Float32 => ncclFloat32,
            RitterFramework.Core.DataType.Float64 => ncclFloat64,
            _ => throw new ArgumentException($"Unsupported tensor data type: {dtype}")
        };
    }

    /// <summary>
    /// Get NCCL error string
    /// </summary>
    public static string GetErrorString(int error)
    {
        try
        {
            byte[] buffer = new byte[1024];
            GCHandle handle = GCHandle.Alloc(buffer, GCHandleType.Pinned);
            IntPtr bufferPtr = handle.AddrOfPinnedObject();

            int result = ncclGetErrorString(error, bufferPtr, buffer.Length);
            if (result == NCCL_SUCCESS)
            {
                string errorStr = Encoding.ASCII.GetString(buffer).TrimEnd('\0');
                handle.Free();
                return errorStr;
            }

            handle.Free();
        }
        catch
        {
            // Ignore errors
        }

        return $"NCCL error code: {error}";
    }
}
