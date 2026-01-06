using MLFramework.Distributed;
using System;
using System.Runtime.InteropServices;
using System.Text;

namespace MLFramework.Distributed.NCCL
{
    /// <summary>
    /// P/Invoke declarations for NCCL library functions.
    /// </summary>
    internal static class NCCLNative
    {
        private const string NCCLLib = "nccl";

        /// <summary>
        /// Generates a unique ID for NCCL communicator initialization.
        /// </summary>
        [DllImport(NCCLLib, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern int ncclGetUniqueId([Out] byte[] uniqueId);

        /// <summary>
        /// Initializes an NCCL communicator with the given rank.
        /// </summary>
        [DllImport(NCCLLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int ncclCommInitRank(
            ref IntPtr comm,
            int nranks,
            byte[] uniqueId,
            int rank);

        /// <summary>
        /// Destroys an NCCL communicator.
        /// </summary>
        [DllImport(NCCLLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int ncclCommDestroy(IntPtr comm);

        /// <summary>
        /// Performs an AllReduce operation.
        /// </summary>
        [DllImport(NCCLLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int ncclAllReduce(
            IntPtr sendbuff,
            IntPtr recvbuff,
            ulong count,
            ncclDataType_t datatype,
            ncclRedOp_t op,
            IntPtr comm,
            IntPtr stream);

        /// <summary>
        /// Performs a Broadcast operation.
        /// </summary>
        [DllImport(NCCLLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int ncclBroadcast(
            IntPtr sendbuff,
            IntPtr recvbuff,
            ulong count,
            ncclDataType_t datatype,
            int root,
            IntPtr comm,
            IntPtr stream);

        /// <summary>
        /// Performs a barrier operation to synchronize all processes.
        /// </summary>
        [DllImport(NCCLLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int ncclBarrier(
            IntPtr comm,
            IntPtr stream);

        /// <summary>
        /// Gets the error string for an NCCL error code.
        /// </summary>
        [DllImport(NCCLLib, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        private static extern IntPtr ncclGetErrorString(int error);

        /// <summary>
        /// Checks the error code and throws a CommunicationException if not successful.
        /// </summary>
        public static void CheckError(int error, int rank, string operation)
        {
            if (error != 0)
            {
                var errorMsg = GetErrorString(error);
                throw new CommunicationException(
                    $"NCCL operation '{operation}' failed: {errorMsg}",
                    rank,
                    "NCCL");
            }
        }

        /// <summary>
        /// Gets the error string for an NCCL error code.
        /// </summary>
        private static string GetErrorString(int error)
        {
            try
            {
                IntPtr errorPtr = ncclGetErrorString(error);
                if (errorPtr != IntPtr.Zero)
                {
                    return Marshal.PtrToStringAnsi(errorPtr) ?? $"NCCL error code: {error}";
                }
            }
            catch
            {
                // Ignore errors getting error string
            }

            return $"NCCL error code: {error}";
        }
    }

    /// <summary>
    /// NCCL data type enumeration.
    /// </summary>
    internal enum ncclDataType_t
    {
        ncclInt8 = 0,
        ncclChar = 0,
        ncclUint8 = 1,
        ncclInt32 = 2,
        ncclInt = 2,
        ncclUint32 = 3,
        ncclInt64 = 4,
        ncclUint64 = 5,
        ncclFloat16 = 6,
        ncclHalf = 6,
        ncclFloat32 = 7,
        ncclFloat = 7,
        ncclFloat64 = 8,
        ncclDouble = 8
    }

    /// <summary>
    /// NCCL reduction operation enumeration.
    /// </summary>
    internal enum ncclRedOp_t
    {
        ncclSum = 0,
        ncclProd = 1,
        ncclMax = 2,
        ncclMin = 3,
        ncclAvg = 4
    }
}
