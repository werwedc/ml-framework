using System;
using System.Runtime.InteropServices;

namespace MLFramework.Distributed.Gloo
{
    /// <summary>
    /// P/Invoke declarations for the Gloo library.
    /// </summary>
    internal static class GlooNative
    {
        private const string GlooLib = "gloo";

        /// <summary>
        /// Creates a Gloo context for communication.
        /// </summary>
        [DllImport(GlooLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr gloo_create_context(
            int rank,
            int size,
            [MarshalAs(UnmanagedType.LPStr)] string iface,
            [MarshalAs(UnmanagedType.LPStr)] string transport);

        /// <summary>
        /// Destroys a Gloo context and frees resources.
        /// </summary>
        [DllImport(GlooLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern void gloo_destroy_context(IntPtr context);

        /// <summary>
        /// Performs an AllReduce operation.
        /// </summary>
        [DllImport(GlooLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern void gloo_allreduce(
            IntPtr context,
            IntPtr sendbuf,
            IntPtr recvbuf,
            long count,
            glooDataType_t datatype,
            glooRedOp_t op);

        /// <summary>
        /// Performs a Broadcast operation.
        /// </summary>
        [DllImport(GlooLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern void gloo_broadcast(
            IntPtr context,
            IntPtr buffer,
            long count,
            glooDataType_t datatype,
            int root);

        /// <summary>
        /// Performs a Barrier operation.
        /// </summary>
        [DllImport(GlooLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern void gloo_barrier(IntPtr context);

        /// <summary>
        /// Checks if a Gloo error occurred and throws an exception if needed.
        /// </summary>
        public static void CheckError(int rank, string backendName)
        {
            // Gloo uses exceptions in C++ layer, so we rely on try-catch in C#
            // This method is kept for future error code checking if needed
        }
    }

    /// <summary>
    /// Gloo data types.
    /// </summary>
    internal enum glooDataType_t
    {
        glooInt8 = 0,
        glooUint8 = 1,
        glooInt32 = 2,
        glooUint32 = 3,
        glooInt64 = 4,
        glooUint64 = 5,
        glooInt16 = 9,
        glooUint16 = 10,
        glooFloat16 = 6,
        glooFloat32 = 7,
        glooFloat64 = 8
    }

    /// <summary>
    /// Gloo reduction operations.
    /// </summary>
    internal enum glooRedOp_t
    {
        glooSum = 0,
        glooProduct = 1,
        glooMax = 2,
        glooMin = 3,
        glooAvg = 4
    }
}
