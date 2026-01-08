namespace MLFramework.Communication.Backends.Native;

using System.Runtime.InteropServices;

/// <summary>
/// P/Invoke declarations for MPI library
/// </summary>
internal static class MPINative
{
    private const string MPI_LIBRARY = "mpi";

    // MPI predefined handles
    public static readonly IntPtr MPI_COMM_WORLD = new IntPtr(0x44000000);
    public static readonly IntPtr MPI_COMM_NULL = IntPtr.Zero;

    // MPI data types
    public const int MPI_BYTE = 0;
    public const int MPI_CHAR = 1;
    public const int MPI_INT = 2;
    public const int MPI_LONG = 3;
    public const int MPI_FLOAT = 4;
    public const int MPI_DOUBLE = 5;
    public const int MPI_UINT8 = 6;
    public const int MPI_UINT16 = 7;
    public const int MPI_UINT32 = 8;
    public const int MPI_UINT64 = 9;

    // MPI reduce operations
    public const int MPI_MAX = 6;
    public const int MPI_MIN = 7;
    public const int MPI_SUM = 8;
    public const int MPI_PROD = 9;
    public const int MPI_LAND = 10;
    public const int MPI_LOR = 11;

    // MPI init and finalize
    [DllImport(MPI_LIBRARY, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern int MPI_Init(IntPtr argc, IntPtr argv);

    [DllImport(MPI_LIBRARY, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern int MPI_Finalize();

    [DllImport(MPI_LIBRARY, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern int MPI_Initialized(out int flag);

    [DllImport(MPI_LIBRARY, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern int MPI_Finalized(out int flag);

    // MPI communicator management
    [DllImport(MPI_LIBRARY, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern int MPI_Comm_rank(IntPtr comm, out int rank);

    [DllImport(MPI_LIBRARY, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern int MPI_Comm_size(IntPtr comm, out int size);

    // MPI collective operations
    [DllImport(MPI_LIBRARY, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern int MPI_Bcast(
        IntPtr buffer,
        int count,
        int datatype,
        int root,
        IntPtr comm);

    [DllImport(MPI_LIBRARY, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern int MPI_Reduce(
        IntPtr sendbuf,
        IntPtr recvbuf,
        int count,
        int datatype,
        int op,
        int root,
        IntPtr comm);

    [DllImport(MPI_LIBRARY, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern int MPI_Allreduce(
        IntPtr sendbuf,
        IntPtr recvbuf,
        int count,
        int datatype,
        int op,
        IntPtr comm);

    [DllImport(MPI_LIBRARY, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern int MPI_Allgather(
        IntPtr sendbuf,
        int sendcount,
        int sendtype,
        IntPtr recvbuf,
        int recvcount,
        int recvtype,
        IntPtr comm);

    [DllImport(MPI_LIBRARY, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern int MPI_Reduce_scatter(
        IntPtr sendbuf,
        IntPtr recvbuf,
        int[] recvcounts,
        int datatype,
        int op,
        IntPtr comm);

    [DllImport(MPI_LIBRARY, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern int MPI_Barrier(IntPtr comm);

    // MPI point-to-point operations
    [DllImport(MPI_LIBRARY, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern int MPI_Send(
        IntPtr buf,
        int count,
        int datatype,
        int dest,
        int tag,
        IntPtr comm);

    [DllImport(MPI_LIBRARY, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern int MPI_Recv(
        IntPtr buf,
        int count,
        int datatype,
        int source,
        int tag,
        IntPtr comm,
        ref MPI_Status status);

    [DllImport(MPI_LIBRARY, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern int MPI_Probe(
        int source,
        int tag,
        IntPtr comm,
        ref MPI_Status status);

    [DllImport(MPI_LIBRARY, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern int MPI_Get_count(
        ref MPI_Status status,
        int datatype,
        out int count);

    // MPI status structure
    [StructLayout(LayoutKind.Sequential)]
    public struct MPI_Status
    {
        public int MPI_SOURCE;
        public int MPI_TAG;
        public int MPI_ERROR;
        public int _count; // Internal field

        public MPI_Status(int source, int tag, int error)
        {
            MPI_SOURCE = source;
            MPI_TAG = tag;
            MPI_ERROR = error;
            _count = 0;
        }
    }

    /// <summary>
    /// Get MPI datatype from DataType enum
    /// </summary>
    public static int GetMPIDatatype(RitterFramework.Core.DataType dtype)
    {
        return dtype switch
        {
            RitterFramework.Core.DataType.Float32 => MPI_FLOAT,
            RitterFramework.Core.DataType.Float64 => MPI_DOUBLE,
            RitterFramework.Core.DataType.Int32 => MPI_INT,
            RitterFramework.Core.DataType.Int64 => MPI_LONG,
            RitterFramework.Core.DataType.Int16 => MPI_SHORT,
            RitterFramework.Core.DataType.Int8 => MPI_CHAR,
            RitterFramework.Core.DataType.UInt8 => MPI_UINT8,
            RitterFramework.Core.DataType.Bool => MPI_BYTE,
            _ => throw new ArgumentException($"Unsupported type: {dtype}")
        };
    }

    private const int MPI_SHORT = 12;

    /// <summary>
    /// Check if MPI library is available
    /// </summary>
    public static bool IsMPIAvailable()
    {
        try
        {
            // Check if MPI is already initialized
            int flag;
            int result = MPI_Initialized(out flag);
            if (result != 0)
            {
                // If we can't even call MPI_Initialized, MPI library isn't available
                return false;
            }

            // Try to initialize if not already
            if (flag == 0)
            {
                result = MPI_Init(IntPtr.Zero, IntPtr.Zero);
                if (result != 0)
                {
                    return false;
                }
                MPI_Finalize();
            }

            return true;
        }
        catch
        {
            return false;
        }
    }
}
