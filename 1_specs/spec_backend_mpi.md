# Spec: MPI Backend Implementation

## Overview
Implement MPI (Message Passing Interface) backend for CPU/GPU communication primitives.

## Dependencies
- `spec_communication_interfaces.md`
- `spec_collective_basic.md`
- `spec_collective_advanced.md`
- `spec_point_to_point.md`

## Technical Requirements

### 1. MPI Backend Class
Wrapper around MPI library for CPU and GPU communication.

```csharp
namespace MLFramework.Communication.Backends
{
    /// <summary>
    /// MPI backend for CPU/GPU communication
    /// </summary>
    public class MPIBackend : IAsyncCommunicationBackend, IPointToPointCommunication
    {
        private readonly IntPtr _comm; // MPI communicator handle
        private readonly int _rank;
        private readonly int _worldSize;
        private readonly CommunicationConfig _config;
        private readonly DeviceType _deviceType;
        private readonly bool _isInitialized;
        private bool _disposed;

        public int Rank => _rank;
        public int WorldSize => _worldSize;
        public string BackendName => "MPI";
        public DeviceType Device => _deviceType;
        public bool IsInitialized => _isInitialized;

        /// <summary>
        /// Initialize MPI backend
        /// </summary>
        /// <param name="config">Configuration</param>
        public MPIBackend(CommunicationConfig config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));

            // Initialize MPI if not already initialized
            if (!IsMPIInitialized())
            {
                MPIInit();
            }

            // Get rank and world size
            _rank = GetMPIRank();
            _worldSize = GetMPIWorldSize();

            // Create communicator (default: MPI_COMM_WORLD)
            _comm = MPICreateComm();

            _isInitialized = true;
            _deviceType = DetectDeviceType();
        }

        public void Broadcast<T>(Tensor<T> tensor, int rootRank)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            // Determine if tensor is on CPU or GPU
            var deviceType = GetTensorDeviceType(tensor);

            if (deviceType == DeviceType.CUDA)
            {
                // Use MPI with CUDA-aware MPI if available
                MPIBroadcastCUDA(tensor, rootRank);
            }
            else
            {
                // Standard CPU broadcast
                MPIBroadcastCPU(tensor, rootRank);
            }
        }

        public Tensor<T> Reduce<T>(Tensor<T> tensor, ReduceOp operation, int rootRank)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            var deviceType = GetTensorDeviceType(tensor);

            if (deviceType == DeviceType.CUDA)
            {
                return MPIReduceCUDA(tensor, operation, rootRank);
            }
            else
            {
                return MPIReduceCPU(tensor, operation, rootRank);
            }
        }

        public Tensor<T> AllReduce<T>(Tensor<T> tensor, ReduceOp operation)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            var deviceType = GetTensorDeviceType(tensor);

            if (deviceType == DeviceType.CUDA)
            {
                return MPIAllReduceCUDA(tensor, operation);
            }
            else
            {
                return MPIAllReduceCPU(tensor, operation);
            }
        }

        public Tensor<T> AllGather<T>(Tensor<T> tensor)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            var deviceType = GetTensorDeviceType(tensor);

            if (deviceType == DeviceType.CUDA)
            {
                return MPIAllGatherCUDA(tensor);
            }
            else
            {
                return MPIAllGatherCPU(tensor);
            }
        }

        public Tensor<T> ReduceScatter<T>(Tensor<T> tensor, ReduceOp operation)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            var deviceType = GetTensorDeviceType(tensor);

            if (deviceType == DeviceType.CUDA)
            {
                return MPIReduceScatterCUDA(tensor, operation);
            }
            else
            {
                return MPIReduceScatterCPU(tensor, operation);
            }
        }

        public void Barrier()
        {
            MPIBarrier(_comm);
        }

        // Point-to-point operations
        public void Send<T>(Tensor<T> tensor, int destinationRank, int tag = 0)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            var deviceType = GetTensorDeviceType(tensor);

            if (deviceType == DeviceType.CUDA)
            {
                MPISendCUDA(tensor, destinationRank, tag);
            }
            else
            {
                MPISendCPU(tensor, destinationRank, tag);
            }
        }

        public Tensor<T> Receive<T>(int sourceRank, int tag = 0)
        {
            var deviceType = _deviceType;

            if (deviceType == DeviceType.CUDA)
            {
                return MPIReceiveCUDA<T>(sourceRank, tag);
            }
            else
            {
                return MPIReceiveCPU<T>(sourceRank, tag);
            }
        }

        public Tensor<T> Receive<T>(int sourceRank, Tensor<T> template, int tag = 0)
        {
            if (template == null)
                throw new ArgumentNullException(nameof(template));

            var deviceType = GetTensorDeviceType(template);

            if (deviceType == DeviceType.CUDA)
            {
                return MPIReceiveCUDAWithShape(sourceRank, template, tag);
            }
            else
            {
                return MPIReceiveCPUWithShape(sourceRank, template, tag);
            }
        }

        public ICommunicationHandle SendAsync<T>(Tensor<T> tensor, int destinationRank, int tag = 0)
        {
            var task = Task.Run(() => Send(tensor, destinationRank, tag));
            return new AsyncCommunicationHandle(task);
        }

        public ICommunicationHandle ReceiveAsync<T>(int sourceRank, int tag = 0)
        {
            var task = Task.Run(() => Receive<T>(sourceRank, tag));
            return new AsyncCommunicationHandle(task);
        }

        public ICommunicationHandle ReceiveAsync<T>(int sourceRank, Tensor<T> template, int tag = 0)
        {
            var task = Task.Run(() => Receive(sourceRank, template, tag));
            return new AsyncCommunicationHandle(task);
        }

        public MessageInfo? Probe(int sourceRank, int tag = 0)
        {
            return MPIProbe(sourceRank, tag);
        }

        // Async operations
        public ICommunicationHandle BroadcastAsync<T>(Tensor<T> tensor, int rootRank)
        {
            var task = Task.Run(() => Broadcast(tensor, rootRank));
            return new AsyncCommunicationHandle(task);
        }

        public ICommunicationHandle AllReduceAsync<T>(Tensor<T> tensor, ReduceOp operation)
        {
            var task = Task.Run(() => AllReduce(tensor, operation));
            return new AsyncCommunicationHandle(task);
        }

        public ICommunicationHandle BarrierAsync()
        {
            var task = Task.Run(() => Barrier());
            return new AsyncCommunicationHandle(task);
        }

        // MPI implementation methods (placeholders)
        private void MPIInit()
        {
            // Call MPI_Init or MPI_Init_thread
            MPINative.MPI_Init(IntPtr.Zero, IntPtr.Zero);
        }

        private bool IsMPIInitialized()
        {
            int flag;
            MPINative.MPI_Initialized(out flag);
            return flag != 0;
        }

        private int GetMPIRank()
        {
            int rank;
            MPINative.MPI_Comm_rank(_comm, out rank);
            return rank;
        }

        private int GetMPIWorldSize()
        {
            int size;
            MPINative.MPI_Comm_size(_comm, out size);
            return size;
        }

        private IntPtr MPICreateComm()
        {
            // Use MPI_COMM_WORLD
            return MPINative.MPI_COMM_WORLD;
        }

        private void MPIBroadcastCPU<T>(Tensor<T> tensor, int rootRank)
        {
            // P/Invoke to MPI_Bcast
            var dataPtr = GetDataPointer(tensor);
            var count = tensor.Shape.TotalSize;
            var datatype = MPINative.GetMPIDatatype<T>();

            MPINative.MPI_Bcast(dataPtr, count, datatype, rootRank, _comm);
        }

        private void MPIBroadcastCUDA<T>(Tensor<T> tensor, int rootRank)
        {
            // Use CUDA-aware MPI if available
            if (IsCudaAwareMPI())
            {
                var dataPtr = GetCudaDataPointer(tensor);
                var count = tensor.Shape.TotalSize;
                var datatype = MPINative.GetMPIDatatype<T>();

                MPINative.MPI_Bcast(dataPtr, count, datatype, rootRank, _comm);
            }
            else
            {
                // Fallback to CPU: copy to CPU, broadcast, copy back
                MPIBroadcastCPUWithCUDA<T>(tensor, rootRank);
            }
        }

        private Tensor<T> MPIReduceCPU<T>(Tensor<T> tensor, ReduceOp operation, int rootRank)
        {
            // P/Invoke to MPI_Reduce
            var result = CreateTensorLike(tensor);
            var sendPtr = GetDataPointer(tensor);
            var recvPtr = GetDataPointer(result);
            var count = tensor.Shape.TotalSize;
            var datatype = MPINative.GetMPIDatatype<T>();
            var mpiOp = MapReduceOp(operation);

            MPINative.MPI_Reduce(sendPtr, recvPtr, count, datatype, mpiOp, rootRank, _comm);
            return result;
        }

        private Tensor<T> MPIAllReduceCPU<T>(Tensor<T> tensor, ReduceOp operation)
        {
            var result = CreateTensorLike(tensor);
            var sendPtr = GetDataPointer(tensor);
            var recvPtr = GetDataPointer(result);
            var count = tensor.Shape.TotalSize;
            var datatype = MPINative.GetMPIDatatype<T>();
            var mpiOp = MapReduceOp(operation);

            MPINative.MPI_Allreduce(sendPtr, recvPtr, count, datatype, mpiOp, _comm);
            return result;
        }

        private void MPISendCPU<T>(Tensor<T> tensor, int destinationRank, int tag)
        {
            var dataPtr = GetDataPointer(tensor);
            var count = tensor.Shape.TotalSize;
            var datatype = MPINative.GetMPIDatatype<T>();

            MPINative.MPI_Send(dataPtr, count, datatype, destinationRank, tag, _comm);
        }

        private Tensor<T> MPIReceiveCPU<T>(int sourceRank, int tag)
        {
            // Probe for message size first
            var status = MPINative.MPI_Status();
            MPINative.MPI_Probe(sourceRank, tag, _comm, ref status);

            int count;
            MPINative.MPI_Get_count(ref status, MPINative.GetMPIDatatype<T>(), out count);

            var result = CreateTensor<T>(count);
            var dataPtr = GetDataPointer(result);

            MPINative.MPI_Recv(dataPtr, count, MPINative.GetMPIDatatype<T>(),
                             sourceRank, tag, _comm, ref status);

            return result;
        }

        private void MPIBarrier(IntPtr comm)
        {
            MPINative.MPI_Barrier(comm);
        }

        // Helper methods
        private DeviceType GetTensorDeviceType<T>(Tensor<T> tensor)
        {
            return tensor.Device;
        }

        private DeviceType DetectDeviceType()
        {
            // Detect if CUDA is available
            return DeviceType.CPU;
        }

        private bool IsCudaAwareMPI()
        {
            // Check if CUDA-aware MPI is available
            return false; // Placeholder
        }

        private IntPtr GetDataPointer<T>(Tensor<T> tensor)
        {
            // Get pointer to tensor data
            return IntPtr.Zero; // Placeholder
        }

        private IntPtr GetCudaDataPointer<T>(Tensor<T> tensor)
        {
            // Get CUDA pointer to tensor data
            return IntPtr.Zero; // Placeholder
        }

        private Tensor<T> CreateTensorLike<T>(Tensor<T> template)
        {
            // Create new tensor with same shape/type
            return null; // Placeholder
        }

        private Tensor<T> CreateTensor<T>(long count)
        {
            // Create new tensor with given size
            return null; // Placeholder
        }

        private int MapReduceOp(ReduceOp op)
        {
            return op switch
            {
                ReduceOp.Sum => MPINative.MPI_SUM,
                ReduceOp.Product => MPINative.MPI_PROD,
                ReduceOp.Max => MPINative.MPI_MAX,
                ReduceOp.Min => MPINative.MPI_MIN,
                _ => throw new ArgumentException($"Unsupported reduce operation: {op}")
            };
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                MPINative.MPI_Finalize();
                _disposed = true;
            }
        }
    }
}
```

### 2. MPI Backend Factory
Factory for creating MPI backend instances.

```csharp
namespace MLFramework.Communication.Backends
{
    /// <summary>
    /// Factory for MPI backend
    /// </summary>
    public class MPIBackendFactory : ICommunicationBackendFactory
    {
        private static readonly Lazy<bool> _isAvailable = new Lazy<bool>(CheckMPIAvailability);

        public int Priority => 50; // Medium priority (works everywhere)

        public bool IsAvailable()
        {
            return _isAvailable.Value;
        }

        public ICommunicationBackend Create(CommunicationConfig config)
        {
            if (!IsAvailable())
            {
                throw new CommunicationException("MPI is not available on this system");
            }

            return new MPIBackend(config);
        }

        /// <summary>
        /// Check if MPI is available
        /// </summary>
        private static bool CheckMPIAvailability()
        {
            try
            {
                // Try to load MPI library
                // Check for MPI environment variables
                var rank = Environment.GetEnvironmentVariable("OMPI_COMM_WORLD_RANK");
                return rank != null || MPINative.IsMPIAvailable();
            }
            catch
            {
                return false;
            }
        }
    }
}
```

### 3. MPI P/Invoke Interop
Native interop layer for MPI library.

```csharp
namespace MLFramework.Communication.Backends.Native
{
    /// <summary>
    /// P/Invoke declarations for MPI library
    /// </summary>
    internal static class MPINative
    {
        private const string MPI_LIBRARY = "mpi";

        // MPI predefined handles
        public static readonly IntPtr MPI_COMM_WORLD = IntPtr.Zero;
        public static readonly IntPtr MPI_COMM_NULL = IntPtr.Zero;

        // MPI data types
        public const int MPI_BYTE = 0;
        public const int MPI_CHAR = 1;
        public const int MPI_INT = 2;
        public const int MPI_LONG = 3;
        public const int MPI_FLOAT = 4;
        public const int MPI_DOUBLE = 5;

        // MPI reduce operations
        public const int MPI_MAX = 6;
        public const int MPI_MIN = 7;
        public const int MPI_SUM = 8;
        public const int MPI_PROD = 9;
        public const int MPI_LAND = 10;
        public const int MPI_LOR = 11;

        // MPI init and finalize
        [DllImport(MPI_LIBRARY, CallingConvention = CallingConvention.Cdecl)]
        public static extern int MPI_Init(IntPtr argc, IntPtr argv);

        [DllImport(MPI_LIBRARY, CallingConvention = CallingConvention.Cdecl)]
        public static extern int MPI_Finalize();

        [DllImport(MPI_LIBRARY, CallingConvention = CallingConvention.Cdecl)]
        public static extern int MPI_Initialized(out int flag);

        [DllImport(MPI_LIBRARY, CallingConvention = CallingConvention.Cdecl)]
        public static extern int MPI_Finalized(out int flag);

        // MPI communicator management
        [DllImport(MPI_LIBRARY, CallingConvention = CallingConvention.Cdecl)]
        public static extern int MPI_Comm_rank(IntPtr comm, out int rank);

        [DllImport(MPI_LIBRARY, CallingConvention = CallingConvention.Cdecl)]
        public static extern int MPI_Comm_size(IntPtr comm, out int size);

        // MPI collective operations
        [DllImport(MPI_LIBRARY, CallingConvention = CallingConvention.Cdecl)]
        public static extern int MPI_Bcast(
            IntPtr buffer,
            int count,
            int datatype,
            int root,
            IntPtr comm);

        [DllImport(MPI_LIBRARY, CallingConvention = CallingConvention.Cdecl)]
        public static extern int MPI_Reduce(
            IntPtr sendbuf,
            IntPtr recvbuf,
            int count,
            int datatype,
            int op,
            int root,
            IntPtr comm);

        [DllImport(MPI_LIBRARY, CallingConvention = CallingConvention.Cdecl)]
        public static extern int MPI_Allreduce(
            IntPtr sendbuf,
            IntPtr recvbuf,
            int count,
            int datatype,
            int op,
            IntPtr comm);

        [DllImport(MPI_LIBRARY, CallingConvention = CallingConvention.Cdecl)]
        public static extern int MPI_Allgather(
            IntPtr sendbuf,
            int sendcount,
            int sendtype,
            IntPtr recvbuf,
            int recvcount,
            int recvtype,
            IntPtr comm);

        [DllImport(MPI_LIBRARY, CallingConvention = CallingConvention.Cdecl)]
        public static extern int MPI_Reduce_scatter(
            IntPtr sendbuf,
            IntPtr recvbuf,
            int[] recvcounts,
            int datatype,
            int op,
            IntPtr comm);

        [DllImport(MPI_LIBRARY, CallingConvention = CallingConvention.Cdecl)]
        public static extern int MPI_Barrier(IntPtr comm);

        // MPI point-to-point operations
        [DllImport(MPI_LIBRARY, CallingConvention = CallingConvention.Cdecl)]
        public static extern int MPI_Send(
            IntPtr buf,
            int count,
            int datatype,
            int dest,
            int tag,
            IntPtr comm);

        [DllImport(MPI_LIBRARY, CallingConvention = CallingConvention.Cdecl)]
        public static extern int MPI_Recv(
            IntPtr buf,
            int count,
            int datatype,
            int source,
            int tag,
            IntPtr comm,
            ref MPI_Status status);

        [DllImport(MPI_LIBRARY, CallingConvention = CallingConvention.Cdecl)]
        public static extern int MPI_Probe(
            int source,
            int tag,
            IntPtr comm,
            ref MPI_Status status);

        [DllImport(MPI_LIBRARY, CallingConvention = CallingConvention.Cdecl)]
        public static extern int MPI_Get_count(
            ref MPI_Status status,
            int datatype,
            out int count);

        // MPI status structure
        public struct MPI_Status
        {
            public int MPI_SOURCE;
            public int MPI_TAG;
            public int MPI_ERROR;

            public MPI_Status(int source, int tag, int error)
            {
                MPI_SOURCE = source;
                MPI_TAG = tag;
                MPI_ERROR = error;
            }
        }

        /// <summary>
        /// Get MPI datatype from C# type
        /// </summary>
        public static int GetMPIDatatype<T>()
        {
            var type = typeof(T);

            if (type == typeof(byte))
                return MPI_BYTE;
            if (type == typeof(char))
                return MPI_CHAR;
            if (type == typeof(int))
                return MPI_INT;
            if (type == typeof(long))
                return MPI_LONG;
            if (type == typeof(float))
                return MPI_FLOAT;
            if (type == typeof(double))
                return MPI_DOUBLE;

            throw new ArgumentException($"Unsupported type: {type.Name}");
        }

        /// <summary>
        /// Check if MPI library is available
        /// </summary>
        public static bool IsMPIAvailable()
        {
            try
            {
                // Try to initialize and finalize
                MPI_Init(IntPtr.Zero, IntPtr.Zero);
                MPI_Finalize();
                return true;
            }
            catch
            {
                return false;
            }
        }
    }
}
```

### 4. MPI Configuration
Configuration specific to MPI backend.

```csharp
namespace MLFramework.Communication.Backends
{
    /// <summary>
    /// MPI-specific configuration
    /// </summary>
    public class MPIConfig
    {
        /// <summary>
        /// Use CUDA-aware MPI for GPU communication
        /// </summary>
        public bool UseCudaAwareMPI { get; set; } = true;

        /// <summary>
        /// Number of threads for MPI initialization
        /// </summary>
        public int ThreadLevel { get; set; } = 1; // MPI_THREAD_SINGLE

        /// <summary>
        /// Enable MPI profiling
        /// </summary>
        public bool EnableProfiling { get; set; } = false;

        /// <summary>
        /// MPI buffer size for non-blocking operations
        /// </summary>
        public int BufferSize { get; set; } = 65536; // 64KB

        /// <summary>
        /// Enable collective algorithm tuning
        /// </summary>
        public bool EnableTuning { get; set; } = false;

        /// <summary>
        /// Apply MPI configuration
        /// </summary>
        public void Apply()
        {
            // Set MPI environment variables
            if (EnableTuning)
            {
                Environment.SetEnvironmentVariable("I_MPI_ADJUST_ALLREDUCE", "2");
            }

            if (EnableProfiling)
            {
                Environment.SetEnvironmentVariable("I_MPI_STATS", "1");
            }
        }
    }
}
```

## Implementation Notes

1. **File Structure:**
   - `src/MLFramework/Communication/Backends/MPIBackend.cs`
   - `src/MLFramework/Communication/Backends/MPIBackendFactory.cs`
   - `src/MLFramework/Communication/Backends/Native/MPINative.cs`
   - `src/MLFramework/Communication/Backends/MPIConfig.cs`

2. **Design Decisions:**
   - Support both CPU and GPU communication
   - Use CUDA-aware MPI if available for GPU tensors
   - Fallback to CPU-based communication if CUDA-aware not available
   - Implement both point-to-point and collective operations

3. **Error Handling:**
   - Throw CommunicationException for MPI errors
   - Handle CUDA-aware MPI unavailability gracefully
   - Validate tensor device before operations

4. **Performance Considerations:**
   - Use pinned memory for efficient transfers
   - Probe for message size before receive
   - Support collective algorithm tuning
   - Thread-safe MPI initialization

## Testing Requirements
- Mock tests for MPI backend interface
- Tests for factory with/without MPI availability
- Tests for CPU and GPU communication paths
- Tests for point-to-point operations
- Tests for collective operations

## Success Criteria
- MPI backend compiles with P/Invoke declarations
- Factory correctly detects MPI availability
- Both CPU and GPU communication paths work
- Point-to-point operations work correctly
- Collective operations work correctly
- Configuration applied correctly
