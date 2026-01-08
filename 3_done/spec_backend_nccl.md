# Spec: NCCL Backend Implementation

## Overview
Implement NCCL (NVIDIA Collective Communications Library) backend for GPU communication primitives.

## Dependencies
- `spec_communication_interfaces.md`
- `spec_collective_basic.md`
- `spec_collective_advanced.md`

## Technical Requirements

### 1. NCCL Backend Class
Wrapper around NCCL library for NVIDIA GPUs.

```csharp
namespace MLFramework.Communication.Backends
{
    /// <summary>
    /// NCCL backend for NVIDIA GPU communication
    /// </summary>
    public class NCCLBackend : IAsyncCommunicationBackend
    {
        private readonly IntPtr _comm; // NCCL communicator handle
        private readonly int _rank;
        private readonly int _worldSize;
        private readonly CommunicationConfig _config;
        private readonly DeviceType _deviceType;
        private bool _disposed;

        public int Rank => _rank;
        public int WorldSize => _worldSize;
        public string BackendName => "NCCL";
        public DeviceType Device => _deviceType;

        /// <summary>
        /// Initialize NCCL backend
        /// </summary>
        /// <param name="rank">Rank of this process</param>
        /// <param name="worldSize">Total number of processes</param>
        /// <param name="config">Configuration</param>
        public NCCLBackend(int rank, int worldSize, CommunicationConfig config)
        {
            _rank = rank;
            _worldSize = worldSize;
            _config = config ?? throw new ArgumentNullException(nameof(config));
            _deviceType = DeviceType.CUDA;

            // Initialize NCCL communicator (via P/Invoke)
            _comm = InitializeNCCLComm(rank, worldSize);
        }

        /// <summary>
        /// Initialize NCCL communicator (P/Invoke wrapper)
        /// </summary>
        private IntPtr InitializeNCCLComm(int rank, int worldSize)
        {
            // This would use P/Invoke to call NCCL C API
            // Placeholder for actual implementation
            return IntPtr.Zero;
        }

        public void Broadcast<T>(Tensor<T> tensor, int rootRank)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            if (!IsTensorOnGPU(tensor))
                throw new ArgumentException("Tensor must be on GPU for NCCL operations");

            // Call NCCL broadcast via P/Invoke
            NCCLBroadcast(tensor, rootRank);
        }

        public Tensor<T> Reduce<T>(Tensor<T> tensor, ReduceOp operation, int rootRank)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            if (!IsTensorOnGPU(tensor))
                throw new ArgumentException("Tensor must be on GPU for NCCL operations");

            return NCCLReduce(tensor, operation, rootRank);
        }

        public Tensor<T> AllReduce<T>(Tensor<T> tensor, ReduceOp operation)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            if (!IsTensorOnGPU(tensor))
                throw new ArgumentException("Tensor must be on GPU for NCCL operations");

            return NCCLAllReduce(tensor, operation);
        }

        public Tensor<T> AllGather<T>(Tensor<T> tensor)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            if (!IsTensorOnGPU(tensor))
                throw new ArgumentException("Tensor must be on GPU for NCCL operations");

            return NCCLAllGather(tensor);
        }

        public Tensor<T> ReduceScatter<T>(Tensor<T> tensor, ReduceOp operation)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            if (!IsTensorOnGPU(tensor))
                throw new ArgumentException("Tensor must be on GPU for NCCL operations");

            return NCCLReduceScatter(tensor, operation);
        }

        public void Barrier()
        {
            NCCLBarrier();
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

        // NCCL P/Invoke methods (placeholders)
        private void NCCLBroadcast<T>(Tensor<T> tensor, int rootRank)
        {
            // P/Invoke to NCCL broadcast
            throw new NotImplementedException("NCCL broadcast P/Invoke not implemented");
        }

        private Tensor<T> NCCLReduce<T>(Tensor<T> tensor, ReduceOp operation, int rootRank)
        {
            // P/Invoke to NCCL reduce
            throw new NotImplementedException("NCCL reduce P/Invoke not implemented");
        }

        private Tensor<T> NCCLAllReduce<T>(Tensor<T> tensor, ReduceOp operation)
        {
            // P/Invoke to NCCL all-reduce
            throw new NotImplementedException("NCCL all-reduce P/Invoke not implemented");
        }

        private Tensor<T> NCCLAllGather<T>(Tensor<T> tensor)
        {
            // P/Invoke to NCCL all-gather
            throw new NotImplementedException("NCCL all-gather P/Invoke not implemented");
        }

        private Tensor<T> NCCLReduceScatter<T>(Tensor<T> tensor, ReduceOp operation)
        {
            // P/Invoke to NCCL reduce-scatter
            throw new NotImplementedException("NCCL reduce-scatter P/Invoke not implemented");
        }

        private void NCCLBarrier()
        {
            // P/Invoke to NCCL barrier
            throw new NotImplementedException("NCCL barrier P/Invoke not implemented");
        }

        private bool IsTensorOnGPU<T>(Tensor<T> tensor)
        {
            // Check if tensor is on CUDA device
            return tensor.Device == DeviceType.CUDA;
        }

        private NCCLReduceOperation MapReduceOp(ReduceOp op)
        {
            return op switch
            {
                ReduceOp.Sum => NCCLReduceOperation.NCCL_SUM,
                ReduceOp.Product => NCCLReduceOperation.NCCL_PROD,
                ReduceOp.Max => NCCLReduceOperation.NCCL_MAX,
                ReduceOp.Min => NCCLReduceOperation.NCCL_MIN,
                ReduceOp.Avg => NCCLReduceOperation.NCCL_AVG,
                _ => throw new ArgumentException($"Unsupported reduce operation: {op}")
            };
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                // Cleanup NCCL communicator
                CleanupNCCLComm(_comm);
                _disposed = true;
            }
        }

        private void CleanupNCCLComm(IntPtr comm)
        {
            // P/Invoke to NCCL destroy communicator
        }
    }

    /// <summary>
    /// NCCL reduce operations
    /// </summary>
    internal enum NCCLReduceOperation
    {
        NCCL_SUM = 0,
        NCCL_PROD = 1,
        NCCL_MAX = 2,
        NCCL_MIN = 3,
        NCCL_AVG = 4
    }
}
```

### 2. NCCL Backend Factory
Factory for creating NCCL backend instances.

```csharp
namespace MLFramework.Communication.Backends
{
    /// <summary>
    /// Factory for NCCL backend
    /// </summary>
    public class NCCLBackendFactory : ICommunicationBackendFactory
    {
        private static readonly Lazy<bool> _isAvailable = new Lazy<bool>(CheckNCCLAvailability);

        public int Priority => 100; // Highest priority for NVIDIA GPUs

        public bool IsAvailable()
        {
            return _isAvailable.Value;
        }

        public ICommunicationBackend Create(CommunicationConfig config)
        {
            if (!IsAvailable())
            {
                throw new CommunicationException("NCCL is not available on this system");
            }

            // Get rank and world size from environment or MPI
            int rank = GetRank();
            int worldSize = GetWorldSize();

            return new NCCLBackend(rank, worldSize, config);
        }

        /// <summary>
        /// Check if NCCL is available
        /// </summary>
        private static bool CheckNCCLAvailability()
        {
            try
            {
                // Try to load NCCL library
                // Check for CUDA availability
                return true; // Placeholder
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Get rank from environment or MPI
        /// </summary>
        private int GetRank()
        {
            // Check environment variables (e.g., RANK, OMPI_COMM_WORLD_RANK)
            string rankStr = Environment.GetEnvironmentVariable("RANK") ??
                           Environment.GetEnvironmentVariable("OMPI_COMM_WORLD_RANK") ??
                           Environment.GetEnvironmentVariable("WORLD_RANK");

            if (int.TryParse(rankStr, out int rank))
            {
                return rank;
            }

            // Default to rank 0
            return 0;
        }

        /// <summary>
        /// Get world size from environment or MPI
        /// </summary>
        private int GetWorldSize()
        {
            // Check environment variables
            string sizeStr = Environment.GetEnvironmentVariable("WORLD_SIZE") ??
                            Environment.GetEnvironmentVariable("OMPI_COMM_WORLD_SIZE");

            if (int.TryParse(sizeStr, out int size))
            {
                return size;
            }

            // Default to single process
            return 1;
        }
    }
}
```

### 3. NCCL P/Invoke Interop
Native interop layer for NCCL library.

```csharp
namespace MLFramework.Communication.Backends.Native
{
    /// <summary>
    /// P/Invoke declarations for NCCL library
    /// </summary>
    internal static class NCCLNative
    {
        private const string NCCL_LIBRARY = "nccl";

        // NCCL result codes
        public const int NCCL_SUCCESS = 0;

        // NCCL reduce operations
        public const int NCCL_SUM = 0;
        public const int NCCL_PROD = 1;
        public const int NCCL_MAX = 2;
        public const int NCCL_MIN = 3;
        public const int NCCL_AVG = 4;

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

        /// <summary>
        /// Get NCCL datatype from C# type
        /// </summary>
        public static int GetNCCLDataType<T>()
        {
            var type = typeof(T);

            if (type == typeof(sbyte) || type == typeof(char))
                return ncclInt8;
            if (type == typeof(byte))
                return ncclUint8;
            if (type == typeof(int))
                return ncclInt32;
            if (type == typeof(uint))
                return ncclUint32;
            if (type == typeof(long))
                return ncclInt64;
            if (type == typeof(ulong))
                return ncclUint64;
            if (type == typeof(float))
                return ncclFloat32;
            if (type == typeof(double))
                return ncclFloat64;

            throw new ArgumentException($"Unsupported type: {type.Name}");
        }
    }
}
```

### 4. NCCL Configuration
Configuration specific to NCCL backend.

```csharp
namespace MLFramework.Communication.Backends
{
    /// <summary>
    /// NCCL-specific configuration
    /// </summary>
    public class NCCLConfig
    {
        /// <summary>
        /// Use NCCL's ring-based all-reduce (default)
        /// </summary>
        public bool UseRingAllReduce { get; set; } = true;

        /// <summary>
        /// Use NCCL's tree-based all-reduce
        /// </summary>
        public bool UseTreeAllReduce { get; set; } = false;

        /// <summary>
        /// Threshold for switching from ring to tree all-reduce (bytes)
        /// </summary>
        public long TreeThresholdBytes { get; set; } = 1024 * 1024; // 1MB

        /// <summary>
        /// Number of channels for multi-rail communication
        /// </summary>
        public int NumChannels { get; set; } = 1;

        /// <summary>
        /// Enable NCCL debugging
        /// </summary>
        public bool EnableDebug { get; set; } = false;

        /// <summary>
        /// NCCL buffer size for multi-threaded communication
        /// </summary>
        public int BufferSize { get; set; } = 4194304; // 4MB

        /// <summary>
        /// Set NCCL environment variable
        /// </summary>
        public static void SetEnvironmentVariable(string key, string value)
        {
            Environment.SetEnvironmentVariable(key, value);
        }

        /// <summary>
        /// Apply NCCL configuration
        /// </summary>
        public void Apply()
        {
            SetEnvironmentVariable("NCCL_DEBUG", EnableDebug ? "INFO" : "WARN");
            SetEnvironmentVariable("NCCL_BUFFSIZE", BufferSize.ToString());

            if (NumChannels > 1)
            {
                SetEnvironmentVariable("NCCL_NCHANNELS", NumChannels.ToString());
            }
        }
    }
}
```

## Implementation Notes

1. **File Structure:**
   - `src/MLFramework/Communication/Backends/NCCLBackend.cs`
   - `src/MLFramework/Communication/Backends/NCCLBackendFactory.cs`
   - `src/MLFramework/Communication/Backends/Native/NCCLNative.cs`
   - `src/MLFramework/Communication/Backends/NCCLConfig.cs`

2. **Design Decisions:**
   - Use P/Invoke to call NCCL C API
   - Detect rank/world size from environment variables
   - Support both ring and tree all-reduce algorithms
   - Validate tensors are on GPU before operations

3. **Error Handling:**
   - Throw CommunicationException for NCCL errors
   - Validate tensor device before operations
   - Check NCCL availability before creating backend

4. **Performance Considerations:**
   - Use pinned memory for efficient transfers
   - Support GPU-direct for zero-copy transfers
   - Multi-channel configuration for better bandwidth utilization
   - Optimize all-reduce algorithm based on tensor size

## Testing Requirements
- Mock tests for NCCL backend interface
- Tests for factory with/without NCCL availability
- Tests for environment variable parsing
- Tests for NCCL configuration application

## Success Criteria
- NCCL backend compiles with P/Invoke declarations
- Factory correctly detects NCCL availability
- Environment variables parsed correctly
- Configuration applied correctly
- Interface methods implemented correctly
