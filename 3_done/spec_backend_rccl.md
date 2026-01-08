# Spec: RCCL Backend Implementation

## Overview
Implement RCCL (ROCm Collective Communications Library) backend for AMD GPU communication primitives.

## Dependencies
- `spec_communication_interfaces.md`
- `spec_collective_basic.md`
- `spec_collective_advanced.md`

## Technical Requirements

### 1. RCCL Backend Class
Wrapper around RCCL library for AMD GPUs.

```csharp
namespace MLFramework.Communication.Backends
{
    /// <summary>
    /// RCCL backend for AMD GPU communication
    /// </summary>
    public class RCCLBackend : IAsyncCommunicationBackend
    {
        private readonly IntPtr _comm; // RCCL communicator handle
        private readonly int _rank;
        private readonly int _worldSize;
        private readonly CommunicationConfig _config;
        private readonly DeviceType _deviceType;
        private bool _disposed;

        public int Rank => _rank;
        public int WorldSize => _worldSize;
        public string BackendName => "RCCL";
        public DeviceType Device => _deviceType;

        /// <summary>
        /// Initialize RCCL backend
        /// </summary>
        /// <param name="rank">Rank of this process</param>
        /// <param name="worldSize">Total number of processes</param>
        /// <param name="config">Configuration</param>
        public RCCLBackend(int rank, int worldSize, CommunicationConfig config)
        {
            _rank = rank;
            _worldSize = worldSize;
            _config = config ?? throw new ArgumentNullException(nameof(config));
            _deviceType = DeviceType.ROCm;

            // Initialize RCCL communicator (via P/Invoke)
            _comm = InitializeRCCLComm(rank, worldSize);
        }

        /// <summary>
        /// Initialize RCCL communicator (P/Invoke wrapper)
        /// </summary>
        private IntPtr InitializeRCCLComm(int rank, int worldSize)
        {
            // This would use P/Invoke to call RCCL C API
            // Placeholder for actual implementation
            return IntPtr.Zero;
        }

        public void Broadcast<T>(Tensor<T> tensor, int rootRank)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            if (!IsTensorOnGPU(tensor))
                throw new ArgumentException("Tensor must be on GPU for RCCL operations");

            // Call RCCL broadcast via P/Invoke
            RCCLBroadcast(tensor, rootRank);
        }

        public Tensor<T> Reduce<T>(Tensor<T> tensor, ReduceOp operation, int rootRank)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            if (!IsTensorOnGPU(tensor))
                throw new ArgumentException("Tensor must be on GPU for RCCL operations");

            return RCCLReduce(tensor, operation, rootRank);
        }

        public Tensor<T> AllReduce<T>(Tensor<T> tensor, ReduceOp operation)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            if (!IsTensorOnGPU(tensor))
                throw new ArgumentException("Tensor must be on GPU for RCCL operations");

            return RCCLAllReduce(tensor, operation);
        }

        public Tensor<T> AllGather<T>(Tensor<T> tensor)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            if (!IsTensorOnGPU(tensor))
                throw new ArgumentException("Tensor must be on GPU for RCCL operations");

            return RCCLAllGather(tensor);
        }

        public Tensor<T> ReduceScatter<T>(Tensor<T> tensor, ReduceOp operation)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            if (!IsTensorOnGPU(tensor))
                throw new ArgumentException("Tensor must be on GPU for RCCL operations");

            return RCCLReduceScatter(tensor, operation);
        }

        public void Barrier()
        {
            RCCLBarrier();
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

        // RCCL P/Invoke methods (placeholders)
        private void RCCLBroadcast<T>(Tensor<T> tensor, int rootRank)
        {
            // P/Invoke to RCCL broadcast
            throw new NotImplementedException("RCCL broadcast P/Invoke not implemented");
        }

        private Tensor<T> RCCLReduce<T>(Tensor<T> tensor, ReduceOp operation, int rootRank)
        {
            // P/Invoke to RCCL reduce
            throw new NotImplementedException("RCCL reduce P/Invoke not implemented");
        }

        private Tensor<T> RCCLAllReduce<T>(Tensor<T> tensor, ReduceOp operation)
        {
            // P/Invoke to RCCL all-reduce
            throw new NotImplementedException("RCCL all-reduce P/Invoke not implemented");
        }

        private Tensor<T> RCCLAllGather<T>(Tensor<T> tensor)
        {
            // P/Invoke to RCCL all-gather
            throw new NotImplementedException("RCCL all-gather P/Invoke not implemented");
        }

        private Tensor<T> RCCLReduceScatter<T>(Tensor<T> tensor, ReduceOp operation)
        {
            // P/Invoke to RCCL reduce-scatter
            throw new NotImplementedException("RCCL reduce-scatter P/Invoke not implemented");
        }

        private void RCCLBarrier()
        {
            // P/Invoke to RCCL barrier
            throw new NotImplementedException("RCCL barrier P/Invoke not implemented");
        }

        private bool IsTensorOnGPU<T>(Tensor<T> tensor)
        {
            // Check if tensor is on ROCm device
            return tensor.Device == DeviceType.ROCm;
        }

        private RCCLReduceOperation MapReduceOp(ReduceOp op)
        {
            return op switch
            {
                ReduceOp.Sum => RCCLReduceOperation.RCCL_SUM,
                ReduceOp.Product => RCCLReduceOperation.RCCL_PROD,
                ReduceOp.Max => RCCLReduceOperation.RCCL_MAX,
                ReduceOp.Min => RCCLReduceOperation.RCCL_MIN,
                ReduceOp.Avg => RCCLReduceOperation.RCCL_AVG,
                _ => throw new ArgumentException($"Unsupported reduce operation: {op}")
            };
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                // Cleanup RCCL communicator
                CleanupRCCLComm(_comm);
                _disposed = true;
            }
        }

        private void CleanupRCCLComm(IntPtr comm)
        {
            // P/Invoke to RCCL destroy communicator
        }
    }

    /// <summary>
    /// RCCL reduce operations
    /// </summary>
    internal enum RCCLReduceOperation
    {
        RCCL_SUM = 0,
        RCCL_PROD = 1,
        RCCL_MAX = 2,
        RCCL_MIN = 3,
        RCCL_AVG = 4
    }
}
```

### 2. RCCL Backend Factory
Factory for creating RCCL backend instances.

```csharp
namespace MLFramework.Communication.Backends
{
    /// <summary>
    /// Factory for RCCL backend
    /// </summary>
    public class RCCLBackendFactory : ICommunicationBackendFactory
    {
        private static readonly Lazy<bool> _isAvailable = new Lazy<bool>(CheckRCCLAvailability);

        public int Priority => 90; // High priority for AMD GPUs

        public bool IsAvailable()
        {
            return _isAvailable.Value;
        }

        public ICommunicationBackend Create(CommunicationConfig config)
        {
            if (!IsAvailable())
            {
                throw new CommunicationException("RCCL is not available on this system");
            }

            // Get rank and world size from environment or MPI
            int rank = GetRank();
            int worldSize = GetWorldSize();

            return new RCCLBackend(rank, worldSize, config);
        }

        /// <summary>
        /// Check if RCCL is available
        /// </summary>
        private static bool CheckRCCLAvailability()
        {
            try
            {
                // Try to load RCCL library
                // Check for ROCm availability
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

### 3. RCCL P/Invoke Interop
Native interop layer for RCCL library.

```csharp
namespace MLFramework.Communication.Backends.Native
{
    /// <summary>
    /// P/Invoke declarations for RCCL library
    /// </summary>
    internal static class RCCLNative
    {
        private const string RCCL_LIBRARY = "rccl";

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

        /// <summary>
        /// Get RCCL datatype from C# type
        /// </summary>
        public static int GetRCCLDataType<T>()
        {
            var type = typeof(T);

            if (type == typeof(sbyte) || type == typeof(char))
                return rcclInt8;
            if (type == typeof(byte))
                return rcclUint8;
            if (type == typeof(int))
                return rcclInt32;
            if (type == typeof(uint))
                return rcclUint32;
            if (type == typeof(long))
                return rcclInt64;
            if (type == typeof(ulong))
                return rcclUint64;
            if (type == typeof(float))
                return rcclFloat32;
            if (type == typeof(double))
                return rcclFloat64;

            throw new ArgumentException($"Unsupported type: {type.Name}");
        }
    }
}
```

### 4. RCCL Configuration
Configuration specific to RCCL backend.

```csharp
namespace MLFramework.Communication.Backends
{
    /// <summary>
    /// RCCL-specific configuration
    /// </summary>
    public class RCCLConfig
    {
        /// <summary>
        /// Use RCCL's ring-based all-reduce (default)
        /// </summary>
        public bool UseRingAllReduce { get; set; } = true;

        /// <summary>
        /// Use RCCL's tree-based all-reduce
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
        /// Enable RCCL debugging
        /// </summary>
        public bool EnableDebug { get; set; } = false;

        /// <summary>
        /// RCCL buffer size for multi-threaded communication
        /// </summary>
        public int BufferSize { get; set; } = 4194304; // 4MB

        /// <summary>
        /// Set RCCL environment variable
        /// </summary>
        public static void SetEnvironmentVariable(string key, string value)
        {
            Environment.SetEnvironmentVariable(key, value);
        }

        /// <summary>
        /// Apply RCCL configuration
        /// </summary>
        public void Apply()
        {
            SetEnvironmentVariable("RCCL_DEBUG", EnableDebug ? "INFO" : "WARN");
            SetEnvironmentVariable("RCCL_BUFFSIZE", BufferSize.ToString());

            if (NumChannels > 1)
            {
                SetEnvironmentVariable("RCCL_NCHANNELS", NumChannels.ToString());
            }
        }
    }
}
```

## Implementation Notes

1. **File Structure:**
   - `src/MLFramework/Communication/Backends/RCCLBackend.cs`
   - `src/MLFramework/Communication/Backends/RCCLBackendFactory.cs`
   - `src/MLFramework/Communication/Backends/Native/RCCLNative.cs`
   - `src/MLFramework/Communication/Backends/RCCLConfig.cs`

2. **Design Decisions:**
   - Use P/Invoke to call RCCL C API
   - Detect rank/world size from environment variables
   - Support both ring and tree all-reduce algorithms
   - Validate tensors are on GPU before operations

3. **Error Handling:**
   - Throw CommunicationException for RCCL errors
   - Validate tensor device before operations
   - Check RCCL availability before creating backend

4. **Performance Considerations:**
   - Use pinned memory for efficient transfers
   - Support GPU-direct for zero-copy transfers
   - Multi-channel configuration for better bandwidth utilization
   - Optimize all-reduce algorithm based on tensor size

## Testing Requirements
- Mock tests for RCCL backend interface
- Tests for factory with/without RCCL availability
- Tests for environment variable parsing
- Tests for RCCL configuration application

## Success Criteria
- RCCL backend compiles with P/Invoke declarations
- Factory correctly detects RCCL availability
- Environment variables parsed correctly
- Configuration applied correctly
- Interface methods implemented correctly
