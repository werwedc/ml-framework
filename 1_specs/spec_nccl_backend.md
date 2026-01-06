# Spec: NCCL Backend Implementation

## Overview
Implement the NVIDIA NCCL backend for communication primitives. NCCL provides optimized GPU-to-GPU communication on NVIDIA hardware.

## Requirements
- Wrap NCCL library functions for AllReduce, Broadcast, and Barrier
- Handle NCCL initialization and cleanup
- Support both CUDA tensors
- Provide NCCL availability detection
- Handle NCCL-specific error codes

## Classes

### 1. NCCLBackend Class
```csharp
public class NCCLBackend : ICommunicationBackend, IDisposable
{
    private IntPtr _comm;  // ncclComm_t
    private int _rank;
    private int _worldSize;
    private bool _initialized;

    public string Name => "NCCL";
    public bool IsAvailable => CheckAvailability();

    public NCCLBackend()
    {
        _initialized = false;
        _comm = IntPtr.Zero;
    }

    /// <summary>
    /// Initialize NCCL communicator.
    /// Must be called before any communication operations.
    /// </summary>
    public void Initialize()
    {
        // Read rank and world size from environment
        _rank = GetEnvVar("RANK", 0);
        _worldSize = GetEnvVar("WORLD_SIZE", 1);

        // Get device ID for this rank (assume rank 0 = device 0, etc.)
        var deviceId = _rank;
        SetDevice(deviceId);

        // Initialize NCCL communicator
        var uniqueId = GetUniqueId();
        NCCLInit(uniqueId, _worldSize, _rank, ref _comm);

        _initialized = true;
    }

    /// <summary>
    /// Finalize NCCL communicator and free resources.
    /// </summary>
    public void Finalize()
    {
        if (_initialized)
        {
            NCCLEnd(_comm);
            _comm = IntPtr.Zero;
            _initialized = false;
        }
    }

    /// <summary>
    /// Check if NCCL is available on this system.
    /// </summary>
    public static bool CheckAvailability()
    {
        try
        {
            // Try to load NCCL library
            if (!LoadNCLLLibrary())
                return false;

            // Check for CUDA availability
            if (!CUDA.IsAvailable())
                return false;

            return true;
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Get the unique ID for NCCL initialization.
    /// Rank 0 generates and broadcasts, others receive.
    /// </summary>
    private static NCCLUniqueId GetUniqueId();

    /// <summary>
    /// Set CUDA device for this rank.
    /// </summary>
    private void SetDevice(int deviceId);

    /// <summary>
    /// Get integer environment variable or default value.
    /// </summary>
    private static int GetEnvVar(string name, int defaultValue);

    public void Dispose()
    {
        Finalize();
    }
}
```

### 2. NCCLUniqueId Struct
```csharp
/// <summary>
/// NCCL unique ID for initialization (128 bytes).
/// </summary>
[StructLayout(LayoutKind.Sequential, Size = 128)]
public struct NCCLUniqueId
{
    private byte[] _data;

    public NCCLUniqueId(byte[] data)
    {
        if (data.Length != 128)
            throw new ArgumentException("NCCLUniqueId must be 128 bytes");
        _data = (byte[])data.Clone();
    }

    public byte[] Data => _data;

    public static NCCLUniqueId Generate()
    {
        var id = new byte[128];
        // Use NCCL's ncclGetUniqueId
        NCCLGetUniqueId(id);
        return new NCCLUniqueId(id);
    }
}
```

### 3. NCCLProcessGroup Class
```csharp
/// <summary>
/// Process group implementation using NCCL backend.
/// </summary>
public class NCCLProcessGroup : IProcessGroup
{
    private readonly NCCLBackend _backend;

    public NCCLProcessGroup(NCCLBackend backend)
    {
        _backend = backend;
        backend.Initialize();
    }

    public int Rank => _backend._rank;
    public int WorldSize => _backend._worldSize;
    public ICommunicationBackend Backend => _backend;

    public void AllReduce(Tensor tensor, ReduceOp op = ReduceOp.Sum)
    {
        var tensorPtr = tensor.Storage.DataPointer;
        var numElements = tensor.NumElements;
        var dataType = GetNCCLDataType(tensor.DType);

        NCCLAllReduce(tensorPtr, tensorPtr, numElements, dataType,
                      GetNCCLOp(op), _backend._comm, IntPtr.Zero);
    }

    public Task AllReduceAsync(Tensor tensor, ReduceOp op = ReduceOp.Sum)
    {
        var tensorPtr = tensor.Storage.DataPointer;
        var numElements = tensor.NumElements;
        var dataType = GetNCCLDataType(tensor.DType);

        var stream = CUDA.GetDefaultStream();
        NCCLAllReduce(tensorPtr, tensorPtr, numElements, dataType,
                      GetNCCLOp(op), _backend._comm, stream.Ptr);

        // Return task that completes when CUDA stream finishes
        return stream.RecordEvent().Task;
    }

    public void Broadcast(Tensor tensor, int root = 0)
    {
        var tensorPtr = tensor.Storage.DataPointer;
        var numElements = tensor.NumElements;
        var dataType = GetNCCLDataType(tensor.DType);

        NCCLBroadcast(tensorPtr, tensorPtr, numElements, dataType,
                      root, _backend._comm, IntPtr.Zero);
    }

    public Task BroadcastAsync(Tensor tensor, int root = 0)
    {
        var tensorPtr = tensor.Storage.DataPointer;
        var numElements = tensor.NumElements;
        var dataType = GetNCCLDataType(tensor.DType);

        var stream = CUDA.GetDefaultStream();
        NCCLBroadcast(tensorPtr, tensorPtr, numElements, dataType,
                      root, _backend._comm, stream.Ptr);

        return stream.RecordEvent().Task;
    }

    public void Barrier()
    {
        NCCLBarrier(_backend._comm, IntPtr.Zero);
    }

    public Task BarrierAsync()
    {
        var stream = CUDA.GetDefaultStream();
        NCCLBarrier(_backend._comm, stream.Ptr);
        return stream.RecordEvent().Task;
    }

    public void Destroy()
    {
        _backend.Finalize();
    }

    private ncclDataType_t GetNCCLDataType(DataType dtype);
    private ncclRedOp_t GetNCCLOp(ReduceOp op);
}
```

## P/Invoke Declarations

```csharp
internal static class NCCLNative
{
    private const string NCCLLib = "nccl";

    [DllImport(NCCLLib, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ncclGetUniqueId(byte[] uniqueId);

    [DllImport(NCCLLib, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ncclCommInitRank(
        ref IntPtr comm,
        int nranks,
        byte[] uniqueId,
        int rank);

    [DllImport(NCCLLib, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ncclCommDestroy(IntPtr comm);

    [DllImport(NCCLLib, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ncclAllReduce(
        IntPtr sendbuff,
        IntPtr recvbuff,
        ulong count,
        ncclDataType_t datatype,
        ncclRedOp_t op,
        IntPtr comm,
        IntPtr stream);

    [DllImport(NCCLLib, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ncclBroadcast(
        IntPtr buff,
        IntPtr buff,
        ulong count,
        ncclDataType_t datatype,
        int root,
        IntPtr comm,
        IntPtr stream);

    [DllImport(NCCLLib, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ncclBarrier(
        IntPtr comm,
        IntPtr stream);

    public static void CheckError(int error)
    {
        if (error != 0)
        {
            var errorMsg = GetErrorString(error);
            throw new CommunicationException(errorMsg, 0, "NCCL");
        }
    }

    private static string GetErrorString(int error);
}

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

internal enum ncclRedOp_t
{
    ncclSum = 0,
    ncclProd = 1,
    ncclMax = 2,
    ncclMin = 3,
    ncclAvg = 4
}
```

## Implementation Details

### Initialization Flow

1. **Read Environment**: Get `RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`
2. **Set CUDA Device**: Each rank sets its device (rank 0 = device 0, etc.)
3. **Generate Unique ID**: Rank 0 generates NCCL unique ID and broadcasts to others
4. **Initialize Comm**: Each rank initializes its NCCL communicator with the unique ID

### Unique ID Broadcasting

**Rank 0**:
```csharp
var uniqueId = NCCLUniqueId.Generate();
// Broadcast uniqueId via TCP (using MASTER_ADDR:MASTER_PORT)
BroadcastUniqueId(uniqueId);
```

**Other Ranks**:
```csharp
var uniqueId = ReceiveUniqueIdFromMaster();
// Initialize with received unique ID
```

**Broadcasting**: Use TCP sockets or existing Gloo backend for initial handshake

### Error Handling

- Check all NCCL function return codes
- Convert NCCL errors to CommunicationException
- Include rank information in error messages
- Handle timeout scenarios

### CUDA Stream Management

- Use default CUDA stream for synchronous operations
- For async operations, create events to track completion
- Ensure proper CUDA device context before operations

### Data Type Mapping

```csharp
private ncclDataType_t GetNCCLDataType(DataType dtype)
{
    return dtype switch
    {
        DataType.Float32 => ncclDataType_t.ncclFloat32,
        DataType.Float64 => ncclDataType_t.ncclFloat64,
        DataType.Int32 => ncclDataType_t.ncclInt32,
        DataType.Int64 => ncclDataType_t.ncclInt64,
        DataType.Float16 => ncclDataType_t.ncclFloat16,
        _ => throw new ArgumentException($"Unsupported dtype: {dtype}")
    };
}
```

### Reduction Operation Mapping

```csharp
private ncclRedOp_t GetNCCLOp(ReduceOp op)
{
    return op switch
    {
        ReduceOp.Sum => ncclRedOp_t.ncclSum,
        ReduceOp.Avg => ncclRedOp_t.ncclAvg,
        ReduceOp.Max => ncclRedOp_t.ncclMax,
        ReduceOp.Min => ncclRedOp_t.ncclMin,
        ReduceOp.Product => ncclRedOp_t.ncclProd,
        _ => throw new ArgumentException($"Unsupported op: {op}")
    };
}
```

## Success Criteria
- [ ] NCCL backend initializes correctly
- [ ] AllReduce produces correct results
- [ ] Broadcast works from rank 0 to all ranks
- [ ] Barrier synchronizes all ranks
- [ ] Async operations complete correctly
- [ ] Error handling converts NCCL errors properly
- [ ] Availability detection works correctly

## Dependencies
- spec_communication_backend_interface.md (interfaces)
- spec_process_group.md (process group concept)
- Existing CUDA integration (from src/)
- NCCL library (external dependency)

## Testing
- Unit tests in spec_ddp_tests.md will verify:
  - Correct initialization
  - AllReduce correctness
  - Broadcast correctness
  - Barrier synchronization
  - Async operation completion
  - Error handling for invalid inputs
- Integration tests require multiple GPUs

## External Dependencies

- **NCCL Library**: Must be installed and available on system
- **CUDA**: Required for GPU tensor support
- **Library Loading**: Need to dynamically load nccl.so (Linux) or nccl64.dll (Windows)

## Notes

- NCCL is Linux-only officially (Windows support is experimental)
- Alternative: Use Gloo backend for Windows support
- For AMD GPUs, RCCL backend would be similar implementation
