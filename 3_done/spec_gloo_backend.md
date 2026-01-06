# Spec: Gloo Backend Implementation

## Overview
Implement the Gloo backend for communication primitives. Gloo provides CPU and multi-GPU communication, and works on both Linux and Windows, making it a good fallback when NCCL is not available.

## Requirements
- Wrap Gloo library functions for AllReduce, Broadcast, and Barrier
- Handle Gloo initialization and cleanup
- Support both CPU and CUDA tensors
- Provide Gloo availability detection
- Handle Gloo-specific error codes

## Classes

### 1. GlooBackend Class
```csharp
public class GlooBackend : ICommunicationBackend, IDisposable
{
    private IntPtr _context;  // gloo::rendezvous::Context
    private int _rank;
    private int _worldSize;
    private bool _initialized;

    public string Name => "Gloo";
    public bool IsAvailable => CheckAvailability();

    public GlooBackend()
    {
        _initialized = false;
        _context = IntPtr.Zero;
    }

    /// <summary>
    /// Initialize Gloo context.
    /// Must be called before any communication operations.
    /// </summary>
    public void Initialize()
    {
        // Read environment variables
        _rank = GetEnvVar("RANK", 0);
        _worldSize = GetEnvVar("WORLD_SIZE", 1);
        var iface = GetEnvVar("GLOO_IFACE", "eth0");
        var transport = GetEnvVar("GLOO_DEVICE_TRANSPORT", "tcp");

        // Create Gloo context
        _context = CreateContext(_rank, _worldSize, iface, transport);

        _initialized = true;
    }

    /// <summary>
    /// Finalize Gloo context and free resources.
    /// </summary>
    public void Finalize()
    {
        if (_initialized)
        {
            DestroyContext(_context);
            _context = IntPtr.Zero;
            _initialized = false;
        }
    }

    /// <summary>
    /// Check if Gloo is available on this system.
    /// </summary>
    public static bool CheckAvailability()
    {
        try
        {
            // Try to load Gloo library
            if (!LoadGlooLibrary())
                return false;

            return true;
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Get integer environment variable or default value.
    /// </summary>
    private static int GetEnvVar(string name, int defaultValue);

    /// <summary>
    /// Get string environment variable or default value.
    /// </summary>
    private static string GetEnvVar(string name, string defaultValue);

    public void Dispose()
    {
        Finalize();
    }
}
```

### 2. GlooProcessGroup Class
```csharp
/// <summary>
/// Process group implementation using Gloo backend.
/// </summary>
public class GlooProcessGroup : IProcessGroup
{
    private readonly GlooBackend _backend;

    public GlooProcessGroup(GlooBackend backend)
    {
        _backend = backend;
        backend.Initialize();
    }

    public int Rank => _backend._rank;
    public int WorldSize => _backend._worldSize;
    public ICommunicationBackend Backend => _backend;

    public void AllReduce(Tensor tensor, ReduceOp op = ReduceOp.Sum)
    {
        if (tensor.Device.IsCPU)
        {
            AllReduceCPU(tensor, op);
        }
        else if (tensor.Device.IsCUDA)
        {
            AllReduceCUDA(tensor, op);
        }
        else
        {
            throw new ArgumentException($"Unsupported device: {tensor.Device}");
        }
    }

    public Task AllReduceAsync(Tensor tensor, ReduceOp op = ReduceOp.Sum)
    {
        // Gloo's async support is limited, so we wrap sync operation
        return Task.Run(() => AllReduce(tensor, op));
    }

    public void Broadcast(Tensor tensor, int root = 0)
    {
        if (tensor.Device.IsCPU)
        {
            BroadcastCPU(tensor, root);
        }
        else if (tensor.Device.IsCUDA)
        {
            BroadcastCUDA(tensor, root);
        }
        else
        {
            throw new ArgumentException($"Unsupported device: {tensor.Device}");
        }
    }

    public Task BroadcastAsync(Tensor tensor, int root = 0)
    {
        return Task.Run(() => Broadcast(tensor, root));
    }

    public void Barrier()
    {
        GlooBarrier(_backend._context);
    }

    public Task BarrierAsync()
    {
        return Task.Run(() => Barrier());
    }

    public void Destroy()
    {
        _backend.Finalize();
    }

    private void AllReduceCPU(Tensor tensor, ReduceOp op);
    private void AllReduceCUDA(Tensor tensor, ReduceOp op);
    private void BroadcastCPU(Tensor tensor, int root);
    private void BroadcastCUDA(Tensor tensor, int root);
}
```

### 3. GlooAllReduce Class (Internal)
```csharp
/// <summary>
/// Implements AllReduce using Gloo's ring algorithm.
/// </summary>
internal class GlooAllReduce
{
    private readonly IntPtr _context;

    public GlooAllReduce(IntPtr context)
    {
        _context = context;
    }

    /// <summary>
    /// Perform AllReduce on a CPU tensor using Gloo's ring algorithm.
    /// </summary>
    public void AllReduceCPU(Tensor tensor, ReduceOp op)
    {
        var dataType = GetGlooDataType(tensor.DType);
        var opType = GetGlooOp(op);
        var ptr = tensor.Storage.DataPointer;
        var count = tensor.NumElements;

        GlooAllReduceNative(_context, ptr, ptr, count, dataType, opType);
    }

    /// <summary>
    /// Perform AllReduce on a CUDA tensor.
    /// Gloo transfers to CPU, reduces, and transfers back.
    /// </summary>
    public void AllReduceCUDA(Tensor tensor, ReduceOp op)
    {
        // Copy to CPU
        var cpuTensor = tensor.To(Device.CPU);

        // Reduce on CPU
        AllReduceCPU(cpuTensor, op);

        // Copy back to CUDA
        tensor.Copy_(cpuTensor);
    }

    private glooDataType_t GetGlooDataType(DataType dtype);
    private glooRedOp_t GetGlooOp(ReduceOp op);
}
```

## P/Invoke Declarations

```csharp
internal static class GlooNative
{
    private const string GlooLib = "gloo";

    // Context management
    [DllImport(GlooLib, CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr gloo_create_context(
        int rank,
        int size,
        string iface,
        string transport);

    [DllImport(GlooLib, CallingConvention = CallingConvention.Cdecl)]
    public static extern void gloo_destroy_context(IntPtr context);

    // AllReduce
    [DllImport(GlooLib, CallingConvention = CallingConvention.Cdecl)]
    public static extern void gloo_allreduce(
        IntPtr context,
        IntPtr sendbuf,
        IntPtr recvbuf,
        long count,
        glooDataType_t datatype,
        glooRedOp_t op);

    // Broadcast
    [DllImport(GlooLib, CallingConvention = CallingConvention.Cdecl)]
    public static extern void gloo_broadcast(
        IntPtr context,
        IntPtr buffer,
        long count,
        glooDataType_t datatype,
        int root);

    // Barrier
    [DllImport(GlooLib, CallingConvention = CallingConvention.Cdecl)]
    public static extern void gloo_barrier(IntPtr context);

    public static void CheckError()
    {
        // Gloo uses exceptions in C++ layer, so we need exception handling wrapper
    }
}

internal enum glooDataType_t
{
    glooInt8 = 0,
    glooUint8 = 1,
    glooInt32 = 2,
    glooUint32 = 3,
    glooInt64 = 4,
    glooUint64 = 5,
    glooFloat16 = 6,
    glooFloat32 = 7,
    glooFloat64 = 8
}

internal enum glooRedOp_t
{
    glooSum = 0,
    glooProduct = 1,
    glooMax = 2,
    glooMin = 3,
    glooAvg = 4
}
```

## Implementation Details

### Initialization Flow

1. **Read Environment**: Get `RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`, `GLOO_IFACE`, `GLOO_DEVICE_TRANSPORT`
2. **Create Context**: Initialize Gloo rendezvous context
3. **Connect**: Establish connections with all other ranks via TCP

### Network Configuration

**Default Interface**: `eth0` (configurable via `GLOO_IFACE`)

**Transport Options**:
- `tcp`: TCP transport (most compatible)
- `ib`: InfiniBand (for HPC clusters)
- `rocm`: AMD ROCm GPU support

### CPU vs CUDA Handling

**CPU Tensors**:
- Direct communication using Gloo's CPU operations
- Most efficient path

**CUDA Tensors**:
- Copy tensor to CPU
- Perform AllReduce on CPU
- Copy result back to CUDA
- Less efficient than NCCL, but provides compatibility

**Alternative** (future optimization):
- Use Gloo's CUDA backend if available
- Direct GPU-to-GPU communication via Gloo

### Device Transport

Gloo needs to know how to handle CUDA tensors. Options:

1. **CPU-only**: Copy all tensors to CPU (current implementation)
2. **CUDA-enabled**: Use Gloo's CUDA transport for direct GPU communication

For this spec, we implement CPU-only transport first. CUDA transport can be added later.

### Error Handling

- Wrap Gloo C++ exceptions in C# exception handling
- Convert to CommunicationException with rank and backend info
- Handle network errors gracefully

### Rendezvous

Gloo uses a rendezvous protocol to establish connections:
1. All processes connect to a master (rank 0)
2. Master coordinates connection establishment
3. Pairwise connections are established in a mesh

## Usage Example

```csharp
// Initialize Gloo backend (works on Windows and Linux)
var backend = new GlooBackend();
backend.Initialize();

// Create process group
var processGroup = new GlooProcessGroup(backend);

// Use for communication
var tensor = Tensor.Random(1000);
processGroup.AllReduce(tensor, ReduceOp.Sum);
```

## Success Criteria
- [ ] Gloo backend initializes correctly on both Linux and Windows
- [ ] AllReduce produces correct results for CPU tensors
- [ ] AllReduce works for CUDA tensors (via CPU fallback)
- [ ] Broadcast works from rank 0 to all ranks
- [ ] Barrier synchronizes all ranks
- [ ] Error handling works correctly
- [ ] Availability detection works correctly

## Dependencies
- spec_communication_backend_interface.md (interfaces)
- spec_process_group.md (process group concept)
- Existing tensor operations (for CPU/CUDA conversion)
- Gloo library (external dependency)

## Testing
- Unit tests in spec_ddp_tests.md will verify:
  - Correct initialization
  - AllReduce correctness (CPU and CUDA)
  - Broadcast correctness
  - Barrier synchronization
  - Error handling
  - Cross-platform compatibility

## External Dependencies

- **Gloo Library**: Must be installed and available on system
- **Python**: Gloo is built with Python bindings, may need Python installation
- **Library Loading**: Need to dynamically load libgloo.so (Linux) or gloo.dll (Windows)

## Comparison with NCCL

| Feature | NCCL | Gloo |
|---------|------|------|
| Platform | Linux only | Linux, Windows |
| GPU Support | Optimized | Via CPU or CUDA backend |
| Performance | Best for NVIDIA GPUs | Good for CPU, slower for GPU |
| Availability | Requires NVIDIA hardware | Works on any hardware |

Gloo is recommended when:
- Running on Windows
- Using non-NVIDIA hardware
- CPU-only distributed training
- Need cross-platform compatibility

NCCL is recommended when:
- Training on NVIDIA GPUs on Linux
- Maximum performance is required
- Multi-GPU training is primary use case
