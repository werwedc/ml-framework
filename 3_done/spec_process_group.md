# Spec: Process Group Management

## Overview
Implement the core process group management system that tracks distributed training processes, handles initialization, and provides the main API for communication primitives.

## Requirements
- Track process rank and world size across all workers
- Initialize process groups with specified backend
- Handle graceful cleanup of process groups
- Support singleton process group per process
- Provide validation for rank and world size consistency

## Classes

### 1. ProcessGroup Class
```csharp
public class ProcessGroup : IProcessGroup, IDisposable
{
    private static ProcessGroup _defaultProcessGroup;
    private readonly ICommunicationBackend _backend;
    private readonly int _rank;
    private readonly int _worldSize;

    // Private constructor - use static factory methods
    private ProcessGroup(ICommunicationBackend backend, int rank, int worldSize);

    // Public properties
    public override int Rank => _rank;
    public override int WorldSize => _worldSize;
    public override ICommunicationBackend Backend => _backend;

    // Static factory method for initialization
    public static ProcessGroup Init(BackendType backendType,
                                   string initMethod = "env");

    // Get the default process group
    public static ProcessGroup Default => _defaultProcessGroup;

    // Destroy current process group
    public static void Destroy();

    // Communication primitives (delegate to backend)
    public override void AllReduce(Tensor tensor, ReduceOp op = ReduceOp.Sum);
    public override void Broadcast(Tensor tensor, int root = 0);
    public override void Barrier();
    public override Task AllReduceAsync(Tensor tensor, ReduceOp op = ReduceOp.Sum);
    public override Task BroadcastAsync(Tensor tensor, int root = 0);
    public override Task BarrierAsync();

    public override void Destroy();

    // IDisposable implementation
    public void Dispose();
}
```

### 2. BackendType Enum
```csharp
public enum BackendType
{
    NCCL,   // NVIDIA Collective Communications Library
    Gloo,   // CPU and multi-GPU communication
    MPI,    // Message Passing Interface (future)
    RCCL    // AMD ROCm (future)
}
```

### 3. BackendFactory Class
```csharp
public static class BackendFactory
{
    public static ICommunicationBackend CreateBackend(BackendType backendType)
    {
        return backendType switch
        {
            BackendType.NCCL => new NCCLBackend(),
            BackendType.Gloo => new GlooBackend(),
            _ => throw new ArgumentException($"Unsupported backend: {backendType}")
        };
    }

    public static bool IsBackendAvailable(BackendType backendType)
    {
        return backendType switch
        {
            BackendType.NCCL => NCCLBackend.IsAvailable(),
            BackendType.Gloo => GlooBackend.IsAvailable(),
            _ => false
        };
    }
}
```

## Implementation Details

### Initialization Methods

**"env" Method (Environment Variables)**
- Read `RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT` from environment
- Standard approach for most ML frameworks
- Used by torch.distributed and similar systems

**"tcp" Method (TCP-based initialization)**
- Discovery via master process
- Alternative for custom launchers
- Future implementation

### Singleton Pattern
- Only one active process group per process
- Attempting to create multiple groups throws InvalidOperationException
- Destroy must be called before creating a new group

### Error Handling
- Validate rank is in range [0, worldSize - 1]
- Validate worldSize > 0
- Check backend availability before initialization
- Throw CommunicationException for initialization failures

### Cleanup
- Destroy backend resources
- Clear static default process group
- Handle double-destroy gracefully

## Environment Variables

### Standard Variables (used by "env" init method)
- `RANK`: This process's rank (int)
- `WORLD_SIZE`: Total number of processes (int)
- `MASTER_ADDR`: Address of rank 0 (string, default: "127.0.0.1")
- `MASTER_PORT`: Port for initialization (int, default: 29500)
- `NCCL_DEBUG`: NCCL debug level (optional, for NCCL backend)

## Usage Example

```csharp
// Initialize from environment (typical usage with torch.multiprocessing equivalent)
var processGroup = ProcessGroup.Init(BackendType.NCCL, initMethod: "env");

Console.WriteLine($"Rank: {processGroup.Rank}, World Size: {processGroup.WorldSize}");

// Communication
var tensor = Tensor.Random(1000);
processGroup.AllReduce(tensor, ReduceOp.Sum);

// Access default process group
var defaultGroup = ProcessGroup.Default;

// Cleanup
ProcessGroup.Destroy();
```

## Success Criteria
- [ ] ProcessGroup can initialize from environment variables
- [ ] Singleton pattern prevents multiple active groups
- [ ] All communication primitives delegate correctly to backend
- [ ] Proper cleanup and disposal
- [ ] Environment variable validation with clear error messages
- [ ] Backend factory creates correct backend types

## Dependencies
- spec_communication_backend_interface.md (must be implemented first)
- Existing Tensor class

## Testing
- Unit tests in spec_ddp_tests.md will verify:
  - Environment variable parsing
  - Singleton behavior
  - Communication delegation
  - Error handling
  - Cleanup behavior
