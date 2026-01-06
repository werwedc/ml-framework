# Spec: Communication Backend Interface

## Overview
Define the abstraction layer for communication primitives used in distributed training. This interface allows multiple backend implementations (NCCL, Gloo, etc.) to be plugged in.

## Requirements
- Define interfaces for core communication primitives: AllReduce, Broadcast, Barrier
- Support both synchronous and asynchronous operations
- Tensor-agnostic design (works with any tensor type)
- Support different reduction operations (Sum, Avg, Max, Min)

## Classes and Interfaces

### 1. ICommunicationBackend Interface
```csharp
public interface ICommunicationBackend
{
    string Name { get; }
    bool IsAvailable { get; }
    void Initialize();
    void Finalize();
}
```

### 2. IProcessGroup Interface
```csharp
public interface IProcessGroup
{
    int Rank { get; }
    int WorldSize { get; }
    ICommunicationBackend Backend { get; }

    // Synchronous communication primitives
    void AllReduce(Tensor tensor, ReduceOp op = ReduceOp.Sum);
    void Broadcast(Tensor tensor, int root = 0);
    void Barrier();

    // Asynchronous versions
    Task AllReduceAsync(Tensor tensor, ReduceOp op = ReduceOp.Sum);
    Task BroadcastAsync(Tensor tensor, int root = 0);
    Task BarrierAsync();

    void Destroy();
}
```

### 3. ReduceOp Enum
```csharp
public enum ReduceOp
{
    Sum,
    Avg,
    Max,
    Min,
    Product
}
```

### 4. CommunicationException Class
```csharp
public class CommunicationException : Exception
{
    public int Rank { get; }
    public string BackendName { get; }

    public CommunicationException(string message, int rank, string backendName)
        : base(message)
    {
        Rank = rank;
        BackendName = backendName;
    }
}
```

## Implementation Notes

### Tensor Compatibility
- Work with the existing Tensor class from the framework
- Support both CPU and CUDA tensors
- Backend should handle tensor device placement

### Error Handling
- All communication methods should throw CommunicationException on failure
- Include rank and backend information for debugging
- Handle timeout scenarios

### Async Operations
- Use Task-based async pattern
- Ensure thread-safety for concurrent async calls
- Provide cancellation token support

## Success Criteria
- [ ] Interface definitions compile without errors
- [ ] Clear separation between process group and backend concerns
- [ ] Support for both sync and async operations
- [ ] Comprehensive enum for reduction operations
- [ ] Proper exception hierarchy for communication errors

## Dependencies
- Existing Tensor class (from src/)
- No external communication library dependencies yet (those come in backend implementations)

## Testing
- Unit tests will verify interface contracts (implemented in spec_ddp_tests.md)
- Mock implementations for testing process group logic without actual communication
