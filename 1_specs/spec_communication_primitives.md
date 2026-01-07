# Spec: Communication Primitives Wrapper for Tensor Parallelism

## Overview
Implement a high-level wrapper for collective communication operations required by Tensor Parallelism (TP). This provides the foundation for all tensor parallel operations by abstracting backend-specific details (NCCL, RCCL, MPI) and offering async operations for communication-computation overlap.

## Context
Tensor Parallelism requires efficient collective operations to synchronize data across devices during both forward and backward passes. This spec focuses on creating a clean, async-capable API that hides backend complexity while enabling performance optimization.

## Implementation Details

### 1. Core Interface: `ICommunicator`

```csharp
namespace MLFramework.Distributed.Communication;

public interface ICommunicator : IDisposable
{
    /// <summary>
    /// Gets the world size (total number of processes/ranks)
    /// </summary>
    int WorldSize { get; }

    /// <summary>
    /// Gets the rank of this process (0 to WorldSize-1)
    /// </summary>
    int Rank { get; }

    /// <summary>
    /// Performs all-reduce operation: sums tensors across all ranks and distributes result to all
    /// </summary>
    Task<Tensor> AllReduceAsync(Tensor tensor, ReduceOperation operation = ReduceOperation.Sum);

    /// <summary>
    /// Performs all-gather operation: gathers tensor shards from all ranks and concatenates
    /// </summary>
    Task<Tensor> AllGatherAsync(Tensor tensor, int dim = 0);

    /// <summary>
    /// Performs reduce-scatter operation: reduces then scatters result chunks
    /// </summary>
    Task<Tensor> ReduceScatterAsync(Tensor tensor, ReduceOperation operation = ReduceOperation.Sum);

    /// <summary>
    /// Broadcasts a tensor from a specific rank to all other ranks
    /// </summary>
    Task<Tensor> BroadcastAsync(Tensor tensor, int root);

    /// <summary>
    /// Barrier: synchronizes all ranks
    /// </summary>
    Task BarrierAsync();
}
```

### 2. Enum for Reduction Operations

```csharp
public enum ReduceOperation
{
    Sum,
    Max,
    Min,
    Product,
    Avg
}
```

### 3. Backend Abstraction Classes

#### 3.1 Base Backend Class

```csharp
public abstract class CommunicatorBackend : ICommunicator
{
    protected readonly int _worldSize;
    protected readonly int _rank;

    protected CommunicatorBackend(int worldSize, int rank)
    {
        _worldSize = worldSize;
        _rank = rank;
    }

    public int WorldSize => _worldSize;
    public int Rank => _rank;

    public abstract Task<Tensor> AllReduceAsync(Tensor tensor, ReduceOperation operation);
    public abstract Task<Tensor> AllGatherAsync(Tensor tensor, int dim);
    public abstract Task<Tensor> ReduceScatterAsync(Tensor tensor, ReduceOperation operation);
    public abstract Task<Tensor> BroadcastAsync(Tensor tensor, int root);
    public abstract Task BarrierAsync();

    public abstract void Dispose();
}
```

#### 3.2 Mock/CPU Backend (for testing)

```csharp
public class MockCommunicator : CommunicatorBackend
{
    private readonly Dictionary<int, Tensor> _sharedMemory;

    public MockCommunicator(int worldSize, int rank)
        : base(worldSize, rank)
    {
        _sharedMemory = new Dictionary<int, Tensor>();
    }

    public override async Task<Tensor> AllReduceAsync(Tensor tensor, ReduceOperation operation)
    {
        // Store this rank's tensor
        _sharedMemory[_rank] = tensor;

        // Wait for all ranks to contribute (simulated delay)
        await Task.Delay(1);

        // Collect all tensors
        var tensors = _sharedMemory.Values.OrderBy(k => _sharedMemory.Keys.First()).ToList();
        var result = tensor.Clone();

        // Apply reduction operation
        for (int i = 1; i < tensors.Count; i++)
        {
            switch (operation)
            {
                case ReduceOperation.Sum:
                    result += tensors[i];
                    break;
                case ReduceOperation.Max:
                    result = Tensor.Maximum(result, tensors[i]);
                    break;
                // ... other operations
            }
        }

        return result;
    }

    // Implement other methods similarly...
}
```

#### 3.3 NCCL Backend (stub for future GPU implementation)

```csharp
public class NCCLCommunicator : CommunicatorBackend
{
    private IntPtr _communicatorHandle;

    public NCCLCommunicator(int worldSize, int rank, int device)
        : base(worldSize, rank)
    {
        // Initialize NCCL communicator
        // _communicatorHandle = NCCL.InitComm(worldSize, rank, device);
        throw new NotImplementedException("GPU backend not yet implemented");
    }

    public override Task<Tensor> AllReduceAsync(Tensor tensor, ReduceOperation operation)
    {
        // Call NCCL all-reduce with async callback
        throw new NotImplementedException();
    }

    // ... other methods
}
```

### 4. Communicator Factory

```csharp
public static class CommunicatorFactory
{
    public static ICommunicator Create(string backend = "mock", Dictionary<string, object>? config = null)
    {
        return backend.ToLowerInvariant() switch
        {
            "mock" => new MockCommunicator(
                config?.GetValueOrDefault("world_size", 1) as int? ?? 1,
                config?.GetValueOrDefault("rank", 0) as int? ?? 0
            ),
            "nccl" => new NCCLCommunicator(/* ... */),
            _ => throw new ArgumentException($"Unknown backend: {backend}")
        };
    }
}
```

### 5. Process Group Support (for advanced topologies)

```csharp
public class ProcessGroup
{
    private readonly ICommunicator _globalCommunicator;
    private readonly List<int> _ranks;
    private readonly int _localRank;
    private readonly Dictionary<int, int> _globalToLocalRank;

    public ProcessGroup(ICommunicator globalComm, List<int> ranks, int myGlobalRank)
    {
        _globalCommunicator = globalComm;
        _ranks = ranks;
        _globalToLocalRank = ranks.Select((rank, idx) => (rank, idx))
                                   .ToDictionary(x => x.rank, x => x.idx);
        _localRank = _globalToLocalRank[myGlobalRank];
    }

    public int WorldSize => _ranks.Count;
    public int LocalRank => _localRank;

    public Task<Tensor> AllReduceAsync(Tensor tensor, ReduceOperation operation)
    {
        // Filter operations to only involve ranks in this process group
        // For now, delegate to global communicator with subset
        return _globalCommunicator.AllReduceAsync(tensor, operation);
    }
}
```

## Files to Create

### Source Files
- `src/MLFramework/Distributed/Communication/ICommunicator.cs`
- `src/MLFramework/Distributed/Communication/ReduceOperation.cs`
- `src/MLFramework/Distributed/Communication/CommunicatorBackend.cs`
- `src/MLFramework/Distributed/Communication/MockCommunicator.cs`
- `src/MLFramework/Distributed/Communication/NCCLCommunicator.cs`
- `src/MLFramework/Distributed/Communication/CommunicatorFactory.cs`
- `src/MLFramework/Distributed/Communication/ProcessGroup.cs`

### Test Files
- `tests/MLFramework.Tests/Distributed/Communication/MockCommunicatorTests.cs`

## Test Requirements

1. **All-Reduce Tests**
   - Test Sum operation across 2-4 mock ranks
   - Test Max, Min, Product operations
   - Verify result tensor shape and values

2. **All-Gather Tests**
   - Test gathering sharded tensors along different dimensions
   - Verify concatenated output matches expected shape

3. **Broadcast Tests**
   - Test broadcasting from different root ranks
   - Verify all ranks receive identical tensors

4. **Process Group Tests**
   - Test creating subgroups from larger communicator
   - Verify operations only affect group members

## Dependencies
- Existing `Tensor` class in the framework
- .NET Task-based async patterns
- Dictionary and LINQ collections

## Success Criteria
- [ ] Mock communicator correctly simulates all-reduce, all-gather, broadcast
- [ ] Async operations return Tasks that complete properly
- [ ] Process groups can be created and used
- [ ] Unit tests pass for all communication operations
- [ ] Factory correctly instantiates appropriate backend
- [ ] IDisposable pattern properly implemented for cleanup

## Estimated Time
45-60 minutes
