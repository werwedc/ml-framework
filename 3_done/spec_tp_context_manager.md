# Spec: Tensor Parallelism Context Manager

## Overview
Implement the `TensorParallel` context manager that initializes, manages, and scopes tensor parallelism across the application. This provides a clean API for setting up TP groups, managing rank/world size state, and ensuring proper cleanup.

## Context
Tensor Parallelism requires global state management (world size, rank, communicator) that must be accessible to TP layers but properly scoped to avoid interference between different experiments or models. The context manager pattern provides RAII-style initialization and cleanup.

## Implementation Details

### 1. TensorParallel Context Class

```csharp
namespace MLFramework.Distributed.TensorParallel;

public class TensorParallelContext : IDisposable
{
    private static TensorParallelContext? _current;
    private static readonly object _lock = new object();

    private readonly ICommunicator _communicator;
    private readonly int _worldSize;
    private readonly int _rank;
    private readonly bool _ownsCommunicator;
    private readonly List<TensorParallelGroup> _processGroups;

    public int WorldSize => _worldSize;
    public int Rank => _rank;
    public ICommunicator Communicator => _communicator;
    public static TensorParallelContext? Current => _current;

    private TensorParallelContext(ICommunicator communicator, bool ownsCommunicator)
    {
        _communicator = communicator;
        _worldSize = communicator.WorldSize;
        _rank = communicator.Rank;
        _ownsCommunicator = ownsCommunicator;
        _processGroups = new List<TensorParallelGroup>();

        // Set as current context
        lock (_lock)
        {
            _current = this;
        }
    }

    /// <summary>
    /// Initializes tensor parallelism with a new communicator
    /// </summary>
    public static TensorParallelContext Initialize(int worldSize, int rank, string backend = "mock")
    {
        var config = new Dictionary<string, object>
        {
            ["world_size"] = worldSize,
            ["rank"] = rank
        };
        var communicator = CommunicatorFactory.Create(backend, config);
        return new TensorParallelContext(communicator, ownsCommunicator: true);
    }

    /// <summary>
    /// Initializes tensor parallelism with an existing communicator
    /// </summary>
    public static TensorParallelContext Initialize(ICommunicator communicator)
    {
        return new TensorParallelContext(communicator, ownsCommunicator: false);
    }

    /// <summary>
    /// Creates a process group for a subset of ranks
    /// </summary>
    public TensorParallelGroup CreateProcessGroup(List<int> ranks)
    {
        var group = new TensorParallelGroup(_communicator, ranks, _rank);
        _processGroups.Add(group);
        return group;
    }

    /// <summary>
    /// Gets or creates the default TP group (all ranks)
    /// </summary>
    public TensorParallelGroup DefaultGroup { get; } = new TensorParallelGroup(null, null, -1);

    public void Dispose()
    {
        // Cleanup process groups
        foreach (var group in _processGroups)
        {
            group.Dispose();
        }
        _processGroups.Clear();

        // Dispose communicator if we own it
        if (_ownsCommunicator)
        {
            _communicator.Dispose();
        }

        // Clear current context
        lock (_lock)
        {
            if (_current == this)
            {
                _current = null;
            }
        }
    }
}
```

### 2. Simplified API Wrapper

```csharp
public static class TensorParallel
{
    /// <summary>
    /// Initializes TP with default settings (for common use cases)
    /// </summary>
    public static TensorParallelContext Initialize(int worldSize, int rank, string backend = "mock")
    {
        return TensorParallelContext.Initialize(worldSize, rank, backend);
    }

    /// <summary>
    /// Gets the current TP context (throws if not initialized)
    /// </summary>
    public static TensorParallelContext GetContext()
    {
        return TensorParallelContext.Current
            ?? throw new InvalidOperationException(
                "TensorParallel context not initialized. Call TensorParallel.Initialize() first.");
    }

    /// <summary>
    /// Tries to get the current TP context (returns null if not initialized)
    /// </summary>
    public static TensorParallelContext? TryGetContext()
    {
        return TensorParallelContext.Current;
    }

    /// <summary>
    /// Checks if TP is currently active
    /// </summary>
    public static bool IsInitialized => TensorParallelContext.Current != null;

    /// <summary>
    /// Gets the world size from current context
    /// </summary>
    public static int GetWorldSize() => GetContext().WorldSize;

    /// <summary>
    /// Gets the rank from current context
    /// </summary>
    public static int GetRank() => GetContext().Rank;

    /// <summary>
    /// Gets the communicator from current context
    /// </summary>
    public static ICommunicator GetCommunicator() => GetContext().Communicator;
}
```

### 3. Process Group Class (Refined from communication spec)

```csharp
public class TensorParallelGroup : IDisposable
{
    private readonly ICommunicator? _globalCommunicator;
    private readonly List<int>? _ranks;
    private readonly int _globalRank;
    private readonly bool _isDefaultGroup;

    public int WorldSize { get; }
    public int LocalRank { get; }
    public bool InGroup { get; }

    /// <summary>
    /// Creates a process group for a subset of ranks
    /// </summary>
    public TensorParallelGroup(ICommunicator? globalComm, List<int>? ranks, int myGlobalRank)
    {
        if (globalComm == null && ranks == null)
        {
            // Default group (all ranks in TP context)
            _isDefaultGroup = true;
            _globalCommunicator = null;
            _ranks = null;
            _globalRank = -1;

            // Will be set when context is available
            var ctx = TensorParallel.TryGetContext();
            if (ctx != null)
            {
                WorldSize = ctx.WorldSize;
                LocalRank = ctx.Rank;
                InGroup = true;
            }
            else
            {
                WorldSize = 1;
                LocalRank = 0;
                InGroup = true;
            }
        }
        else
        {
            _isDefaultGroup = false;
            _globalCommunicator = globalComm ?? throw new ArgumentNullException(nameof(globalComm));
            _ranks = ranks ?? throw new ArgumentNullException(nameof(ranks));
            _globalRank = myGlobalRank;

            // Check if this rank is in the group
            InGroup = _ranks.Contains(_globalRank);

            if (InGroup)
            {
                WorldSize = _ranks.Count;
                LocalRank = _ranks.IndexOf(_globalRank);
            }
            else
            {
                WorldSize = 0;
                LocalRank = -1;
            }
        }
    }

    public Task<Tensor> AllReduceAsync(Tensor tensor, ReduceOperation operation)
    {
        if (_isDefaultGroup)
        {
            var ctx = TensorParallel.GetContext();
            return ctx.Communicator.AllReduceAsync(tensor, operation);
        }
        else if (InGroup)
        {
            // Filter to only ranks in this group
            return _globalCommunicator!.AllReduceAsync(tensor, operation);
        }
        else
        {
            // This rank is not in the group, return tensor unchanged
            return Task.FromResult(tensor);
        }
    }

    public Task<Tensor> AllGatherAsync(Tensor tensor, int dim = 0)
    {
        if (_isDefaultGroup)
        {
            var ctx = TensorParallel.GetContext();
            return ctx.Communicator.AllGatherAsync(tensor, dim);
        }
        else if (InGroup)
        {
            return _globalCommunicator!.AllGatherAsync(tensor, dim);
        }
        else
        {
            return Task.FromResult(tensor);
        }
    }

    public void Dispose()
    {
        // Cleanup if needed
    }
}
```

### 4. TP State Helper

```csharp
public static class TPState
{
    /// <summary>
    /// Helper method to safely execute code only on specific ranks
    /// </summary>
    public static Task ExecuteOnRankAsync(int targetRank, Func<Task> action)
    {
        var rank = TensorParallel.GetRank();
        if (rank == targetRank)
        {
            return action();
        }
        return Task.CompletedTask;
    }

    /// <summary>
    /// Helper method to execute code only on the master (rank 0)
    /// </summary>
    public static Task ExecuteOnMasterAsync(Func<Task> action)
    {
        return ExecuteOnRankAsync(0, action);
    }

    /// <summary>
    /// Executes code on all ranks, but waits for all to complete
    /// </summary>
    public static async Task ExecuteOnAllAsync(Func<Task> action)
    {
        await Task.WhenAll(
            Enumerable.Range(0, TensorParallel.GetWorldSize())
                     .Select(rank => TPState.ExecuteOnRankAsync(rank, action))
        );

        // Barrier to ensure all ranks have completed
        await TensorParallel.GetCommunicator().BarrierAsync();
    }
}
```

## Files to Create

### Source Files
- `src/MLFramework/Distributed/TensorParallel/TensorParallelContext.cs`
- `src/MLFramework/Distributed/TensorParallel/TensorParallel.cs` (static helper)
- `src/MLFramework/Distributed/TensorParallel/TensorParallelGroup.cs`
- `src/MLFramework/Distributed/TensorParallel/TPState.cs`

### Test Files
- `tests/MLFramework.Tests/Distributed/TensorParallel/TensorParallelContextTests.cs`
- `tests/MLFramework.Tests/Distributed/TensorParallel/TensorParallelGroupTests.cs`

## Test Requirements

1. **Context Initialization Tests**
   - Test Initialize(int, int) creates valid context
   - Test Initialize(ICommunicator) uses provided communicator
   - Test WorldSize and Rank properties return correct values

2. **Context Scope Tests**
   - Test nested contexts (should throw or replace)
   - Test Dispose clears current context
   - Test multiple sequential contexts

3. **Static API Tests**
   - Test GetContext() throws when not initialized
   - Test TryGetContext() returns null when not initialized
   - Test IsInitialized flag

4. **Process Group Tests**
   - Test CreateProcessGroup() creates valid groups
   - Test InGroup property correctly identifies membership
   - Test operations work only on group members
   - Test default group uses all ranks

5. **TPState Helper Tests**
   - Test ExecuteOnRankAsync executes on correct rank only
   - Test ExecuteOnMasterAsync executes on rank 0 only
   - Test ExecuteOnAllAsync synchronizes all ranks

## Dependencies
- `ICommunicator` and `CommunicatorFactory` from communication primitives
- .NET IDisposable pattern
- .NET Task-based async

## Success Criteria
- [ ] Context can be initialized and disposed correctly
- [ ] WorldSize and Rank accessible through context and static API
- [ ] Process groups can be created and filtered correctly
- [ ] TPState helpers execute on correct ranks
- [ ] Dispose properly cleans up communicator ownership
- [ ] Unit tests cover all major scenarios
- [ ] Thread-safe access to current context

## Estimated Time
45-60 minutes
