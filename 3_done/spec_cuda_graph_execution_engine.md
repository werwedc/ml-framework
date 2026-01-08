# Spec: CUDA Graph Execution Engine

## Overview
Implement the graph execution engine that instantiates and replays captured CUDA graphs. This spec completes the CUDAGraph class implementation and adds graph instantiation and execution logic.

## Requirements

### 1. Complete CUDAGraph Implementation
Finish the CUDAGraph class stub from the previous spec.

```csharp
public class CUDAGraph : ICUDAGraph
{
    private IntPtr _cuGraph;
    private IntPtr _cuGraphExec;
    private readonly object _lock = new object();

    public string GraphId { get; }
    public CUDAGraphState State { get; private set; }

    public CUDAGraph(IntPtr graph)
    {
        _cuGraph = graph;
        _cuGraphExec = IntPtr.Zero;
        GraphId = Guid.NewGuid().ToString();
        State = CUDAGraphState.Created;
    }

    /// <summary>
    /// Instantiates the graph for execution
    /// </summary>
    private void Instantiate()
    {
        if (_cuGraphExec != IntPtr.Zero)
            return;

        lock (_lock)
        {
            // Double-check pattern
            if (_cuGraphExec != IntPtr.Zero)
                return;

            var result = CUDADriver.cuGraphInstantiate(
                out IntPtr graphExec,
                _cuGraph,
                IntPtr.Zero,
                0);

            CheckResult(result);
            _cuGraphExec = graphExec;
            State = CUDAGraphState.Ready;
        }
    }

    /// <summary>
    /// Executes the captured graph
    /// </summary>
    public void Execute(CUDAStream stream)
    {
        if (State == CUDAGraphState.Disposed)
            throw new ObjectDisposedException(nameof(CUDAGraph));

        // Instantiate graph if not already done
        Instantiate();

        State = CUDAGraphState.Executing;

        try
        {
            var result = CUDADriver.cuGraphLaunch(_cuGraphExec, stream.NativeHandle);
            CheckResult(result);
        }
        finally
        {
            State = CUDAGraphState.Ready;
        }
    }

    /// <summary>
    /// Validates that the graph is executable
    /// </summary>
    public CUDAGraphValidationResult Validate()
    {
        var errors = new List<string>();
        var warnings = new List<string>();

        // Get number of nodes in the graph
        var result = CUDADriver.cuGraphGetNodes(_cuGraph, out IntPtr nodes, out ulong nodeCount);
        CheckResult(result);

        // Basic validation
        if (nodeCount == 0)
        {
            errors.Add("Graph contains no operations");
        }

        // Check if graph can be instantiated
        try
        {
            Instantiate();
        }
        catch (CUDADriverException ex)
        {
            errors.Add($"Graph instantiation failed: {ex.Message}");
        }

        return new CUDAGraphValidationResult
        {
            IsValid = errors.Count == 0,
            Errors = errors,
            Warnings = warnings,
            OperationCount = (int)nodeCount
        };
    }

    public void Dispose()
    {
        lock (_lock)
        {
            if (_cuGraphExec != IntPtr.Zero)
            {
                CUDADriver.cuGraphExecDestroy(_cuGraphExec);
                _cuGraphExec = IntPtr.Zero;
            }

            if (_cuGraph != IntPtr.Zero)
            {
                CUDADriver.cuGraphDestroy(_cuGraph);
                _cuGraph = IntPtr.Zero;
            }

            State = CUDAGraphState.Disposed;
        }
    }

    private void CheckResult(CUResult result)
    {
        if (result != CUResult.Success)
        {
            throw new CUDADriverException($"CUDA error: {result}");
        }
    }

    // Internal property for weight updates
    internal IntPtr GraphExecHandle => _cuGraphExec;
}
```

### 2. Additional CUDA Driver API Bindings
Add bindings for graph operations.

```csharp
internal static partial class CUDADriver
{
    [DllImport("nvcuda", EntryPoint = "cuGraphInstantiate")]
    public static extern CUResult cuGraphInstantiate(
        out IntPtr graphExec,
        IntPtr graph,
        IntPtr nodeParams,
        long errorLogSize);

    [DllImport("nvcuda", EntryPoint = "cuGraphLaunch")]
    public static extern CUResult cuGraphLaunch(IntPtr graphExec, IntPtr stream);

    [DllImport("nvcuda", EntryPoint = "cuGraphExecDestroy")]
    public static extern CUResult cuGraphExecDestroy(IntPtr graphExec);

    [DllImport("nvcuda", EntryPoint = "cuGraphDestroy")]
    public static extern CUResult cuGraphDestroy(IntPtr graph);

    [DllImport("nvcuda", EntryPoint = "cuGraphGetNodes")]
    public static extern CUResult cuGraphGetNodes(
        IntPtr graph,
        out IntPtr nodes,
        out ulong numNodes);
}
```

## Implementation Details

### File Structure
- **File**: `src/CUDA/Graphs/CUDAGraph.cs` (complete implementation)
- **File**: `src/CUDA/Driver/CUDADriver.cs` (extend with execution methods)

### Dependencies
- ICUDAGraph interface (from spec_cuda_graph_core_interfaces)
- CUDAGraphState enum (from spec_cuda_graph_core_interfaces)
- CUDAGraphValidationResult class (from spec_cuda_graph_core_interfaces)
- CUDAStream class (existing)
- CUDADriverException class (existing)
- CUDAGraphCapture class (from spec_cuda_graph_capture_api)

### Threading Considerations
- Use double-check locking for lazy instantiation
- Lock around Execute calls if graph is being instantiated concurrently
- Thread-safe state management

### Error Handling
- Throw ObjectDisposedException if Execute is called after Dispose
- Throw CUDADriverException for CUDA API failures
- Handle instantiation failures gracefully
- Ensure proper cleanup in all error paths

## Success Criteria
- Graph can be successfully instantiated
- Graph execution launches on specified CUDA stream
- Multiple executions of the same graph work correctly
- Graph validation returns correct results
- Resources are properly disposed
- Thread-safe for concurrent execution

## Testing Requirements

### Unit Tests
- Test graph instantiation
- Test graph execution
- Test multiple executions of the same graph
- Test validation with empty graph
- Test validation with valid graph
- Test Dispose cleans up resources
- Test Execute after Dispose throws exception

### Integration Tests
- Test full capture -> instantiate -> execute cycle
- Test execution across multiple streams
- Test concurrent execution from multiple threads
- Test execution with actual kernel launches (requires GPU)
