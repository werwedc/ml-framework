# Spec: CUDA Graph Capture API

## Overview
Implement the core graph capture mechanism that records kernel launches and creates a reusable CUDA graph. This spec focuses on the capture functionality using CUDA driver APIs.

## Requirements

### 1. CUDAGraphCapture Class
Implement the graph capture class.

```csharp
public class CUDAGraphCapture : ICUDAGraphCapture
{
    private readonly IntPtr _cuStream;
    private IntPtr _cuGraph;
    private bool _isCapturing;

    public bool IsCapturing => _isCapturing;

    public CUDAGraphCapture()
    {
        _cuGraph = IntPtr.Zero;
        _cuStream = IntPtr.Zero;
        _isCapturing = false;
    }

    /// <summary>
    /// Begins capturing kernel launches from the specified stream
    /// </summary>
    public void BeginCapture(CUDAStream stream)
    {
        if (_isCapturing)
            throw new InvalidOperationException("Capture is already in progress");

        _cuStream = stream.NativeHandle;

        // Call CUDA driver API: cuStreamBeginCapture
        var result = CUDADriver.cuStreamBeginCapture(_cuStream, CUDACaptureMode.CaptureModeThreadLocal);
        CheckResult(result);

        _isCapturing = true;
    }

    /// <summary>
    /// Ends capture and instantiates the graph
    /// </summary>
    public ICUDAGraph EndCapture()
    {
        if (!_isCapturing)
            throw new InvalidOperationException("No capture in progress");

        // Call CUDA driver API: cuStreamEndCapture
        var result = CUDADriver.cuStreamEndCapture(_cuStream, out IntPtr graph);
        CheckResult(result);

        _cuGraph = graph;
        _isCapturing = false;

        return new CUDAGraph(_cuGraph);
    }

    public void Dispose()
    {
        if (_isCapturing)
        {
            // Abort capture if disposing while capturing
            CUDADriver.cuStreamEndCapture(_cuStream, out _);
        }
        _isCapturing = false;
    }

    private void CheckResult(CUResult result)
    {
        if (result != CUResult.Success)
        {
            throw new CUDADriverException($"CUDA error: {result}");
        }
    }
}
```

### 2. CUDA Driver API Bindings
Add necessary CUDA driver API bindings.

```csharp
internal static class CUDADriver
{
    [DllImport("nvcuda", EntryPoint = "cuStreamBeginCapture")]
    public static extern CUResult cuStreamBeginCapture(IntPtr stream, CUDACaptureMode mode);

    [DllImport("nvcuda", EntryPoint = "cuStreamEndCapture")]
    public static extern CUResult cuStreamEndCapture(IntPtr stream, out IntPtr graph);

    [DllImport("nvcuda", EntryPoint = "cuGraphInstantiate")]
    public static extern CUResult cuGraphInstantiate(
        out IntPtr graphExec,
        IntPtr graph,
        IntPtr nodeParams,
        long errorLogSize);
}

public enum CUDACaptureMode
{
    CaptureModeThreadLocal = 0,
    CaptureModeGlobal = 1,
    CaptureModeRelaxed = 2
}
```

### 3. CUDAGraph Stub
Create a stub CUDAGraph class that will be fully implemented in the Execution Engine spec.

```csharp
internal class CUDAGraph : ICUDAGraph
{
    private IntPtr _cuGraph;
    private IntPtr _cuGraphExec;

    public string GraphId { get; }
    public CUDAGraphState State { get; private set; }

    public CUDAGraph(IntPtr graph)
    {
        _cuGraph = graph;
        _cuGraphExec = IntPtr.Zero;
        GraphId = Guid.NewGuid().ToString();
        State = CUDAGraphState.Created;
    }

    public void Execute(CUDAStream stream)
    {
        throw new NotImplementedException("Execution will be implemented in next spec");
    }

    public CUDAGraphValidationResult Validate()
    {
        throw new NotImplementedException("Validation will be implemented in next spec");
    }

    public void Dispose()
    {
        if (_cuGraph != IntPtr.Zero)
        {
            // TODO: Call cuGraphDestroy when graph instance is created
            _cuGraph = IntPtr.Zero;
        }
    }
}
```

## Implementation Details

### File Structure
- **File**: `src/CUDA/Graphs/CUDAGraphCapture.cs`
- **File**: `src/CUDA/Graphs/CUDAGraph.cs` (stub, fully implemented later)
- **File**: `src/CUDA/Driver/CUDADriver.cs` (extend with capture methods)

### Dependencies
- ICUDAGraphCapture interface (from spec_cuda_graph_core_interfaces)
- ICUDAGraph interface (from spec_cuda_graph_core_interfaces)
- CUDAGraphState enum (from spec_cuda_graph_core_interfaces)
- CUDAStream class (existing)
- CUDADriverException class (existing)
- System.Runtime.InteropServices for P/Invoke

### Error Handling
- Throw InvalidOperationException if BeginCapture is called while already capturing
- Throw InvalidOperationException if EndCapture is called without BeginCapture
- Throw CUDADriverException for CUDA API failures
- Ensure proper cleanup in Dispose if capture is aborted

## Success Criteria
- CUDAGraphCapture successfully begins and ends capture
- CUDA driver API bindings are correctly defined
- Capture state is properly managed
- Resources are cleaned up on disposal
- Integration with existing CUDAStream type works correctly

## Testing Requirements

### Unit Tests
- Test BeginCapture/EndCapture happy path
- Test double BeginCapture throws exception
- Test EndCapture without BeginCapture throws exception
- Test Dispose while capturing aborts capture
- Test CUDA error handling

### Integration Tests
- Test capture with actual kernel launches (requires GPU)
- Test capture across multiple CUDA streams
